"""Bounded local GPU groups; detached jobs are recorded and safely adopted."""
import argparse,fcntl,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import common as C

def process(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
        return dict(pid=int(pid),birth_ticks=int(stat[19]),state=stat[0],command=cmd)
    except (FileNotFoundError,ProcessLookupError):return None

def alive(record):
    p=process(record['pid']);return bool(p and p['state']!='Z' and p['birth_ticks']==record['birth_ticks'] and p['command']==record['command'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--retry-failed',action='store_true');p.add_argument('--groups',required=True,help='Disjoint comma-separated groups, separated by semicolons');a=p.parse_args()
    groups=[C.parse_devices(g) for g in a.groups.split(';')];flat=sum(groups,[])
    if len(set(flat))!=len(flat):raise ValueError('GPU groups overlap')
    paths=C.Paths(a.config);paths.setup('');C.verify_package();jobs={};children={};blocked={};cooldown={};stop=[False]
    def handle(*_):stop[0]=True
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    marker=paths.out('jobs.json')
    if marker.exists():
        old=C.read(marker)
        if old['groups']!=groups:raise RuntimeError('Existing queue uses different groups; preserve its configuration')
        jobs={int(k):v for k,v in old['active'].items()};blocked=old.get('blocked',{})
        if a.retry_failed:
            C.append(paths.out('retry_history.jsonl'),dict(previous_blocked=blocked,time=C.utc()));blocked={}
    def record(state):
        value=dict(state=state,controller_pid=os.getpid(),groups=groups,active={str(k):v for k,v in jobs.items()},blocked=blocked,time=C.utc())
        C.write(marker,value);C.write(paths.out('status.json'),value)
    def done(job):
        method=job['method'];world=len(job['devices'])
        if job['phase']=='smoke':
            f=paths.out('checks',f'{method}_world{world}.json')
            return f.exists() and C.read(f).get('identity')==paths.identity(method,world) and C.read(f).get('passed')
        if job['phase']=='train':
            f=paths.out('runs',method,'DONE.json');return f.exists() and C.read(f).get('identity')==paths.identity(method,world) and C.read(f).get('passed')
        return paths.out('evaluation',method,job['dataset'],'DONE.json').exists()
    def launch(slot,phase,method,dataset=None):
        key=f'{phase}_{method}'+('_'+dataset if dataset else '')
        if key in blocked or time.time()<cooldown.get(key,0):return False
        devices=groups[slot];cmd=[sys.executable,'-u','-B',str(C.ROOT/({'train':'train.py','smoke':'smoke.py','eval':'evaluate.py'}[phase])),'--config',str(paths.config),'--method',method,'--devices',','.join(map(str,devices))]
        if dataset:cmd+=['--dataset',dataset]
        if phase=='train' and paths.out('runs',method,'last.pt').exists():cmd+=['--resume']
        log=paths.out('logs',key+'_'+str(time.time_ns())+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('x') as stream:child=subprocess.Popen(cmd,cwd=C.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process(child.pid)
        if identity is None:raise RuntimeError('Child exited before PID registration; inspect '+str(log))
        job=dict(key=key,phase=phase,method=method,dataset=dataset,devices=devices,log=str(log),started=C.utc(),port=None,**identity)
        jobs[slot]=job;children[slot]=child;C.append(paths.out('launches.jsonl'),dict(event='started',**job));record('running');return True
    with paths.out('supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        C.write(paths.out('supervisor.json'),dict(pid=os.getpid(),command=sys.argv,groups=groups,time=C.utc(),port=None))
        paths.identity('vast',len(groups[0]))
        if not paths.out('cache/READY.json').exists():raise RuntimeError('Run prepare.py and cache.py before starting the queue')
        last_report=0
        try:
            while True:
                for slot,job in list(jobs.items()):
                    child=children.get(slot);code=child.poll() if child else None
                    if (child and code is None) or (not child and alive(job)):continue
                    complete=done(job);jobs.pop(slot);children.pop(slot,None)
                    C.append(paths.out('launches.jsonl'),dict(event='finished',key=job['key'],exit_code=code,complete=complete,time=C.utc()))
                    if not complete:
                        state=paths.out('runs',job['method'],'status.json');paused=job['phase']=='train' and state.exists() and C.read(state).get('state')=='paused_resource_or_signal' and C.read(state).get('time','')>=job['started']
                        text=Path(job['log']).read_text()[-8000:]
                        if code==75 or paused or 'GPU admission:' in text:cooldown[job['key']]=time.time()+60
                        else:blocked[job['key']]=dict(job=job,exit_code=code,time=C.utc())
                if stop[0]:
                    # Keep live children running; restart the supervisor to adopt them.
                    record('supervisor_stopped_children_continue');return
                busy={j['method'] for j in jobs.values() if j['phase'] in ('train','smoke')}
                for slot,devices in enumerate(groups):
                    if slot in jobs:continue
                    candidate=next((m for m in C.protocol()['methods'] if m not in busy and not paths.out('runs',m,'DONE.json').exists() and 'train_'+m not in blocked and 'smoke_'+m not in blocked),None)
                    if candidate:
                        proof=dict(phase='smoke',method=candidate,devices=devices)
                        if launch(slot,'train' if done(proof) else 'smoke',candidate):busy.add(candidate)
                        continue
                    busy_eval={(j['method'],j['dataset']) for j in jobs.values() if j['phase']=='eval'}
                    item=next(((m,d) for m in C.protocol()['methods'] if paths.out('runs',m,'DONE.json').exists() for d in C.protocol()['evaluation']['datasets'] if (m,d) not in busy_eval and not paths.out('evaluation',m,d,'DONE.json').exists() and f'eval_{m}_{d}' not in blocked),None)
                    if item:launch(slot,'eval',*item)
                if time.monotonic()-last_report>60:
                    subprocess.run([sys.executable,'-B',str(C.ROOT/'report.py'),'--config',str(paths.config)],check=True);last_report=time.monotonic()
                count=sum(paths.out('evaluation',m,d,'DONE.json').exists() for m in C.protocol()['methods'] for d in C.protocol()['evaluation']['datasets'])
                if count==15 and not jobs:record('complete');return
                pending_train=any(not paths.out('runs',m,'DONE.json').exists() and 'train_'+m not in blocked and 'smoke_'+m not in blocked for m in C.protocol()['methods'])
                pending_eval=any(paths.out('runs',m,'DONE.json').exists() and not paths.out('evaluation',m,d,'DONE.json').exists() and f'eval_{m}_{d}' not in blocked for m in C.protocol()['methods'] for d in C.protocol()['evaluation']['datasets'])
                if blocked and not jobs and not pending_train and not pending_eval:record('failed');raise RuntimeError('Blocked jobs; inspect preserved logs and jobs.json')
                record('running');time.sleep(5)
        except BaseException as exc:record('failed');C.write(paths.out('ERROR.json'),dict(error=repr(exc),traceback=traceback.format_exc(),time=C.utc()));raise
if __name__=='__main__':main()
