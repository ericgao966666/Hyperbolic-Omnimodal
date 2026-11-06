"""Start the coordinator detached, recording the exact command/PID/log."""
import argparse,fcntl,os,subprocess,sys,time
import common as C

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--groups',required=True);p.add_argument('--retry-failed',action='store_true');a=p.parse_args();paths=C.Paths(a.config);paths.setup('')
    with paths.out('supervisor.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd=[sys.executable,'-u','-B',str(C.ROOT/'run.py'),'--config',str(paths.config),'--groups',a.groups];log=paths.out('logs','supervisor_'+str(time.time_ns())+'.log');log.parent.mkdir(exist_ok=True)
    if a.retry_failed:cmd.append('--retry-failed')
    with log.open('x') as stream:proc=subprocess.Popen(cmd,cwd=C.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    C.write(paths.out('launch.json'),dict(pid=proc.pid,command=cmd,log=str(log),started=C.utc(),port=None));print('PID:',proc.pid,'log:',log)
if __name__=='__main__':main()
