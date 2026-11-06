"""5 head epochs +25 joint epochs; exact stage-aware resume and periodic validation."""
import argparse,copy,fcntl,os,random,signal,time,traceback,uuid
from pathlib import Path
import common as C

def worker(rank,args):
    import numpy as np,torch,torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import System,optimizer,active_state,load_active,lr_factor,probe
    from inputs import Pool,tokens,add_text,global_batches,train_loader
    from validation import evaluate
    import amp_update as A
    paths=C.Paths(args.config);world=len(args.devices);device=C.init_rank(rank,args.devices,args.rendezvous,paths)
    folder=paths.out('runs',args.method);folder.mkdir(parents=True,exist_ok=True);identity=paths.identity(args.method,world)
    stop=[False]
    def handle(*_):stop[0]=True
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGINT,handle)
    C.write(folder/f'runtime_rank{rank}.json',dict(pid=os.getpid(),gpu=args.devices[rank],rank=rank,world=world,command=__import__('sys').argv,port=None,time=C.utc()))
    random.seed(50);np.random.seed(50);torch.manual_seed(50);torch.cuda.manual_seed_all(50)
    model=System(paths,args.method).to(device);step=0;phase='warmup';restore=None
    scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    if (folder/'last.pt').exists():
        if not args.resume:raise RuntimeError('Checkpoint exists; pass --resume')
        ck=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if ck['identity']!=identity:raise RuntimeError('Resume identity/world/environment differs')
        step=ck['step'];phase=ck['phase'];model.set_phase(phase);load_active(model,ck['model']);opt=optimizer(model);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);restore=ck['rng'][rank];del ck
    else:
        opt=optimizer(model);random.seed(50+rank);np.random.seed(50+rank);torch.manual_seed(50+rank);torch.cuda.manual_seed_all(50+rank)
    saved_rng=restore or A.rng_capture(device)
    ddp=DDP(model,device_ids=[rank],find_unused_parameters=True,broadcast_buffers=False,gradient_as_bucket_view=True);A.rng_restore(saved_rng,device)
    if rank==0:
        C.write(folder/'identity.json',identity)
        if not (folder/'model_audit.json').exists():C.write(folder/'model_audit.json',dict(load=model.audit,warmup_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups]))
        if args.resume:C.append(folder/'resumes.jsonl',dict(step=step,phase=phase,time=C.utc()))
    ids=tokens(paths,'train');pool=Pool(paths,'train');started=time.monotonic();resumed_step=step
    def checkpoint():
        rng=C.gather(A.rng_capture(device),rank,world)
        if rank==0:
            paths.save(folder/'last.pt',dict(identity=identity,step=step,phase=phase,model=active_state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng))
            C.write(folder/'checkpoint.json',dict(step=step,phase=phase,sha=C.sha(folder/'last.pt'),time=C.utc()))
        dist.barrier()
    def validate_epoch(epoch):
        dest=folder/'validation'/f'epoch_{epoch:03d}.json'
        if C.broadcast(dest.exists() if rank==0 else None):return
        full_itm=epoch in C.protocol()['evaluation']['itm_epochs']
        if rank==0:C.write(folder/'status.json',dict(state='validating',epoch=epoch,step=step,itm=full_itm,time=C.utc()))
        value=evaluate(model,paths,rank,world,itm=full_itm,cache_check=epoch==5)
        if rank==0:
            value.update(epoch=epoch,step=step,phase=phase,identity=identity,time=C.utc());C.write(dest,value)
            print(C.utc(),args.method,'epoch',epoch,'ITC',value['itc']['mean']['R@1'],'ITM',value.get('itm',{}).get('mean',{}).get('R@1'),flush=True)
            if epoch and (epoch<=5 or epoch in C.protocol()['checkpoints']['joint_model_epochs']):
                snap=folder/'snapshots'/f'epoch_{epoch:03d}.pt'
                if not snap.exists():paths.save(snap,dict(identity=identity,epoch=epoch,step=step,phase=phase,model=active_state(model)))
                if full_itm:
                    best=folder/'BEST.json';score=value['itm']['mean']['R@1']
                    if not best.exists() or score>C.read(best)['R1']:C.write(best,dict(epoch=epoch,R1=score,weights=str(snap.relative_to(paths.output_root)),phase=phase,primary_test_epoch=30))
        dist.barrier()
    try:
        if step%322==0:validate_epoch(step//322)
        while step<9660:
            if step>=1610 and phase=='warmup':
                old=copy.deepcopy(opt.state_dict());rng=A.rng_capture(device);del ddp,opt
                model.set_phase('joint');opt=optimizer(model,old)
                mapped=sum(len(s) for s in opt.state.values());del old
                ddp=DDP(model,device_ids=[rank],find_unused_parameters=True,broadcast_buffers=False,gradient_as_bucket_view=True);A.rng_restore(rng,device);phase='joint'
                if rank==0:C.write(folder/'HANDOFF.json',dict(step=step,head_optimizer_states=len(opt.state),joint_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],time=C.utc()))
                checkpoint()
            epoch=step//322+1;ddp.train()
            batches=global_batches(82608,rank,world,epoch,step%322) if phase=='warmup' else train_loader(paths,rank,world,epoch,step%322)
            for raw in batches:
                if step%5==0:
                    free,_=torch.cuda.mem_get_info();reserve=float(paths.resources.get('gpu_reserve_gib',16))*C.GIB
                    bytes_needed=sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)*3+2*C.GIB
                    import shutil
                    bad=stop[0] or free<reserve or (rank==0 and shutil.disk_usage(paths.output_root).free<paths.reserve+bytes_needed)
                    if A.any_rank(bad,device):
                        checkpoint()
                        if rank==0:C.write(folder/'status.json',dict(state='paused_resource_or_signal',step=step,phase=phase,epoch=epoch,time=C.utc()))
                        return 75
                if phase=='warmup':idx=raw;batch=__import__('easydict').EasyDict(pooled=pool.fetch(idx,device))
                else:idx=raw.index.tolist();batch=C.to_device(raw,device)
                add_text(batch,ids,idx,device)
                for g in opt.param_groups:g['lr']=g['initial_lr']*lr_factor(step,g['kind'])
                def retry(event):C.append(folder/f'amp_retry_rank{rank}.jsonl',dict(event,time=C.utc()))
                loss,terms,norm,scaler,grad,amp=A.update_batch(ddp,model,opt,scaler,batch,device,step,probe=probe if step in (0,1610) else None,on_retry=retry)
                step+=1;values=torch.tensor([float(loss.detach()),float(norm),*[float(v.detach()) for v in terms.values()]],device=device);dist.all_reduce(values);values/=world
                if rank==0 and (step==1 or step%10==0 or step==1611):
                    event=dict(state='training',method=args.method,phase=phase,epoch=epoch,step=step,total_steps=9660,time=C.utc(),loss=float(values[0]),grad_norm=float(values[1]),terms=dict(zip(terms,[float(x) for x in values[2:]])),learning_rates={g['kind']:g['lr'] for g in opt.param_groups},amp=amp,gradients=grad,elapsed_seconds=time.monotonic()-started,resumed_step=resumed_step,peak_reserved_gib=torch.cuda.max_memory_reserved()/C.GIB)
                    C.write(folder/'status.json',event);C.append(folder/'training_metrics.jsonl',event);print(C.utc(),args.method,phase,step,event['loss'],flush=True)
                del loss,terms,batch,raw
                if step%322==0:checkpoint();validate_epoch(step//322)
            del batches
        if rank==0:
            final=folder/'final.pt';snapshot=folder/'snapshots/epoch_030.pt'
            if not final.exists():os.link(snapshot,final)
            ck=torch.load(final,map_location='cpu',weights_only=True,mmap=True);actual=model.state_dict()
            if ck['step']!=9660 or ck['identity']!=identity or any(not torch.equal(v,actual[k].detach().cpu()) for k,v in ck['model'].items()):raise RuntimeError('Final checkpoint reload mismatch')
            C.write(folder/'DONE.json',dict(passed=True,step=9660,identity=identity,final_sha=C.sha(final),final_reload_verified=True,time=C.utc()))
            C.write(folder/'status.json',dict(state='completed',step=9660,epoch=30,time=C.utc()))
        dist.barrier();return 0
    except BaseException as exc:
        if isinstance(exc,A.ReplayFailure):checkpoint()
        C.write(folder/f'FAILED_rank{rank}_{os.getpid()}.json',dict(step=step,phase=phase,error=repr(exc),traceback=traceback.format_exc(),time=C.utc()));raise
    finally:dist.destroy_process_group()

def entry(rank,args):
    code=worker(rank,args)
    if code:raise SystemExit(code)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--method',choices=('vast','gram','pmrl'),required=True);p.add_argument('--devices',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.devices=C.parse_devices(a.devices);paths=C.Paths(a.config);paths.setup(','.join(map(str,a.devices)));identity=paths.identity(a.method,len(a.devices))
    proof=paths.out('checks',f'{a.method}_world{len(a.devices)}.json')
    if not proof.exists() or C.read(proof).get('identity')!=identity or not C.read(proof).get('passed'):raise RuntimeError('Current-source real-GPU stage/resume smoke required for this method and world size')
    folder=paths.out('runs',a.method);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():print('Already completed');return
    a.rendezvous=str(Path(os.environ['TMPDIR'])/('train_'+uuid.uuid4().hex))
    import torch.multiprocessing as mp
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:mp.spawn(entry,args=(a,),nprocs=len(a.devices),join=True)
        except mp.ProcessExitedException as exc:
            if exc.exit_code==75:raise SystemExit(75)
            raise
if __name__=='__main__':main()
