"""Mandatory real-input, full-global256 stage transition and exact resume test."""
import argparse,copy,fcntl,gc,os,random,uuid
from pathlib import Path
import common as C

def worker(rank,a):
    import numpy as np,torch,torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import System,optimizer,active_state,load_active,lr_factor,probe
    from inputs import Pool,tokens,add_text,global_batches,train_loader
    from validation import evaluate
    import amp_update as A
    paths=C.Paths(a.config);world=len(a.devices);device=C.init_rank(rank,a.devices,a.rendezvous,paths);identity=paths.identity(a.method,world)
    folder=paths.out('checks',a.method+'_world'+str(world));folder.mkdir(parents=True,exist_ok=True)
    try:
        random.seed(50);np.random.seed(50);torch.manual_seed(50);torch.cuda.manual_seed_all(50)
        model=System(paths,a.method).to(device);opt=optimizer(model);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        ddp=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True);pool=Pool(paths,'train');ids=tokens(paths,'train')
        for step,idx in zip(range(3),global_batches(82608,rank,world,1)):
            b=add_text(__import__('easydict').EasyDict(pooled=pool.fetch(idx,device)),ids,idx,device)
            for g in opt.param_groups:g['lr']=g['initial_lr']*lr_factor(step,g['kind'])
            _,_,_,scaler,_,_=A.update_batch(ddp,model,opt,scaler,b,device,step,probe=probe)
        # All warmup optimizer parameters must be heads, with no encoder/ITM state.
        assert all(g['kind']=='head' for g in opt.param_groups)
        before=A.rng_capture(device);warmval=evaluate(model,paths,rank,world,limit=64);after=A.rng_capture(device)
        assert torch.equal(before['torch'],after['torch']) and torch.equal(before['cuda'],after['cuda'])
        old=copy.deepcopy(opt.state_dict());old_count=len(old['state']);del ddp,opt,b
        model.set_phase('joint');opt=optimizer(model,old);assert len(opt.state)==old_count;del old
        ddp=DDP(model,device_ids=[rank],find_unused_parameters=True,broadcast_buffers=False,gradient_as_bucket_view=True)
        raw=evaluate(model,paths,rank,world,itm=True,limit=64,cache_check=True)
        for k in ('R@1','R@5','R@10'):assert abs(raw['itc']['mean'][k]-warmval['itc']['mean'][k])<=.2
        loader=train_loader(paths,rank,world,6);iterator=iter(loader)
        for i in range(3):
            b=next(iterator);idx=b.index.tolist();b=add_text(C.to_device(b,device),ids,idx,device)
            for g in opt.param_groups:g['lr']=g['initial_lr']*lr_factor(1610+i,g['kind'])
            _,_,_,scaler,grad,_=A.update_batch(ddp,model,opt,scaler,b,device,1610+i,probe=probe)
            if i==1:
                rng=C.gather(A.rng_capture(device),rank,world)
                if rank==0:paths.save(folder/'roundtrip.pt',dict(model=active_state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=rng))
                dist.barrier()
        expected=active_state(model);ck=torch.load(folder/'roundtrip.pt',map_location='cpu',weights_only=False,mmap=True)
        # Reconstruct exactly as a process restart, releasing old Adam/DDP storage.
        del ddp,opt,model;gc.collect();torch.cuda.empty_cache()
        model=System(paths,a.method).to(device);model.set_phase('joint');load_active(model,ck['model'])
        opt=optimizer(model);opt.load_state_dict(ck['optimizer']);scaler=torch.amp.GradScaler('cuda',init_scale=1024.);scaler.load_state_dict(ck['scaler'])
        ddp=DDP(model,device_ids=[rank],find_unused_parameters=True,broadcast_buffers=False,gradient_as_bucket_view=True);A.rng_restore(ck['rng'][rank],device)
        for g in opt.param_groups:g['lr']=g['initial_lr']*lr_factor(1612,g['kind'])
        A.update_batch(ddp,model,opt,scaler,b,device,1612)
        current=model.state_dict()
        if any(not torch.equal(v,current[k].detach().cpu()) for k,v in expected.items()):raise RuntimeError('GPU resume was not bitwise equal')
        results=C.gather(dict(rank=rank,gpu=a.devices[rank],passed=True,resume_bitwise_equal=True,cache_max_abs=raw['cache_max_abs'],gradients=grad,peak_reserved_gib=torch.cuda.max_memory_reserved()/C.GIB,initialization=model.audit),rank,world)
        if rank==0:C.write(paths.out('checks',f'{a.method}_world{world}.json'),dict(passed=True,identity=identity,global_batch=256,results=results,time=C.utc()))
        dist.barrier()
        del ck,expected,current,iterator,loader
        if rank==0:
            # This exact scratch checkpoint was created by this smoke and has no training result.
            p=folder/'roundtrip.pt';C.append(paths.out('cleanup.jsonl'),dict(path=str(p),reason='completed smoke scratch',time=C.utc()));p.unlink()
        dist.barrier()
    finally:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--method',required=True,choices=('vast','gram','pmrl'));p.add_argument('--devices',required=True);a=p.parse_args()
    a.devices=C.parse_devices(a.devices);paths=C.Paths(a.config);paths.setup(','.join(map(str,a.devices)));C.verify_package();a.rendezvous=str(Path(os.environ['TMPDIR'])/('smoke_'+uuid.uuid4().hex))
    import torch.multiprocessing as mp
    with paths.out(f'smoke_{a.method}.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);mp.spawn(worker,args=(a,),nprocs=len(a.devices),join=True)
if __name__=='__main__':main()
