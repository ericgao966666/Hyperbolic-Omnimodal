"""One shared, exact-checkpoint frozen VAST pool for all three head stages."""
import argparse,uuid
from pathlib import Path
import common as C

def worker(rank,args):
    import numpy as np,torch,torch.distributed as dist
    from models import System
    from inputs import media_loader,tokens,add_text
    paths=C.Paths(args.config);device=C.init_rank(rank,args.devices,args.rendezvous,paths);world=len(args.devices)
    torch.manual_seed(50);model=System(paths,'vast').to(device).eval().requires_grad_(False)
    try:
        for name in ('train','val'):
            n=len(C.manifest(name)['videos']);positions=list(range(rank,n,world));arrays={k:np.empty((len(positions),d),dtype=np.float16) for k,d in [('t',768),('v',1408),('a',768)]};cursor=0;ids=tokens(paths,name)
            with torch.no_grad():
                for batch in media_loader(paths,name,rank,world,16):
                    idx=batch.index.tolist();batch=C.to_device(batch,device);add_text(batch,ids,idx,device)
                    with torch.autocast('cuda',dtype=torch.float16):pooled=model.pooled(batch)
                    for k in arrays:
                        value=pooled[k].half().cpu().numpy()
                        if not np.isfinite(value).all():raise RuntimeError('Nonfinite frozen features')
                        arrays[k][cursor:cursor+len(idx)]=value
                    cursor+=len(idx)
                    if rank==0 and cursor%256==0:print(name,cursor,'/',len(positions),flush=True)
            if cursor!=len(positions):raise RuntimeError('Incomplete feature cache')
            folder=paths.out('cache/shards',name);folder.mkdir(parents=True,exist_ok=True)
            for k,a in arrays.items():np.save(folder/f'{k}_rank{rank}.npy',a,allow_pickle=False)
            dist.barrier()
            if rank==0:
                dest=paths.out('cache',name);dest.mkdir(exist_ok=True)
                for k,d in [('t',768),('v',1408),('a',768)]:
                    full=np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype=np.float16,shape=(n,d))
                    for r in range(world):full[r::world]=np.load(folder/f'{k}_rank{r}.npy',mmap_mode='r')
                    full.flush();del full
            dist.barrier()
        if rank==0:
            files={str(p.relative_to(paths.out('cache'))):C.sha(p) for name in ('train','val') for p in paths.out('cache',name).glob('*.npy')}
            C.write(paths.out('cache/READY.json'),dict(passed=True,prepared_sha=C.sha(paths.out('PREPARED.json')),package_sha=C.verify_package(),world=world,batch_per_rank=16,files=files,stats={rel:[paths.out('cache',rel).stat().st_size,paths.out('cache',rel).stat().st_mtime_ns] for rel in files}))
    finally:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--devices',required=True);a=p.parse_args();a.devices=C.parse_devices(a.devices)
    paths=C.Paths(a.config);paths.setup(','.join(map(str,a.devices)));C.verify_package()
    if not paths.out('PREPARED.json').exists():raise RuntimeError('Run prepare first')
    paths.identity('vast',len(a.devices))
    if paths.out('cache/READY.json').exists():
        r=C.read(paths.out('cache/READY.json'))
        if r['prepared_sha']!=C.sha(paths.out('PREPARED.json')) or r['package_sha']!=C.verify_package():raise RuntimeError('Existing cache belongs to different inputs/code')
        for rel,expected in r['files'].items():
            if C.sha(paths.out('cache',rel))!=expected:raise RuntimeError('Existing cache changed')
        print('Shared cache already verified');return
    paths.guard(3*C.GIB);a.rendezvous=str(Path(__import__('os').environ['TMPDIR'])/('cache_'+uuid.uuid4().hex))
    import torch.multiprocessing as mp
    with paths.out('cache.lock').open('a') as f:
        __import__('fcntl').flock(f,__import__('fcntl').LOCK_EX|__import__('fcntl').LOCK_NB);mp.spawn(worker,args=(a,),nprocs=len(a.devices),join=True)
if __name__=='__main__':main()
