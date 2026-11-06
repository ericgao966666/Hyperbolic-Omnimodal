"""Only completed fixed15 validation-selected models may enter the five fixed test cohorts."""
import argparse,fcntl,os,uuid
from pathlib import Path
import common as C

def worker(rank,a):
    import torch,torch.distributed as dist
    from models import System,load_active
    from validation import evaluate
    paths=C.Paths(a.config);device=C.init_rank(rank,a.devices,a.rendezvous,paths)
    try:
        run=paths.out('runs',a.method);done=C.read(run/'DONE.json');final=run/'final.pt'
        if done['identity']!=paths.identity(a.method,done['identity']['world']):raise RuntimeError('Final model provenance differs')
        if done['step']!=4830 or not done['final_reload_verified'] or C.sha(final)!=done['final_sha']:raise RuntimeError('Unverified fixed-final model')
        if C.sha(run/'SELECTION.json')!=done['selection_sha']:raise RuntimeError('Validation selection changed')
        choice=C.read(run/'SELECTION.json')['choices']['retrieval'];selected=Path(choice['checkpoint'])
        if C.sha(selected)!=choice['checkpoint_sha']:raise RuntimeError('Selected model changed')
        model=System(paths,a.method);model.set_phase('joint');ck=torch.load(selected,map_location='cpu',weights_only=True,mmap=True);load_active(model,ck['model']);del ck
        model.to(device).eval().requires_grad_(False);folder=paths.out('evaluation',a.method,a.dataset);folder.mkdir(parents=True,exist_ok=True)
        if (folder/'DONE.json').exists():
            d=C.read(folder/'DONE.json')
            if d['checkpoint_sha']!=choice['checkpoint_sha'] or C.sha(folder/'results.json')!=d['results_sha']:raise RuntimeError('Existing test provenance mismatch')
            return
        result=evaluate(model,paths,rank,len(a.devices),name=a.dataset,itm=True,progress=folder/'progress.json',save_scores=folder/'scores.pt')
        if rank==0:
            result.update(method=a.method,dataset=a.dataset,epoch=choice['epoch'],step=choice['step'],budget_epochs=15,budget_steps=4830,checkpoint_sha=choice['checkpoint_sha'],identity=done['identity'],time=C.utc())
            C.write(folder/'results.json',result);C.write(folder/'DONE.json',dict(passed=True,checkpoint_sha=choice['checkpoint_sha'],results_sha=C.sha(folder/'results.json'),scores_sha=C.sha(folder/'scores.pt'),time=C.utc()))
    finally:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--method',required=True,choices=('vast','gram','pmrl'));p.add_argument('--dataset',required=True,choices=C.protocol()['evaluation']['datasets']);p.add_argument('--devices',required=True);a=p.parse_args()
    a.devices=C.parse_devices(a.devices);paths=C.Paths(a.config);paths.setup(','.join(map(str,a.devices)));C.verify_package()
    folder=paths.out('evaluation',a.method,a.dataset);folder.mkdir(parents=True,exist_ok=True)
    a.rendezvous=str(Path(os.environ['TMPDIR'])/('eval_'+uuid.uuid4().hex))
    import torch.multiprocessing as mp
    with (folder/'eval.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);mp.spawn(worker,args=(a,),nprocs=len(a.devices),join=True)
if __name__=='__main__':main()
