"""CPU audit: all native methods load the identical VAST shared parameters."""
import argparse,gc
import common as C

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();paths=C.Paths(a.config);paths.setup('')
    if C.sha(paths.vast_checkpoint)!=C.protocol()['initialization']['sha256']:raise RuntimeError('Wrong VAST checkpoint')
    import torch
    from models import System,HEADS
    torch.set_num_threads(2);results=[]
    for method in C.protocol()['methods']:
        torch.manual_seed(50);model=System(paths,method)
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        if any(not any(n=='net.'+h or n.startswith('net.'+h+'.') for h in HEADS[method]) for n in names):raise RuntimeError('Unexpected warmup parameters')
        row=dict(method=method,**model.audit,warmup_names=names);results.append(row);print(method,row,flush=True);del model;gc.collect()
    if len({r['shared_loaded_sha'] for r in results})!=1:raise RuntimeError('Common VAST parameters differ across models')
    C.write(paths.out('checks/CPU_MODELS.json'),dict(passed=True,results=results,package_sha=C.verify_package(),time=C.utc()))
if __name__=='__main__':main()
