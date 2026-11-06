"""Validation-only native baseline checkpoint selection; training always reaches15."""
from pathlib import Path
import common as C

def choose(rows):
    if not rows:raise ValueError('No joint validation rows')
    ret=max(rows,key=lambda r:(r['itm']['mean']['R@1'],r['itc']['mean']['R@1'],-r['epoch']))
    itc=max(rows,key=lambda r:(r['itc']['mean']['R@1'],r['itm']['mean']['R@1'],-r['epoch']))
    return dict(retrieval_epoch=ret['epoch'],itc_epoch=itc['epoch'],candidate_epochs=[r['epoch'] for r in rows],early_stopping=False)

def update(paths,folder,*,keep_latest=True,final=False):
    folder=Path(folder);rows=[C.read(p) for p in sorted((folder/'validation').glob('epoch_*.json'))];rows=[r for r in rows if 6<=r['epoch']<=15]
    if not rows:return
    d=choose(rows);keep={d['retrieval_epoch'],d['itc_epoch']}
    if keep_latest:keep.add(max(r['epoch'] for r in rows))
    choices={}
    for role,key in [('retrieval','retrieval_epoch'),('itc','itc_epoch')]:
        epoch=d[key];p=folder/'snapshots'/f'epoch_{epoch:03d}.pt';meta=C.read(p.with_suffix('.json'))
        if not p.exists():raise RuntimeError('Selected model missing')
        choices[role]=dict(epoch=epoch,step=epoch*322,checkpoint=str(p),checkpoint_sha=meta['sha'])
        C.write(folder/('BEST_'+role.upper()+'.json'),dict(**choices[role],decision=d))
    C.write(folder/'selection_current.json',dict(**d,retained_epochs=sorted(keep),time=C.utc()))
    if final:
        if set(r['epoch'] for r in rows)!=set(range(6,16)):raise RuntimeError('All ten joint epochs required')
        for c in choices.values():
            if C.sha(c['checkpoint'])!=c['checkpoint_sha']:raise RuntimeError('Selected checkpoint changed')
        C.write(folder/'SELECTION.json',dict(choices=choices,decision=d,budget_epochs=15,training_steps=4830,time=C.utc()))
    for p in sorted((folder/'snapshots').glob('epoch_*.pt')):
        epoch=int(p.stem.split('_')[-1])
        if epoch>5 and epoch not in keep:
            if not p.resolve().is_relative_to(paths.out('runs')) or p.is_symlink():raise ValueError('Only new run-owned snapshots may be pruned')
            C.append(folder/'retention.jsonl',dict(path=str(p),epoch=epoch,reason='cannot become future best ITM or ITC',decision=d,time=C.utc()));p.unlink()
    return d
