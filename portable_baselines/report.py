"""Live CSV/Markdown summaries; optional standalone validation plots."""
import argparse,csv
import common as C

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();paths=C.Paths(a.config);curves=[];test=[];states=[]
    for method in C.protocol()['methods']:
        folder=paths.out('runs',method)
        if (folder/'status.json').exists():states.append(dict(method=method,**C.read(folder/'status.json')))
        for f in sorted((folder/'validation').glob('epoch_*.json')):
            v=C.read(f);curves.append(dict(method=method,epoch=v['epoch'],step=v['step'],phase=v['phase'],**{'ITC_'+k:x for k,x in v['itc']['mean'].items()},**{'ITM_'+k:v.get('itm',{}).get('mean',{}).get(k) for k in ('R@1','R@5','R@10')}))
        for name in C.protocol()['evaluation']['datasets']:
            folder=paths.out('evaluation',method,name)
            if not (folder/'DONE.json').exists():continue
            d=C.read(folder/'DONE.json')
            if C.sha(folder/'results.json')!=d['results_sha']:raise RuntimeError('Test result changed')
            v=C.read(folder/'results.json');test.append(dict(method=method,dataset=name,selected_epoch=v['epoch'],budget_epochs=15,videos=v['videos'],captions=v['caption_queries'],**{'ITC_'+k:x for k,x in v['itc']['mean'].items()},**{'ITM_'+k:x for k,x in v['itm']['mean'].items()}))
    folder=paths.out('reports');folder.mkdir(exist_ok=True)
    for name,rows in [('validation.csv',curves),('tests.csv',test)]:
        if rows:
            with (folder/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# VAST initialization / fixed15 baselines','',f'Updated: {C.utc()}','',f'Completed training: {sum(paths.out("runs",m,"DONE.json").exists() for m in C.protocol()["methods"])}/3; tests: {len(test)}/15.','',
        '| Method | Phase | Epoch | Step | State |','| --- | --- | ---: | ---: | --- |']
    for s in states:lines.append(f'| {s["method"]} | {s.get("phase", "")} | {s.get("epoch", "")} | {s.get("step", "")} | {s["state"]} |')
    lines+=['','[Validation CSV](validation.csv) · [Final tests CSV](tests.csv)','', 'Validation curves are diagnostic. Training budget is fixed15 with no early stopping; primary test model is validation ITM best. Baselines have no native entailment-cone metric.','']
    (folder/'REPORT.md').write_text('\n'.join(lines));C.write(folder/'summary.json',dict(states=states,validation=curves,tests=test,time=C.utc()))
    if curves:
        try:
            import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        except ImportError:return
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,key in zip(axes,('ITC_R@1','ITM_R@1')):
            for m in C.protocol()['methods']:
                rows=[r for r in curves if r['method']==m and r.get(key) is not None];ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],marker='o',label=m)
            ax.set(xlabel='Total epoch',ylabel=key+' (%)');ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(folder/'validation.png',dpi=180);fig.savefig(folder/'validation.pdf');plt.close(fig)
if __name__=='__main__':main()
