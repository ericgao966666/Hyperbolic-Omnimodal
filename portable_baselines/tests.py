"""CPU regression: manifest identity, native loss gradients, transfer and ranking."""
import ast,copy,json,os,tempfile,uuid
from pathlib import Path
import common as C

def native_worker(rank,rendezvous):
    import torch,torch.distributed as dist
    from torch import nn
    from easydict import EasyDict as E
    from unittest.mock import patch
    from models import native_class
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    class Bert(nn.Module):
        def forward(self,input_ids,attention_mask,encoder_hidden_states):return E(last_hidden_state=encoder_hidden_states.mean(1)[:,None,:].expand(-1,input_ids.shape[1],-1))
    class Match(nn.Module):
        def forward(self,x):s=x.float().sum(-1);return torch.stack((s,-s),-1)
    results=[]
    try:
        for method in ('vast','gram','pmrl'):
            cls=native_class(method);net=cls.__new__(cls);nn.Module.__init__(net)
            net.config=E(alignment_only=True,pmrl_global_negatives=True);net.tau1=.05;net.tau2=.1;net.ret_pairwise_stage1_train=False;net.lambda_itm=.1;net.itm_ratio=.1
            net.contra_temp=nn.Parameter(torch.tensor(.07));net.itm_head=Match();net.multimodal_encoder=E(bert=Bert())
            torch.manual_seed(50+rank);features={k:torch.nn.functional.normalize(torch.randn(4,8),dim=-1).requires_grad_() for k in ('t','v','a','va')}
            b=E(raw_captions=['']*4,caption_tokens=E(input_ids=torch.ones(4,3,dtype=torch.long),attention_mask=torch.ones(4,3,dtype=torch.long)),condition_feats_va=torch.randn(4,2,4),**{'feat_'+k:v for k,v in features.items()})
            warm=net.forward_ret(E(b),'ret%tva',True);net.config.alignment_only=False
            with patch.object(torch.Tensor,'cuda',lambda x,*args,**kwargs:x):full=net.forward_ret(E(b),'ret%tva',True)
            full_alignment=sum(v for k,v in full.items() if k!='loss_itm');warm_alignment=sum(warm.values())
            torch.testing.assert_close(warm_alignment,full_alignment,rtol=0,atol=0)
            params=[*features.values(),net.contra_temp]
            a=torch.autograd.grad(warm_alignment,params,allow_unused=True,retain_graph=True);bgrad=torch.autograd.grad(full_alignment,params,allow_unused=True)
            for x,y in zip(a,bgrad):
                assert (x is None)==(y is None)
                if x is not None:assert torch.isfinite(x).all();torch.testing.assert_close(x,y,rtol=0,atol=0)
            results.append(dict(method=method,alignment_value_and_gradients_equal=True))
        C.write(C.ROOT/'.local'/f'cpu_rank{rank}.json',results)
    finally:dist.destroy_process_group()

def main():
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    import torch
    from torch import nn
    from models import optimizer,lr_factor
    from validation import retrieval_pools,recall
    from inputs import global_batches
    torch.set_num_threads(2);checks=[]
    assert C.protocol()['epochs']==15 and C.protocol()['total_steps']==4830 and not C.protocol()['early_stopping']
    assert C.protocol()['evaluation']['itm_epochs']==[0,*range(5,16)]
    from select_checkpoints import choose
    history=[dict(epoch=6,itm=dict(mean={'R@1':50.}),itc=dict(mean={'R@1':38.})),dict(epoch=7,itm=dict(mean={'R@1':51.}),itc=dict(mean={'R@1':37.})),dict(epoch=15,itm=dict(mean={'R@1':49.}),itc=dict(mean={'R@1':35.}))]
    selected=choose(history);assert selected['retrieval_epoch']==7 and selected['itc_epoch']==6
    checks.append('fixed4830 steps, no early stopping; every-joint-epoch ITM; validation selection differs from final model')
    for p in C.ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    manifest={k:C.manifest(k) for k in C.read(C.ROOT/'manifests/index.json')}
    groups=[]
    for name,m in manifest.items():
        assert len({r['clip_id'] for r in m['videos']})==len(m['videos'])
        assert {q['video_index'] for q in m['queries']}==set(range(len(m['videos'])))
        if name in ('train','val','vast'):groups.append({r['source_video'] for r in m['videos']})
    assert not any(a&b for i,a in enumerate(groups) for b in groups[i+1:]);checks.append('all seven logical manifests, exact counts, caption mapping, no source-video overlap')
    for world in (1,2,4,8):
        parts=[list(global_batches(82608,r,world,6)) for r in range(world)];ref=list(global_batches(82608,0,1,6))
        assert all(sum([parts[r][i] for r in range(world)],[])==row for i,row in enumerate(ref))
    checks.append('same global256 order for world1/2/4/8')
    class Tiny(nn.Module):
        def __init__(self):super().__init__();self.net=nn.Module();self.net.contra_head_t=nn.Linear(3,2);self.net.backbone=nn.Linear(2,2);self.net.backbone.requires_grad_(False)
    model=Tiny();opt=optimizer(model)
    for _ in range(3):opt.zero_grad();model.net.contra_head_t(torch.ones(4,3)).square().mean().backward();opt.step()
    before=copy.deepcopy(opt.state_dict());weights=copy.deepcopy(model.state_dict());model.net.backbone.requires_grad_(True);new=optimizer(model,before)
    assert len(new.state)==len(opt.state)
    for p in model.net.contra_head_t.parameters():
        for k,v in opt.state[p].items():torch.testing.assert_close(new.state[p][k],v,atol=0,rtol=0)
    assert all(p not in new.state for p in model.net.backbone.parameters())
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items());assert lr_factor(1610,'other')==1/322
    checks.append('head moments transfer exactly; no backbone optimizer history; weights unchanged')
    scores=torch.tensor([[4.,1.],[3.,2.],[1.,4.]]);mapping=torch.tensor([0,0,1]);assert recall(retrieval_pools(scores,mapping))['mean']['R@1']==100.
    assert recall(retrieval_pools(torch.tensor([[0.,1.],[1.,0.]]),torch.arange(2),1))['mean']['R@1']==0.
    checks.append('all captions accepted as positives; missing TopK positives remain misses')
    folder=C.ROOT/'.local';folder.mkdir(exist_ok=True)
    # file rendezvous outside long Unix socket path components, inside workspace.
    import tempfile,torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(prefix='.native-tests-',dir=C.ROOT.parent.parent) as tmp:
        mp.spawn(native_worker,args=(str(Path(tmp)/'gloo'),),nprocs=2,join=True)
    checks.append('two-rank author warmup vs joint alignment values and gradients exactly match for all three methods')
    C.write(folder/'CPU_TESTS.json',dict(passed=True,checks=checks,ranks=[C.read(folder/f'cpu_rank{r}.json') for r in range(2)]));print(json.dumps(dict(passed=True,checks=checks),indent=2))
if __name__=='__main__':main()
