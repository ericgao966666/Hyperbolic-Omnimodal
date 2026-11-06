"""Same VAST initialization; native objectives with explicit stage boundaries."""
import copy,hashlib,importlib,importlib.util,math,os,sys,types
import torch
from torch import nn
from torch.nn import functional as F
from easydict import EasyDict as E
import common as C

HEADS={'vast':('contra_head_t','contra_head_va','contra_temp'),'gram':('contra_head_t','contra_head_v','contra_head_a','contra_temp'),'pmrl':('contra_head_t','contra_head_v','contra_head_a')}
def native_class(method):
    if method=='pmrl':
        from model.pmrl import PMRL
        return PMRL
    source=C.ROOT/'vendor/authors'/method.upper()
    if method=='gram':
        spec=importlib.util.spec_from_file_location('utils.volume',source/'utils/volume.py');m=importlib.util.module_from_spec(spec);sys.modules['utils.volume']=m;spec.loader.exec_module(m)
    name='native_author_'+method;p=types.ModuleType(name);p.__path__=[str(source/'model')];sys.modules[name]=p
    return getattr(importlib.import_module(name+'.'+method),method.upper())

def checkpoint_policy():
    import torch.utils.checkpoint as cp
    original=getattr(cp.checkpoint,'_native_original',cp.checkpoint)
    def wrapped(function,*args,**kwargs):kwargs['use_reentrant']=False;return original(function,*args,**kwargs)
    wrapped._native_original=original;cp.checkpoint=wrapped
    for name,module in list(sys.modules.items()):
        if name.startswith(('native_author_','model.')) and callable(getattr(module,'checkpoint',None)):module.checkpoint=wrapped

def parameter_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class System(nn.Module):
    def __init__(self,paths,method):
        super().__init__();self.method=method;self.phase='warmup';cfg=C.read(C.ROOT/'configs/model.json');cfg.update(alignment_only=True,tau1=.05,tau2=.1,pmrl_global_negatives=True)
        work=paths.out('constructor');work.mkdir(exist_ok=True);link=work/'pretrained_weights'
        if link.exists() or link.is_symlink():
            if link.resolve()!=paths.assets_root:raise RuntimeError('Constructor assets changed')
        else:
            try:link.symlink_to(paths.assets_root,target_is_directory=True)
            except FileExistsError:
                if link.resolve()!=paths.assets_root:raise
        cls=native_class(method);previous=os.getcwd();original=torch.load
        def cpu_load(*args,**kwargs):kwargs.setdefault('map_location','cpu');kwargs.setdefault('weights_only',False);return original(*args,**kwargs)
        try:
            os.chdir(work);torch.load=cpu_load;self.net=cls(E(cfg))
        finally:os.chdir(previous);torch.load=original
        checkpoint_policy()
        for module in self.net.modules():module.subset_non_reentrant=True
        raw=torch.load(paths.vast_checkpoint,map_location='cpu',weights_only=False,mmap=True)
        state=self.net.modify_checkpoint(raw.get('model',raw));missing,unexpected=self.net.load_state_dict(state,strict=False)
        allowed={'vision_encoder.logit_scale'}|({'contra_head_d.linear.weight'} if method=='gram' else set())
        if set(missing)-allowed or set(unexpected)-{'vision_encoder.text.logit_scale'}:raise RuntimeError(f'VAST initialization mismatch: {missing}, {unexpected}')
        actual=self.net.state_dict()
        for name,value in state.items():
            if name in actual and not torch.equal(actual[name],value):raise RuntimeError('Loaded tensor differs: '+name)
        self.audit=dict(checkpoint_sha=C.protocol()['initialization']['sha256'],missing=list(missing),unexpected=list(unexpected),
            shared_loaded_sha=parameter_digest({k:actual[k] for k in state if k in actual}),loaded_keys=len(state))
        if self.audit['shared_loaded_sha']!=C.protocol()['initialization']['shared_loaded_sha']:raise RuntimeError('Shared VAST parameter fingerprint differs from source')
        del raw,state,actual
        self.set_phase('warmup')
    def set_phase(self,phase):
        self.phase=phase;self.net.requires_grad_(False)
        for name in HEADS[self.method]:getattr(self.net,name).requires_grad_(True)
        if phase=='joint':
            for module in (self.net.vision_encoder.visual,self.net.audio_encoder,self.net.multimodal_encoder.bert,self.net.itm_head,self.net.hidden_trans_vision_multimodal,self.net.hidden_trans_audio_multimodal):module.requires_grad_(True)
            for name in ('vision_frame_embedding','audio_frame_embedding','vision_type_embeddings','audio_type_embeddings'):getattr(self.net,name).requires_grad_(True)
            if self.net.multimodal_encoder.bert.pooler is not None:self.net.multimodal_encoder.bert.pooler.requires_grad_(False)
        self.net.config.alignment_only=phase=='warmup';self.train(self.training)
    def train(self,mode=True):
        super().train(mode)
        if self.phase=='warmup':self.net.eval()
        return self
    def pooled(self,b):
        return dict(t=self.net.pool_text_for_contra(self.net.batch_get(b,'caption_output')),v=self.net.pool_vision_for_contra(self.net.batch_get(b,'vision_output')),a=self.net.pool_audio_for_contra(self.net.batch_get(b,'audio_output')))
    def features(self,p):
        z={k:F.normalize(getattr(self.net,'contra_head_'+k)(p[k]),dim=-1) for k in 'tva' if k in p}
        if 'v' in p and 'a' in p:z['va']=F.normalize(self.net.contra_head_va(torch.cat((p['v'],p['a']),-1)),dim=-1)
        return z
    def forward(self,b):
        if self.phase=='warmup':
            z=self.features(b['pooled'])
            b=E(b);b.update({'feat_'+k:v for k,v in z.items()})
        return self.net.forward_ret(b,'ret%tva',compute_loss=True)
    def clamp_parameters(self):pass

def groups(model):
    out={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        kind='vision' if name.startswith('net.vision_encoder.visual.') else 'head' if name.startswith(('net.contra_head_','net.contra_temp')) else 'other'
        wd=.01 if p.ndim>=2 and not name.endswith('.bias') else 0.;key=(kind,wd)
        out.setdefault(key,dict(params=[],names=[],kind=kind,weight_decay=wd,lr={'vision':5e-7,'head':1e-4,'other':2e-5}[kind]))
        out[key]['params'].append(p);out[key]['names'].append(name)
    for g in out.values():g['initial_lr']=g['lr']
    return list(out.values())

def optimizer(model,old=None):
    opt=torch.optim.AdamW(groups(model),betas=(.9,.98),foreach=False)
    if old is not None:
        named={name:old['state'][idx] for g in old['param_groups'] for name,idx in zip(g['names'],g['params']) if idx in old['state']}
        state=opt.state_dict();matched=set()
        for g in state['param_groups']:
            for name,idx in zip(g['names'],g['params']):
                if name in named:state['state'][idx]=copy.deepcopy(named[name]);matched.add(name)
        if matched!=set(named):raise RuntimeError('Optimizer named transfer incomplete')
        opt.load_state_dict(state)
    return opt

def lr_factor(step,kind):
    if kind=='head':return (step+1)/483 if step<483 else .5*(1+math.cos(math.pi*(step-483)/(9660-483)))
    j=step-1610
    return (j+1)/322 if j<322 else .5*(1+math.cos(math.pi*(j-322)/(8050-322)))

def active_state(model):
    keys={n for n,p in model.named_parameters() if p.requires_grad}
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in keys}
def load_active(model,state):
    expected={n for n,p in model.named_parameters() if p.requires_grad}
    if set(state)!=expected:raise RuntimeError('Active model checkpoint keys mismatch')
    model.load_state_dict(state,strict=False)
def probe(model):
    result={}
    for g in groups(model):
        name=g['kind'];result.setdefault(name,0.)
        result[name]+=sum(float(p.grad.detach().float().square().sum()) for p in g['params'] if p.grad is not None)
    if not all(math.isfinite(v) and v>0 for v in result.values()):raise RuntimeError('Missing/nonfinite active gradients: '+str(result))
    return {k:v**.5 for k,v in result.items()}
