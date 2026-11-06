"""Distributed validation and fixed-final tests with full caption coverage."""
import numpy as np
import torch
import common as C
from inputs import Pool,media_loader,tokens,add_text
from amp_update import rng_capture,rng_restore

def retrieval_pools(score,mapping,k=32):
    if score.dtype!=torch.float32 or not torch.isfinite(score).all():raise RuntimeError('Nonfinite or non-FP32 retrieval scores')
    nt,nv=score.shape
    if len(mapping)!=nt or set(mapping.tolist())!=set(range(nv)):raise RuntimeError('Caption mapping does not cover videos')
    f=score.argsort(dim=-1,descending=True,stable=True)[:,:min(k,nv)];r=score.T.argsort(dim=-1,descending=True,stable=True)[:,:min(k,nt)]
    return dict(t2va=dict(indices=f,itc_scores=score.gather(1,f),exact_positive=f==mapping[:,None]),va2t=dict(indices=r,itc_scores=score.T.gather(1,r),exact_positive=mapping[r]==torch.arange(nv)[:,None]))
def recall(pools,field='itc_scores'):
    result={}
    for direction,p in pools.items():
        order=p[field].argsort(dim=-1,descending=True,stable=True);hits=p['exact_positive'].gather(1,order)
        result[direction]={f'R@{k}':float(hits[:,:k].any(1).double().mean()*100) for k in (1,5,10)}
        result[direction]['candidate_R@32']=float(hits.any(1).double().mean()*100)
    return dict(directions=result,mean={k:sum(x[k] for x in result.values())/2 for k in result['t2va']})

def score_matrix(method,z):
    from models import native_class
    if method=='gram':
        native_class('gram')
        from utils.volume import volume_computation3
    scores=torch.empty(len(z['t']),len(z['va']),dtype=torch.float32)
    for start in range(0,len(scores),256):
        t=z['t'][start:start+256].float()
        scores[start:start+len(t)]=-volume_computation3(t,z['v'].float(),z['a'].float()) if method=='gram' else t@z['va'].float().T
    return scores

@torch.no_grad()
def evaluate(model,paths,rank,world,*,name='val',itm=False,limit=None,cache_check=False,progress=None,save_scores=None):
    device=next(model.parameters()).device;state=rng_capture(device);training=model.training;model.eval()
    m=C.manifest(name);nv=len(m['videos']) if limit is None else min(limit,len(m['videos']))
    full_mapping=np.array([q['video_index'] for q in m['queries']],dtype=np.int64);qidx=np.flatnonzero(full_mapping<nv);mapping=torch.tensor(full_mapping[qidx]);nt=len(qidx)
    ids=tokens(paths,name);media={k:[] for k in ('v','a','va')};texts=[];vi=[];ti=[];raw_v=[];raw_a=[];errors=[]
    try:
        cached=model.phase=='warmup' and not itm
        if cached:
            if name!='val':raise RuntimeError('Frozen validation only supports val')
            pool=Pool(paths,'val');positions=list(range(rank,nv,world))
            for start in range(0,len(positions),256):
                idx=positions[start:start+256]
                with torch.autocast('cuda',dtype=torch.float16):z=model.features(pool.fetch(idx,device))
                vi.extend(idx);ti.extend(idx);texts.append(z['t'].float().cpu())
                for k in media:media[k].append(z[k].float().cpu())
        else:
            pool=Pool(paths,'val') if cache_check else None
            for batch in media_loader(paths,name,rank,world,16 if name=='val' else 8,limit):
                idx=batch.index.tolist();vi.extend(idx);batch=C.to_device(batch,device)
                with torch.autocast('cuda',dtype=torch.float16):
                    v=model.net.batch_get(batch,'vision_output');a=model.net.batch_get(batch,'audio_output')
                    pooled=dict(v=model.net.pool_vision_for_contra(v),a=model.net.pool_audio_for_contra(a))
                    if name=='val':
                        add_text(batch,ids,idx,device);pooled['t']=model.net.pool_text_for_contra(model.net.batch_get(batch,'caption_output'))
                    z=model.features(pooled)
                for k in media:media[k].append(z[k].float().cpu())
                if name=='val':ti.extend(idx);texts.append(z['t'].float().cpu())
                if pool:
                    expected=pool.fetch(idx,device)
                    for k in 'tva':
                        actual=pooled[k].half().float();torch.testing.assert_close(actual,expected[k],rtol=.002,atol=.002)
                        errors.append(float((actual-expected[k]).abs().max()))
                if itm:raw_v.append(v.flatten(1,-2).half().cpu());raw_a.append(a.flatten(1,-2).half().cpu())
                if progress and rank==0:C.write(progress,dict(phase='media',local_completed=len(vi),local_total=len(range(rank,nv,world)),time=C.utc()))
            if name!='val':
                positions=list(range(rank,nt,world))
                for start in range(0,len(positions),64):
                    idx=positions[start:start+64];batch=add_text(__import__('easydict').EasyDict(),ids,qidx[idx],device)
                    with torch.autocast('cuda',dtype=torch.float16):p=model.net.pool_text_for_contra(model.net.batch_get(batch,'caption_output'));z=model.features({'t':p})
                    ti.extend(idx);texts.append(z['t'].float().cpu())
        part=dict(vi=vi,ti=ti,features=dict(t=torch.cat(texts),**{k:torch.cat(v) for k,v in media.items()}),cache_errors=errors)
        all_parts=C.gather(part,rank,world);metrics=None;keys=None;ret=None;scores=None
        if rank==0:
            vindex=sum([p['vi'] for p in all_parts],[]);tindex=sum([p['ti'] for p in all_parts],[])
            if sorted(vindex)!=list(range(nv)) or sorted(tindex)!=list(range(nt)):raise RuntimeError('Missing/duplicate evaluation rows')
            vo=torch.tensor(vindex).argsort();to=torch.tensor(tindex).argsort()
            z={k:torch.cat([p['features'][k] for p in all_parts])[to if k=='t' else vo] for k in ('t','v','a','va')}
            scores=score_matrix(model.method,z);ret=retrieval_pools(scores,mapping)
            metrics=dict(itc=recall(ret),videos=nv,caption_queries=nt,cache_max_abs=max(sum([p['cache_errors'] for p in all_parts],[]),default=0.))
            if itm:
                parts=[]
                for direction,r in ret.items():
                    q=torch.arange(len(r['indices']))[:,None].expand_as(r['indices']);r['keys']=q*nv+r['indices'] if direction=='t2va' else r['indices']*nv+q;parts.append(r['keys'].flatten())
                keys=torch.unique(torch.cat(parts),sorted=True)
        if itm:
            keys=C.broadcast(keys);positions=torch.nonzero((keys%nv)%world==rank).flatten();own=keys[positions]
            v=torch.cat(raw_v);a=torch.cat(raw_a);logits=[]
            for start in range(0,len(own),32):
                key=own[start:start+32];t=key//nv;video=key%nv
                if not torch.equal(torch.tensor(vi)[video//world],video):raise RuntimeError('ITM raw-token owner mismatch')
                vt=v[video//world].to(device).reshape(-1,2,257,1408);at=a[video//world].to(device).reshape(-1,1,256,768)
                text=add_text(__import__('easydict').EasyDict(),ids,qidx[t.numpy()],device).caption_tokens
                with torch.autocast('cuda',dtype=torch.float16):
                    cond=torch.cat((model.net.get_multimodal_forward_input_vision(vt),model.net.get_multimodal_forward_input_audio(at)),1)
                    hidden=model.net.multimodal_encoder.bert(input_ids=text.input_ids,attention_mask=text.attention_mask,encoder_hidden_states=cond).last_hidden_state
                    out=model.net.itm_head(hidden[:,0])
                logits.append((out[:,1].float()-out[:,0].float()).cpu())
                if progress and rank==0 and start%1024==0:C.write(progress,dict(phase='itm',local_completed=start+len(key),local_total=len(own),time=C.utc()))
            parts=C.gather(dict(positions=positions,logits=torch.cat(logits) if logits else torch.empty(0)),rank,world)
            if rank==0:
                out=torch.empty(len(keys));coverage=torch.zeros(len(keys),dtype=torch.long)
                for p in parts:out[p['positions']]=p['logits'];coverage[p['positions']]+=1
                if not (coverage==1).all() or not torch.isfinite(out).all():raise RuntimeError('ITM pair coverage/nonfinite scores')
                for r in ret.values():r['itm_scores']=out[torch.searchsorted(keys,r['keys'])]
                metrics['itm']=recall(ret,'itm_scores');metrics['unique_itm_pairs']=len(keys)
        if save_scores and rank==0:paths.save(save_scores,dict(itc_scores=scores,pools=ret,caption_to_video=mapping))
        return C.broadcast(metrics)
    finally:model.train(training);rng_restore(state,device)
