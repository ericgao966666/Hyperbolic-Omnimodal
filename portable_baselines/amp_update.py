"""Bounded AMP retry: identical minibatch/RNG, exactly one optimizer update."""
import copy
import random
import numpy as np
import torch
import torch.distributed as dist


class ReplayFailure(FloatingPointError):
    """All ranks agreed to stop before any optimizer update."""


def rng_capture(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def rng_restore(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],device)


def any_rank(value,device):
    flag=torch.tensor(int(value),device=device)
    if dist.is_initialized():dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag.item())


def fresh_lower_scaler(scaler):
    state=scaler.state_dict();state['scale']=state['scale']*state['backoff_factor'];state['_growth_tracker']=0
    # A fresh object discards UNSCALED per-optimizer state without pretending
    # that an optimizer update happened, or touching GradScaler private fields.
    result=torch.amp.GradScaler('cuda',init_scale=state['scale'],growth_factor=state['growth_factor'],
        backoff_factor=state['backoff_factor'],growth_interval=state['growth_interval'])
    result.load_state_dict(state)
    return result


def update_batch(ddp,model,optimizer,scaler,raw_batch,device,step,*,probe=None,on_retry=None,max_retries=8,min_scale=1e-4):
    initial_rng=rng_capture(device)
    buffers={name:value.detach().clone() for name,value in model.named_buffers()}
    rank=dist.get_rank() if dist.is_initialized() else 0
    scale_start=scaler.get_scale()

    def restore():
        rng_restore(initial_rng,device)
        with torch.no_grad():
            for name,value in model.named_buffers():value.copy_(buffers[name])

    for attempt in range(max_retries+1):
        if attempt:restore()
        attempt_scale=scaler.get_scale()
        optimizer.zero_grad(set_to_none=True)
        # Author batch_get caches encoded tensors in the dictionary. Each retry
        # must get a fresh top-level mapping so the graph is recomputed.
        batch=type(raw_batch)(raw_batch)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            terms=ddp(batch)
        loss=sum(terms.values())
        if any_rank(not bool(torch.isfinite(loss)),device):
            restore();optimizer.zero_grad(set_to_none=True)
            raise ReplayFailure('Nonfinite forward loss; changing loss scale cannot repair the forward pass')
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        bad=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if any_rank(bool(bad),device):
            current=scaler.get_scale()
            if dist.is_initialized():
                lo=torch.tensor(current,device=device);hi=lo.clone()
                dist.all_reduce(lo,op=dist.ReduceOp.MIN);dist.all_reduce(hi,op=dist.ReduceOp.MAX)
                if float(lo)!=float(hi):
                    restore();optimizer.zero_grad(set_to_none=True)
                    raise ReplayFailure('Loss scales differ across ranks')
            retry=attempt<max_retries and current*scaler.get_backoff_factor()>=min_scale
            event=dict(step_before_update=step,attempted_update=step+1,attempt=attempt,rank=rank,
                scale=current,next_scale=current*scaler.get_backoff_factor() if retry else None,
                nonfinite_parameter_count=len(bad),nonfinite_parameter_names=bad[:30],
                loss=float(loss.detach()),retry=retry,optimizer_updates_on_failed_attempt=0)
            if on_retry is not None:on_retry(event)
            del batch,terms,loss
            if not retry:
                restore();optimizer.zero_grad(set_to_none=True)
                raise ReplayFailure('Nonfinite gradient persisted through bounded loss-scale backoff')
            scaler=fresh_lower_scaler(scaler)
            continue
        diagnostics=probe(model) if probe else None
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        if any_rank(not bool(torch.isfinite(norm)),device):
            restore();optimizer.zero_grad(set_to_none=True)
            raise ReplayFailure('Nonfinite unscaled gradient norm')
        scaler.step(optimizer);scaler.update();model.clamp_parameters()
        return loss,terms,norm,scaler,diagnostics,dict(retries=attempt,scale_start=scale_start,scale_used=attempt_scale,scale_next=scaler.get_scale())
    raise AssertionError('Unreachable retry loop exit')
