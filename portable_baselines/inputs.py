"""Fixed cohorts and unchanged deterministic two-frame/one-audio-window inputs."""
import atexit,math
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from easydict import EasyDict as E
import common as C
from data.bounded_video import BoundedVideoReader
from data.audio_mapper import load_audio
from PIL import Image
from torchvision.transforms import functional as VF,InterpolationMode
import torchaudio

class Media(Dataset):
    def __init__(self,paths,name):
        self.paths=paths;self.name=name;self.m=C.manifest(name);self.reader=None
        self.stats=C.read(paths.out('inputs',name,'media_stats.json'))
    def __len__(self):return len(self.m['videos'])
    def __getitem__(self,i):
        r=self.m['videos'][i];files={k:self.paths.media(self.m,r,k) for k in ('video','audio')}
        for k,p in files.items():
            s=p.stat()
            if [s.st_size,s.st_mtime_ns]!=self.stats[i][k]:raise RuntimeError('Media changed after preparation: '+str(p))
        if self.reader is None:
            self.reader=BoundedVideoReader(timeout=30);atexit.register(self.reader.close)
        frames=self.reader.read(files['video'],lambda n,fps:[min(n-1,int((j+.5)*n/2)) for j in range(2)])
        h,w=frames.shape[1:3];scale=256/min(h,w)
        frames=np.stack([np.asarray(Image.fromarray(x).resize((round(w*scale),round(h*scale)),Image.Resampling.BILINEAR)) for x in frames])
        pixels=torch.from_numpy(frames.transpose(0,3,1,2).copy()).float()/255
        pixels=VF.resize(pixels,[224,224],InterpolationMode.BILINEAR,antialias=True)
        pixels=VF.normalize(pixels,[.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711])
        waveform,sr=load_audio(files['audio'])
        if sr!=16000:waveform=torchaudio.transforms.Resample(sr,16000)(waveform)
        bank=torchaudio.compliance.kaldi.fbank(waveform*2**15,num_mel_bins=64,sample_frequency=16000,frame_length=25,frame_shift=10,channel=0)
        bank=(bank-15.41663)/(6.55582*2)
        if not len(bank) or not torch.isfinite(bank).all():raise RuntimeError('Invalid audio: '+str(files['audio']))
        pad=max(1024-len(bank),1024-len(bank)%1024);selected=((len(bank)+pad)//1024-1)//2
        bank=bank[selected*1024:(selected+1)*1024];bank=torch.nn.functional.pad(bank,(0,0,0,1024-len(bank)))
        if pixels.shape!=(2,3,224,224) or bank.shape!=(1024,64) or not torch.isfinite(pixels).all():raise RuntimeError('Invalid media shape')
        return dict(index=i,vision_pixels=pixels,audio_spectrograms=bank.unsqueeze(0))

def collate(rows):return E({k:torch.stack([torch.as_tensor(r[k]) for r in rows]) for k in rows[0]})
def tokens(paths,name):
    return {k:np.load(paths.out('inputs',name,k+'.npy'),mmap_mode='r') for k in ('input_ids','attention_mask')}
def add_text(batch,token_arrays,indices,device):
    batch.caption_tokens=E({k:torch.tensor(np.array(v[indices],copy=True),dtype=torch.long,device=device) for k,v in token_arrays.items()})
    batch.raw_captions=['']*len(indices);return batch

def global_batches(size,rank,world,epoch,start=0):
    order=torch.randperm(size,generator=torch.Generator().manual_seed(50+epoch)).tolist();width=256//world
    for j in range(start,size//256):yield order[j*256+rank*width:j*256+(rank+1)*width]

def train_loader(paths,rank,world,epoch,start=0):
    ds=Media(paths,'train');workers=int(paths.resources.get('workers_per_rank',2))
    return DataLoader(ds,batch_sampler=list(global_batches(len(ds),rank,world,epoch,start)),collate_fn=collate,num_workers=workers,pin_memory=True,
        generator=torch.Generator().manual_seed(50+epoch),**(dict(multiprocessing_context='spawn',timeout=240,prefetch_factor=2) if workers else {}))
def media_loader(paths,name,rank,world,batch_size=16,limit=None):
    ds=Media(paths,name);n=len(ds) if limit is None else min(limit,len(ds));positions=list(range(rank,n,world));workers=int(paths.resources.get('workers_per_rank',2))
    return DataLoader(Subset(ds,positions),batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=workers,pin_memory=True,
        generator=torch.Generator().manual_seed(991),**(dict(multiprocessing_context='spawn',timeout=240) if workers else {}))

class Pool:
    def __init__(self,paths,name):
        receipt=C.read(paths.out('cache/READY.json'))
        if receipt['prepared_sha']!=C.sha(paths.out('PREPARED.json')) or receipt['package_sha']!=C.verify_package():raise RuntimeError('Cache provenance changed')
        for k in 'tva':
            stat=paths.out('cache',name,k+'.npy').stat()
            if [stat.st_size,stat.st_mtime_ns]!=receipt['stats'][name+'/'+k+'.npy']:raise RuntimeError('Cache file changed')
        self.arrays={k:np.load(paths.out('cache',name,k+'.npy'),mmap_mode='r') for k in 'tva'}
    def fetch(self,idx,device):return {k:torch.from_numpy(np.array(v[idx],copy=True)).float().to(device) for k,v in self.arrays.items()}
