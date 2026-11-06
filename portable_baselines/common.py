"""Portable, explicit resource roots and immutable experiment identity."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,sys,tempfile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/code'))
GIB=2**30

def read(p):
    p=Path(p)
    if p.suffix=='.gz':
        import gzip
        return json.loads(gzip.decompress(p.read_bytes()).decode('utf-8'))
    return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def append(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def protocol():return read(ROOT/'protocol.json')
def verify_package():
    files=read(ROOT/'PACKAGE_SHA256.json')
    for rel,expected in files.items():
        p=(ROOT/rel).resolve()
        if not p.is_relative_to(ROOT) or sha(p)!=expected:raise RuntimeError('Package changed: '+rel)
    return digest(files)

def manifest(name):
    path=ROOT/'manifests'/f'{name}.json'
    m=read(path if path.exists() else path.with_suffix('.json.gz'))
    if 'shards' in m:
        videos=[];queries=[]
        for rel in m['shards']:
            part=read(ROOT/rel);offset=len(videos);videos.extend(part['videos'])
            queries.extend(dict(q,video_index=q['video_index']+offset) for q in part['queries'])
        m=dict(m,videos=videos,queries=queries)
    index=read(ROOT/'manifests/index.json')[name]
    if len(m['videos'])!=index['videos'] or len(m['queries'])!=index['queries'] or digest(dict(videos=m['videos'],queries=m['queries']))!=index['logical_sha256']:
        raise RuntimeError('Manifest identity changed: '+name)
    return m

class Paths:
    def __init__(self,config):
        self.config=Path(config).resolve();self.cfg=read(self.config)
        for key in ('data_root','output_root','assets_root','vast_checkpoint'):
            p=Path(self.cfg[key])
            if not p.is_absolute() or '/ABSOLUTE' in str(p):raise ValueError('Set absolute '+key)
            setattr(self,key,p.resolve())
        if self.output_root==self.data_root or not self.output_root.is_relative_to(self.data_root):raise ValueError('Output must be a dedicated child of data_root')
        for d,roots in self.cfg['datasets'].items():
            for kind in ('video','audio'):
                p=Path(roots[kind]).resolve()
                if not p.is_relative_to(self.data_root):raise ValueError('Dataset outside data_root: '+str(p))
                if self.output_root==p or self.output_root.is_relative_to(p) or p.is_relative_to(self.output_root):raise ValueError('Output and media directories overlap')
        if not self.assets_root.is_relative_to(self.data_root) or not self.vast_checkpoint.is_relative_to(self.data_root):raise ValueError('Weights/assets must be under data_root')
        self.resources=self.cfg.get('resources',{})
        self.reserve=float(self.resources.get('disk_reserve_gib',40))*GIB
    def out(self,*parts):
        p=self.output_root.joinpath(*parts)
        if not p.resolve().is_relative_to(self.output_root):raise ValueError('Output path escapes output_root')
        return p
    def media(self,m,row,kind):
        base=Path(self.cfg['datasets'][m['media_dataset']][kind]).resolve()
        filename=row[kind+'_file']
        if Path(filename).name!=filename or filename in ('','.','..'):raise ValueError('Unsafe media filename')
        p=(base/filename).resolve()
        if not p.is_relative_to(base):raise ValueError('Media symlink escapes dataset directory')
        return p
    def setup(self,devices=''):
        self.out().mkdir(parents=True,exist_ok=True)
        temp=self.data_root/('.tmp-native-'+digest(str(self.output_root))[:10]);temp.mkdir(exist_ok=True)
        os.environ.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=devices,
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(temp),
            HF_HOME=str(self.data_root/'.cache/huggingface'),TORCH_HOME=str(self.data_root/'.cache/torch'))
    def identity(self,method,world):
        prep=read(self.out('PREPARED.json'))
        if prep['package_sha']!=verify_package() or prep['config_sha']!=sha(self.config) or prep['runtime_versions']!=runtime_versions():raise RuntimeError('Prepared package/config/environment changed')
        stat=self.vast_checkpoint.stat()
        if [stat.st_size,stat.st_mtime_ns]!=prep['checkpoint_stat']:raise RuntimeError('VAST checkpoint changed after preparation')
        for name,entry in prep['inputs'].items():
            for filename,expected in entry['files'].items():
                if sha(self.out('inputs',name,filename))!=expected:raise RuntimeError('Prepared tokens/media bindings changed')
        for name,expected in prep['asset_stats'].items():
            stat=(self.assets_root/name).stat()
            if [stat.st_size,stat.st_mtime_ns]!=expected:raise RuntimeError('Constructor asset changed')
        return dict(method=method,world=world,seed=50,global_batch=256,package_sha=verify_package(),
                    prepared_sha=sha(self.out('PREPARED.json')),protocol_sha=sha(ROOT/'protocol.json'),
                    checkpoint_sha=prep['checkpoint_sha'],runtime_versions=runtime_versions())
    def guard(self,extra=0):
        if shutil.disk_usage(self.output_root).free<self.reserve+extra:raise RuntimeError('Insufficient disk reserve')
    def save(self,path,value):
        import torch
        path=self.out(Path(path).relative_to(self.output_root));path.parent.mkdir(parents=True,exist_ok=True)
        with self.out('checkpoint_writer.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);self.guard(tensor_bytes(value)+GIB)
            tmp=path.with_name(path.name+f'.{os.getpid()}.partial');torch.save(value,tmp);os.replace(tmp,path)

def tensor_bytes(x):
    import torch
    if torch.is_tensor(x):return x.numel()*x.element_size()
    if isinstance(x,dict):return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x,(tuple,list)):return sum(map(tensor_bytes,x))
    return 0

def runtime_versions():
    import importlib.metadata as md
    return {k:md.version(k) for k in ('torch','torchvision','torchaudio','numpy','transformers','decord')}
def to_device(x,device):
    import torch
    from easydict import EasyDict
    if torch.is_tensor(x):return x.to(device,non_blocking=True)
    if isinstance(x,dict):return EasyDict({k:to_device(v,device) for k,v in x.items()})
    return x

def init_rank(rank,devices,rendezvous,paths):
    import torch,torch.distributed as dist
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    free,total=torch.cuda.mem_get_info();reserve=float(paths.resources.get('gpu_reserve_gib',16))*GIB
    cap=min(float(paths.resources.get('gpu_memory_cap_gib',64))*GIB,total-reserve-GIB)
    if cap<8*GIB or free<cap+reserve:raise RuntimeError('GPU admission: insufficient free memory; use an idle group or adjust resource caps and rerun smoke')
    torch.cuda.set_per_process_memory_fraction(cap/total)
    dist.init_process_group('nccl',init_method='file://'+rendezvous,rank=rank,world_size=len(devices),timeout=datetime.timedelta(minutes=30))
    return torch.device('cuda',rank)

def gather(value,rank,world):
    import torch.distributed as dist
    out=[None]*world if rank==0 else None;dist.gather_object(value,out,dst=0);return out

def broadcast(value):
    import torch.distributed as dist
    x=[value];dist.broadcast_object_list(x,src=0);return x[0]

def parse_devices(text):
    ids=[int(x) for x in text.split(',')]
    if len(ids)!=len(set(ids)) or not ids or any(x<0 for x in ids) or len(ids) not in (1,2,4,8):raise ValueError('Distinct nonnegative devices; supported world sizes:1/2/4/8')
    return ids
