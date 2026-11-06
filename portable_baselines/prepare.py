"""Bind immutable logical manifests to target media; never re-split or drop rows."""
import argparse,fcntl
from pathlib import Path
import numpy as np
import common as C

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--full-decode',action='store_true');a=p.parse_args()
    paths=C.Paths(a.config);paths.setup('');package=C.verify_package();paths.guard(2*C.GIB)
    with paths.out('prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if paths.out('PREPARED.json').exists():
            old=C.read(paths.out('PREPARED.json'))
            if old['package_sha']!=package or old['config_sha']!=C.sha(paths.config):raise RuntimeError('Existing output belongs to another package/config; use a new output_root')
        expected=C.protocol()['initialization']['sha256'];actual=C.sha(paths.vast_checkpoint)
        if actual!=expected:raise RuntimeError('All methods require the exact VAST checkpoint SHA256')
        required=['clip/EVA01_CLIP_g_14_psz14_s11B.pt','beats/BEATs_iter3_plus_AS2M.pt','bert/bert-base-uncased/config.json','bert/bert-base-uncased/vocab.txt']
        bert=paths.assets_root/'bert/bert-base-uncased'
        weights=[name for name in ('model.safetensors','pytorch_model.bin') if (bert/name).is_file()]
        if not weights:raise FileNotFoundError('BERT model.safetensors or pytorch_model.bin required')
        required+=['bert/bert-base-uncased/'+name for name in weights]
        required+=['bert/bert-base-uncased/'+p.name for p in bert.iterdir() if p.is_file() and p.name in ('tokenizer_config.json','special_tokens_map.json','tokenizer.json')]
        for name in required:
            if not (paths.assets_root/name).is_file():raise FileNotFoundError(paths.assets_root/name)
        from transformers import BertTokenizer
        tokenizer=BertTokenizer.from_pretrained(str(paths.assets_root/'bert/bert-base-uncased'),local_files_only=True)
        checks=C.read(C.ROOT/'manifests/token_checksums.json');records={};all_ids=[];all_masks=[];groups=[]
        for name in ('train','val','vast','msrvtt','vatex','activitynet','didemo'):
            m=C.manifest(name);stats=[]
            if name in ('train','val','vast'):groups.append({r['source_video'] for r in m['videos']})
            for r in m['videos']:
                row={}
                for kind in ('video','audio'):
                    path=paths.media(m,r,kind);s=path.stat()
                    if s.st_size!=r[kind+'_bytes']:raise RuntimeError('Media size differs from source (do not silently change cohort): '+str(path))
                    row[kind]=[s.st_size,s.st_mtime_ns]
                stats.append(row)
            folder=paths.out('inputs',name);folder.mkdir(parents=True,exist_ok=True)
            token_parts={k:[] for k in ('input_ids','attention_mask')}
            for start in range(0,len(m['queries']),1024):
                batch=tokenizer([q['caption'] for q in m['queries'][start:start+1024]],padding='max_length',truncation=True,max_length=40,return_tensors='np')
                for k in token_parts:token_parts[k].append(batch[k])
            arrays={k:np.concatenate(v).astype(np.int32 if k=='input_ids' else np.int8) for k,v in token_parts.items()}
            arrays['caption_to_video']=np.array([q['video_index'] for q in m['queries']],dtype=np.int64)
            for k,v in arrays.items():
                f=folder/(k+'.npy');np.save(f,v,allow_pickle=False)
                if name in checks and C.sha(f)!=checks[name][f.name]:raise RuntimeError('Tokenizer/mapping differs from source: '+str(f))
            if name in ('train','val','vast'):all_ids.append(arrays['input_ids']);all_masks.append(arrays['attention_mask'])
            C.write(folder/'media_stats.json',stats)
            records[name]=dict(videos=len(m['videos']),queries=len(m['queries']),files={f.name:C.sha(f) for f in folder.iterdir() if f.suffix in ('.json','.npy')})
            print(name,records[name]['videos'],records[name]['queries'],'paths/sizes/tokens verified',flush=True)
        if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:raise RuntimeError('Source-video split leakage')
        # Compare exact original combined train/val/test token serialization.
        for filename,array in [('token_ids.npy',np.concatenate(all_ids)),('token_mask.npy',np.concatenate(all_masks))]:
            f=paths.out('inputs',filename);np.save(f,array,allow_pickle=False)
            if C.sha(f)!=checks['vast_combined'][filename]:raise RuntimeError('Combined VAST tokens differ')
        for r in C.read(C.ROOT/'manifests/media_samples.json'):
            m=C.manifest(r['dataset']);f=paths.media(m,m['videos'][r['index']],r['kind'])
            if C.sha(f)!=r['sha256']:raise RuntimeError('Source media sample hash differs: '+str(f))
        from inputs import Media
        decoded={}
        for name in records:
            ds=Media(paths,name);positions=range(len(ds)) if a.full_decode else sorted({0,len(ds)//3,2*len(ds)//3,len(ds)-1})
            for i in positions:ds[i]
            if ds.reader:ds.reader.close()
            decoded[name]=len(positions)
        assets={name:C.sha(paths.assets_root/name) for name in required}
        source_assets=C.read(C.ROOT/'configs/assets_checksums.json')
        for name,expected in source_assets.items():
            if name in assets and assets[name]!=expected:raise RuntimeError('Constructor asset differs from source: '+name)
        stat=paths.vast_checkpoint.stat()
        receipt=dict(checkpoint_stat=[stat.st_size,stat.st_mtime_ns],asset_stats={name:[(paths.assets_root/name).stat().st_size,(paths.assets_root/name).stat().st_mtime_ns] for name in required},passed=True,package_sha=package,config_sha=C.sha(paths.config),checkpoint_sha=actual,assets=assets,inputs=records,
            source_media_sample_hashes_verified=True,decoded_counts=decoded,all_media_decoded=a.full_decode,runtime_versions=C.runtime_versions())
        C.write(paths.out('PREPARED.json'),receipt)
        print('Target inputs prepared. GPU preflight still required.',flush=True)
if __name__=='__main__':main()
