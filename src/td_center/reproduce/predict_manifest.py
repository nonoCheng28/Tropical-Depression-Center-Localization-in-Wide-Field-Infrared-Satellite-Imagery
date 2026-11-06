"""Infer explicit checkpoint groups on a frozen or independent sample manifest."""
import argparse,json,os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--spec',type=Path,required=True)
 p.add_argument('--data-root',type=Path,required=True)
 p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--split',choices=['train','val','test'],default='val')
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--device',default='cuda');p.add_argument('--allow-test',action='store_true');p.add_argument('--purpose',default='')
 a=p.parse_args()
 if a.split=='test' and(not a.allow_test or not a.purpose):p.error('test requires --allow-test and --purpose')
 a.out.mkdir(parents=True,exist_ok=True)
 (a.out/'registration.json').write_text(json.dumps(vars(a),default=str,indent=2),encoding='utf-8')
 from td_center.dataloader import IRDataset
 from td_center.model_d import DModel
 from td_center.reproduce.train_d_v2 import DModelV2
 spec=json.loads(a.spec.read_text(encoding='utf-8'));ds=IRDataset(a.data_root,a.manifest.parent,a.split,manifest_path=a.manifest)
 device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA requested but unavailable')
 normalized=[]
 for name,group in spec.items():
  models=[]
  for ck in group['checkpoints']:
   path=Path(ck);path=path if path.is_absolute() else a.spec.parent/path
   c=torch.load(path,map_location='cpu',weights_only=False)
   m=DModel() if group['arch']=='d' else DModelV2();m.load_state_dict(c['model_state_dict']);m.to(device).eval();models.append(m)
  heat={}
  with torch.no_grad():
   for sample in ds:
    x=sample['images'][None].to(device);ys=[m(x) for m in models]
    if group.get('activation')=='sigmoid':ys=[torch.sigmoid(y) for y in ys]
    heat[sample['sample_id']]=torch.stack(ys).mean(0)[0,0].cpu()
  torch.save(heat,a.out/f'{name}_fp32.pt')
  normalized.append({k:(v-v.min())/(v.max()-v.min()+1e-6) for k,v in heat.items()})
  del models
  if device.type=='cuda':torch.cuda.empty_cache()
 if not normalized:raise ValueError('Empty spec')
 ensemble={k:torch.stack([h[k] for h in normalized]).mean(0) for k in normalized[0]}
 torch.save(ensemble,a.out/f'heatmaps_{a.split}_fp32.pt')
 print('Saved',len(ds),'samples,',len(normalized),'emissions')

if __name__=='__main__':main()
