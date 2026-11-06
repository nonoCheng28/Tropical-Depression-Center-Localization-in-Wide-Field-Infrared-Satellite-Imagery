"""One fixed-config decoder for all paper tables, without archived ledger paths."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--heatmaps',type=Path)
 p.add_argument('--loyo-pattern',help='path with {year} placeholder; exactly eight folds')
 p.add_argument('--split',choices=['val','test'],default='val')
 p.add_argument('--manifest',type=Path,help='Independent holdout list; never overwrite frozen lists')
 p.add_argument('--data-root',type=Path,default=Path(os.environ.get('TD_DATA_ROOT','data')))
 p.add_argument('--alpha',type=float,default=1.5);p.add_argument('--sigma',type=float,default=3)
 p.add_argument('--argmax',action='store_true',help='Retain mask; remove trajectory coupling only')
 p.add_argument('--raw',action='store_true',help='Raw heatmap argmax; no mask or Viterbi')
 p.add_argument('--normalize',action='store_true',help='Min-max raw emissions before masking')
 p.add_argument('--candidate-coverage',action='store_true',help='Coverage within 3 degrees of the unmasked top-8 peaks')
 p.add_argument('--allow-test',action='store_true');p.add_argument('--purpose',default='')
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if bool(a.heatmaps)==bool(a.loyo_pattern):p.error('Choose heatmaps OR loyo-pattern')
 if a.split=='test' and (not a.allow_test or not a.purpose):p.error('test requires --allow-test --purpose')
 if a.loyo_pattern and (a.manifest or a.split=='test'):p.error('LOYO is training years only')
 os.environ['TD_DATA_ROOT']=str(a.data_root.resolve())
 a.out.parent.mkdir(parents=True,exist_ok=True)
 reg=a.out.with_suffix('.registration.json')
 if a.split=='test':
  if a.out.exists():p.error('Existing test report will not be overwritten; choose a new registered output')
  reg.write_text(json.dumps(vars(a),default=str,indent=2),encoding='utf-8')
 from td_center.reproduce import besttrack,common,deliver_aided as da
 besttrack.DATA=a.data_root;besttrack.BST=a.data_root/'RSMC_Best_Track_Data/bst_all.txt';besttrack.load.cache_clear()
 torch.set_num_threads(1)
 groups={}
 for year in (range(2016,2024) if a.loyo_pattern else [None]):
  manifest=(common.REPO/f'splits/loyo/{year}_eval.txt') if year else (a.manifest or common.REPO/f'splits/single_center_v1/{a.split}.txt')
  ids=manifest.read_text(encoding='utf-8').split()
  heat=torch.load(Path(a.loyo_pattern.format(year=year)) if year else a.heatmaps,map_location='cpu',weights_only=False)
  if len(ids)!=len(set(ids)) or not set(ids)<=set(heat):raise ValueError('Duplicate/missing sample IDs')
  heat={k:v.float() for k,v in heat.items()}
  if a.normalize:heat={k:(v-v.min())/(v.max()-v.min()+1e-6) for k,v in heat.items()}
  labels=a.data_root/f'single_frame/ir/{"train" if year else a.split}/labels/TD_center/yolo'
  if a.raw:
   errors=[];comp=[]
   for sid in ids:
    h=heat[sid];iy,ix=divmod(int(h.argmax()),h.shape[-1]);pred=np.array([(ix+.5)/h.shape[-1],(iy+.5)/h.shape[-2]])
    target=np.array(list(map(float,(labels/f'{sid}.txt').read_text().split())));delta=(pred-target)*70
    errors.append(np.linalg.norm(delta));comp.append(delta)
   e=np.asarray(errors);c=np.asarray(comp)
   r={'MD':float(e.mean()),'DR3':float((e<=3).mean()),'n':len(ids),'MAE_comp':float(abs(c).mean()),'RMSE_comp':float(np.sqrt((c*c).mean()))}
  else:
   original=da.KAPPA
   try:
    if a.argmax:da.KAPPA=0
    st=da.run(heat,ids,labels,alpha=a.alpha,sigma=a.sigma)
   finally:da.KAPPA=original
   r=da.clean(st,a.alpha,'real',a.sigma)
   counts={storm:sum(sid.split('_')[0]==storm for sid in ids) for storm in st['storm_md']}
   denom=st['MD']*len(ids)
   r['storm_absolute_error_share']={storm:md*counts[storm]/denom if denom else 0 for storm,md in st['storm_md'].items()}
  if a.candidate_coverage:
   from td_center.candidates import local_peak_candidates
   covered=[]
   for sid in ids:
    h=heat[sid];target=np.array(list(map(float,(labels/f'{sid}.txt').read_text().split())))
    peaks=local_peak_candidates(h.numpy(),k=8,separation_deg=3)
    xy=np.array([[(pk['col']+.5)/h.shape[-1],(pk['row']+.5)/h.shape[-2]] for pk in peaks])
    covered.append(bool((np.linalg.norm((xy-target)*70,axis=1)<=3).any()))
   r['candidate_coverage3']=float(np.mean(covered))
  groups[str(year) if year else a.split]=r
 output={'folds':groups,'aggregation':'year_macro' if a.loyo_pattern else 'frame_mean','MD':float(np.mean([r['MD'] for r in groups.values()])), 'DR3':float(np.mean([r['DR3'] for r in groups.values()])), 'n':sum(r['n'] for r in groups.values()),'config':vars(a)}
 a.out.write_text(json.dumps(output,default=str,indent=2),encoding='utf-8');print(json.dumps({k:output[k] for k in ['MD','DR3','n']}))

if __name__=='__main__':main()
