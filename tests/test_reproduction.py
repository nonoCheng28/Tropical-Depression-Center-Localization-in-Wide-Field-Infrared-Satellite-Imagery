import itertools,unittest
from pathlib import Path
import numpy as np
import torch
from td_center.reproduce.common import REPO
from td_center.reproduce.deliver_aided import gaussian_at
from td_center.reproduce.eval_viterbi_decoder import viterbi

class ReproductionTests(unittest.TestCase):
 def test_frozen_partitions_and_year_folds(self):
  def ids(p):return (REPO/p).read_text(encoding='utf-8').split()
  tr,va,te=[ids(f'splits/single_center_v1/{x}.txt') for x in ['train','val','test']]
  self.assertEqual([len(tr),len(va),len(te)],[805,94,121])
  storms=[{s.split('_')[0] for s in x} for x in [tr,va,te]]
  for a,b in itertools.combinations(storms,2):self.assertFalse(a&b)
  held=[]
  for year in range(2016,2024):
   a,b=ids(f'splits/loyo/{year}_train.txt'),ids(f'splits/loyo/{year}_eval.txt')
   self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(tr));held+=b
  self.assertEqual(sorted(held),sorted(tr))
  for year,n in [(2015,67),(2026,23)]:self.assertEqual(len(ids(f'splits/independent_{year}/eval.txt')),n)
 def test_gaussian_amplitude_and_center(self):
  g=gaussian_at(80,(.5,.5),3)
  self.assertEqual(g.shape,(80,80));self.assertAlmostEqual(float(g.max()),1)
  self.assertTrue(torch.allclose(g,g.flip(0)));self.assertLess(float(g[0,0]),1e-6)
 def test_viterbi_matches_exhaustive_paths(self):
  xy=np.array([[0,0],[.02,.01]])
  rows=[{'xy':xy,'sup':np.array(x)} for x in [[1,.2],[.1,1],[1,.3]]]
  for kappa in [0,.003,.1]:
   chosen=viterbi(rows,kappa)
   def score(p):return sum(rows[t]['sup'][i] for t,i in enumerate(p))-kappa*sum(np.sum(((xy[p[t]]-xy[p[t-1]])*70)**2) for t in range(1,len(p)))
   self.assertAlmostEqual(score(chosen),max(score(p) for p in itertools.product(range(2),repeat=3)))
if __name__=='__main__':unittest.main()
