import unittest
import numpy as np
from td_center.candidates import local_peak_candidates


class CandidateTests(unittest.TestCase):
    def test_separated_peaks_and_argmax(self):
        h = np.zeros((80, 80))
        h[10, 10], h[10, 12], h[30, 30], h[60, 60] = 5, 4, 3, 2
        peaks = local_peak_candidates(h, 3)
        self.assertEqual([(p['row'], p['col']) for p in peaks], [(10, 10), (30, 30), (60, 60)])
        self.assertEqual(peaks[0]['lon'], 109+10.5*70/80)

    def test_plateau_boundary_and_negative_scores(self):
        h = np.full((8, 8), -4.)
        h[0:2, 0:2] = -1
        h[7, 7] = -2
        peaks = local_peak_candidates(h, 2)
        self.assertEqual([(p['row'], p['col']) for p in peaks], [(0, 0), (7, 7)])
        self.assertEqual(len(local_peak_candidates(np.ones((8, 8)), 5)), 1)
