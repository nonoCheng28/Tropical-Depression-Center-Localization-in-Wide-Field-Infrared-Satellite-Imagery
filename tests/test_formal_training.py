import math
import unittest

import torch
from torch.utils.data import DataLoader

from td_center.heatmap import make_gaussian_heatmap, decode_argmax
from td_center.metrics import center_metrics, centers_to_lonlat
from td_center.losses import weighted_mse_per_sample
from td_center.training import train_epoch, evaluate


class FormalTrainingTests(unittest.TestCase):
    def test_metric_hand_calculation(self):
        truth = torch.tensor([[0.2, 0.3], [0.4, 0.5]], dtype=torch.float64)
        pred = truth + torch.tensor([[3/70, 4/70], [0, 0]], dtype=torch.float64)
        metrics = center_metrics(pred, truth)
        self.assertAlmostEqual(metrics['mean_distance_deg'], 2.5)
        self.assertAlmostEqual(metrics['mae_deg'], 1.75)
        self.assertAlmostEqual(metrics['rmse_deg'], 2.5)
        self.assertEqual(metrics['detection_rate_at_3deg'], 0.5)
        self.assertEqual(centers_to_lonlat([[0, 0], [1, 1]]).tolist(), [[109., 55.], [179., -15.]])

    def test_threshold_and_invalid_values(self):
        metrics = center_metrics([[3/70, 0]], [[0, 0]])
        self.assertEqual(metrics['detection_rate_at_3deg'], 1.)
        with self.assertRaises(ValueError):
            centers_to_lonlat([[float('nan'), 0]])
        with self.assertRaises(ValueError):
            center_metrics([[0, 0]], [[0, 0], [0, 0]])

    def test_heatmap_roundtrip(self):
        for xy in ([0., 0.], [1., 1.], [.25, .64]):
            center = torch.tensor(xy)
            decoded = decode_argmax(make_gaussian_heatmap(center))
            self.assertTrue(((decoded - center).abs() <= .5/80 + 1e-6).all())

    def test_accumulation_matches_actual_batches_including_tail(self):
        torch.manual_seed(42)
        samples = [{'images': torch.rand(1, 2, 2), 'heatmap': torch.rand(1, 2, 2)} for _ in range(19)]
        for micro in (1, 2, 4, 8, 16):
            accumulated = torch.nn.Conv2d(1, 1, 1).double()
            reference = torch.nn.Conv2d(1, 1, 1).double()
            reference.load_state_dict(accumulated.state_dict())
            data = [{k: v.double() for k, v in s.items()} for s in samples]
            opt = torch.optim.SGD(accumulated.parameters(), lr=.01)
            ref_opt = torch.optim.SGD(reference.parameters(), lr=.01)
            stats = train_epoch(accumulated, DataLoader(data, batch_size=micro), opt, 'cpu')
            for batch in DataLoader(data, batch_size=16):
                ref_opt.zero_grad()
                weighted_mse_per_sample(reference(batch['images']), batch['heatmap']).mean().backward()
                ref_opt.step()
            self.assertEqual(stats['train_n'], 19)
            self.assertEqual(stats['optimizer_updates'], 2)
            for a, b in zip(accumulated.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_validation_is_sample_weighted_and_does_not_update(self):
        model = torch.nn.Conv2d(1, 1, 1)
        samples = [{'images': torch.rand(1, 2, 2), 'heatmap': torch.rand(1, 2, 2),
                    'center': torch.tensor([.4, .6]), 'sample_id': str(i)} for i in range(5)]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        a, rows = evaluate(model, DataLoader(samples, batch_size=1), 'cpu')
        b, _ = evaluate(model, DataLoader(samples, batch_size=3), 'cpu')
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=6)
        self.assertEqual(len(rows), 5)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
