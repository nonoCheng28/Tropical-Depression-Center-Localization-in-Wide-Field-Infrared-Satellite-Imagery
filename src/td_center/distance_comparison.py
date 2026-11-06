"""Paired argmax comparison from saved full-validation predictions."""
import csv
import json
from collections import defaultdict
from td_center.training import write_csv, write_json, sha256


def compare_saved(baseline, experiment):
    configs = [json.loads((p / 'config.json').read_text(encoding='utf-8')) for p in (baseline, experiment)]
    if any(c['smoke'] for c in configs):
        raise ValueError('Cannot compare smoke runs with the formal baseline')
    differences = {k: [configs[0].get(k), configs[1].get(k)] for k in configs[0].keys() | configs[1].keys()
                   if configs[0].get(k) != configs[1].get(k)}
    allowed = {'role', 'loss', 'experiment', 'baseline_run', 'expected_distance'}
    if set(differences) - allowed:
        raise ValueError(f'Unexpected configuration differences: {differences}')
    records = []
    for path in (baseline, experiment):
        with (path / 'best_val_predictions.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        indexed = {r['sample_id']: r for r in rows}
        if len(indexed) != 94 or len(rows) != 94:
            raise ValueError('Expected 94 unique validation predictions')
        records.append(indexed)
    if records[0].keys() != records[1].keys():
        raise ValueError('Validation IDs differ')
    paired, storms = [], defaultdict(list)
    for sid, a in records[0].items():
        b = records[1][sid]
        if any(a[k] != b[k] for k in ('true_x', 'true_y')):
            raise ValueError('Validation labels differ')
        da, db = float(a['distance_deg']), float(b['distance_deg'])
        row = {'sample_id': sid, 'storm_id': sid.split('_')[0], 'storm': sid.split('_')[1],
               'baseline_distance_deg': da, 'experiment_distance_deg': db, 'delta_deg': db-da,
               'rescued': da > 3 and db <= 3, 'harmed': da <= 3 and db > 3}
        paired.append(row)
        storms[(row['storm_id'], row['storm'])].append(row)
    out = experiment / 'baseline_comparison'
    out.mkdir(exist_ok=True)
    write_csv(out / 'paired.csv', paired)
    write_csv(out / 'storms.csv', [{'storm_id': sid, 'storm': name, 'n': len(rs),
        'baseline_MD': sum(r['baseline_distance_deg'] for r in rs)/len(rs),
        'experiment_MD': sum(r['experiment_distance_deg'] for r in rs)/len(rs),
        'rescued': sum(r['rescued'] for r in rs), 'harmed': sum(r['harmed'] for r in rs)}
        for (sid, name), rs in sorted(storms.items())])
    metrics = [json.loads((p / 'best_metrics.json').read_text()) for p in (baseline, experiment)]
    rescued, harmed = sum(r['rescued'] for r in paired), sum(r['harmed'] for r in paired)
    write_json(out / 'summary.json', {'role': 'advanced', 'test_evaluated': False,
        'baseline': metrics[0], 'experiment': metrics[1], 'rescued': rescued, 'harmed': harmed,
        'net_hits': rescued-harmed, 'config_differences': differences,
        'checkpoint_sha256': [sha256(p/'best.pt') for p in (baseline, experiment)]})
    lines = ['# A：期望距离辅助损失对照', '', '角色 advanced；固定 Val94；test 未使用；两者均为 argmax 解码。', '',
             '| 指标 | 原 baseline | A |', '|---|---:|---:|']
    for k in ('mean_distance_deg', 'mae_deg', 'rmse_deg', 'detection_rate_at_3deg'):
        lines.append(f'| {k} | {metrics[0][k]:.6f} | {metrics[1][k]:.6f} |')
    lines += ['', f'救回 {rescued}，误伤 {harmed}，净增 {rescued-harmed}。', '',
              '这是单 seed、validation 选模的探索性结果，不代表独立跨年份测试。新增辅助损失，不改变模型、输入或解码。']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Report: {out / "report.md"}', flush=True)
