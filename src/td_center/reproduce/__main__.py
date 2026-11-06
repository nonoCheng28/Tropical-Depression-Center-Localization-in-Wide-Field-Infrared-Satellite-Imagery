"""Dispatch source-backed paper reproduction commands."""
import argparse,runpy,sys
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('command',choices=['besttrack', 'combine_holdout', 'common', 'decoder_ablation', 'deliver_aided', 'eval_heat_ensemble', 'eval_viterbi_decoder', 'fair_compare', 'fair_compare_loyo', 'fair_compare_pipeline', 'figure_emissions', 'figure_interference', 'kappa_curve', 'nested_kappa', 'predict_arm_heatmaps', 'predict_d_ensemble', 'report_ensemble_kappa', 'train_d_v2', 'train_loyo_fold', 'evaluate', 'predict_manifest', 'build_loyo_manifests'])
 if len(sys.argv)<2 or sys.argv[1] in ('--help','-h'):parser.parse_args()
 command=sys.argv.pop(1)
 if command not in ['besttrack', 'combine_holdout', 'common', 'decoder_ablation', 'deliver_aided', 'eval_heat_ensemble', 'eval_viterbi_decoder', 'fair_compare', 'fair_compare_loyo', 'fair_compare_pipeline', 'figure_emissions', 'figure_interference', 'kappa_curve', 'nested_kappa', 'predict_arm_heatmaps', 'predict_d_ensemble', 'report_ensemble_kappa', 'train_d_v2', 'train_loyo_fold', 'evaluate', 'predict_manifest', 'build_loyo_manifests']:parser.error('Unknown command: '+command)
 runpy.run_module('td_center.reproduce.'+command,run_name='__main__')
if __name__=='__main__':main()
