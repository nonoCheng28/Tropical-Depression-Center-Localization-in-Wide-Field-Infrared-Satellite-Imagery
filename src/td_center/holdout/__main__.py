"""Dispatch source-backed paper reproduction commands."""
import argparse,runpy,sys
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('command',choices=['ahipipe', 'build_dataset', 'eval_holdout', 'hsd', 'make_split', 'merge_multi_center', 'move_excluded', 'verify_dataset'])
 if len(sys.argv)<2 or sys.argv[1] in ('--help','-h'):parser.parse_args()
 command=sys.argv.pop(1)
 if command not in ['ahipipe', 'build_dataset', 'eval_holdout', 'hsd', 'make_split', 'merge_multi_center', 'move_excluded', 'verify_dataset']:parser.error('Unknown command: '+command)
 runpy.run_module('td_center.holdout.'+command,run_name='__main__')
if __name__=='__main__':main()
