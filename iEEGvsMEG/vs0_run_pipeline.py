"""Run selected analyses for all 22 pairings, respecting reusable dependencies."""
from pathlib import Path
import argparse
import json
from vs0_run_analysis import ROOT, ANALYSES, plan, run_analysis
from vs0_utils_pairing import PAIRING_IDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root',type=Path,default=ROOT/'out')
    parser.add_argument('--analyses',nargs='+',choices=('all',*ANALYSES),default=['all'])
    parser.add_argument('--pairing',nargs='+',choices=('all',*PAIRING_IDS),default=None)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--recompute',action='store_true')
    args=parser.parse_args()
    analyses=ANALYSES if args.analyses==['all'] else args.analyses
    for analysis in analyses:
        config=args.out_root/analysis/'config.json'
        for job in plan(config,args.pairing):
            if args.dry_run:print(json.dumps(job));continue
            print(f"Running {analysis} / {job['pairing']}",flush=True)
            print(run_analysis(config,job['pairing'],recompute=args.recompute),flush=True)


if __name__=='__main__':main()
