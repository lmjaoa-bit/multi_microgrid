"""Fixed-data Actor/Critic architecture diagnosis; original models stay frozen."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','confirm','critic','actor','policy','report','all'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',type=Path,default=ROOT/'outputs/diagnostics/critic_probe/expanded_24d_2seeds_2026-10-02')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':
        from diagnostics.capacity_data import prepare_capacity_data
        prepare_capacity_data(ROOT,args.probe,args.output,smoke=args.smoke)
    elif args.stage=='confirm':
        from diagnostics.capacity_confirmation import prepare_confirmation_supplement
        prepare_confirmation_supplement(ROOT,args.output)
    elif args.stage=='all':
        from diagnostics.capacity_confirmation import prepare_confirmation_supplement
        from diagnostics.capacity_fit import run_stage
        prepare_confirmation_supplement(ROOT,args.output)
        for stage in ('critic','actor','policy','report'):run_stage(ROOT,args.output,stage)
    else:
        from diagnostics.capacity_fit import run_stage
        run_stage(ROOT,args.output,args.stage)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
