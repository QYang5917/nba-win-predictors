"""Run a local V2 research bundle on supplied pregame feature snapshots."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from v2_research import predict_bundle, ROOT, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features',type=Path)
    parser.add_argument('--bundle',type=Path,default=ROOT/'reports/v2_research/research_bundle.joblib')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,help='Optional past-OOF calibration bundle')
    parser.add_argument('--recency',type=Path,help='Optional chronological recency-selection bundle')
    args=parser.parse_args()
    bundle=joblib.load(args.bundle)
    features=pd.read_csv(args.features,dtype={'GAME_ID':str})
    with threadpool_limits(limits=2):
        result=predict_bundle(bundle,features)
        if args.calibration:
            from v2_calibration import apply_meta
            calibration=joblib.load(args.calibration)
            if digest(args.bundle)!=calibration['base_bundle_sha256']:
                raise ValueError('Calibration does not match this base model bundle')
            result=apply_meta(calibration['meta'],result)
        if args.recency:
            from v2_recency import predict_bundle as predict_recency
            result['P_recency']=predict_recency(joblib.load(args.recency),features)
    # A separate research file; never silently overwrite a prediction log.
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'{len(result)} research predictions saved to {args.output}')


if __name__=='__main__':
    main()
