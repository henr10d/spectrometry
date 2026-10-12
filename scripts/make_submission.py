import argparse,json
from src.inference import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',default='test.parquet');p.add_argument('--template',default='sample_submission.csv');p.add_argument('--artifacts',default='artifacts');p.add_argument('--output',default='submission.csv');p.add_argument('--config',default='configs/baseline.json');a=p.parse_args()
    run(a.test,a.template,a.artifacts,a.output,json.load(open(a.config)))
