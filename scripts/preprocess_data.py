from gpu_experiment import preprocess_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=384)
args = parser.parse_args()
preprocess_data(args.max_samples)
