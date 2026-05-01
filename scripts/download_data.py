from gpu_experiment import download_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=384)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
download_data(args.max_samples, args.smoke)
