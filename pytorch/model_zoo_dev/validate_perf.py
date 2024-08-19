"""Validate performance of models specified in a config file

This script allows the user to run a set of benchmarks described in a YAML file.
It runs the benchmarks using benchmark_ss.py script and compares the obtained
latency with a threshold value stored in the config file.

Example config.yaml:

tf:
    - densenet_169:
        precision: fp32
        num_threads: 64
        batch_size: 1
        env:
            ENABLE_AIO_IMPLICIT_FP16: 1
            SOME_OTHER_OPTION: 1
        latency: 48.1
    - densenet_169:
        precision: fp16
        num_threads: 64
        batch_size: 1
        latency: 20.0
    - inception_v2:
        precision: fp32
        num_threads: 64
        batch_size: 1
        latency: 10

pytorch:
    - resnet_50_v1:
        precision: fp32
        num_threads: 64
        latency: 9.15
"""

import argparse
import csv
import yaml
import subprocess
import os

from ampere_model_library.utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark models specified in a file.")
    parser.add_argument(
        "config_file", help="Path to a YAML file containing the benchmarks to run.")
    parser.add_argument("--dry-run", action='store_true',
                        help="Instead of running the benchmarks, print the commands that would be run.")
    parser.add_argument("--keep-going", action='store_true',
                        help="Don't exit if a threshold was not met, but continue to run all the specified scenarios.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_file, "rb") as f:
        yaml_dict = yaml.load(f, yaml.CLoader)

    fail = False

    for framework, runs in yaml_dict.items():
        for run in runs:
            for model in run:
                # YAML maps this to a dict with one key. It's possible to have it as just one dict with all the models, but then we would need a different way to specify multiple runs of the same model.
                run = run[model]
                run.setdefault('env', {})
                run.setdefault('batch_size', 1)
                command = f"python3 benchmark_ss.py -f {framework} -m {model} -p {run['precision']} -b {run['batch_size']} -t {run['num_threads']}"
                if args.dry_run:
                    print(
                        f"Command: {command}\nEnv: {run['env']}\nThreshold: {run['latency']} ms\n")
                    continue

                run["env"] = {k: str(v) for k, v in run["env"].items()}
                env = dict(os.environ, **run['env'])

                print(f"Running '{command}' with env: {run['env']}")
                _ = subprocess.run(
                    command.split(), env=env, capture_output=True, text=True)

                with open(f"results/csv_files/{framework}@{model}@{run['precision']}@ss.csv", "r") as f:
                    result = float(list(csv.DictReader(f))
                                    [-1]["latency_mean_ms"])
                success = result < run['latency']
                print(
                    f"Latency within limits: {'✅' if success else '❌'}\nThreshold: {run['latency']} ms\nAchieved: {result} ms\n")
                if not success and not args.keep_going:
                    print_goodbye_message_and_die("Latency threshold not met")
                if not success:
                    fail = True

    if fail:
        print_goodbye_message_and_die("At least one latency threshold not met")


if __name__ == "__main__":
    main()
