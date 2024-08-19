import os
import sys
import json
import time
import argparse
import statistics
import subprocess
import numpy as np
from filelock import FileLock
from pathlib import Path
from benchmark_set import dump_result, MAX_DEVIATION, configs, metrics
from run_utils.misc import SUPPORTED_DTYPES, init_env_variables

TIMEOUT = 24 * 60 * 60


def parse_args():
    parser = argparse.ArgumentParser(description="Run single stream benchmark.")
    parser.add_argument("-f", "--framework",
                        type=str, choices=["tf", "tflite", "pytorch", "ort", "ctranslate"], required=True,
                        help="name of the framework")
    parser.add_argument("-m", "--model_name",
                        type=str, required=True,
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, required=True,
                        help="batch size to feed the model with")
    parser.add_argument("-p", "--precision",
                        type=str, choices=SUPPORTED_DTYPES, required=True,
                        help="precision of the model provided")
    parser.add_argument("-t", "--num_threads",
                        type=int, default=1,
                        help="number of threads to use")
    parser.add_argument("--warm_up",
                        action="store_true",
                        help="just setup the model and run for 15 seconds")
    parser.add_argument("--debug",
                        action="store_true",
                        help="print stdout + stderr of processes?")
    parser.add_argument("--disable_dump",
                        action="store_true",
                        help="don't write results to .csv file")
    parser.add_argument("-i", "--import_all",
                        action="store_true",
                        help="import all available models")
    return parser.parse_args()


class Results:
    def __init__(self, results_dir):
        self._results_dir = results_dir
        self._prev_measurements_count = None
        self._prev_lat_mean = None
        self.stable = False

    def calculate_latencies(self, final_calc=False):
        logs = [log for log in os.listdir(self._results_dir) if "json" in log and "lock" not in log]
        assert len(logs) == 1

        log_filepath = os.path.join(self._results_dir, logs[0])
        with FileLock(f"{log_filepath}.lock", timeout=60):
            with open(log_filepath, "r") as f:
                log = json.load(f)

        measurements_count = len(log["start_times"])
        assert measurements_count == len(log["finish_times"]) == len(log["workload_size"])
        assert measurements_count >= 1

        latencies = [finish - start for start, finish in zip(log["start_times"], log["finish_times"])]

        lat_mean = statistics.mean(latencies)
        lat_median = statistics.median(latencies)
        lat_90th_percentile = np.percentile(latencies, 90)

        lat_mean *= 1000
        lat_median *= 1000
        lat_90th_percentile *= 1000

        if self._prev_measurements_count is not None and measurements_count > self._prev_measurements_count:
            self.stable = abs((lat_mean / self._prev_lat_mean) - 1.) <= MAX_DEVIATION
        self._prev_lat_mean = lat_mean
        self._prev_measurements_count = measurements_count

        if final_calc:
            print("\nLatency mean:   {:.2f} ms".format(lat_mean))
            print("Latency median: {:.2f} ms".format(lat_median))
            print("Latency p90:    {:.2f} ms".format(lat_90th_percentile))
        elif not self.stable:
            print("Result not yet stable - current latency mean: {:.2f} ms".format(lat_mean))

        return {"latency_mean_ms": lat_mean,
                "latency_median_ms": lat_median,
                "latency_90th_percentile_ms": lat_90th_percentile,
                "start_timestamp": log["start_times"][0],
                "finish_timestamp": log["finish_times"][-1]}


def main():
    args = parse_args()

    timeout = TIMEOUT
    if args.warm_up:
        timeout = 15

    results_dir = init_env_variables(args.num_threads)
    results = Results(results_dir)

    cmd = ["numactl", "--cpunodebind=0", "--membind=0",
           "python3", f"benchmark_{args.framework}.py",
           "-m", args.model_name, "-p", args.precision, "-b", str(args.batch_size),
           "--timeout", str(timeout)]
    if args.import_all:
        cmd.append("-i")
    if args.debug:
        proc = subprocess.Popen(cmd)
    else:
        proc = subprocess.Popen(cmd, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

    while proc.poll() is None:
        time.sleep(15)
        try:
            results.calculate_latencies()
        except AssertionError:
            pass
        if results.stable:
            Path(os.path.join(results_dir, "STOP")).touch()
            break

    if proc.wait() != 0:
        print("FAIL: Process returned exit code other than 0 or died!")
        sys.exit(1)

    if not args.warm_up:
        values = results.calculate_latencies(final_calc=True)
        if not args.disable_dump:
            dump_result(
                args.framework, args.model_name, args.precision, "ss",
                configs["ss"],
                {"batch_size": args.batch_size, "num_threads": args.num_threads},
                metrics["ss"],
                values
            )


if __name__ == "__main__":
    main()
