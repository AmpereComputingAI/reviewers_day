import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from filelock import FileLock
from benchmark_set import dump_result, MAX_DEVIATION, configs, metrics, parse_threads_range
from run_utils.misc import SUPPORTED_DTYPES, init_env_variables
from ampere_model_library.utils.misc import print_goodbye_message_and_die

TIMEOUT = 24 * 60 * 60
MIN_MEASUREMENTS_IN_OVERLAP_COUNT = 10
online_threads = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline benchmark.")
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
    parser.add_argument("-r", "--threads_range",
                        type=str, required=True,
                        help="range of threads to use, e.g. '0-63,128-191', threads will be divided between processes "
                             "- hint: 'lscpu | grep NUMA'")
    parser.add_argument("-n", "--num_processes",
                        type=int, default=1,
                        help="number of processes to spawn")
    parser.add_argument("-t", "--num_threads",
                        type=int, default=1,
                        help="number of threads to use per process")
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


def gen_threads_config(num_threads, process_id):
    threads_to_use = [str(t) for t in online_threads[num_threads*process_id:num_threads*(process_id+1)]]
    assert len(threads_to_use) == num_threads
    return " ".join(threads_to_use), ",".join(threads_to_use)


class Results:
    def __init__(self, results_dir, processes_count):
        self._results_dir = results_dir
        self._prev_measurements_count = None
        self._prev_throughput_total = None
        self._processes_count = processes_count
        self.stable = False

    def calculate_throughput(self, final_calc=False):
        logs = [log for log in os.listdir(self._results_dir) if "json" in log and "lock" not in log]
        assert len(logs) == self._processes_count

        loaded_logs = []
        for log in logs:
            log_filepath = os.path.join(self._results_dir, log)
            with FileLock(f"{log_filepath}.lock", timeout=60):
                with open(log_filepath, "r") as f:
                    loaded_logs.append(json.load(f))

        measurements_counts = [(len(log["start_times"]), len(log["finish_times"]), len(log["workload_size"])) for log in
                               loaded_logs]
        assert all(x[0] == x[1] == x[2] and x[0] >= MIN_MEASUREMENTS_IN_OVERLAP_COUNT for x in measurements_counts)
        latest_start = max(log["start_times"][0] for log in loaded_logs)
        earliest_finish = min(log["finish_times"][-1] for log in loaded_logs)

        measurements_completed_in_overlap_total = 0
        throughput_total = 0.
        for log in loaded_logs:
            input_size_processed_per_process = 0
            total_latency_per_process = 0.
            measurements_completed_in_overlap = 0
            for i in range(len(log["start_times"])):
                start = log["start_times"][i]
                finish = log["finish_times"][i]
                if start >= latest_start and finish <= earliest_finish:
                    input_size_processed_per_process += log["workload_size"][i]
                    total_latency_per_process += finish - start
                    measurements_completed_in_overlap += 1
                elif earliest_finish < finish:
                    break
            assert measurements_completed_in_overlap >= MIN_MEASUREMENTS_IN_OVERLAP_COUNT
            measurements_completed_in_overlap_total += measurements_completed_in_overlap
            throughput_total += input_size_processed_per_process / total_latency_per_process

        if self._prev_measurements_count is not None and \
                measurements_completed_in_overlap_total > self._prev_measurements_count:
            self.stable = abs((throughput_total / self._prev_throughput_total) - 1.) <= MAX_DEVIATION
        self._prev_throughput_total = throughput_total
        self._prev_measurements_count = measurements_completed_in_overlap_total

        if final_calc:
            print("\nOverlap time of processes: {:.2f} s".format(earliest_finish - latest_start))
            print("Total throughput: {:.2f} fps".format(throughput_total))
        elif not self.stable:
            print("Result not yet stable - current throughput: {:.2f} fps".format(throughput_total))

        return {"throughput_total": throughput_total,
                "start_timestamp": latest_start,
                "finish_timestamp": earliest_finish}


def main():
    global online_threads

    args = parse_args()

    designated_threads = parse_threads_range(args.threads_range)
    numa_config = subprocess.run(["numactl", "--show"], capture_output=True, text=True, check=True)
    online_threads = [int(t) for t in numa_config.stdout.split("physcpubind: ")[1].split(" \ncpubind:")[0].split()
                      if int(t) in designated_threads]
    if len(online_threads) < args.num_processes * args.num_threads:
        print_goodbye_message_and_die(f"Requested config requires {args.num_processes * args.num_threads} threads, "
                                      f"while only {len(online_threads)} threads are both online and designated")

    results_dir = init_env_variables(args.num_threads)
    results = Results(results_dir, args.num_processes)

    current_subprocesses = list()
    for n in range(args.num_processes):
        aio_numa_cpus, physcpubind = gen_threads_config(args.num_threads, n)
        os.environ["AIO_NUMA_CPUS"] = aio_numa_cpus
        os.environ["DLS_NUMA_CPUS"] = aio_numa_cpus
        cmd = ["numactl", f"--physcpubind={physcpubind}",
               "python3", f"benchmark_{args.framework}.py",
               "-m", args.model_name, "-p", args.precision, "-b", str(args.batch_size),
               "--timeout", str(TIMEOUT)]
        if args.import_all:
            cmd.append("-i")
        if args.debug:
            current_subprocesses.append(subprocess.Popen(cmd))
        else:
            current_subprocesses.append(subprocess.Popen(
                cmd, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb')))

    failure = False
    while not all(p.poll() is not None for p in current_subprocesses):
        time.sleep(15)
        try:
            results.calculate_throughput()
        except AssertionError:
            pass
        failure = any(p.poll() is not None and p.poll() != 0 for p in current_subprocesses)
        if results.stable or failure:
            Path(os.path.join(results_dir, "STOP")).touch()
            break

    if not failure:
        # wait for subprocesses to finish their job if all are alive till now
        failure = any(p.wait() != 0 for p in current_subprocesses)

    if failure:
        print("FAIL: At least one process returned exit code other than 0 or died!")
        sys.exit(1)

    values = results.calculate_throughput(final_calc=True)
    if not args.disable_dump:
        dump_result(
            args.framework, args.model_name, args.precision, "offline",
            configs["offline"],
            {"batch_size": args.batch_size, "num_processes": args.num_processes, "num_threads": args.num_threads},
            metrics["offline"],
            values
        )


if __name__ == "__main__":
    main()
