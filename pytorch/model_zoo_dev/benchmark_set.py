import os
import csv
import json
import sys
import time
import argparse
import subprocess

from results.utils import get_results_path
from ampere_model_library.utils.misc import print_goodbye_message_and_die

MAX_DECLINES_IN_ROW = 3  # number of successively worse results across given dimension after which the script will move
# to the next value in the higher dimension
MAX_DEVIATION = 0.001  # max deviation [abs((value_n+1 / value_n) - 1.)] between sample n and sample n+1
SETUP_TIMEOUT = 15 * 60

CONFIG_SS = ["batch_size", "num_threads"]
METRICS_SS = ["latency_mean_ms", "latency_median_ms", "latency_90th_percentile_ms"]
CONFIG_OFFLINE = ["batch_size", "num_processes", "num_threads"]
METRICS_OFFLINE = ["throughput_total"]
METADATA = ["start_timestamp", "finish_timestamp"]
configs = {"ss": CONFIG_SS, "offline": CONFIG_OFFLINE}
metrics = {"ss": METRICS_SS + METADATA, "offline": METRICS_OFFLINE + METADATA}


def parse_args():
    parser = argparse.ArgumentParser(description="Run set of benchmarks.")
    parser.add_argument("-s", "--scenario",
                        type=str, choices=["ss", "offline"], required=True,
                        help="scenario to run")
    parser.add_argument("-f", "--framework",
                        type=str, choices=["tf", "tflite", "pytorch", "ort", "ctranslate"], required=True,
                        help="name of the framework")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16", "bf16", "int8"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-m", "--model_names",
                        type=str, required=True, nargs="+",
                        help="name of the model")
    parser.add_argument("-t", "--num_threads",
                        type=int, required=True, nargs="+",
                        help="number of threads to use")
    parser.add_argument("-c",
                        type=int, default=1e+9,
                        help="max concurrency")
    parser.add_argument("-b", "--batch_sizes",
                        type=int, required=True, nargs="+",
                        help="batch sizes to cover")
    parser.add_argument("--docker_name",
                        type=str, required=True,
                        help="name of the docker container to use")
    parser.add_argument("--mzdev_dir",
                        type=str, required=True,
                        help="dir to model_zoo_dev in docker")
    parser.add_argument("-r", "--threads_range",
                        type=str,
                        help="range of threads to use in offline mode, e.g. '0-63,128-191', threads will be divided "
                             "between processes - hint: 'lscpu | grep NUMA'")
    parser.add_argument("--pre_cmd",
                        type=str, default="",
                        help="cmd to run in docker before launching subprocesses")
    parser.add_argument("--timeout",
                        type=float, default=900,
                        help="timeout to apply per single benchmark case")
    parser.add_argument("--debug",
                        action="store_true",
                        help="print stdout + stderr of processes?")
    parser.add_argument("--disable_warm_up",
                        action="store_true",
                        help="skip warm-up iteration?")
    parser.add_argument("--exhaustive",
                        action="store_true",
                        help="run all cases? (by default unpromising cases will be dropped)")
    parser.add_argument("--memory",
                        action="store_true",
                        help="measure memory usage?")
    return parser.parse_args()


def parse_threads_range(threads_range: str) -> list[int]:
    threads_range = [s.split("-") for s in threads_range.split(",")]
    if not all([len(s) == 2 for s in threads_range]):
        print("Format of --threads_range argument must be '{idx}-{idx},{idx}-{idx},...', "
              "e.g. '88-88' to use just thread idx 88")
        sys.exit(1)
    designated_threads = []
    for s in threads_range:
        s_0, s_1 = int(s[0]), int(s[1])
        if s_1 < s_0:
            print(f"Range {s_0}-{s_1} is not valid, second value has to be equal to or greater than the first value")
            sys.exit(1)
        designated_threads += [i for i in range(s_0, s_1 + 1)]
    return designated_threads


def dump_result(framework, model_name, precision, scenario, config_fieldnames, config, results_fieldnames, results):
    results_filename = f"{framework}@{model_name}@{precision}@{scenario}.csv"
    results_path = os.path.join(get_results_path(), results_filename)

    if os.path.exists(results_path):
        first_write = False
    else:
        first_write = True
    with open(results_path, "a") as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(config_fieldnames + results_fieldnames)
        writer.writerow([config[key] for key in config_fieldnames] + [results[key] for key in results_fieldnames])
    print(f"Result has been saved - {results_path}")


def dump_memory(framework, model_name, precision, scenario, config, memory_reading):
    results_filename = f"memory_{framework}@{model_name}@{precision}@{scenario}.csv"
    results_path = os.path.join(get_results_path(), results_filename)

    if os.path.exists(results_path):
        first_write = False
    else:
        first_write = True
    with open(results_path, "a") as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(CONFIG_OFFLINE + ["memory_MiB"])
        writer.writerow([config[key] for key in CONFIG_OFFLINE] + [memory_reading])
    print(f"Memory reading has been saved - {results_path}")


def read_latest_result(framework, model_name, precision, scenario, config_fieldnames, results_fieldnames, label):
    results_filename = f"{framework}@{model_name}@{precision}@{scenario}.csv"
    results_path = os.path.join(get_results_path(), results_filename)

    with open(results_path, "r") as f:
        results = list(csv.reader(f))
    assert results[0] == config_fieldnames + results_fieldnames
    return float(results[-1][len(config_fieldnames) + results_fieldnames.index(label)])


def docker_restart(docker_name):
    break_time = 15
    
    def docker_stop():
        if subprocess.run(["docker", "stop", docker_name]).returncode != 0:
            print(f"Stopping docker container {docker_name} failed, retrying in {break_time} seconds.")
            time.sleep(break_time)
            docker_stop()

    def docker_start():
        if subprocess.run(["docker", "start", docker_name]).returncode != 0:
            print(f"Starting docker container {docker_name} failed, retrying in {break_time} seconds.")
            time.sleep(break_time)
            docker_start()
            
    print(f"\nRestarting docker container {docker_name} ...")
    docker_stop()
    docker_start()


class ThroughputTendency:
    def __init__(self):
        self._prev_result = None
        self._declines_count = 0
        self.max_value = 0.

    def is_promising(self, result: float) -> bool:
        if self._prev_result is not None and self._prev_result >= result:
            self._declines_count += 1
        else:
            self._declines_count = 0
        self._prev_result = result
        self.max_value = max(self.max_value, result)
        if self._declines_count >= MAX_DECLINES_IN_ROW:
            print(f"Performance declined {MAX_DECLINES_IN_ROW} times in a row - skipping the remaining cases across the"
                  f" dimension")
            return False
        else:
            return True


def main():
    args = parse_args()

    if args.scenario == "offline":
        if args.threads_range is None:
            print_goodbye_message_and_die("Range of threads to use needs to be set with --threads_range arg when "
                                          "running the offline scenario.")
        num_available_threads = len(parse_threads_range(args.threads_range))
        if num_available_threads < max(args.num_threads):
            print_goodbye_message_and_die(f"Requested number of threads ({max(args.num_threads)}) exceeds threads "
                                          f"available ({num_available_threads})")

    docker_restart(args.docker_name)
    if not args.disable_warm_up:
        print("--- SETUP ITERATION ---")
        time.sleep(3)
        for model in args.model_names:
            cmd = f"cd {args.mzdev_dir};"
            cmd += f" {args.pre_cmd}"
            cmd += f"IGNORE_PERF_CALC_ERROR=1 " \
                   f"python3 benchmark_ss.py " \
                   f"-f {args.framework} " \
                   f"-m {model} " \
                   f"-p {args.precision} " \
                   f"-t {str(min(64, max(args.num_threads)))} " \
                   f"-b 1 " \
                   f"--warm_up " \
                   "--disable_dump"
            if args.debug:
                cmd += " --debug"
            cmd = ["docker", "exec", "-i", args.docker_name, "bash", "-c", cmd]

            print(f"Executing: {' '.join(cmd)}")

            success = False
            start = time.time()
            p = subprocess.Popen(cmd, start_new_session=True)
            while time.time() - start < max(SETUP_TIMEOUT, args.timeout):
                time.sleep(1)
                exit_code = p.poll()
                if exit_code is not None:
                    success = exit_code == 0
                    break
            if success:
                print(f"SUCCESS: {model} prepared")
            else:
                print(f"FAIL: {model} not prepared")
                sys.exit(1)

    print("--- BENCHMARKING ITERATION ---")
    time.sleep(3)
    for model in args.model_names:
        tendency_across_model = ThroughputTendency()
        only_timeouts = False
        for batch_size in sorted(args.batch_sizes):
            tendency_across_bs = ThroughputTendency()
            if only_timeouts:
                break
            only_timeouts = True
            time_outs = 0
            for num_threads in reversed(sorted(args.num_threads)):
                num_processes = min(args.c, int(num_available_threads / num_threads)) if args.scenario == "offline" else 1

                case = f"{num_processes} x {num_threads} [proc x threads], bs = {batch_size}"
                print(f"\nRunning {case}")

                cmd = f"cd {args.mzdev_dir};"
                cmd += f" {args.pre_cmd}"
                if args.scenario == "ss":
                    cmd += f" python3 benchmark_ss.py " \
                           f"-f {args.framework} " \
                           f"-m {model} " \
                           f"-p {args.precision} " \
                           f"-t {str(num_threads)} " \
                           f"-b {str(batch_size)}"
                elif args.scenario == "offline":
                    cmd += f" python3 benchmark_offline.py " \
                           f"-f {args.framework} " \
                           f"-m {model} " \
                           f"-p {args.precision} " \
                           f"-n {str(num_processes)} " \
                           f"-t {str(num_threads)} " \
                           f"-b {str(batch_size)} " \
                           f"-r {args.threads_range}"
                else:
                    print_goodbye_message_and_die(f"Scenario {args.scenario} undefined.")
                if args.debug:
                    cmd += " --debug"
                cmd = ["docker", "exec", "-i", args.docker_name, "bash", "-c", cmd]

                print(f"Executing: {' '.join(cmd)}")
                mem_max = 0
                success = False
                timed_out = True
                start = time.time()
                p = subprocess.Popen(cmd, start_new_session=True)
                while time.time() - start < args.timeout:
                    time.sleep(3)
                    if args.memory:
                        stats = subprocess.check_output(
                            ['docker', 'stats', '--format', 'json', '--no-stream']).decode().split("\n")[:-1]
                        for container in stats:
                            data = json.loads(container)
                            if data["Name"] == args.docker_name:
                                mem = data['MemUsage'].split(" / ")[0].split("iB")[0]
                                prefix = mem[-1]
                                mem_usage = float(mem[:-1])
                                mem_usage *= {"T": 2<<19, "G": 2<<9, "M": 1, "K": 1/(2<<9)}[prefix]
                                mem_max = max(mem_max, mem_usage)
                                break
                    exit_code = p.poll()
                    if exit_code is not None:
                        success = exit_code == 0
                        only_timeouts = timed_out = False
                        time_outs = 2
                        break 

                if success:
                    print(f"SUCCESS: {case}")
                    if args.memory:
                        dump_memory(
                            args.framework, model, args.precision, args.scenario,
                            {"batch_size": batch_size, "num_processes": num_processes, "num_threads": num_threads},
                            mem_max
                        )
                    if args.scenario == "offline" and not args.exhaustive:
                        latest_result = read_latest_result(
                            args.framework, model, args.precision, args.scenario,
                            configs[args.scenario], metrics[args.scenario], "throughput_total"
                        )
                        if not tendency_across_bs.is_promising(latest_result):
                            break
                else:
                    print(f"FAIL: {case}")
                    docker_restart(args.docker_name)
                    if args.memory:
                        dump_memory(
                            args.framework, model, args.precision, args.scenario,
                            {"batch_size": batch_size, "num_processes": num_processes, "num_threads": num_threads},
                            "F"
                        )
                    dump_result(
                        args.framework, model, args.precision, args.scenario,
                        configs[args.scenario],
                        {"batch_size": batch_size, "num_processes": num_processes, "num_threads": num_threads},
                        metrics[args.scenario],
                        {k: "F" for k in metrics[args.scenario]}
                    )
                    if timed_out:
                        time_outs += 1
                    if time_outs >= 3:
                        break
                    
            if args.scenario == "offline" and not args.exhaustive:
                if not tendency_across_model.is_promising(tendency_across_bs.max_value):
                    break
    print("\nDONE")


if __name__ == "__main__":
    main()
