import os
import csv
import sys
import time
import argparse
import importlib
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from results.utils import get_results_path

SAMPLING_RATE = 1.0
POWER_HEADER = ["power_watts", "timestamp"]
POWER_FILENAME = "power.csv"


def log_power(sample_power):
    time.sleep(3)
    power_log_path = os.path.join(get_results_path(), POWER_FILENAME)
    signal_file = Path("/tmp/log_power")
    signal_file.touch()
    x = 0
    period = 1 / SAMPLING_RATE
    with open(power_log_path, "w") as log:
        writer = csv.writer(log)
        writer.writerow(POWER_HEADER)
    try:
        while signal_file.exists():
            time.sleep(max(period-(time.time()-x), 0))
            x = time.time()
            with open(power_log_path, "a") as log:
                writer = csv.writer(log)
                writer.writerow([sample_power(), time.time()])
    except KeyboardInterrupt:
        pass
    print(f"\nLog available at {power_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Collects power samples at {SAMPLING_RATE}Hz.")
    parser.add_argument("-s", "--sampler",
                        type=str, required=True,
                        help="Path to .py file with function sample_power() defined. Function should take no arguments "
                             "and return instantaneous power draw in Watts as float. Response time of the function "
                             f"should not exceed {1/SAMPLING_RATE} seconds and ideally it should be much lower.")
    log_power(importlib.import_module(parser.parse_args().sampler).sample_power)
