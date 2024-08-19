import os
import csv
import sys

for filename in [filename for filename in os.listdir(sys.argv[1]) if ".csv" == filename[-len(".csv"):] and filename != "power.csv"]:
    new_lines = []
    with open(os.path.join(sys.argv[1], filename)) as csv_file:
        csv_reader = csv.reader(csv_file)
        first_line = True
        for line in csv_reader:
            if "offline" in filename:
                if len(line) != 6:
                    if first_line:
                        new_lines.append(['batch_size', 'num_processes', 'num_threads', 'throughput_total', 'start_timestamp', 'finish_timestamp'])
                    else:
                        assert len(line) == 4
                        new_lines.append(line + ['F', 'F'])
                else:
                    new_lines.append(line)
            elif "ss" in filename:
                if len(line) != 7:
                    if first_line:
                        new_lines.append(
                            ['batch_size', 'num_threads', 'latency_mean_ms', 'latency_median_ms', 'latency_90th_percentile_ms', 'start_timestamp', 'finish_timestamp'])
                    else:
                        assert len(line) == 5
                        new_lines.append(line + ['F', 'F'])
                else:
                    new_lines.append(line)
            else:
                assert False
            first_line = False
    with open(os.path.join(sys.argv[1], filename), "w") as f:
        writer = csv.writer(f)
        for line in new_lines:
            writer.writerow(line)