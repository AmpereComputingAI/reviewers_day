import os
import csv
import sys
import argparse
import uuid
import json
import xlsxwriter

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from datetime import datetime
from results.utils import get_results_path
from benchmark_set import CONFIG_SS, CONFIG_OFFLINE, METRICS_SS, METRICS_OFFLINE, METADATA
from power.log_power import POWER_FILENAME, POWER_HEADER

DATA_FORMAT_VERSION = "v1.1"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET_LEN = len(ALPHABET)
PRECISIONS = ["fp32", "fp16", "bf16", "int8"]
SCENARIOS = ["ss", "offline"]
META_DATA_FORMAT = {"framework": 0, "model_name": 1, "precision": 2, "scenario": 3}
HEADERS_CSV = {"ss": CONFIG_SS + METRICS_SS + METADATA, "offline": CONFIG_OFFLINE + METRICS_OFFLINE + METADATA}

headers_json = {"ss": CONFIG_SS + METRICS_SS + ["power_watts"],
                "offline": CONFIG_OFFLINE + METRICS_OFFLINE + ["power_watts"]}
len_configs = {"ss": len(CONFIG_SS), "offline": len(CONFIG_OFFLINE)}
metrics_ss = METRICS_SS
metrics_offline = METRICS_OFFLINE
len_metrics = {"ss": len(METRICS_SS), "offline": len(METRICS_OFFLINE)}
power_data_available = None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate spreadsheet from .csv files with results.")
    parser.add_argument("--details",
                        type=str, required=True, help=".json file with filled out benchmark details")
    parser.add_argument("--results",
                        type=str, default=get_results_path(), help="directory with .csv files")
    parser.add_argument("--do_prune",
                        default=False, action="store_true",
                        help="prune .csv files, i.e. in the case of same configuration covered more than once pick the "
                             "best result")
    return parser.parse_args()


def int_to_excel_column(n):
    """
    Converts int n to a letter value compatible with Excel columns.
    0 -> 'A'
    1 -> 'B'
    26 -> 'AA'
    27 -> 'AB'

    :param n: int, column index
    :return: str, column name
    """

    if n < 0:
        return ""
    return int_to_excel_column((n - ALPHABET_LEN) // ALPHABET_LEN) + ALPHABET[n % ALPHABET_LEN]


def sort_combinations(scenario, combinations: list):
    bs_to_t = dict()
    for i, combination in enumerate(combinations):
        bs = int(combination[HEADERS_CSV[scenario].index("batch_size")])
        t = int(combination[HEADERS_CSV[scenario].index("num_threads")])
        if bs in bs_to_t.keys():
            bs_to_t[bs][t] = i
        else:
            bs_to_t[bs] = {t: i}
    sorted_combinations = list()
    for bs in sorted(bs_to_t.keys()):
        for t in sorted(bs_to_t[bs].keys()):
            sorted_combinations.append(combinations[bs_to_t[bs][t]])
    return sorted_combinations


def prune_csv_results(results_dir, filename, scenario):
    pruned_results = dict()
    with open(os.path.join(results_dir, filename)) as csv_file:
        csv_reader = csv.reader(csv_file)
        first_line = True
        for line in csv_reader:
            if first_line:
                assert line == HEADERS_CSV[scenario], \
                    f"Format of results in file {filename} doesn't match expected format " \
                    f"[{','.join(HEADERS_CSV[scenario])}]"
                first_line = False
                continue
            results = line[len_configs[scenario]:]
            if str(line[:len_configs[scenario]]) in pruned_results.keys():
                alternative_results = pruned_results[str(line[:len_configs[scenario]])]
                assert scenario in SCENARIOS
                if scenario == "ss":
                    try:
                        alternative_value = float(alternative_results[metrics_ss.index("latency_median_ms")])
                    except ValueError:
                        alternative_value = None
                    try:
                        value = float(results[metrics_ss.index("latency_median_ms")])
                    except ValueError:
                        value = None
                    if alternative_value is None or (value is not None and value < alternative_value):
                        pruned_results[str(line[:len_configs[scenario]])] = results
                elif scenario == "offline":
                    try:
                        alternative_value = float(alternative_results[metrics_offline.index("throughput_total")])
                    except ValueError:
                        alternative_value = None
                    try:
                        value = float(results[metrics_offline.index("throughput_total")])
                    except ValueError:
                        value = None
                    if alternative_value is None or (value is not None and value > alternative_value):
                        pruned_results[str(line[:len_configs[scenario]])] = results
                else:
                    assert False
            else:
                pruned_results[str(line[:len_configs[scenario]])] = results
    # re-assemble .csv file
    tmp_filename = "pruned_tmp"
    with open(os.path.join(results_dir, tmp_filename), "w") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS_CSV[scenario])
        for key, value in pruned_results.items():
            writer.writerow(eval(key) + value)
    return tmp_filename


def calc_average_power(power_readings: list[tuple[float, float]], period_start: str, period_end: str):
    try:
        period_start, period_end = float(period_start), float(period_end)
    except ValueError:
        assert period_start == "F" and period_end == "F"
        return "F"
    power_over_period = []
    for reading in power_readings:
        if period_start <= reading[1] <= period_end:
            power_over_period.append(reading[0])
        if reading[1] > period_end:
            break
    assert len(power_over_period) > 0
    return sum(power_over_period) / len(power_over_period)


def load_data(results_dir, do_prune=False):
    def prepare_placeholder():
        framework = meta_data[META_DATA_FORMAT["framework"]]
        model_name = meta_data[META_DATA_FORMAT["model_name"]]
        if framework in data.keys():
            if scenario in data[framework].keys():
                if model_name in data[framework][scenario].keys():
                    data[framework][scenario][model_name][precision] = list()
                else:
                    data[framework][scenario][model_name] = {precision: list(), "combinations": list()}
            else:
                data[framework][scenario] = {model_name: {precision: list(), "combinations": list()}}
        else:
            data[framework] = {scenario: {model_name: {precision: list(), "combinations": list()}}}
        return data[framework][scenario][model_name]

    global power_data_available
    power_data_available = POWER_FILENAME in os.listdir(results_dir)
    power_readings = []
    if power_data_available:
        global metrics_ss, metrics_offline, len_metrics
        metrics_ss, metrics_offline = metrics_ss + ["power_watts"], metrics_offline + ["power_watts"]
        len_metrics = {"ss": len(metrics_ss), "offline": len(metrics_offline)}
        with open(os.path.join(results_dir, POWER_FILENAME)) as csv_file:
            csv_reader = csv.reader(csv_file)
            first_line = True
            for line in csv_reader:
                if first_line:
                    assert line == POWER_HEADER, \
                        f"Format of results in file {POWER_FILENAME} doesn't match expected format " \
                        f"[{','.join(POWER_HEADER)}]"
                    first_line = False
                    continue
                power_readings.append(tuple(float(val) for val in line))

    data = dict()
    precisions_observed = set()
    for filename in [filename for filename in os.listdir(results_dir)
                     if ".csv" == filename[-len(".csv"):] and filename != "power.csv"]:
        meta_data = filename[:-len(".csv")].split("@")
        assert len(meta_data) == len(META_DATA_FORMAT), \
            f"Name of results csv file doesn't match expected format " \
            f"[{'@'.join(META_DATA_FORMAT.keys())}.csv]: {filename}"

        scenario = meta_data[META_DATA_FORMAT["scenario"]]
        precision = meta_data[META_DATA_FORMAT["precision"]]
        precisions_observed.add(precision)
        placeholder = prepare_placeholder()

        if do_prune:
            filename = prune_csv_results(results_dir, filename, scenario)

        local_combinations = list()
        with open(os.path.join(results_dir, filename)) as csv_file:
            csv_reader = csv.reader(csv_file)
            first_line = True
            for line in csv_reader:
                if first_line:
                    assert line == HEADERS_CSV[scenario], \
                        f"Format of results in file {filename} doesn't match expected format " \
                        f"[{','.join(HEADERS_CSV[scenario])}]"
                    first_line = False
                    continue
                elif line[:len_configs[scenario]] not in placeholder["combinations"]:
                    placeholder["combinations"].append(line[:len_configs[scenario]])
                if power_data_available:
                    placeholder[precision].append(line[:-2] + [calc_average_power(power_readings, line[-2], line[-1])])
                else:
                    placeholder[precision].append(line[:-2])
                assert line[:len_configs[scenario]] not in local_combinations, \
                    f"Ambiguous results for config {line[:len_configs[scenario]]} - file: {filename}"
                local_combinations.append(line[:len_configs[scenario]])
    return data, precisions_observed


def add_general_header(details, precisions_observed, worksheet, scenario, unique_id):
    worksheet.write(0, 0, unique_id)
    details = {
        "Date:": details["date_conducted"],
        "Author's email:": details["authors_email"],
        "Benchmark's scripting:": details["scripts"],
        "System:": f"{details['csp_or_system_name']} {details['instance_family']} "
                  f"{details['instance_shape_or_core_count']}",
        "CPU:": f"{details['cpu_manufacturer']} {details['cpu_model']} @ {details['max_cpu_freq']}",
        "Memory:": details["memory"],
        "Linux kernel:": details["linux_kernel"],
        "Software:": details["software_stack"],
        "Other:": details["other"]
    }
    row, col = 2, 1  # arbitrary
    for key, value in details.items():
        worksheet.write(row, col, key)
        worksheet.write(row, col+1, value)
        row += 1
    row, col = row + 2, col + len_configs[scenario] + 2
    for precision in PRECISIONS:
        if precision in precisions_observed:
            worksheet.write(row, col, precision)
            col += len_metrics[scenario] + 1
    return row


def add_model_header(precisions_observed, worksheet, scenario, model_name, row):
    positions = ["model"] + {"ss": CONFIG_SS, "offline": CONFIG_OFFLINE}[scenario]
    col = 1
    for i in range(len(positions)):
        worksheet.write(row, col + i, positions[i])
    col = col + len(positions) + 1
    for precision in PRECISIONS:
        if precision in precisions_observed:
            for i in range(len_metrics[scenario]):
                worksheet.write(row, col + i, {"ss": metrics_ss, "offline": metrics_offline}[scenario][i])
            col += 1 + len_metrics[scenario]
    worksheet.write(row + 1, 1, model_name)


def generate_spreadsheet(details_path, results_dir, do_prune=False):
    try:
        details = json.load(open(details_path, "r"))
    except json.decoder.JSONDecodeError:
        print(f"Modify {details_path} file and run again.")
        sys.exit(1)
    data, precisions_observed = load_data(results_dir, do_prune=do_prune)
    if len(data.keys()) == 0:
        print(f"No data found in {results_dir}/ directory!")
        sys.exit(1)
    details["precisions"] = list(precisions_observed)
    unique_id = str(uuid.uuid4())
    filename = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_{details['cpu_manufacturer']}_" \
               f"{details['cpu_model']}_{details['csp_or_system_name']}_{details['instance_family']}_{unique_id}"
    filename = filename.replace('/', '-').replace(' ', '_')
    with open(f"{filename}.json", "w") as json_file:
        json.dump(
            {"details": details,
             "results": data,
             "power_data_available": power_data_available,
             "uuid": unique_id,
             "version": DATA_FORMAT_VERSION,
             "format": headers_json},
            json_file)
    workbook = xlsxwriter.Workbook(f"{filename}.xlsx")
    number_format = workbook.add_format({"num_format": "#,##0.00"})
    for framework in data.keys():
        for scenario in data[framework].keys():
            worksheet = workbook.add_worksheet(f"{scenario} - {framework}")
            row = add_general_header(details, precisions_observed, worksheet, scenario, unique_id) + 2
            for model_name in sorted(data[framework][scenario].keys()):
                add_model_header(precisions_observed, worksheet, scenario, model_name, row)
                row += 1
                col = 2
                combination_to_row_offset = dict()
                sorted_combinations = sort_combinations(scenario, data[framework][scenario][model_name]["combinations"])
                for i, combination in enumerate(sorted_combinations):
                    for j in range(len_configs[scenario]):
                        worksheet.write(row + i, col + j, int(combination[j]))
                    bs = int(combination[HEADERS_CSV[scenario].index("batch_size")])
                    t = int(combination[HEADERS_CSV[scenario].index("num_threads")])
                    if bs in combination_to_row_offset.keys():
                        combination_to_row_offset[bs][t] = i
                    else:
                        combination_to_row_offset[bs] = {t: i}
                worksheet.write(row + len(sorted_combinations), col+len_configs[scenario]-1,
                                {"ss": "min", "offline": "max"}[scenario])

                col += 1 + len_configs[scenario]
                for precision in PRECISIONS:
                    if precision not in precisions_observed \
                            or precision not in data[framework][scenario][model_name].keys():
                        continue
                    for result in data[framework][scenario][model_name][precision]:
                        for i in range(len_metrics[scenario]):
                            try:
                                value = float(result[len_configs[scenario] + i])
                            except ValueError:
                                value = "F"
                            worksheet.write(
                                row + combination_to_row_offset[
                                    int(result[HEADERS_CSV[scenario].index("batch_size")])][
                                    int(result[HEADERS_CSV[scenario].index("num_threads")])
                                ],
                                col + i,
                                value,
                                number_format
                            )
                    for i in range(len_metrics[scenario]-power_data_available):
                        worksheet.write_formula(
                            row + len(sorted_combinations), col + i,
                            "=" + {'ss': 'MIN', 'offline': 'MAX'}[scenario] +
                            f"({int_to_excel_column(col+i)}{row+1}:"
                            f"{int_to_excel_column(col+i)}{row+len(sorted_combinations)})",
                            number_format
                        )
                    col += 1 + len_metrics[scenario]
                row += 2 + len(data[framework][scenario][model_name]["combinations"])
    workbook.close()
    print("Files\n\n" + 8 * " " + f"- {filename}.json\n" + 8 * " " + f"- {filename}.xlsx\n\n" + 20 * " " + "created!")
    print(f"\nRemember to keep both of them!")


if __name__ == "__main__":
    args = parse_args()
    generate_spreadsheet(args.details, args.results, args.do_prune)
