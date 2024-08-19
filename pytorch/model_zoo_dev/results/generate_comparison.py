import json
import os
import sys
import argparse
import xlsxwriter
from xlsxwriter.exceptions import DuplicateWorksheetName

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

from pathlib import Path
from datetime import datetime
from results.generate_spreadsheet import DATA_FORMAT_VERSION, PRECISIONS, SCENARIOS, int_to_excel_column, \
    sort_combinations, headers_json, len_configs

COLORS = {"ratio": "#f0e5c5", "target": "#cadcc4", "ref": "#bdd0e1", "background": "#dcdcdc"}
precisions = ["most efficient"] + PRECISIONS
precisions_observed = []


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison based on .json data files created with "
                                                 "generate_spreadsheet.py script. 'Target' systems are systems that "
                                                 "will be compared with every 'reference' system. Example --- target: "
                                                 "A, B; reference: C, D, E --- generated sheets will be: "
                                                 "A vs C, A vs D, A vs E and B vs C, B vs D, B vs E --- ratios "
                                                 "calculated will be increasing in value with A/B getting advantage "
                                                 "over C/D/E.")
    parser.add_argument("--target",
                        required=True, type=str, nargs="+", help="names of .json files containing data on target "
                                                                 "systems")
    parser.add_argument("--reference",
                        required=True, type=str, nargs="+", help="names of .json files containing data on reference "
                                                                 "systems")
    return parser.parse_args()


def check_if_files_exist(filenames):
    for filename in filenames:
        if not Path(filename).exists():
            print(f"File {filename} doesn't exist!")
            sys.exit(1)


class Pointer:
    def __init__(self, default_row=1, default_col=1):
        self.default_row = default_row
        self.default_col = default_col
        self.row = default_row
        self.col = default_col
        self._sanity_check()

    def position(self):
        return self.row, self.col

    def _sanity_check(self):
        assert self.default_row >= 0 and self.default_col >= 0
        assert self.row >= 0 and self.col >= 0

    def set_default_row(self, row):
        self.default_row = row
        self.row = row
        self._sanity_check()
        return self

    def set_default_col(self, col):
        self.default_col = col
        self.col = col
        self._sanity_check()
        return self

    def reset_col(self):
        self.col = self.default_col
        return self

    def reset_row(self):
        self.row = self.default_row
        return self

    def down(self, n=1):
        self.row += n
        self._sanity_check()
        return self

    def up(self, n=1):
        self.row -= n
        self._sanity_check()
        return self

    def left(self, n=1):
        self.col -= n
        self._sanity_check()
        return self

    def right(self, n=1):
        self.col += n
        self._sanity_check()
        return self


def add_systems_header(pointer, worksheet, target_data, ref_data, bold, colors):
    def write_row(title, target, ref, key):
        worksheet.write(*pointer.position(), target[key], colors["target"])
        pointer.right()
        worksheet.write(*pointer.position(), None, colors["target"])
        pointer.right()
        worksheet.write(*pointer.position(), ref[key], colors["ref"])
        pointer.right()
        worksheet.write(*pointer.position(), None, colors["ref"])
        pointer.right()
        worksheet.write(*pointer.position(), title)
        pointer.reset_col().down()

    worksheet.write(*pointer.position(), target_data["details"]["cpu_model"], bold)
    pointer.right(2)
    worksheet.write(*pointer.position(), ref_data["details"]["cpu_model"], bold)
    pointer.reset_col().down()

    write_row("#  Raw data spreadsheet id", target_data, ref_data, "uuid")

    worksheet.write(*pointer.position(),
                    f"https://amperemail.sharepoint.com/sites/AmpereAIdatarepo/_layouts/15/search.aspx/siteall?q="
                    f"{target_data['uuid']}", colors["target"])
    pointer.right()
    worksheet.write(*pointer.position(), None, colors["target"])
    pointer.right()
    worksheet.write(*pointer.position(),
                    f"https://amperemail.sharepoint.com/sites/AmpereAIdatarepo/_layouts/15/search.aspx/siteall?q="
                    f"{ref_data['uuid']}", colors["ref"])
    pointer.right()
    worksheet.write(*pointer.position(), None, colors["ref"])
    pointer.right()
    worksheet.write(*pointer.position(), "#  Look up raw data spreadsheet in the data repo")
    pointer.reset_col().down()

    write_row("#  Date", target_data["details"], ref_data["details"], "date_conducted")
    write_row("#  Author's email", target_data["details"], ref_data["details"], "authors_email")
    write_row("#  Benchmark's scripting", target_data["details"], ref_data["details"], "scripts")

    worksheet.write(*pointer.position(), f"{target_data['details']['csp_or_system_name']} "
                                         f"{target_data['details']['instance_family']} "
                                         f"{target_data['details']['instance_shape_or_core_count']}",
                    colors["target"])
    pointer.right()
    worksheet.write(*pointer.position(), None, colors["target"])
    pointer.right()
    worksheet.write(*pointer.position(), f"{ref_data['details']['csp_or_system_name']} "
                                         f"{ref_data['details']['instance_family']} "
                                         f"{ref_data['details']['instance_shape_or_core_count']}",
                    colors["ref"])
    pointer.right()
    worksheet.write(*pointer.position(), None, colors["ref"])
    pointer.right()
    worksheet.write(*pointer.position(), "#  System")
    pointer.reset_col().down()

    worksheet.write(*pointer.position(), f"{target_data['details']['cpu_manufacturer']} "
                                         f"{target_data['details']['cpu_model']} @ "
                                         f"{target_data['details']['max_cpu_freq']}", colors["target"])
    pointer.right()
    worksheet.write(*pointer.position(), None, colors["target"])
    pointer.right()
    worksheet.write(*pointer.position(), f"{ref_data['details']['cpu_manufacturer']} "
                                         f"{ref_data['details']['cpu_model']} @ "
                                         f"{ref_data['details']['max_cpu_freq']}", colors["ref"])
    pointer.right()
    worksheet.write(*pointer.position(), None, colors["ref"])
    pointer.right()
    worksheet.write(*pointer.position(), "#  CPU")
    pointer.reset_col().down()

    write_row("#  Memory", target_data["details"], ref_data["details"], "memory")
    write_row("#  Linux kernel", target_data["details"], ref_data["details"], "linux_kernel")
    write_row("#  Software", target_data["details"], ref_data["details"], "software_stack")
    write_row("#  Other", target_data["details"], ref_data["details"], "other")

    pointer.reset_col()


def add_scenario_framework_header(pointer, worksheet, scenario, framework, bold):
    disclaimer = {"ss": "single-process latency", "offline": "multi-process throughput"}[scenario]
    worksheet.write(*pointer.position(), f"{scenario} - {framework}{3 * ' '}[{disclaimer}]", bold)
    if scenario == "offline":
        pointer.down()
        worksheet.write(*pointer.position(), "ips = instances / second", bold)
        pointer.up()


def add_map(pointer, worksheet, frameworks):
    worksheet.write(*pointer.position(), "MAP")
    cells = {scenario: {} for scenario in SCENARIOS}
    for framework in frameworks:
        pointer.right()
        worksheet.write(*pointer.position(), framework)
        for i, scenario in enumerate(SCENARIOS):
            cells[scenario][framework] = (pointer.row + i + 1, pointer.col)
    pointer.reset_col()
    for scenario in SCENARIOS:
        pointer.down()
        worksheet.write(*pointer.position(), scenario)
    return cells


def add_model_header(
        pointer, worksheet, scenario, is_power_comparison, target_name, ref_name,
        bold_bg, colors, align_left, align_right):
    unit = {"ss": "ms", "offline": "ips"}[scenario]
    pointer.right(3)
    for precision in precisions:
        if precision in precisions_observed:
            worksheet.write(*pointer.position(), precision, bold_bg)
            pointer.right()
            for _ in range(2):
                worksheet.write(*pointer.position(), None, colors["background"])
                pointer.right()
            if is_power_comparison:
                for _ in range(5):
                    worksheet.write(*pointer.position(), None, colors["background"])
                    pointer.right()
            pointer.right()
    pointer.reset_col().down()
    pointer.right(3)
    for precision in precisions:
        if precision in precisions_observed:
            worksheet.write(*pointer.position(), None, colors["background"])
            pointer.right()
            for pos in [target_name, ref_name]:
                worksheet.write(*pointer.position(), pos, colors["background"])
                pointer.right()
            if is_power_comparison:
                worksheet.write(*pointer.position(), None, colors["background"])
                pointer.right()
                for _ in range(2):
                    for pos in [target_name, ref_name]:
                        worksheet.write(*pointer.position(), pos, colors["background"])
                        pointer.right()
            pointer.right()
    pointer.reset_col().down()
    for pos in ["model", "num threads"]:
        worksheet.write(*pointer.position(), pos)
        pointer.right()
    for precision in precisions:
        if precision in precisions_observed:
            pointer.right()
            if is_power_comparison:
                worksheet.write(*pointer.position(), "perf / power ratio", align_left)
                pointer.right()
                for _ in range(2):
                    worksheet.write(*pointer.position(), "ips / Watt", align_right)
                    pointer.right()
            worksheet.write(*pointer.position(), "perf ratio", align_left)
            pointer.right()
            for _ in range(2):
                worksheet.write(*pointer.position(), unit, align_right)
                pointer.right()
            if is_power_comparison:
                for _ in range(2):
                    worksheet.write(*pointer.position(), "Watt", align_right)
                    pointer.right()
    pointer.reset_col()


def get_union_of_keys(a, b):
    return sorted(set(list(a.keys()) + list(b.keys())))


def get_intersection_of_keys(a, b):
    return sorted([item for item in a.keys() if item in b.keys()])


class ComparisonError(Exception):
    pass


def get_smallest_common_bs(target_results, ref_results):
    sorted_target_results = sort_combinations("ss", target_results["combinations"])
    bs_target = [combination[headers_json["ss"].index("batch_size")] for combination in sorted_target_results]
    bs_ref = [combination[headers_json["ss"].index("batch_size")] for combination in ref_results["combinations"]]
    for bs in bs_target:
        if bs in bs_ref:
            return bs
    raise ComparisonError


def extract_model_data(scenario, target_results, ref_results):
    if scenario == "ss":
        rows = {}
        best = {precision: {name: {"perf": float("inf"), "power": None}
                            for name in ["target", "ref"]} for precision in precisions}
        bs = get_smallest_common_bs(target_results, ref_results)
        combinations = target_results["combinations"] + \
                       [x for x in ref_results["combinations"] if x not in target_results["combinations"]]
        for combination in sort_combinations(scenario, combinations):
            if bs == combination[headers_json["ss"].index("batch_size")]:
                threads = combination[headers_json["ss"].index("num_threads")]
                rows[threads] = {"most efficient": {
                    "target": {"perf": float("inf"), "power": None}, "ref": {"perf": float("inf"), "power": None}}}
                for precision in PRECISIONS:
                    rows[threads][precision] = {}
                    for name, results in [("target", target_results), ("ref", ref_results)]:
                        try:
                            results_precision = results[precision]
                        except KeyError:
                            continue
                        rows[threads][precision][name] = {}
                        for result in results_precision:
                            if combination == result[:len_configs[scenario]]:
                                try:
                                    latency = float(result[headers_json[scenario].index("latency_median_ms")])
                                except ValueError:
                                    continue
                                try:
                                    power = float(result[headers_json[scenario].index("power_watts")])
                                except IndexError:
                                    power = None
                                rows[threads][precision][name]["perf"] = latency
                                rows[threads][precision][name]["power"] = power
                                if rows[threads]["most efficient"][name]["perf"] > latency:
                                    rows[threads]["most efficient"][name]["perf"] = latency
                                    rows[threads]["most efficient"][name]["power"] = power
                                if best[precision][name]["perf"] > latency:
                                    best[precision][name]["perf"] = latency
                                    best[precision][name]["power"] = power
                                if best["most efficient"][name]["perf"] > latency:
                                    best["most efficient"][name]["perf"] = latency
                                    best["most efficient"][name]["power"] = power
        rows["best"] = best
        return rows
    elif scenario == "offline":
        rows = {
            "per thread": {"most efficient":
                               {"target": {"perf": 0., "power": None}, "ref": {"perf": 0., "power": None}}},
            "full system": {"most efficient":
                                {"target": {"perf": 0., "power": None}, "ref": {"perf": 0., "power": None}}}}
        for precision in PRECISIONS:
            rows["per thread"][precision] = {}
            rows["full system"][precision] = {}
            for name, results in [("target", target_results), ("ref", ref_results)]:
                thread_count = None
                max_throughput = 0.
                power = None
                try:
                    results_precision = results[precision]
                except KeyError:
                    continue
                rows["per thread"][precision][name] = {}
                rows["full system"][precision][name] = {}
                for result in results_precision:
                    try:
                        throughput = float(result[headers_json[scenario].index("throughput_total")])
                        if throughput > max_throughput:
                            max_throughput = throughput
                            try:
                                power = float(result[headers_json[scenario].index("power_watts")])
                            except IndexError:
                                power = None
                            thread_count = int(result[headers_json[scenario].index("num_processes")]) * int(
                                result[headers_json[scenario].index("num_threads")])
                    except ValueError:
                        continue

                rows["full system"][precision][name]["perf"] = max_throughput
                rows["full system"][precision][name]["power"] = power
                if max_throughput > rows["full system"]["most efficient"][name]["perf"]:
                    rows["full system"]["most efficient"][name]["perf"] = max_throughput
                    rows["full system"]["most efficient"][name]["power"] = power

                per_thread_throughput = max_throughput / thread_count
                rows["per thread"][precision][name]["perf"] = per_thread_throughput
                if per_thread_throughput > rows["per thread"]["most efficient"][name]["perf"]:
                    rows["per thread"]["most efficient"][name]["perf"] = per_thread_throughput

        return rows
    else:
        assert False


def load_data_from_json(filepath):
    data = json.load(open(filepath, "r"))
    assert data["version"] == DATA_FORMAT_VERSION
    return data, data["details"]["cpu_model"]


def generate_comparison(target_systems, reference_systems):
    global precisions_observed
    check_if_files_exist(target_systems + reference_systems)
    filename = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_comparison_spreadsheet.xlsx"
    print(f"Creating spreadsheet {filename} ...")
    workbook = xlsxwriter.Workbook(filename)
    colors = {key: workbook.add_format({"bg_color": color}) for key, color in COLORS.items()}
    bold = workbook.add_format({"bold": True})
    bold_bg = workbook.add_format({"bold": True, "bg_color": COLORS["background"]})
    align_left = workbook.add_format({"align": "left", "bg_color": COLORS["background"]})
    align_right = workbook.add_format({"align": "right", "bg_color": COLORS["background"]})
    number_formats = {key: workbook.add_format({"num_format": "#,##0.00", "bg_color": COLORS[key]})
                      for key in ["target", "ref"]}
    number_formats["ratio"] = workbook.add_format(
        {"num_format": "#,##0.00", "bg_color": COLORS["ratio"], "align": "center"})
    number_formats_highlighted = {key: workbook.add_format(
        {"num_format": "#,##0.00", "font_color": "red", "bg_color": COLORS[key]}) for key in
        ["ratio", "target", "ref"]}

    for target in target_systems:
        target_data, target_name = load_data_from_json(target)
        for ref in reference_systems:
            ref_data, ref_name = load_data_from_json(ref)
            for i in range(100):
                # 31 chars is the limit of sheet name in xlsx
                sheet_name = f"<{i}> {target_name[:11]} vs {ref_name[:11]}"
                try:
                    worksheet = workbook.add_worksheet(sheet_name)
                    break
                except DuplicateWorksheetName:
                    assert i < 99
                    continue
            print(f"Generating sheet {sheet_name} ....")
            pointer = Pointer()
            add_systems_header(pointer, worksheet, target_data, ref_data, bold, colors)
            pointer.down(2)
            power_comparison = target_data["power_data_available"] == ref_data["power_data_available"] == True
            target_results, ref_results = target_data["results"], ref_data["results"]
            frameworks = get_union_of_keys(target_results, ref_results)
            precisions_observed = ["most efficient"] + \
                                  list(set(target_data["details"]["precisions"] + ref_data["details"]["precisions"]))
            map_cells = add_map(pointer, worksheet, frameworks)
            pointer.down(3)
            for scenario in SCENARIOS:
                for framework in frameworks:
                    try:
                        target_models, ref_models = \
                            target_results[framework][scenario], ref_results[framework][scenario]
                    except KeyError:
                        continue
                    start_cell = f"{int_to_excel_column(pointer.col)}{pointer.row + 1}"
                    lowest_row, rightmost_col = pointer.position()
                    add_scenario_framework_header(pointer, worksheet, scenario, framework, bold)
                    pointer.down(2)
                    for model in get_intersection_of_keys(target_models, ref_models):
                        try:
                            model_data = extract_model_data(scenario, target_models[model], ref_models[model])
                        except ComparisonError:
                            continue
                        add_model_header(
                            pointer, worksheet, scenario, power_comparison, target_name, ref_name,
                            bold_bg, colors, align_left, align_right
                        )
                        pointer.down()
                        worksheet.write(*pointer.position(), model)
                        pointer.set_default_row(pointer.row)
                        pointer.set_default_col(2)
                        for threads in model_data.keys():
                            worksheet.write(*pointer.position(), threads)
                            for precision in precisions:
                                if precision in precisions_observed:
                                    pointer.right(2)
                                    if power_comparison:
                                        if_error = f"IFERROR({int_to_excel_column(pointer.col + 1)}{pointer.row + 1}/" \
                                                   f"{int_to_excel_column(pointer.col + 2)}{pointer.row + 1}, \"\")"
                                        worksheet.write_formula(
                                            *pointer.position(),
                                            f"=IF({if_error}=0, \"\", {if_error})",
                                            number_formats["ratio"]
                                        )
                                        for name in ["target", "ref"]:
                                            pointer.right()
                                            formula = f"{int_to_excel_column(pointer.col + 3)}{pointer.row + 1}/" \
                                                      f"{int_to_excel_column(pointer.col + 5)}{pointer.row + 1}"
                                            if scenario == "ss":
                                                formula = "1000/" + formula
                                            worksheet.write(
                                                *pointer.position(),
                                                f"=IFERROR({formula}, \"\")",
                                                number_formats[name]
                                            )
                                        pointer.right()
                                    if scenario == "ss":
                                        if_error = f"IFERROR({int_to_excel_column(pointer.col + 2)}{pointer.row + 1}/" \
                                                   f"{int_to_excel_column(pointer.col + 1)}{pointer.row + 1}, \"\")"
                                    elif scenario == "offline":
                                        if_error = f"IFERROR({int_to_excel_column(pointer.col + 1)}{pointer.row + 1}/" \
                                                   f"{int_to_excel_column(pointer.col + 2)}{pointer.row + 1}, \"\")"
                                    else:
                                        assert False
                                    worksheet.write_formula(
                                        *pointer.position(),
                                        f"=IF({if_error}=0, \"\", {if_error})",
                                        number_formats["ratio"]
                                    )
                                    for name in ["target", "ref"]:
                                        pointer.right()
                                        try:
                                            value = model_data[threads][precision][name]["perf"]
                                            if value != 0:
                                                if scenario == "ss" and \
                                                        value == model_data["best"][precision][name]["perf"]:
                                                    worksheet.write(
                                                        *pointer.position(), value,
                                                        number_formats_highlighted[name]
                                                    )
                                                else:
                                                    worksheet.write(
                                                        *pointer.position(), value,
                                                        number_formats[name]
                                                    )
                                        except (KeyError, TypeError):
                                            worksheet.write(
                                                *pointer.position(), None,
                                                number_formats[name]
                                            )
                                    if power_comparison:
                                        for name in ["target", "ref"]:
                                            pointer.right()
                                            try:
                                                worksheet.write(
                                                    *pointer.position(), model_data[threads][precision][name]["power"],
                                                    number_formats[name]
                                                )
                                            except KeyError:
                                                worksheet.write(
                                                    *pointer.position(), None,
                                                    number_formats[name]
                                                )
                            rightmost_col = max(rightmost_col, pointer.col)
                            lowest_row = max(lowest_row, pointer.row)
                            pointer.reset_col().down()
                        pointer.set_default_col(1).down()
                    finish_cell = f"{int_to_excel_column(rightmost_col)}{lowest_row + 1}"
                    worksheet.write_url(
                        *map_cells[scenario][framework],
                        url=f"internal:'{sheet_name}'!{start_cell}:{finish_cell}",
                        string=f"{start_cell}:{finish_cell}")
                    pointer.down()
    workbook.close()
    print("\nComparison generated.")


if __name__ == "__main__":
    args = parse_args()
    generate_comparison(args.target, args.reference)
