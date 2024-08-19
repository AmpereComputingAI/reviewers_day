import datetime
import os
import sys

from O365 import Account
from O365.excel import WorkBook

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
    return int_to_excel_column((n - 26) // 26) + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[n % 26]

def get_worksheet(filename, worksheet_name, credentials, tenant_id):
    """
    Gets a worksheet from Office 365 SharePoint

    :param filename: str, name of the spreadsheet file, eg. "performance_thresholds.xlsx"
    :param worksheet_name: str, name of the worksheet inside of the spreadsheet file
    :param credentials: (str, str), a tuple containing O365 application (client) id and O365 secret key
    :param tenant_id: str, directory (tenant) id
    :return: O365.excel.WorkSheet, requested WorkSheet object
    """

    account = Account(credentials, tenant_id=tenant_id, auth_flow_type="credentials")

    if not account.is_authenticated:
        account.authenticate()
    sharepoint = account.sharepoint().get_site("root", "sites/AmpereAIdatarepo")
    document_library_id = next((library.object_id for library in sharepoint.list_document_libraries() if library.name == "Documents"), None)
    root_folder = sharepoint.get_document_library(document_library_id).get_root_folder()
    benchmarks_folder = next((folder for folder in root_folder.get_items() if folder.name == "Benchmarks"), None)
    excel_file = next((item for item in benchmarks_folder.get_items() if item.name == filename), None)

    return WorkBook(excel_file).get_worksheet(worksheet_name)

def check_threshold(filename, worksheet_name, model, framework, precision, perf_metrics):
    """
    :param filename: str, name of the spreadsheet file, eg. "performance_thresholds.xlsx"
    :param worksheet_name: str, name of the worksheet inside of the spreadsheet file
    :param model: str, name of the model
    :param framework: str, name of the framework (tf, pytorch, ort)
    :param precision: str, requested precision (fp32, fp16, int8)
    :param perf_metrics: dict of str: int, dict containing performance metrics
    """
    
    model_col_idx = None
    model_col_name = None
    precision_col_name = None
    model_row = None
    next_framework_column = None
    last_checked_column = "ZZ"
    last_checked_row = 1000

    ws = get_worksheet(filename, worksheet_name, (os.environ['O365_APP_ID'], os.environ['O365_SECRET']), os.environ['O365_TENANT_ID'])
    
    if ws is None:
        print(f"FAIL: Performance check requested but worksheet not found")
        sys.exit(1)
    
    for col_idx, v in enumerate(ws.get_range(f'A2:{last_checked_column}2').values[0]):
        if model_col_name is None and v == framework:
            model_col_idx = col_idx
            model_col_name = int_to_excel_column(col_idx)
            continue
        if model_col_name is not None:
            if v != '':
                next_framework_column = int_to_excel_column(col_idx)
                break

    if model_col_name is None:
        print(f"FAIL: Performance check requested but framework '{framework}' not found in the worksheet")
        sys.exit(1)
    
    # It stops working when there are a lot of precision/framework combinations used. Just change the last_checked_column to a further one if necessary
    if next_framework_column is None:
        next_framework_column = last_checked_column
    
    for precision_idx, v in enumerate(ws.get_range(f'{model_col_name}3:{next_framework_column}3').values[0]):
        if v == precision:
            precision_col_name = int_to_excel_column(model_col_idx + precision_idx)
            break
    
    if precision_col_name is None:
        print(f"FAIL: Performance check requested but precision '{precision}' not found in the worksheet")
        sys.exit(1)
    
    # It will not work correctly if we test over last_checked_row - 3 models per framework, which should not happen soon. Just increase the number if it does
    model_col = ws.get_range(f"{model_col_name}4:{model_col_name}{last_checked_row}").get_used_range()
    for idx, v in enumerate(model_col.values):
        if model == v[0]:
            model_row = model_col.get_cell(idx, 0).row_index + 1
            break

    if model_row is None:
        print(f"FAIL: Performance check requested but model '{model}' not found in the worksheet")
        sys.exit(1)

    cell_value = ws.get_range(f"{precision_col_name}{model_row}").values[0][0]
    
    if cell_value == '':
            print(f"FAIL: Performance check requested but no valid threshold value found")
            sys.exit(1)

    threshold = float(cell_value)
    if perf_metrics['mean_lat_ms'] > threshold:
        print(f"FAIL: Latency does not meet the threshold! Mean latency: {perf_metrics['mean_lat_ms']} ms, threshold: {threshold} ms")
        sys.exit(1)

def write_result_to_excel(filename, worksheet_name, model, framework, precision, batch_size, perf_metrics):
    """
    Writes a benchmark result to an Excel file in Office 365 OneDrive

    :param filename: str, name of the spreadsheet file, eg. "performance_thresholds.xlsx"
    :param worksheet_name: str, name of the worksheet inside of the spreadsheet file
    :param model: str, name of the model
    :param framework: str, name of the framework (tf, pytorch, ort)
    :param precision: str, used precision (fp32, fp16, int8)
    :param batch_size: int, used batch size
    :param perf_metrics: dict of str: int, dict containing performance metrics

    :return: bool, True if written successfully, False otherwise
    """

    max_rows = 100_000 # 1_048_576 is the hard limit.

    ws = get_worksheet(filename, worksheet_name, (os.environ['O365_APP_ID'], os.environ['O365_SECRET']), os.environ['O365_TENANT_ID'])
    if ws is None:
        return False

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results = [perf_metrics[key] for key in ["mean_lat_ms", "median_lat_ms", "90th_percentile_lat_ms", "mean_throughput", "median_throughput", "90th_percentile_throughput"]]
    values = [timestamp, batch_size, os.environ["AIO_NUM_THREADS"], framework, model, precision, *results]
    
    last_column = int_to_excel_column(len(values) - 1)
    
    # Move all the rows down and insert new data at the top
    ws.get_range(f"A2:{last_column}2").insert_range("DOWN")
    first_row = ws.get_range(f"A2:{last_column}2")
    first_row.values = [values] # Has to be a list of lists

    # Remove the last row if it would exceed the limit
    last_row = ws.get_range(f"A{max_rows}:{last_column}{max_rows}")
    last_row.values = [['']*len(values)]

    first_row.update()
    last_row.update()
    return True