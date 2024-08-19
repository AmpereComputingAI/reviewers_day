import os


def get_results_path():
    """
    A function returning absolute path to results dir.

    :return: str, path to results dir
    """
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "csv_files")
    os.makedirs(path, exist_ok=True)
    return path
