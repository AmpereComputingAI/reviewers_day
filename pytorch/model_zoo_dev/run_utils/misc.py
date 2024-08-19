import os
import pathlib
import importlib.util
import sys
import subprocess
import Levenshtein as lev
from downloads.utils import get_downloads_path
from ampere_model_library.utils.misc import print_goodbye_message_and_die

SUPPORTED_DTYPES = ["fp32", "fp16", "bf16", "int8"]
if os.getenv("ENABLE_AIO_IMPLICIT_FP16", "0") == "1":
    os.environ["AIO_IMPLICIT_FP16_TRANSFORM_FILTER"] = ".*"
    os.environ["ALLOW_IMPLICIT_FP16"] = "1"


class Dataset:
    def __init__(self):
        self.unique_args = {}

    def prepare(self):
        pass

    def handler(self, model, batch_size, num_runs, timeout):
        self.prepare()
        model.download_model_maybe()
        return model.run_func(**model.match_args(batch_size, num_runs, timeout, self.unique_args))

    def check_accuracy(self, metrics_dict):
        # should print whether accuracy test is passed or not by comparing obtained score with ref value,
        # sys.exit(1) should be executed in case of fail
        raise NotImplementedError


def download_file(link, file_name, target_dir_path):
    """
    A function downloading the file and then moving it to target dir.

    :param link: str, link to the file
    :param file_name: str, name of the file to be downloaded
    :param target_dir_path: path that the downloaded file should be moved to
    """
    try:
        subprocess.run(["wget", link])
        subprocess.run(["mv", file_name, str(target_dir_path)])
    except KeyboardInterrupt:
        subprocess.run(["rm", file_name])


def untar_file(file_name, target_dir_path):
    """
    A function unpacking the tarball file and then removing it.

    :param file_name: str, name of the file to be unpacked
    :param target_dir_path: path to the downloads directory
    :return: PathLib.path, path to unpacked file
    """
    path_to_file = pathlib.Path(target_dir_path, file_name)
    try:
        subprocess.run(["tar", "xf", path_to_file, "-C", target_dir_path])
        subprocess.run(["rm", path_to_file])
    except KeyboardInterrupt:
        subprocess.run(["rm", path_to_file])


def unzip_file(file_name, target_dir_path):
    """
    A function unpacking the zip file and then removing it.

    :param file_name: str, name of the file to be unpacked
    :param target_dir_path: path to the downloads directory
    """
    path_to_file = pathlib.Path(target_dir_path, file_name)
    try:
        subprocess.run(["unzip", path_to_file, "-d", target_dir_path])
        subprocess.run(["rm", path_to_file])
    except KeyboardInterrupt:
        subprocess.run(["rm", path_to_file])


def list_available_models(model_name, precision, model_names):
    """
    A function listing models available in given precision.

    (and helping you find the droids you are looking for)

    :param model_name: str, name of the model as requested
    :param precision: str, precision of the model requested
    :param model_names: list or dict of keys with available models at given precision
    """
    model_names_list = list(model_names)
    model_names_list.sort()

    best_match = 0
    best_score = 0.0
    for i, model_name_1 in enumerate(model_names_list):
        match_score = lev.ratio(model_name.lower(), model_name_1.lower())
        if match_score > best_score:
            best_match = i
            best_score = match_score

    print(f"\nAvailable {precision} models:\n")
    for i, model_name_1 in enumerate(model_names_list):
        if i == best_match and best_score > 0.9:
            print(f"  {model_name_1} <--------- that's the one?")
        else:
            print(f"  {model_name_1}")


def lazy_import(name):
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def init_env_variables(num_threads):
    os.environ["IGNORE_DATASET_LIMITS"] = "1"

    os.environ["AIO_NUM_THREADS"] = str(num_threads)
    os.environ["DLS_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    results_dir = os.path.join(os.getcwd(), "cache")
    os.environ["RESULTS_DIR"] = results_dir
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        for filepath in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, filepath))
    else:
        os.mkdir(results_dir)
    return results_dir


class Model:
    """
    A class offering facilities for a given model.
    """

    def __init__(self, dataset, run_func, default_bs,
                 link=None, file_path=None, model_name=None, model_args=None, test_num_runs=1000):
        self.dataset = dataset
        self.run_func = run_func
        self.link = link
        if file_path is None:
            self.path_to = None
        else:
            self.path_to = pathlib.Path(get_downloads_path(), file_path)

        self.name = model_name
        assert type(default_bs) is int
        self.default_bs = default_bs
        if model_args is None:
            self.model_args = {}
        else:
            self.model_args = model_args
        self.test_num_runs = test_num_runs

    def get_test_num_runs(self, batch_size_requested):
        if batch_size_requested is None:
            return self.test_num_runs
        else:
            bs_ratio = batch_size_requested / self.default_bs
            return int(self.test_num_runs / bs_ratio)

    def get_available_args(self, batch_size_requested):
        available_args = self.run_func.__code__.co_varnames[:self.run_func.__code__.co_argcount]
        if "batch_size" not in available_args and batch_size_requested is not None and \
                batch_size_requested != self.default_bs:
            print_goodbye_message_and_die(
                f"Model doesn't allow batch size setting. The only available BS is: {self.default_bs}")
        return available_args

    def match_args(self, batch_size, num_runs, timeout, dataset_args):
        general_args = {
            "model_path": str(self.path_to),
            "model_name": self.name,
            "batch_size": batch_size if batch_size is not None else self.default_bs,
            "num_runs": num_runs,
            "timeout": timeout,
        }

        args = dict()
        for arg in self.get_available_args(batch_size):
            if arg in general_args.keys():
                args[arg] = general_args[arg]
            elif arg in dataset_args.keys():
                args[arg] = dataset_args[arg]
            elif arg in self.model_args.keys():
                args[arg] = self.model_args[arg]
            else:
                print_goodbye_message_and_die(f"Required argument {arg} of {self.run_func} unknown.")

        return args

    def download_model_maybe(self):
        """
        A function downloading the model in precision requested.

        :return: PathLib.path, path to downloaded model
        """
        if self.link is None:
            return
        file_name = self.link.split("/")[-1]
        if not self.path_to.is_file() and not self.path_to.is_dir():
            download_file(self.link, file_name, get_downloads_path())
            if "tar" in file_name:
                untar_file(file_name, get_downloads_path())
            if "zip" in file_name:
                unzip_file(file_name, get_downloads_path())
