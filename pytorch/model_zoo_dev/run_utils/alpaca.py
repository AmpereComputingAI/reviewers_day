import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path


class AlpacaInstruct:
    """
    A class initializing and handling the test on Alpaca Instruct dataset.
    """
    def __init__(self, exact_match_ref, f1_ref):
        self.name = 'AlpacaInstruct'
        self.__exact_match_ref = exact_match_ref
        self.__f1_ref = f1_ref
        self.__threshold = 0.05

        self.__file_name = "alpaca_data.json"
        self.__link = "https://ampereaimodelzoo.s3.amazonaws.com/alpaca_data.json"

        self.__dir_path = pathlib.Path(get_downloads_path(), "alpaca_instruct")

    def init_alpaca_instruct(self):
        """
        A function initializing Alpaca Instruct dataset.

        If dataset is not detected on the device a download will be attempted.

        :return: path to Alpaca Instruct dataset file
        """

        if not self.__dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__link])
                subprocess.run(["mkdir", str(self.__dir_path)])
                subprocess.run(["mv", self.__file_name, str(self.__dir_path)])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__file_name])

        return pathlib.Path(self.__dir_path, self.__file_name)

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading Alpaca Instruct dataset if needed
            2. downloading the model if needed
            3. feeding Alpaca-related networks' run functions with proper arguments

        :param model: Model object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        unique_args = {"dataset_path": self.init_alpaca_instruct()}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected Alpaca Instruct metrics
        """
        exact_match = metrics_dict["exact_match"]
        f1 = metrics_dict["f1"]
        exact_match_loss = 1.0 - exact_match / self.__exact_match_ref
        f1_loss = 1.0 - f1 / self.__f1_ref
        if exact_match_loss > self.__threshold or f1_loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" Exact match: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__exact_match_ref, exact_match, exact_match_loss * 100))
            print(" F1 score: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__f1_ref, f1, f1_loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
