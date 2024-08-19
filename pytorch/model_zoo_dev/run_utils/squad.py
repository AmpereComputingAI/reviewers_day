import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path
from ampere_model_library.utils.misc import print_goodbye_message_and_die


class SQuAD_v1_1:
    """
    A class initializing and handling the test on SQuAD dataset.
    """
    def __init__(self, exact_match_ref, f1_ref):
        self.name = 'SQuAD'
        self.__exact_match_ref = exact_match_ref
        self.__f1_ref = f1_ref
        self.__threshold = 0.05

        self.__squad_file_name = "dev-v1.1.json"
        self.__squad_link = "https://ampereaimodelzoo.s3.amazonaws.com/dev-v1.1.json"

        self.__squad_dir_path = pathlib.Path(get_downloads_path(), "squad_v1.1")

    def init_squad(self):
        """
        A function initializing Squad v1.1 dataset.

        If dataset is not detected on the device a download will be attempted.

        :return: path to Squad dataset file
        """

        if not self.__squad_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__squad_link])
                subprocess.run(["mkdir", str(self.__squad_dir_path)])
                subprocess.run(["mv", self.__squad_file_name, str(self.__squad_dir_path)])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__squad_file_name])

        return pathlib.Path(self.__squad_dir_path, self.__squad_file_name)

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading Squad dataset if needed
            2. downloading the model if needed
            3. feeding Squad-related networks' run functions with proper arguments

        :param model: Model object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        unique_args = {"squad_path": self.init_squad()}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected Squad v1.1 metrics
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
