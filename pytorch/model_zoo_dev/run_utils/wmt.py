import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path
from ampere_model_library.utils.misc import print_goodbye_message_and_die


class WMT:
    """
    A class initializing and handling the test on WMT dataset.
    """
    def __init__(self, bleu_ref):
        self.name = 'WMT'
        self.__bleu_ref = bleu_ref
        self.__threshold = 0.05

        self.__wmt_file_name = "training-parallel-nc-v9.tgz"
        self.__wmt_link = "https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"
        self.__tokenizer_file_name = "en_de_sp.model"
        self.__tokenizer_link = "https://ampereaidevelop.s3.eu-central-1.amazonaws.com/en_de_sp.model"

        self.__wmt_dir_path = pathlib.Path(get_downloads_path(), "wmt")

    def init_wmt(self):
        """
        A function initializing WMT dataset.

        If dataset is not detected on the device a download will be attempted.

        :return: path to WMT dataset file, path to WMT targets file
        """

        if not self.__wmt_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__wmt_link])
                subprocess.run(["mkdir", str(self.__wmt_dir_path)])
                subprocess.run(["tar", "-xf", self.__wmt_file_name, "-C", str(self.__wmt_dir_path)])
                subprocess.run(["rm", self.__wmt_file_name])
                subprocess.run(["wget", self.__tokenizer_link])
                subprocess.run(["mv", self.__tokenizer_file_name, str(self.__wmt_dir_path)])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__wmt_file_name])

        return pathlib.Path(self.__wmt_dir_path, "training", "news-commentary-v9.de-en.en"), pathlib.Path(self.__wmt_dir_path, "training", "news-commentary-v9.de-en.de"), str(pathlib.Path(self.__wmt_dir_path, self.__tokenizer_file_name))

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading WMT dataset if needed
            2. downloading the model if needed
            3. feeding WMT-related networks' run functions with proper arguments

        :param model: Model object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """
        wmt_dataset_path, wmt_targets_path, tokenizer_path = self.init_wmt()
        unique_args = {"dataset_path": wmt_dataset_path, "targets_path": wmt_targets_path, "tokenizer_path": tokenizer_path}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected WMT metrics
        """
        bleu = metrics_dict["bleu"]
        bleu_loss = 1.0 - bleu / self.__bleu_ref
        if bleu_loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" Exact match: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__bleu_ref, bleu, bleu_loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
