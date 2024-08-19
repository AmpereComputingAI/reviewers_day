import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path


class KiTS19:
    """
    A class initializing and handling the test on KiTS19 dataset.
    """
    def __init__(self, mean_kidney_ref, mean_tumor_ref):
        self.name = "KiTS19"
        self.__mean_kidney_ref = mean_kidney_ref
        self.__mean_tumor_ref = mean_tumor_ref
        self.__threshold = 0.05

        self.__kits_file_name = "kits19_reduced.tar.gz"
        self.__kits_link = "https://ampereaimodelzoo.s3.amazonaws.com/kits19_reduced.tar.gz"

        self.__kits_dir_path = pathlib.Path(get_downloads_path(), "kits19")

    def init_kits(self):
        """
        A function initializing KiTS 2019 dataset.

        If dataset is not detected on the device a download will be attempted.

        :return: path to KiTS 2019 dataset dir
        """

        if not self.__kits_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__kits_link])
                subprocess.run(["tar", "-xf", self.__kits_file_name, "-C", str(get_downloads_path())])
                subprocess.run(["rm", self.__kits_file_name])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__kits_file_name])

        return self.__kits_dir_path

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading KiTS dataset if needed
            2. downloading the model if needed
            3. feeding KiTS-related networks' run functions with proper arguments

        :param model: ModelClass object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        unique_args = {"kits_path": self.init_kits()}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected KiTS 2019 metrics
        """
        mean_kidney_acc = metrics_dict["mean_kidney_acc"]
        mean_tumor_acc = metrics_dict["mean_tumor_acc"]
        mean_kidney_acc_loss = 1.0 - mean_kidney_acc / self.__mean_kidney_ref
        mean_tumor_acc_loss = 1.0 - mean_tumor_acc / self.__mean_tumor_ref
        if mean_kidney_acc_loss > self.__threshold or mean_tumor_acc_loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" Mean kidney accuracy: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__mean_kidney_ref, mean_kidney_acc, mean_kidney_acc_loss * 100))
            print(" Mean tumor accuracy: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__mean_tumor_ref, mean_tumor_acc, mean_tumor_acc_loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
