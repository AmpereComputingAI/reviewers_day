import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path


class BraTS19:
    """
    A class initializing and handling the test on BraTS19 dataset.
    """
    def __init__(self, mean_whole_tumor_ref, mean_tumor_core_ref, mean_enhancing_tumor_ref):
        self.name = 'BraTS19'
        self.__mean_whole_tumor_ref = mean_whole_tumor_ref
        self.__mean_tumor_core_ref = mean_tumor_core_ref
        self.__mean_enhancing_tumor_ref = mean_enhancing_tumor_ref
        self.__threshold = 0.05

        self.__brats_file_name = "brats_19.tar.gz"
        self.__brats_link = "https://ampereaimodelzoo.s3.amazonaws.com/brats_19.tar.gz"

        self.__brats_dir_path = pathlib.Path(get_downloads_path(), "brats_19")

    def init_brats(self):
        """
        A function initializing BraTS 2019 dataset.

        If dataset is not detected on the device a download will be attempted.

        :return: path to BraTS 2019 dataset dir
        """

        if not self.__brats_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__brats_link])
                subprocess.run(["tar", "-xf", self.__brats_file_name, "-C", str(get_downloads_path())])
                subprocess.run(["rm", self.__brats_file_name])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__brats_file_name])

        return self.__brats_dir_path

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading BraTS dataset if needed
            2. downloading the model if needed
            3. feeding BraTS-related networks' run functions with proper arguments

        :param model: ModelClass object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        unique_args = {"dataset_path": self.init_brats()}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected BraTS 2019 metrics
        """
        mean_whole_tumor_acc = metrics_dict["mean_whole_tumor_acc"]
        mean_tumor_core_acc = metrics_dict["mean_tumor_core_acc"]
        mean_enhancing_tumor_acc = metrics_dict["mean_enhancing_tumor_acc"]

        mean_whole_tumor_acc_loss = 1.0 - mean_whole_tumor_acc / self.__mean_whole_tumor_ref
        mean_tumor_core_acc_loss = 1.0 - mean_tumor_core_acc / self.__mean_tumor_core_ref
        mean_enhancing_tumor_acc_loss = 1.0 - mean_enhancing_tumor_acc / self.__mean_enhancing_tumor_ref

        losses = [mean_whole_tumor_acc_loss, mean_tumor_core_acc_loss, mean_enhancing_tumor_acc_loss]
        if any([loss > self.__threshold for loss in losses]):
            print("\nTEST FAILED!\n")
            print(" Mean whole tumor acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__mean_whole_tumor_ref, mean_whole_tumor_acc, mean_whole_tumor_acc_loss * 100))
            print(" Mean tumor core acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__mean_tumor_core_ref, mean_tumor_core_acc, mean_tumor_core_acc_loss * 100))
            print(" Mean enhancing tumor acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__mean_enhancing_tumor_ref, mean_enhancing_tumor_acc, mean_enhancing_tumor_acc_loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")

