import pathlib
import subprocess
import sys
from downloads.utils import get_downloads_path


class Criteo:

    def __init__(self, accuracy_ref):
        self.name = "Criteo"
        self.accuracy_ref = accuracy_ref
        self.__threshold = 0.05
        self.__criteo_file_name = "criteo_preprocessed.tar.gz"
        self.__criteo_link = "https://ampereaimodelzoo.s3.amazonaws.com/criteo_preprocessed.tar.gz"
        self.__criteo_dir_path = pathlib.Path(get_downloads_path(), "criteo")

    def init_criteo(self):
        """
        A function initializing Criteo dataset.

        If dataset is not detected on the device a download will be attempted.

        :return: path to Criteo dataset dir
        """
        if not self.__criteo_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__criteo_link])
                subprocess.run(["mkdir", str(self.__criteo_dir_path)])
                subprocess.run(["tar", "-xf", self.__criteo_file_name, "-C", str(self.__criteo_dir_path)])
                subprocess.run(["rm", self.__criteo_file_name])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__criteo_file_name])

        return pathlib.Path(self.__criteo_dir_path, "criteo_preprocessed")

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading Criteo dataset if needed
            2. downloading the model if needed
            3. feeding Criteo-related networks' run functions with proper arguments

        :param model: Model object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        dataset_path = self.init_criteo()
        unique_args = {"dataset_path": dataset_path, "debug": model.model_args["debug"]}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dict containing accuracy metrics
        """
        accuracy = metrics_dict["accuracy"]
        loss = 1.0 - accuracy / self.accuracy_ref
        if loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" Acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.accuracy_ref, accuracy, loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
