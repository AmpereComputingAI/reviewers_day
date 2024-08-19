import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path


class ImageNet:
    """
    A class initializing and handling the test on ImageNet 2012 Validation dataset.
    """
    def __init__(self, top_1_ref, top_5_ref):
        self.name = 'ImageNet'
        self.__top_1_ref = top_1_ref
        self.__top_5_ref = top_5_ref
        self.__threshold = 0.05

        self.__imagenet_file_name = "ILSVRC2012_onspecta.tar.gz"
        self.__imagenet_link = "https://ampereaimodelzoo.s3.amazonaws.com/ILSVRC2012_onspecta.tar.gz"
        self.__labels_file_name = "imagenet_labels_onspecta.txt"
        self.__labels_link = "https://ampereaimodelzoo.s3.amazonaws.com/imagenet_labels_onspecta.txt"

        self.__imagenet_dir_path = pathlib.Path(get_downloads_path(), "imagenet")
        self.__labels_file_path = pathlib.Path(self.__imagenet_dir_path, self.__labels_file_name)

    def init_imagenet(self):
        """
        A function initializing ImageNet 2012 Validation dataset.

        If dataset or annotations are not detected on the device a download will be attempted.

        :return: path to ImageNet dataset dir, path to ImageNet labels file
        """

        if not self.__imagenet_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__imagenet_link])
                subprocess.run(["mkdir", str(self.__imagenet_dir_path)])
                subprocess.run(["tar", "-xf", self.__imagenet_file_name, "-C", str(self.__imagenet_dir_path)])
                subprocess.run(["rm", self.__imagenet_file_name])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__imagenet_file_name])
        if not self.__labels_file_path.is_file():
            try:
                subprocess.run(["wget", self.__labels_link])
                subprocess.run(["mv", self.__labels_file_name, str(self.__imagenet_dir_path)])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__labels_file_name])

        return pathlib.Path(self.__imagenet_dir_path, "ILSVRC2012_onspecta"), self.__labels_file_path

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading ImageNet dataset if needed
            2. downloading the model if needed
            3. feeding ImageNet-related networks' run functions with proper arguments

        :param model: ModelClass object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        imagenet_dir_path, labels_file_path = self.init_imagenet()
        unique_args = {"images_path": imagenet_dir_path, "labels_path": labels_file_path}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected Top-1, Top-5 metrics
        """
        top_1 = metrics_dict["top_1_acc"]
        top_5 = metrics_dict["top_5_acc"]
        t1_loss = 1.0 - top_1 / self.__top_1_ref
        t5_loss = 1.0 - top_5 / self.__top_5_ref
        if t1_loss > self.__threshold or t5_loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" Top-1 acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__top_1_ref, top_1, t1_loss * 100))
            print(" Top-5 acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__top_5_ref, top_5, t5_loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
