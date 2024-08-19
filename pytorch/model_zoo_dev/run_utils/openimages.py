import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path


class OpenImages:
    """
    A class initializing and handling the validation OpenImages Dataset.
    """
    def __init__(self, openimages_map_ref):
        self.name = 'OpenImages'
        self.__openimages_map_ref = openimages_map_ref
        self.__threshold = 0.05

        self.__openimages_file_name = "openimages-ampere.tar.gz"
        self.__openimages_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/openimages-ampere.tar.gz"
        self.__annotations_file_name = "openimages-mlperf.json"

        self.__openimages_dir_path = pathlib.Path(get_downloads_path(), "openimages-ampere")
        self.__annotations_file_path = pathlib.Path(self.__openimages_dir_path, "openimages-ampere", "annotations", self.__annotations_file_name)

    def init_openimages(self):
        """
        A function initializing OpenImages Validation dataset.

        If dataset or annotations are not detected on the device a download will be attempted.

        :return: path to OpenImages dataset dir, path to OpenImages annotations file
        """
        if not self.__openimages_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__openimages_link])
                subprocess.run(["mkdir", str(self.__openimages_dir_path)])
                subprocess.run(["tar", "-xf", self.__openimages_file_name, "-C", str(self.__openimages_dir_path)])
                subprocess.run(["rm", self.__openimages_file_name])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__openimages_file_name])

        return pathlib.Path(self.__openimages_dir_path, "openimages-ampere", "validation", "data"), self.__annotations_file_path

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading OpenImages dataset if needed
            2. downloading the model if needed
            3. feeding OpenImages-related networks' run functions with proper arguments

        :param model: Model object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        openimages_dir_path, anno_file_path = self.init_openimages()
        unique_args = {"images_path": openimages_dir_path, "anno_path": anno_file_path}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected OpenImages mAP metric
        """
        openimages_map = metrics_dict["coco_map"]
        map_loss = 1.0 - openimages_map / self.__openimages_map_ref
        if map_loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" OpenImages mAP acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__openimages_map_ref, openimages_map, map_loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
