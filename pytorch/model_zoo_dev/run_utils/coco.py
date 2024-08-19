import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path


class COCO:
    """
    A class initializing and handling the test COCO Dataset.
    """
    def __init__(self, coco_map_ref):
        self.name = 'COCO'
        self.__coco_map_ref = coco_map_ref
        self.__threshold = 0.05

        self.__coco_file_name = "COCO2014_onspecta.tar.gz"
        self.__coco_link = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_onspecta.tar.gz"
        self.__annotations_file_name = "COCO2014_anno_onspecta.json"
        self.__annotations_link = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_anno_onspecta.json"

        self.__coco_dir_path = pathlib.Path(get_downloads_path(), "coco")
        self.__annotations_file_path = pathlib.Path(self.__coco_dir_path, self.__annotations_file_name)

    def init_coco(self):
        """
        A function initializing COCO 2014 Validation dataset.

        If dataset or annotations are not detected on the device a download will be attempted.

        :return: path to COCO dataset dir, path to COCO annotations file
        """
        if not self.__coco_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__coco_link])
                subprocess.run(["mkdir", str(self.__coco_dir_path)])
                subprocess.run(["tar", "-xf", self.__coco_file_name, "-C", str(self.__coco_dir_path)])
                subprocess.run(["rm", self.__coco_file_name])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__coco_file_name])
        if not self.__annotations_file_path.is_file():
            try:
                subprocess.run(["wget", self.__annotations_link])
                subprocess.run(["mv", self.__annotations_file_name, str(self.__coco_dir_path)])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__annotations_file_name])

        return pathlib.Path(self.__coco_dir_path, "COCO2014_onspecta"), self.__annotations_file_path

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading COCO dataset if needed
            2. downloading the model if needed
            3. feeding COCO-related networks' run functions with proper arguments

        :param model: ModelClass object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        coco_dir_path, anno_file_path = self.init_coco()
        unique_args = {"images_path": coco_dir_path, "anno_path": anno_file_path}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected COCO mAP metric
        """
        coco_map = metrics_dict["coco_map"]
        map_loss = 1.0 - coco_map / self.__coco_map_ref
        if map_loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" COCO mAP acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__coco_map_ref, coco_map, map_loss * 100))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
