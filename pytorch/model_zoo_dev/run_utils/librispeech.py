import sys
import pathlib
import subprocess
from downloads.utils import get_downloads_path
from run_utils.misc import Dataset


class LibriSpeech:
    """
    A class initializing and handling the test on LibriSpeech dataset.
    """
    def __init__(self, accuracy_ref):
        self.name = 'LibriSpeech'
        self.__accuracy_ref = accuracy_ref
        self.__threshold = 0.05

        self.__librispeech_file_name = "LibriSpeech.tar.gz"
        self.__librispeech_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/LibriSpeech.tar.gz"

        self.__librispeech_dir_path = pathlib.Path(get_downloads_path(), "librispeech")

    def init_librispeech(self):
        """
        A function initializing LibriSpeech dataset.

        If dataset is not detected on the device a download will be attempted.

        :return: path to LibriSpeech dataset dir
        """

        if not self.__librispeech_dir_path.is_dir():
            try:
                subprocess.run(["wget", self.__librispeech_link])
                subprocess.run(["mkdir", str(self.__librispeech_dir_path)])
                subprocess.run(["tar", "-xf", self.__librispeech_file_name, "-C", str(self.__librispeech_dir_path)])
                subprocess.run(["rm", self.__librispeech_file_name])
            except KeyboardInterrupt:
                subprocess.run(["rm", self.__librispeech_file_name])

        return pathlib.Path(self.__librispeech_dir_path, "LibriSpeech")

    def handler(self, model, batch_size, num_runs, timeout):
        """
        A function taking care of:
            1. downloading LibriSpeech dataset if needed
            2. downloading the model if needed
            3. feeding LibriSpeech-related networks' run functions with proper arguments

        :param model: Model object
        :param batch_size: int, batch size to be used
        :param num_runs: number of runs to go with
        :param timeout: timeout (gets overridden by number of runs)
        :return: dicts with acc & perf metrics
        """

        librispeech_dir_path = self.init_librispeech()
        unique_args = {"dataset_path": librispeech_dir_path}
        model.download_model_maybe()

        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, metrics_dict):
        """
        A function calculating relative deviation from reference value and deciding on test result re threshold (5%).

        :param metrics_dict: dict, dictionary containing expected accuracy
        """
        accuracy = metrics_dict["accuracy"]
        accuracy_loss = 1.0 - accuracy / self.__accuracy_ref

        if accuracy_loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" Accuracy: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.__accuracy_ref, accuracy, accuracy_loss * 100))

            sys.exit(1)
        else:
            print("\nTEST PASSED\n")


class LibriSpeechASRDummy(Dataset):
    def __init__(self, ref_wer_score):
        super().__init__()
        self.name = "hf-internal-testing/librispeech_asr_dummy"
        self.ref_wer_score = ref_wer_score
        self.__threshold = 0.05

    def check_accuracy(self, metrics_dict):
        wer_score = metrics_dict["wer_score"]
        loss = 1.0 - self.ref_wer_score / wer_score 
        if loss > self.__threshold:
            print("\nTEST FAILED!\n")
            print(" Acc: expected = {:.3f}, observed = {:.3f}, relative loss = {:.0f}%".format(
                self.ref_wer_score, wer_score, loss * 100))
            sys.exit(1)
        elif not 0. <= wer_score <= 1.0:
            print("\nTEST FAILED!\n")
            print(" Accuracy result should be in range <0., 1.>, observed = {:.3f}".format(wer_score))
            sys.exit(1)
        else:
            print("\nTEST PASSED\n")
