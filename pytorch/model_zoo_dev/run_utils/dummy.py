import sys


class Dummy:
    """
    A dummy placeholder for dataset handling.
    """

    def __init__(self):
        self.name = 'Dummy'

    def handler(self, model, batch_size, num_runs, timeout):

        unique_args = {}
        model.download_model_maybe()
        return model.run_func(**model.match_args(batch_size, num_runs, timeout, unique_args))

    def check_accuracy(self, outputs_array):
        print("Accuracy check not available!")
        sys.exit(1)
