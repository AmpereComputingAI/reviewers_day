import argparse
import models.ctranslate as ctranslate_models
from ampere_model_library.utils.misc import print_warning_message
from run_utils.misc import SUPPORTED_DTYPES


def parse_args():
    parser = argparse.ArgumentParser(description="Test TensorFlow model accuracy.")
    parser.add_argument("-m", "--model_name",
                        type=str, required=True,
                        help="name of the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=SUPPORTED_DTYPES, required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        help="batch size to feed the model with")
    parser.add_argument("-i", "--import_all",
                        action="store_true",
                        help="import all available models")
    return parser.parse_args()


def main():
    args = parse_args()
    model = ctranslate_models.get_model(args.model_name, args.precision, args.import_all)
    acc_metrics, _ = model.dataset.handler(model, args.batch_size, model.get_test_num_runs(args.batch_size), None)
    model.dataset.check_accuracy(acc_metrics)


if __name__ == "__main__":
    main()
