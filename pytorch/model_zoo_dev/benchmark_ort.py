import argparse
import models.ort as ort_models
from run_utils.misc import SUPPORTED_DTYPES


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX Runtime benchmark.")
    parser.add_argument("-m", "--model_name",
                        type=str, required=True,
                        help="name of the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=SUPPORTED_DTYPES, required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=15.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="num of runs to execute")
    parser.add_argument("-i", "--import_all",
                        action="store_true",
                        help="import all available models")
    return parser.parse_args()


def main():
    args = parse_args()
    model = ort_models.get_model(args.model_name, args.precision, args.import_all)
    model.dataset.handler(model, args.batch_size, args.num_runs, args.timeout)


if __name__ == "__main__":
    main()
