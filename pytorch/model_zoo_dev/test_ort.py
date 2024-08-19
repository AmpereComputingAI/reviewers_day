import argparse
import models.ort as ort_models

from run_utils.o365 import check_threshold, write_result_to_excel
from run_utils.misc import SUPPORTED_DTYPES


def parse_args():
    parser = argparse.ArgumentParser(description="Test ONNX Runtime model accuracy.")
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
    parser.add_argument("-c", "--check_performance",
                        action="store_true",
                        help="compare the benchmark result against threshold set in Excel")
    return parser.parse_args()


def main():
    args = parse_args()
    model = ort_models.get_model(args.model_name, args.precision, args.import_all)
    acc_metrics, perf_metrics = model.dataset.handler(model, args.batch_size, model.get_test_num_runs(args.batch_size),
                                                      None)
    model.dataset.check_accuracy(acc_metrics)
    if args.check_performance:
        write_result_to_excel("AIO CI Perf Test.xlsx", "Results", args.model_name, "ort", args.precision,
                              args.batch_size or model.default_bs, perf_metrics)
        # check_threshold("AIO CI Perf Test.xlsx", "Latency Targets", args.model_name, "ort", args.precision, perf_metrics)


if __name__ == "__main__":
    main()
