import argparse
import models.tf as tf_models
import models.tflite as tflite_models
import models.ort as ort_models
import models.pytorch as pytorch_models
import models.ctranslate as ctranslate_models
from ampere_model_library.utils.misc import print_warning_message


def parse_args():
    parser = argparse.ArgumentParser(description="Test all models.")
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        help="batch size to feed the model with")
    return parser.parse_args()


def main():
    def run_model(models, models_collection, precision):
        for model_name in models_collection.keys():
            model = models.get_model(model_name, precision)
            acc_metrics, _ = model.dataset.handler(
                model, args.batch_size, model.get_test_num_runs(args.batch_size), None)
            model.dataset.check_accuracy(acc_metrics)

    args = parse_args()

    print("\nRunning available TF models in fp32 precision ...\n")
    run_model(tf_models, tf_models.FP32_MODELS, "fp32")

    print("\nRunning available TF models in fp16 precision ...\n")
    run_model(tf_models, tf_models.FP16_MODELS, "fp16")

    print("\nRunning available TFLite models in int8 precision ...\n")
    run_model(tflite_models, tflite_models.INT8_MODELS, "int8")

    print("\nRunning available ORT models in fp32 precision ...\n")
    run_model(ort_models, ort_models.FP32_MODELS, "fp32")

    print("\nRunning available ORT models in fp16 precision ...\n")
    run_model(ort_models, ort_models.FP16_MODELS, "fp16")

    print("\nRunning available PYTORCH models in fp32 precision ...\n")
    run_model(pytorch_models, pytorch_models.FP32_MODELS, "fp32")

    print("\nRunning available CTRANSLATE models in fp32 precision ...\n")
    run_model(ctranslate_models, ctranslate_models.FP32_MODELS, "fp32")

    print("\nRunning available CTRANSLATE models in fp16 precision ...\n")
    run_model(ctranslate_models, ctranslate_models.FP16_MODELS, "fp16")

    print("\nRunning available CTRANSLATE models in int8 precision ...\n")
    run_model(ctranslate_models, ctranslate_models.INT8_MODELS, "int8")


if __name__ == "__main__":
    main()
