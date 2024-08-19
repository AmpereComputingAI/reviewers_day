import os
import sys
import run_utils.imagenet as imagenet_utils
import run_utils.coco as coco_utils
from run_utils.misc import Model, list_available_models, lazy_import

try:
    from ampere_model_library.utils.misc import print_goodbye_message_and_die
except ModuleNotFoundError or ImportError as e:
    print(e)
    print("\ngit submodule update --init --recursive")
    sys.exit(1)

aml_path = os.path.dirname(os.path.realpath(__file__))[:-len("models")]
aml_path += "ampere_model_library"
sys.path.append(aml_path)

########################################################################################################################
# import new model here \/ \/ \/
########################################################################################################################

try:

    run_densenet_169 = lazy_import("ampere_model_library.computer_vision.classification.densenet_169.run")
    run_inception_resnet_v2 = lazy_import("ampere_model_library.computer_vision.classification.inception_resnet_v2.run")
    run_inception_v2 = lazy_import("ampere_model_library.computer_vision.classification.inception_v2.run")
    run_inception_v3 = lazy_import("ampere_model_library.computer_vision.classification.inception_v3.run")
    run_mobilenet_v1 = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v1.run")
    run_nasnet_mobile = lazy_import("ampere_model_library.computer_vision.classification.nasnet_mobile.run")
    run_resnet_50_v2 = lazy_import("ampere_model_library.computer_vision.classification.resnet_50_v2.run")
    run_mobilenet_v2 = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v2.run")
    run_nasnet_large = lazy_import("ampere_model_library.computer_vision.classification.nasnet_large.run")
    run_resnet_50_v15 = lazy_import("ampere_model_library.computer_vision.classification.resnet_50_v15.run")
    run_squeezenet = lazy_import("ampere_model_library.computer_vision.classification.squeezenet.run")
    run_vgg_16 = lazy_import("ampere_model_library.computer_vision.classification.vgg_16.run")
    run_vgg_19 = lazy_import("ampere_model_library.computer_vision.classification.vgg_19.run")

    run_ssd_mobilenet_v2 = lazy_import("ampere_model_library.computer_vision.object_detection.ssd_mobilenet_v2.run")
except ModuleNotFoundError or ImportError as e:
    print(e)
    print("\ngit submodule update --init --recursive")
    sys.exit(1)


########################################################################################################################
# add new model to proper dict \/ \/ \/
# model class params: dataset, run_func, default_bs, link=None, file_path=None, model_name=None, model_args=dict(), test_num_runs=1000
########################################################################################################################


FP32_MODELS = {
    # to be populated
}

INT8_MODELS = {

    # classification models \/ \/ \/ #

    "densenet_169": {
        "dataset": imagenet_utils.ImageNet(0.690, 0.893),
        "run_func": run_densenet_169,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/densenet_169_tflite_int8.tflite",
        "file_path": "densenet_169_tflite_int8.tflite"
    },
    "inception_resnet_v2": {
        "dataset": imagenet_utils.ImageNet(0.775, 0.935),
        "run_func": run_inception_resnet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_resnet_v2_tflite_int8.tflite",
        "file_path": "inception_resnet_v2_tflite_int8.tflite"
    },
    "inception_v2": {
        "dataset": imagenet_utils.ImageNet(0.723, 0.903),
        "run_func": run_inception_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v2_tflite_int8.tflite",
        "file_path": "inception_v2_tflite_int8.tflite"
    },
    "inception_v3": {
        "dataset": imagenet_utils.ImageNet(0.757, 0.932),
        "run_func": run_inception_v3,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v3_tflite_int8.tflite",
        "file_path": "inception_v3_tflite_int8.tflite"
    },
    "mobilenet_v1": {
        "dataset": imagenet_utils.ImageNet(0.701, 0.888),
        "run_func": run_mobilenet_v1,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v1_tflite_int8.tflite",
        "file_path": "mobilenet_v1_tflite_int8.tflite"
    },
    "mobilenet_v2": {
        "dataset": imagenet_utils.ImageNet(0.690, 0.903),
        "run_func": run_mobilenet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v2_tflite_int8.tflite",
        "file_path": "mobilenet_v2_tflite_int8.tflite"
    },
    "nasnet_large": {
        "dataset": imagenet_utils.ImageNet(0.808, 0.958),
        "run_func": run_nasnet_large,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/nasnet_large_tflite_int8.tflite",
        "file_path": "nasnet_large_tflite_int8.tflite"
    },
    "nasnet_mobile": {
        "dataset": imagenet_utils.ImageNet(0.730, 0.906),
        "run_func": run_nasnet_mobile,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mnasnet_tflite_int8.tflite",
        "file_path": "mnasnet_tflite_int8.tflite"
    },
    "resnet_50_v2": {
        "dataset": imagenet_utils.ImageNet(0.674, 0.855),
        "run_func": run_resnet_50_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v2_tflite_int8.tflite",
        "file_path": "resnet_50_v2_tflite_int8.tflite"
    },
    "resnet_50_v1.5": {
        "dataset": imagenet_utils.ImageNet(0.744, 0.913),
        "run_func": run_resnet_50_v15,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v15_tflite_int8.tflite",
        "file_path": "resnet_50_v15_tflite_int8.tflite"
    },
    "squeezenet": {
        "dataset": imagenet_utils.ImageNet(0.47, 0.714),
        "run_func": run_squeezenet,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/squeezenet_tflite_int8.tflite",
        "file_path": "squeezenet_tflite_int8.tflite"
    },
    "vgg_16": {
        "dataset": imagenet_utils.ImageNet(0.655, 0.874),
        "run_func": run_vgg_16,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/vgg_16_tflite_int8.tflite",
        "file_path": "vgg_16_tflite_int8.tflite"
    },
    "vgg_19": {
        "dataset": imagenet_utils.ImageNet(0.668, 0.874),
        "run_func": run_vgg_19,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/vgg_19_tflite_int8.tflite",
        "file_path": "vgg_19_tflite_int8.tflite"
    },

    # object detection models \/ \/ \/ #

    "ssd_mobilenet_v2": {
        "dataset": coco_utils.COCO(0.194),
        "run_func": run_ssd_mobilenet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_mobilenet_v2_tflite_int8.tflite",
        "file_path": "ssd_mobilenet_v2_tflite_int8.tflite"
    },
}


########################################################################################################################
########################################################################################################################


def get_model(model_name, precision, import_all=False):
    """
    A function returning a ModelClass of requested model at given precision.

    :param model_name: str, name of the model
    :param precision: str, precision of the model
    :return: ModelClass object
    """
    if import_all:
        for model in FP32_MODELS.values():
            model["run_func"].run_tflite_fp32
        for model in INT8_MODELS.values():
            model["run_func"].run_tflite_int8
    if precision == "fp32":
        try:
            config = FP32_MODELS[model_name]
            config["run_func"] = config["run_func"].run_tflite_fp32
            return Model(**config)
        except KeyError:
            list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "fp16":
        if os.getenv("ALLOW_IMPLICIT_FP16", "0") == "1":
            try:
                config = FP32_MODELS[model_name]
                config["run_func"] = config["run_func"].run_tflite_fp32
                return Model(**config)
            except KeyError:
                list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "bf16":
        if os.getenv("ALLOW_IMPLICIT_BF16", "0") == "1":
            try:
                config = FP32_MODELS[model_name]
                config["run_func"] = config["run_func"].run_tflite_fp32
                return Model(**config)
            except KeyError:
                list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "int8":
        try:
            config = INT8_MODELS[model_name]
            config["run_func"] = config["run_func"].run_tflite_int8
            return Model(**config)
        except KeyError:
            list_available_models(model_name, precision, INT8_MODELS.keys())
    print_goodbye_message_and_die(f"Model {model_name} not available in {precision} precision!")
