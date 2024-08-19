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
    run_densenet_121 = lazy_import("ampere_model_library.computer_vision.classification.densenet_121.run")
    run_inception_v2 = lazy_import("ampere_model_library.computer_vision.classification.inception_v2.run")
    run_mobilenet_v2 = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v2.run")
    run_resnet_50_v1 = lazy_import("ampere_model_library.computer_vision.classification.resnet_50_v1.run")
    run_resnet_50_v15 = lazy_import("ampere_model_library.computer_vision.classification.resnet_50_v15.run")
    run_shufflenet = lazy_import("ampere_model_library.computer_vision.classification.shufflenet.run")
    run_vgg_16 = lazy_import("ampere_model_library.computer_vision.classification.vgg_16.run")

    run_ssd_mobilenet_v2 = lazy_import("ampere_model_library.computer_vision.object_detection.ssd_mobilenet_v2.run")
    run_yolo_v3 = lazy_import("ampere_model_library.computer_vision.object_detection.yolo_v3.run")
except ModuleNotFoundError or ImportError as e:
    print(e)
    print("\ngit submodule update --init --recursive")
    sys.exit(1)


########################################################################################################################
# add new model to proper dict \/ \/ \/
# model class params: dataset, run_func, default_bs, link=None, file_path=None, model_name=None, model_args=dict(), test_num_runs=1000
########################################################################################################################

FP32_MODELS = {
    # classification models

    "densenet_121": {
        "dataset": imagenet_utils.ImageNet(0.72, 0.916),
        "run_func": run_densenet_121,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/densenet121.onnx",
        "file_path": "densenet121.onnx",
        "test_num_runs": 557
    },
    "mobilenet_v2": {
        "dataset": imagenet_utils.ImageNet(0.693, 0.886),
        "run_func": run_mobilenet_v2,
        "default_bs": 1,
        "link": "https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz",
        "file_path": "mobilenetv2-1.0/mobilenetv2-1.0.onnx",
        "test_num_runs": 900
    },
    "resnet_50_v1": {
        "dataset": imagenet_utils.ImageNet(0.694, 0.919),
        "run_func": run_resnet_50_v1,
        "default_bs": 1,
        "link": "https://zenodo.org/records/4735647/files/resnet50_v1.onnx",
        "file_path": "resnet50_v1.onnx",
        "test_num_runs": 62
    },
    "shufflenet": {
        "dataset": imagenet_utils.ImageNet(0.593, 0.822),
        "run_func": run_shufflenet,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/shufflenet.onnx",
        "file_path": "shufflenet.onnx"
    },
    "vgg_16": {
        "dataset": imagenet_utils.ImageNet(0.608, 0.825),
        "run_func": run_vgg_16,
        "default_bs": 1,
        "link": "https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz",
        "file_path": "vgg16/vgg16.onnx",
        "test_num_runs": 462
    },

    # object detection models

    "ssd_mobilenet_v2": {
        "dataset": coco_utils.COCO(0.264),
        "run_func": run_ssd_mobilenet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/tf2onnx_ssd_mobilenet_v2.onnx",
        "file_path": "tf2onnx_ssd_mobilenet_v2.onnx",
        "test_num_runs": 200
    },
    "yolo_v3": {
        "dataset": coco_utils.COCO(0.363),
        "run_func": run_yolo_v3,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/yolov3-10.onnx",
        "file_path": "yolov3-10.onnx",
        "test_num_runs": 30
    }
}

FP16_MODELS = {
    # classification models

    "inception_v2": {
        "dataset": imagenet_utils.ImageNet(0.712, 0.894),
        "run_func": run_inception_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v2_fp16.onnx",
        "file_path": "inception_v2_fp16.onnx",
        "test_num_runs": 170
    },
    "mobilenet_v2": {
        "dataset": imagenet_utils.ImageNet(0.689, 0.906),
        "run_func": run_mobilenet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v2_fp16.onnx",
        "file_path": "mobilenet_v2_fp16.onnx",
        "test_num_runs": 415
    },
    "resnet_50_v1.5": {
        "dataset": imagenet_utils.ImageNet(0.738, 0.925),
        "run_func": run_resnet_50_v15,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v1.5_fp16.onnx",
        "file_path": "resnet_50_v1.5_fp16.onnx",
        "test_num_runs": 80
    },
    "vgg_16": {
        "dataset": imagenet_utils.ImageNet(0.663, 0.877),
        "run_func": run_vgg_16,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/vgg_16_frozen_fp16.onnx",
        "file_path": "vgg_16_frozen_fp16.onnx",
        "test_num_runs": 928
    }
}


def get_model(model_name, precision, import_all=False):
    """
    A function returning a ModelClass of requested model at given precision.

    :param model_name: str, name of the model
    :param precision: str, precision of the model
    :return: ModelClass object
    """
    if import_all:
        for model in FP32_MODELS.values():
            model["run_func"].run_ort_fp32
        for model in FP16_MODELS.values():
            model["run_func"].run_ort_fp16
    if precision == "fp32":
        try:
            config = FP32_MODELS[model_name]
            config["run_func"] = config["run_func"].run_ort_fp32
            return Model(**config)
        except KeyError:
            list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "fp16":
        try:
            config = FP16_MODELS[model_name]
            config["run_func"] = config["run_func"].run_ort_fp16
            return Model(**config)
        except KeyError:
            if os.getenv("ALLOW_IMPLICIT_FP16", "0") == "1":
                try:
                    config = FP32_MODELS[model_name]
                    config["run_func"] = config["run_func"].run_ort_fp32
                    return Model(**config)
                except KeyError:
                    list_available_models(model_name, precision, FP32_MODELS.keys())
            list_available_models(model_name, precision, FP16_MODELS.keys())
    elif precision == "bf16":
        if os.getenv("ALLOW_IMPLICIT_BF16", "0") == "1":
            try:
                config = FP32_MODELS[model_name]
                config["run_func"] = config["run_func"].run_ort_fp32
                return Model(**config)
            except KeyError:
                list_available_models(model_name, precision, FP32_MODELS.keys())
    print_goodbye_message_and_die(f"Model {model_name} not available in {precision} precision!")
