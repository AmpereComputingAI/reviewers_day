import os
import sys
import run_utils.imagenet as imagenet_utils
import run_utils.coco as coco_utils
import run_utils.squad as squad_utils
import run_utils.kits as kits_utils
import run_utils.brats as brats_utils
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
    run_mobilenet_v2 = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v2.run")
    run_resnet_50_v15 = lazy_import("ampere_model_library.computer_vision.classification.resnet_50_v15.run")
    run_nasnet_large = lazy_import("ampere_model_library.computer_vision.classification.nasnet_large.run")
    run_inception_resnet_v2 = lazy_import("ampere_model_library.computer_vision.classification.inception_resnet_v2.run")
    run_inception_v2 = lazy_import("ampere_model_library.computer_vision.classification.inception_v2.run")
    run_inception_v3 = lazy_import("ampere_model_library.computer_vision.classification.inception_v3.run")
    run_inception_v4 = lazy_import("ampere_model_library.computer_vision.classification.inception_v4.run")
    run_mobilenet_v1 = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v1.run")
    run_nasnet_mobile = lazy_import("ampere_model_library.computer_vision.classification.nasnet_mobile.run")
    run_resnet_50_v2 = lazy_import("ampere_model_library.computer_vision.classification.resnet_50_v2.run")
    run_resnet_101_v2 = lazy_import("ampere_model_library.computer_vision.classification.resnet_101_v2.run")
    run_squeezenet = lazy_import("ampere_model_library.computer_vision.classification.squeezenet.run")
    run_vgg_16 = lazy_import("ampere_model_library.computer_vision.classification.vgg_16.run")
    run_vgg_19 = lazy_import("ampere_model_library.computer_vision.classification.vgg_19.run")

    run_ssd_mobilenet_v1 = lazy_import("ampere_model_library.computer_vision.object_detection.ssd_mobilenet_v1.run")
    run_ssd_mobilenet_v2 = lazy_import("ampere_model_library.computer_vision.object_detection.ssd_mobilenet_v2.run")
    run_ssd_inception_v2 = lazy_import("ampere_model_library.computer_vision.object_detection.ssd_inception_v2.run")
    run_ssd_resnet_34 = lazy_import("ampere_model_library.computer_vision.object_detection.ssd_resnet_34.run")
    run_yolo_v4_tiny = lazy_import("ampere_model_library.computer_vision.object_detection.yolo_v4_tiny.run")

    run_3d_unet_kits = lazy_import("ampere_model_library.computer_vision.semantic_segmentation.unet_3d.kits_19.run")
    run_3d_unet_brats = lazy_import("ampere_model_library.computer_vision.semantic_segmentation.unet_3d.brats_19.run")

    run_bert_base = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.bert_base.run")
    run_bert_large_mlperf = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.bert_large.run_mlperf")
    run_bert_large_hf = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.bert_large.run_huggingface")
    run_distilbert = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.distilbert_base.run")
    run_electra = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.electra_large.run")
    run_longformer = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.longformer_base.run")
    run_roberta = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.roberta_base.run")


except ModuleNotFoundError or ImportError as e:
    print(e)
    print("\ngit submodule update --init --recursive")
    sys.exit(1)

########################################################################################################################
# add new model to proper dict \/ \/ \/
# model class params: dataset, run_func, default_bs, link=None, file_path=None, model_name=None, model_args=dict(), test_num_runs=1000
########################################################################################################################


FP32_MODELS = {

    # nlp models

    "bert_base_uc_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.068, 0.082),
        "run_func": run_bert_base,
        "default_bs": 1,
        "model_name": "madlag/bert-base-uncased-squadv1-x2.44-f87.7-d26-hybrid-filled-v1"
    },
    "bert_base_c_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.291, 0.374),
        "run_func": run_bert_base,
        "default_bs": 1,
        "model_name": "salti/bert-base-multilingual-cased-finetuned-squad"
    },
    "bert_large_uc_wwm_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.730, 0.839),
        "run_func": run_bert_large_hf,
        "default_bs": 1,
        "model_name": "bert-large-uncased-whole-word-masking-finetuned-squad"
    },
    "bert_large_c_wwm_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.738, 0.839),
        "run_func": run_bert_large_hf,
        "default_bs": 1,
        "model_name": "bert-large-cased-whole-word-masking-finetuned-squad"
    },
    "bert_large_mlperf_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.750, 0.817),
        "run_func": run_bert_large_mlperf,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/bert_large_tf_fp32.pb",
        "file_path": "bert_large_tf_fp32.pb",
        "test_num_runs": 24
    },
    "distilbert_base_uc_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.697, 0.796),
        "run_func": run_distilbert,
        "default_bs": 1,
        "model_name": "distilbert-base-uncased-distilled-squad"
    },
    "distilbert_base_c_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.673, 0.791),
        "run_func": run_distilbert,
        "default_bs": 1,
        "model_name": "distilbert-base-cased-distilled-squad"
    },
    "electra_large_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.722, 0.810),
        "run_func": run_electra,
        "default_bs": 1,
        "model_name": "ahotrod/electra_large_discriminator_squad2_512"
    },
    "roberta_base_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.437, 0.510),
        "run_func": run_roberta,
        "default_bs": 1,
        "model_name": "thatdramebaazguy/roberta-base-squad"
    },

    # classification models

    "densenet_169": {
        "dataset": imagenet_utils.ImageNet(0.733, 0.927),
        "run_func": run_densenet_169,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/densenet_169_tf_fp32.pb",
        "file_path": "densenet_169_tf_fp32.pb",
        "test_num_runs": 768
    },
    "inception_resnet_v2": {
        "dataset": imagenet_utils.ImageNet(0.774, 0.934),
        "run_func": run_inception_resnet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_resnet_v2_tf_fp32.pb",
        "file_path": "inception_resnet_v2_tf_fp32.pb"
    },
    "inception_v2": {
        "dataset": imagenet_utils.ImageNet(0.73, 0.908),
        "run_func": run_inception_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v2_tf_fp32.pb",
        "file_path": "inception_v2_tf_fp32.pb"
    },
    "inception_v3": {
        "dataset": imagenet_utils.ImageNet(0.757, 0.932),
        "run_func": run_inception_v3,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v3_tf_fp32.pb",
        "file_path": "inception_v3_tf_fp32.pb"
    },
    "inception_v4": {
        "dataset": imagenet_utils.ImageNet(0.778, 0.952),
        "run_func": run_inception_v4,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v4_tf_fp32.pb",
        "file_path": "inception_v4_tf_fp32.pb"
    },
    "mobilenet_v1": {
        "dataset": imagenet_utils.ImageNet(0.701, 0.888),
        "run_func": run_mobilenet_v1,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v1_tf_fp32.pb",
        "file_path": "mobilenet_v1_tf_fp32.pb"
    },
    "mobilenet_v2": {
        "dataset": imagenet_utils.ImageNet(0.700, 0.905),
        "run_func": run_mobilenet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v2_tf_fp32.pb",
        "file_path": "mobilenet_v2_tf_fp32.pb"
    },
    "nasnet_large": {
        "dataset": imagenet_utils.ImageNet(0.802, 0.957),
        "run_func": run_nasnet_large,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/nasnet_large_tf_fp32.pb",
        "file_path": "nasnet_large_tf_fp32.pb"
    },
    "nasnet_mobile": {
        "dataset": imagenet_utils.ImageNet(0.739, 0.909),
        "run_func": run_nasnet_mobile,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mnasnet_tf_fp32.pb",
        "file_path": "mnasnet_tf_fp32.pb"
    },
    "resnet_50_v2": {
        "dataset": imagenet_utils.ImageNet(0.685, 0.877),
        "run_func": run_resnet_50_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v2_tf_fp32.pb",
        "file_path": "resnet_50_v2_tf_fp32.pb"
    },
    "resnet_50_v1.5": {
        "dataset": imagenet_utils.ImageNet(0.752, 0.924),
        "run_func": run_resnet_50_v15,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v15_tf_fp32.pb",
        "file_path": "resnet_50_v15_tf_fp32.pb",
        "test_num_runs": 992
    },
    "resnet_101_v2": {
        "dataset": imagenet_utils.ImageNet(0.740, 0.929),
        "run_func": run_resnet_101_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_101_v2_tf_fp32.pb",
        "file_path": "resnet_101_v2_tf_fp32.pb"
    },
    "squeezenet": {
        "dataset": imagenet_utils.ImageNet(0.49, 0.729),
        "run_func": run_squeezenet,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/squeezenet_tf_fp32.pb",
        "file_path": "squeezenet_tf_fp32.pb"
    },
    "vgg_16": {
        "dataset": imagenet_utils.ImageNet(0.658, 0.877),
        "run_func": run_vgg_16,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/vgg_16_tf_fp32.pb",
        "file_path": "vgg_16_tf_fp32.pb"
    },
    "vgg_19": {
        "dataset": imagenet_utils.ImageNet(0.666, 0.874),
        "run_func": run_vgg_19,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/vgg_19_tf_fp32.pb",
        "file_path": "vgg_19_tf_fp32.pb"
    },

    # # object detection models

    "ssd_mobilenet_v2": {
        "dataset": coco_utils.COCO(0.20757853684399008),
        "run_func": run_ssd_mobilenet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_mobilenet_v2_tf_fp32.pb",
        "file_path": "ssd_mobilenet_v2_tf_fp32.pb"
    },
    "ssd_inception_v2": {
        "dataset": coco_utils.COCO(0.229),
        "run_func": run_ssd_inception_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_inception_v2_tf_fp32.pb",
        "file_path": "ssd_inception_v2_tf_fp32.pb"
    },
    "yolo_v4_tiny": {
        "dataset": coco_utils.COCO(0.231),
        "run_func": run_yolo_v4_tiny,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/yolo_v4_tiny_tf_fp32.tar.gz",
        "file_path": "yolo_v4_tiny_tf_fp32",
        "test_num_runs": 918
    },
    "ssd_mobilenet_v1": {
        "dataset": coco_utils.COCO(0.245),
        "run_func": run_ssd_mobilenet_v1,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_mobilenet_v1_tf_fp32.pb",
        "file_path": "ssd_mobilenet_v1_tf_fp32.pb",
        "test_num_runs": 992
    },
    "ssd_resnet_34": {
        "dataset": coco_utils.COCO(0.272),
        "run_func": run_ssd_resnet_34,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_resnet_34_tf_fp32.pb",
        "file_path": "ssd_resnet_34_tf_fp32.pb",
        "test_num_runs": 85
    },

    # # semantic segmentation

    "3d_unet_kits": {
        "dataset": kits_utils.KiTS19(0.927, 0.837),
        "run_func": run_3d_unet_kits,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/3d_unet_tf_fp32.tar.gz",
        "file_path": "3d_unet_tf_fp32"
    },
    "3d_unet_brats": {
        "dataset": brats_utils.BraTS19(0.880, 0.931, 0.851),
        "run_func": run_3d_unet_brats,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/3d_unet_brats_tf_fp32.pb",
        "file_path": "3d_unet_brats_tf_fp32.pb",
        "test_num_runs": 10
    }
}

FP16_MODELS = {

    # nlp models

    "bert_large_mlperf_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.760, 0.825),
        "run_func": run_bert_large_mlperf,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/bert_large_tf_fp16.pb",
        "file_path": "bert_large_tf_fp16.pb",
        "test_num_runs": 25
    },

    # classification models

    "densenet_169": {
        "dataset": imagenet_utils.ImageNet(0.736, 0.926),
        "run_func": run_densenet_169,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/densenet_169_tf_fp16.pb",
        "file_path": "densenet_169_tf_fp16.pb"
    },
    "inception_resnet_v2": {
        "dataset": imagenet_utils.ImageNet(0.775, 0.935),
        "run_func": run_inception_resnet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_resnet_v2_tf_fp16.pb",
        "file_path": "inception_resnet_v2_tf_fp16.pb"
    },
    "inception_v2": {
        "dataset": imagenet_utils.ImageNet(0.728, 0.909),
        "run_func": run_inception_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v2_tf_fp16.pb",
        "file_path": "inception_v2_tf_fp16.pb"
    },
    "inception_v4": {
        "dataset": imagenet_utils.ImageNet(0.778, 0.952),
        "run_func": run_inception_v4,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/inception_v4_tf_fp16.pb",
        "file_path": "inception_v4_tf_fp16.pb"
    },
    "mobilenet_v2": {
        "dataset": imagenet_utils.ImageNet(0.604, 0.938),
        "run_func": run_mobilenet_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mobilenet_v2_tf_fp16.pb",
        "file_path": "mobilenet_v2_tf_fp16.pb",
        "test_num_runs": 64
    },
    "nasnet_large": {
        "dataset": imagenet_utils.ImageNet(0.806, 0.967),
        "run_func": run_nasnet_large,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/nasnet_large_tf_fp16.pb",
        "file_path": "nasnet_large_tf_fp16.pb"
    },
    "nasnet_mobile": {
        "dataset": imagenet_utils.ImageNet(0.739, 0.910),
        "run_func": run_nasnet_mobile,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/mnasnet_tf_fp16.pb",
        "file_path": "mnasnet_tf_fp16.pb"
    },
    "resnet_50_v1.5": {
        "dataset": imagenet_utils.ImageNet(0.751, 0.923),
        "run_func": run_resnet_50_v15,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v15_tf_fp16.pb",
        "file_path": "resnet_50_v15_tf_fp16.pb",
        "test_num_runs": 992
    },
    "resnet_101_v2": {
        "dataset": imagenet_utils.ImageNet(0.737, 0.931),
        "run_func": run_resnet_101_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_101_v2_tf_fp16.pb",
        "file_path": "resnet_101_v2_tf_fp16.pb"
    },
    "vgg_16": {
        "dataset": imagenet_utils.ImageNet(0.657, 0.877),
        "run_func": run_vgg_16,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/vgg_16_tf_fp16.pb",
        "file_path": "vgg_16_tf_fp16.pb"
    },
    "vgg_19": {
        "dataset": imagenet_utils.ImageNet(0.670, 0.871),
        "run_func": run_vgg_19,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/vgg_19_tf_fp16.pb",
        "file_path": "vgg_19_tf_fp16.pb"
    },

    # object detection models

    "ssd_inception_v2": {
        "dataset": coco_utils.COCO(0.228),
        "run_func": run_ssd_inception_v2,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_inception_v2_tf_fp16.pb",
        "file_path": "ssd_inception_v2_tf_fp16.pb",
        "test_num_runs": 736
    },
    "ssd_mobilenet_v1": {
        "dataset": coco_utils.COCO(0.244),
        "run_func": run_ssd_mobilenet_v1,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_mobilenet_v1_tf_fp16.pb",
        "file_path": "ssd_mobilenet_v1_tf_fp16.pb",
        "test_num_runs": 992
    },
    "ssd_resnet_34": {
        "dataset": coco_utils.COCO(0.272),
        "run_func": run_ssd_resnet_34,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/ssd_resnet_34_tf_fp16.pb",
        "file_path": "ssd_resnet_34_tf_fp16.pb"
    },

    # semantic segmentation

    "3d_unet_brats": {
        "dataset": brats_utils.BraTS19(0.914, 0.868, 0.777),
        "run_func": run_3d_unet_brats,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/3d_unet_brats_tf_fp16.pb",
        "file_path": "3d_unet_brats_tf_fp16.pb",
        "test_num_runs": 67
    }
}

BF16_MODELS = {
    "resnet_50_v1.5": {
        "dataset": imagenet_utils.ImageNet(0.752, 0.924),
        "run_func": run_resnet_50_v15,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.amazonaws.com/resnet_50_v15_tf_bf16.pb",
        "file_path": "resnet_50_v15_tf_bf16.pb",
        "test_num_runs": 992
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
            model["run_func"].run_tf_fp32
        for model in FP16_MODELS.values():
            model["run_func"].run_tf_fp16
        for model in BF16_MODELS.values():
            model["run_func"].run_tf_bf16
    if precision == "fp32":
        try:
            config = FP32_MODELS[model_name]
            config["run_func"] = config["run_func"].run_tf_fp32
            return Model(**config)
        except KeyError:
            list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "fp16":
        try:
            config = FP16_MODELS[model_name]
            config["run_func"] = config["run_func"].run_tf_fp16
            return Model(**config)
        except KeyError:
            if os.getenv("ALLOW_IMPLICIT_FP16", "0") == "1":
                try:
                    config = FP32_MODELS[model_name]
                    config["run_func"] = config["run_func"].run_tf_fp32
                    return Model(**config)
                except KeyError:
                    list_available_models(model_name, precision, FP32_MODELS.keys())
            list_available_models(model_name, precision, FP16_MODELS.keys())
    elif precision == "bf16":
        try:
            config = BF16_MODELS[model_name]
            config["run_func"] = config["run_func"].run_tf_bf16
            return Model(**config)
        except KeyError:
            if os.getenv("ALLOW_IMPLICIT_BF16", "0") == "1":
                try:
                    config = FP32_MODELS[model_name]
                    config["run_func"] = config["run_func"].run_tf_fp32
                    return Model(**config)
                except KeyError:
                    list_available_models(model_name, precision, FP32_MODELS.keys())
            list_available_models(model_name, precision, BF16_MODELS.keys())
    print_goodbye_message_and_die(f"Model {model_name} not available in {precision} precision!")
