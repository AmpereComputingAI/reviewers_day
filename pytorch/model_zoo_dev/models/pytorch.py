import os
import sys

import torch

import run_utils.alpaca as alpaca_utils
import run_utils.coco as coco_utils
import run_utils.brats as brats_utils
import run_utils.criteo as criteo_utils
import run_utils.imagenet as imagenet_utils
import run_utils.dummy as dummy_utils
import run_utils.librispeech as librispeech_utils
import run_utils.openimages as openimages_utils
import run_utils.squad as squad_utils
from downloads.utils import get_downloads_path
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
torch.hub.set_dir(get_downloads_path())

########################################################################################################################
# import new model here \/ \/ \/
########################################################################################################################

try:

    run_alexnet = lazy_import("ampere_model_library.computer_vision.classification.alexnet.run")
    run_densenet_121 = lazy_import("ampere_model_library.computer_vision.classification.densenet_121.run")
    run_googlenet = lazy_import("ampere_model_library.computer_vision.classification.googlenet.run")
    run_inception_v3 = lazy_import("ampere_model_library.computer_vision.classification.inception_v3.run")
    run_mobilenet_v2 = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v2.run")
    run_mobilenet_v3_large = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v3_large.run")
    run_mobilenet_v3_small = lazy_import("ampere_model_library.computer_vision.classification.mobilenet_v3_small.run")
    run_nasnet_mobile = lazy_import("ampere_model_library.computer_vision.classification.nasnet_mobile.run")
    run_resnet_18 = lazy_import("ampere_model_library.computer_vision.classification.resnet_18.run")
    run_resnet_50_v15 = lazy_import("ampere_model_library.computer_vision.classification.resnet_50_v15.run")
    run_shufflenet = lazy_import("ampere_model_library.computer_vision.classification.shufflenet.run")
    run_squeezenet = lazy_import("ampere_model_library.computer_vision.classification.squeezenet.run")
    run_vgg_16 = lazy_import("ampere_model_library.computer_vision.classification.vgg_16.run")

    run_ssd_vgg_16 = lazy_import("ampere_model_library.computer_vision.object_detection.ssd_vgg_16.run")
    run_retinanet = lazy_import(
        "ampere_model_library.computer_vision.object_detection.retinanet_mlperf.run")
    run_retinanet_torchvision = lazy_import(
        "ampere_model_library.computer_vision.object_detection.retinanet_torchvision.run")
    run_yolo_v5 = lazy_import("ampere_model_library.computer_vision.object_detection.yolo_v5.run")
    run_yolo_v8 = lazy_import("ampere_model_library.computer_vision.object_detection.yolo_v8.run")

    run_dlrm = lazy_import("ampere_model_library.recommendation.dlrm.run")
    run_3d_unet_brats = lazy_import("ampere_model_library.computer_vision.semantic_segmentation.unet_3d.brats_19.run")

    run_rnnt = lazy_import("ampere_model_library.speech_recognition.rnnt.run")
    run_whisper = lazy_import("ampere_model_library.speech_recognition.whisper.run")
    run_whisper_hf = lazy_import("ampere_model_library.speech_recognition.whisper.run_hf")

    run_alpaca = lazy_import("ampere_model_library.natural_language_processing.text_generation.alpaca.run")
    run_mixtral = lazy_import("ampere_model_library.natural_language_processing.text_generation.mixtral.run")
    run_llama2 = lazy_import("ampere_model_library.natural_language_processing.text_generation.llama2.run")
    run_bert_base_graphcore = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.bert_base.run")
    run_bert_large_mlperf = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.bert_large.run_mlperf")
    run_roberta = lazy_import(
        "ampere_model_library.natural_language_processing.extractive_question_answering.roberta_base.run")

    run_dlrm_torchbench = lazy_import("ampere_model_library.recommendation.dlrm_torchbench.run")

    run_stable_diffusion = lazy_import("ampere_model_library.text_to_image.stable_diffusion.run")

except ModuleNotFoundError or ImportError as e:
    print(e)
    print("\ngit submodule update --init --recursive")
    sys.exit(1)

########################################################################################################################
# add new model to proper dict \/ \/ \/
# model class params: dataset, run_func, default_bs, link=None, file_path=None, model_name=None, model_args=dict(), test_num_runs=1000
########################################################################################################################

FP32_MODELS = {
    # nlp
    "alpaca": {
        "dataset": alpaca_utils.AlpacaInstruct(0.100, 0.317),
        "run_func": run_alpaca,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/alpaca_recovered.tar.gz",
        "file_path": "alpaca_recovered",
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 50,
    },
    "bert_base_graphcore": {
        "dataset": squad_utils.SQuAD_v1_1(0.767, 0.824),
        "run_func": run_bert_base_graphcore,
        "default_bs": 1,
        "model_name": "jimypbr/bert-base-uncased-squad",
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 287
    },
    "bert_large_mlperf_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.750, 0.817),
        "run_func": run_bert_large_mlperf,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/bert_large_pytorch_fp32.pytorch",
        "file_path": "bert_large_pytorch_fp32.pytorch",
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 24
    },
    "llama2_7b": {
        "dataset": alpaca_utils.AlpacaInstruct(0., 0.290),
        "run_func": run_llama2,
        "default_bs": 1,
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "test_num_runs": 50,
    },
    "llama2_13b": {
        "dataset": alpaca_utils.AlpacaInstruct(0., 0.164),
        "run_func": run_llama2,
        "default_bs": 1,
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "test_num_runs": 50,
    },
    "llama3_8b": {
        "dataset": alpaca_utils.AlpacaInstruct(0., 0.164),
        "run_func": run_llama2,
        "default_bs": 1,
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "test_num_runs": 50,
    },
    "roberta_base_squad": {
        "dataset": squad_utils.SQuAD_v1_1(0.828, 0.918),
        "run_func": run_roberta,
        "default_bs": 1,
        "model_name": "thatdramebaazguy/roberta-base-squad",
        "model_args": {"disable_jit_freeze": False}
    },

    # classification models
    "alexnet": {
        "dataset": imagenet_utils.ImageNet(0.519, 0.765),
        "run_func": run_alexnet,
        "default_bs": 1,
        "model_name": 'alexnet',
        "model_args": {"disable_jit_freeze": False},
    },
    "densenet_121": {
        "dataset": imagenet_utils.ImageNet(0.717, 0.905),
        "run_func": run_densenet_121,
        "default_bs": 1,
        "model_name": 'densenet121',
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 346
    },
    "googlenet": {
        "dataset": imagenet_utils.ImageNet(0.681, 0.874),
        "run_func": run_googlenet,
        "default_bs": 1,
        "model_name": 'googlenet',
        "model_args": {"disable_jit_freeze": False},
    },
    "inception_v3": {
        "dataset": imagenet_utils.ImageNet(0.765, 0.932),
        "run_func": run_inception_v3,
        "default_bs": 1,
        "model_name": 'inception_v3',
        "model_args": {"disable_jit_freeze": False},
    },
    "mobilenet_v2": {
        "dataset": imagenet_utils.ImageNet(0.651, 0.870),
        "run_func": run_mobilenet_v2,
        "default_bs": 1,
        "model_name": 'mobilenet_v2',
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 192
    },
    "mobilenet_v3_large": {
        "dataset": imagenet_utils.ImageNet(0.684, 0.864),
        "run_func": run_mobilenet_v3_large,
        "default_bs": 1,
        "model_name": 'mobilenet_v3_large',
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 272
    },
    "mobilenet_v3_small": {
        "dataset": imagenet_utils.ImageNet(0.629, 0.847),
        "run_func": run_mobilenet_v3_small,
        "default_bs": 1,
        "model_name": 'mobilenet_v3_small',
        "model_args": {"disable_jit_freeze": False},
    },
    "nasnet_mobile": {
        "dataset": imagenet_utils.ImageNet(0.724, 0.917),
        "run_func": run_nasnet_mobile,
        "default_bs": 1,
        "model_name": 'mnasnet1_0',
        "model_args": {"disable_jit_freeze": False}
    },
    "resnet_18": {
        "dataset": imagenet_utils.ImageNet(0.667, 0.881),
        "run_func": run_resnet_18,
        "default_bs": 1,
        "model_name": 'resnet18',
        "model_args": {"disable_jit_freeze": False},
    },
    "resnet_50_v1.5": {
        "dataset": imagenet_utils.ImageNet(0.717, 0.904),
        "run_func": run_resnet_50_v15,
        "default_bs": 1,
        "model_name": 'resnet50',
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 448
    },
    "shufflenet": {
        "dataset": imagenet_utils.ImageNet(0.637, 0.851),
        "run_func": run_shufflenet,
        "default_bs": 1,
        "model_name": 'shufflenet_v2_x1_0',
        "model_args": {"disable_jit_freeze": False},
    },
    "squeezenet": {
        "dataset": imagenet_utils.ImageNet(0.554, 0.799),
        "run_func": run_squeezenet,
        "default_bs": 1,
        "model_name": 'squeezenet1_0',
        "model_args": {"disable_jit_freeze": False},
    },
    "vgg_16": {
        "dataset": imagenet_utils.ImageNet(0.661, 0.896),
        "run_func": run_vgg_16,
        "default_bs": 1,
        "model_name": 'vgg16',
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 192
    },

    # object detection models
    "ssd_vgg_16": {
        "dataset": coco_utils.COCO(0.251),
        "run_func": run_ssd_vgg_16,
        "default_bs": 1,
        "model_name": 'ssd300_vgg16_coco',
        "model_args": {"disable_jit_freeze": False},
    },
    "retinanet": {
        "dataset": openimages_utils.OpenImages(0.478),
        "run_func": run_retinanet,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/resnext50_32x4d_fpn.pth",
        "file_path": "resnext50_32x4d_fpn.pth",
        "model_args": {"disable_jit_freeze": False},
    },
    "retinanet_torchvision": {
        "dataset": coco_utils.COCO(0.251),
        "run_func": run_retinanet_torchvision,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
    },
    "yolo_v5_n": {
        "dataset": coco_utils.COCO(0.438),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
        "file_path": "yolov5n.pt",
        "test_num_runs": 100
    },
    "yolo_v5_s": {
        "dataset": coco_utils.COCO(0.438),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        "file_path": "yolov5s.pt",
        "test_num_runs": 100
    },
    "yolo_v5_m": {
        "dataset": coco_utils.COCO(0.492),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
        "file_path": "yolov5m.pt",
        "test_num_runs": 100
    },
    "yolo_v5_l": {
        "dataset": coco_utils.COCO(0.527),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
        "file_path": "yolov5l.pt",
        "test_num_runs": 100
    },
    "yolo_v5_x": {
        "dataset": coco_utils.COCO(0.541),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt",
        "file_path": "yolov5x.pt",
        "test_num_runs": 100
    },
    "yolo_v8_n": {
        "dataset": coco_utils.COCO(0.354),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "file_path": "yolov8n.pt",
        "test_num_runs": 463
    },
    "yolo_v8_s": {
        "dataset": coco_utils.COCO(0.353),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "file_path": "yolov8s.pt",
        "test_num_runs": 465
    },
    "yolo_v8_m": {
        "dataset": coco_utils.COCO(0.536),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "file_path": "yolov8m.pt",
        "test_num_runs": 155
    },
    "yolo_v8_l": {
        "dataset": coco_utils.COCO(0.553),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
        "file_path": "yolov8l.pt",
        "test_num_runs": 85
    },
    "yolo_v8_x": {
        "dataset": coco_utils.COCO(0.553),
        "run_func": run_yolo_v8,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "link": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        "file_path": "yolov8x.pt",
        "test_num_runs": 85
    },

    # recommendation models
    "dlrm_debug": {
        "dataset": criteo_utils.Criteo(0.7882),
        "run_func": run_dlrm,
        "default_bs": 2048,
        "link": "https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt",
        "file_path": "tb0875_10M.pt",
        "model_args": {"debug": True, "disable_jit_freeze": False}
    },
    "dlrm": {
        "dataset": criteo_utils.Criteo(0.782),
        "run_func": run_dlrm,
        "default_bs": 2048,
        "link": "https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt",
        "file_path": "tb00_40M.pt",
        "model_args": {"debug": False, "disable_jit_freeze": False}
    },
    "dlrm_torchbench": {
        "dataset": dummy_utils.Dummy(),
        "run_func": run_dlrm_torchbench,
        "default_bs": 2048
    },

    # text to image models
    "stable_diffusion": {
        "dataset": dummy_utils.Dummy(),
        "run_func": run_stable_diffusion,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/v2-1_512-ema-pruned.ckpt",
        "file_path": "v2-1_512-ema-pruned.ckpt",
        "model_args": {"config": "ampere_model_library/text_to_image/stable_diffusion/stablediffusion/"
                                 "configs/stable-diffusion/intel/v2-inference-fp32.yaml",
                       "steps": 10, "scale": 9},
        "test_num_runs": 5
    },

    # semantic segmentation models
    "3d_unet_brats": {
        "dataset": brats_utils.BraTS19(0.876, 0.92, 0.848),
        "run_func": run_3d_unet_brats,
        "default_bs": 1,
        "link": "https://zenodo.org/record/3904106/files/fold_1.zip",
        "file_path": "nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1/fold_1/model_best.model"
    },

    # speech recognition models
    "whisper_tiny.en_openai": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.155),
        "run_func": run_whisper,
        "default_bs": 1,
        "model_name": "tiny.en",
        "test_num_runs": 30
    },
    "whisper_base.en_openai": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.158),
        "run_func": run_whisper,
        "default_bs": 1,
        "model_name": "base.en",
        "test_num_runs": 30
    },
    "whisper_small.en_openai": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.146),
        "run_func": run_whisper,
        "default_bs": 1,
        "model_name": "small.en",
        "test_num_runs": 30
    },
    "whisper_medium.en_openai": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.16),
        "run_func": run_whisper,
        "default_bs": 1,
        "model_name": "medium.en",
        "test_num_runs": 30
    },
    "whisper_large_openai": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.124),
        "run_func": run_whisper,
        "default_bs": 1,
        "model_name": "large",
        "test_num_runs": 30
    },
    "whisper_tiny.en_hf": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.111),
        "run_func": run_whisper_hf,
        "default_bs": 1,
        "model_name": "openai/whisper-tiny.en",
        "test_num_runs": 72
    },
    "whisper_base.en_hf": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.117),
        "run_func": run_whisper_hf,
        "default_bs": 1,
        "model_name": "openai/whisper-base.en",
        "test_num_runs": 30
    },
    "whisper_small.en_hf": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.102),
        "run_func": run_whisper_hf,
        "default_bs": 1,
        "model_name": "openai/whisper-small.en",
        "test_num_runs": 30
    },
    "whisper_medium.en_hf": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.073),
        "run_func": run_whisper_hf,
        "default_bs": 1,
        "model_name": "openai/whisper-medium.en",
        "test_num_runs": 30
    },
    "whisper_large_v3_hf": {
        "dataset": librispeech_utils.LibriSpeechASRDummy(0.051),
        "run_func": run_whisper_hf,
        "default_bs": 1,
        "model_name": "openai/whisper-large-v3",
        "test_num_runs": 30
    },
    # "rnnt": {
    #     "dataset": librispeech_utils.LibriSpeech(0.921),
    #     "run_func": run_rnnt,
    #     "default_bs": 1,
    #     "link": "https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt",
    #     "file_path": "DistributedDataParallel_1576581068.9962234-epoch-100.pt",
    #     "model_args": {"disable_jit_freeze": False},
    # },

}

FP16_MODELS = {
    # nlp
    "alpaca": {
        "dataset": alpaca_utils.AlpacaInstruct(0.100, 0.317),
        "run_func": run_alpaca,
        "default_bs": 1,
        "link": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/alpaca_recovered.tar.gz",
        "file_path": "alpaca_recovered",
        "test_num_runs": 50,
    },
    "llama2_7b": {
        "dataset": alpaca_utils.AlpacaInstruct(0., 0.290),
        "run_func": run_llama2,
        "default_bs": 1,
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "test_num_runs": 50,
    },
    "llama2_13b": {
        "dataset": alpaca_utils.AlpacaInstruct(0., 0.164),
        "run_func": run_llama2,
        "default_bs": 1,
        "model_name": "meta-llama/Llama-2-13b-chat-hf",
        "test_num_runs": 50,
    },
    "llama3_8b": {
        "dataset": alpaca_utils.AlpacaInstruct(0., 0.164),
        "run_func": run_llama2,
        "default_bs": 1,
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "test_num_runs": 50,
    },
    "mixtral": {
        "dataset": alpaca_utils.AlpacaInstruct(0.100, 0.317),
        "run_func": run_mixtral,
        "default_bs": 1,
        "model_args": {"disable_jit_freeze": False},
        "test_num_runs": 50,
    },
}


########################################################################################################################
########################################################################################################################


def get_model(model_name, precision, import_all=False):
    """
    A function returning a Model of requested model at given precision.

    :param model_name: str, name of the model
    :param precision: str, precision of the model
    :return: ModelClass object
    """
    if import_all:
        for model in FP32_MODELS.values():
            model["run_func"].run_pytorch_fp32
    if precision == "fp32":
        try:
            config = FP32_MODELS[model_name]
            config["run_func"] = config["run_func"].run_pytorch_fp32
            return Model(**config)
        except KeyError:
            list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "fp16":
        try:
            config = FP16_MODELS[model_name]
            config["run_func"] = config["run_func"].run_pytorch_fp16
            return Model(**config)
        except KeyError:
            if os.getenv("ALLOW_IMPLICIT_FP16", "0") == "1":
                try:
                    config = FP32_MODELS[model_name]
                    config["run_func"] = config["run_func"].run_pytorch_fp32
                    return Model(**config)
                except KeyError:
                    list_available_models(model_name, precision, FP32_MODELS.keys())
            list_available_models(model_name, precision, FP16_MODELS.keys())
    elif precision == "bf16":
        if os.getenv("ALLOW_IMPLICIT_BF16", "0") == "1":
            try:
                config = FP32_MODELS[model_name]
                config["run_func"] = config["run_func"].run_pytorch_fp32
                return Model(**config)
            except KeyError:
                list_available_models(model_name, precision, FP32_MODELS.keys())
    print_goodbye_message_and_die(f"Model {model_name} not available in {precision} precision!")
