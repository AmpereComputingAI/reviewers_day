import os
import sys


import run_utils.wmt as wmt_utils
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


########################################################################################################################
# import new model here \/ \/ \/
########################################################################################################################

try:

    run_wmtende = lazy_import("ampere_model_library.natural_language_processing.translation.wmtende.run")


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
    "wmtende": {
        "dataset": wmt_utils.WMT(29.025),
        "run_func": run_wmtende,
        "default_bs": 1,
        "link": "https://ampereaidevelop.s3.eu-central-1.amazonaws.com/ctrans_model_fp32.tar.gz",
        "file_path": "ctrans_model_fp32",
        "model_args": {"constant_input": False}
    },

    "wmtende_constant": {
        "dataset": wmt_utils.WMT(0.072),
        "run_func": run_wmtende,
        "default_bs": 1,
        "link": "https://ampereaidevelop.s3.eu-central-1.amazonaws.com/ctrans_model_fp32.tar.gz",
        "file_path": "ctrans_model_fp32",
        "model_args": {"constant_input": True}
    },
}

FP16_MODELS = {

    # nlp
    "wmtende": {
        "dataset": wmt_utils.WMT(29.025),
        "run_func": run_wmtende,
        "default_bs": 1,
        "link": "https://ampereaidevelop.s3.eu-central-1.amazonaws.com/ctrans_model_fp16.tar.gz",
        "file_path": "ctrans_model_fp16",
        "model_args": {"constant_input": False}
    },

    "wmtende_constant": {
        "dataset": wmt_utils.WMT(0.072),
        "run_func": run_wmtende,
        "default_bs": 1,
        "link": "https://ampereaidevelop.s3.eu-central-1.amazonaws.com/ctrans_model_fp16.tar.gz",
        "file_path": "ctrans_model_fp16",
        "model_args": {"constant_input": True}
    },
}

INT8_MODELS = {

    # nlp
    "wmtende": {
        "dataset": wmt_utils.WMT(29.017),
        "run_func": run_wmtende,
        "default_bs": 1,
        "link": "https://ampereaidevelop.s3.eu-central-1.amazonaws.com/ctrans_model_int8.tar.gz",
        "file_path": "ctrans_model_int8",
        "model_args": {"constant_input": False}
    },

    "wmtende_constant": {
        "dataset": wmt_utils.WMT(0.072),
        "run_func": run_wmtende,
        "default_bs": 1,
        "link": "https://ampereaidevelop.s3.eu-central-1.amazonaws.com/ctrans_model_int8.tar.gz",
        "file_path": "ctrans_model_int8",
        "model_args": {"constant_input": True}
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
            model["run_func"].run_ctranslate_fp32
        for model in FP16_MODELS.values():
            model["run_func"].run_ctranslate_fp16
        for model in INT8_MODELS.values():
            model["run_func"].run_ctranslate_int8
    if precision == "fp32":
        try:
            config = FP32_MODELS[model_name]
            config["run_func"] = config["run_func"].run_ctranslate_fp32
            return Model(**config)
        except KeyError:
            list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "fp16":
        try:
            config = FP16_MODELS[model_name]
            config["run_func"] = config["run_func"].run_ctranslate_fp16
            return Model(**config)
        except KeyError:
            if os.getenv("ALLOW_IMPLICIT_FP16", "0") == "1":
                try:
                    config = FP32_MODELS[model_name]
                    config["run_func"] = config["run_func"].run_ctranslate_fp32
                    return Model(**config)
                except KeyError:
                    list_available_models(model_name, precision, FP32_MODELS.keys())
            list_available_models(model_name, precision, FP16_MODELS.keys())
    elif precision == "bf16":
        if os.getenv("ALLOW_IMPLICIT_BF16", "0") == "1":
            try:
                config = FP32_MODELS[model_name]
                config["run_func"] = config["run_func"].run_ctranslate_fp32
                return Model(**config)
            except KeyError:
                list_available_models(model_name, precision, FP32_MODELS.keys())
    elif precision == "int8":
        try:
            config = INT8_MODELS[model_name]
            config["run_func"] = config["run_func"].run_ctranslate_int8
            return Model(**config)
        except KeyError:
            list_available_models(model_name, precision, INT8_MODELS.keys())
    print_goodbye_message_and_die(f"Model {model_name} not available in {precision} precision!")
