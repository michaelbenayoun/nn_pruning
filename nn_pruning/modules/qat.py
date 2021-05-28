import copy

import torch
from torch import nn
from torch.quantization import (
    QConfig,
    FakeQuantize,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    get_default_qat_qconfig,
)

from .qat_modules import _QAT_MAPPING
from .qat_ops import qat_op_patcher

# TensorFlow Lite Quantization Specs
# https://www.tensorflow.org/lite/performance/quantization_spec?hl=en
# For activations: int8 asymmetric per-tensor [-128, 127] range
# For weights: int8 symmetric per-tensor [-127, 127] range
_TFLITE_QAT_CONFIG = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        dtype=torch.qint8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
    ),
    weight=FakeQuantize.with_args(
        observer=MinMaxObserver,
        dtype=torch.qint8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric
    )
)
_ONNX_QAT_CONFIG = QConfig(
    activation=FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=0,
        quant_max=255,
        reduce_range=True,
    ),
    weight=FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        reduce_range=False,
        qscheme=torch.per_tensor_symmetric
    )
)
_DEFAULT_CONFIG = get_default_qat_qconfig("fbgemm")
_DEFAULT_MOBILE = get_default_qat_qconfig("qnnpack")

_QCONFIG_MAPPING = {
    "default": _DEFAULT_CONFIG,
    "fbgemm": _DEFAULT_CONFIG,
    "mobile": _DEFAULT_MOBILE,
    "qnnpack": _DEFAULT_MOBILE,
    "tflite": _TFLITE_QAT_CONFIG,
    "onnx_test": _ONNX_QAT_CONFIG,
}


def create_qconfig(qconfig_name):
    qconfig = _QCONFIG_MAPPING.get(qconfig_name, None)
    if qconfig is None:
        raise ValueError(
            f"qconfig name must be in {_QCONFIG_MAPPING.keys()}, {qconfig_name} was provided"
        )
    return copy.deepcopy(qconfig)


def prepare_for_qat(model, qconfig_name: str = None, qconfig: QConfig = None):
    if qconfig_name and qconfig:
        raise ValueError("either specify a qconfig name or a qconfig, but not both")
    if qconfig_name:
        qconfig = create_qconfig(qconfig_name)
    model.train()
    qat_op_patcher(model, qconfig=qconfig)
    model.qconfig = qconfig
    model = torch.quantization.prepare_qat(model, mapping=_QAT_MAPPING)
    return model
