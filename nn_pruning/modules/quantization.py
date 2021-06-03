import operator

import torch
from torch.quantization import (
    float_qparams_weight_only_qconfig,
    get_default_qat_qconfig,
    get_default_qconfig,
)
from torch.quantization.quantize_fx import (
    convert_fx,
    prepare_fx,
    prepare_qat_fx,
)
from transformers.modeling_fx_utils import symbolic_trace

from .quantization_config import create_qconfig
from .quantization_transformations import (
    compose_transformations,
    change_attention_mask_value,
    change_truediv_to_mul_when_possible,
)


def _prepare(
    model,
    training,
    torch_prepare_fn,
    input_names=None,
    batch_size=1,
    sequence_length=128,
    num_choices=-1,
    qconfig_dict=None,
    target=None,
):
    """Helper function that prepares the model for either static quantization or QAT."""
    if training:
        model.train()
    else:
        model.eval()

    traced = symbolic_trace(
        model, input_names=input_names, batch_size=batch_size, sequence_length=sequence_length, num_choices=num_choices
    )

    pre_prepare_transformation = compose_transformations(
        change_attention_mask_value,
        change_truediv_to_mul_when_possible,
    )
    pre_prepare_transformation(traced)
    prepare_custom_config_dict = {"preserved_attributes": ["config", "dummy_inputs"]}
    prepared_model = torch_prepare_fn(traced, qconfig_dict, prepare_custom_config_dict)

    if target:
        prepared_model = target.prepare(prepared_model, qconfig_dict=qconfig_dict)

    return prepared_model


def prepare_static(
    model,
    input_names=None,
    batch_size=1,
    sequence_length=128,
    num_choices=-1,
    qconfig_name=None,
    qconfig_dict=None,
    target=None,
):
    if qconfig_name and qconfig_dict:
        raise ValueError("either specify a qconfig name or a qconfig dict, but not both")
    if qconfig_name is None and qconfig_dict is None:
        if target is None:
            raise ValueError("a qconfig name, a qconfig dict or a target need to be specified")
        else:
            print(f"Using default qconfig dict for target {target.__name__}")
            qconfig_dict = target.default_qconfig_dict()
    if qconfig_name:
        qconfig_dict = create_qconfig(qconfig_name, mode="static")

    if target:
        target.validate_qconfig_dict(qconfig_dict)

    return _prepare(
        model,
        False,
        prepare_fx,
        input_names=input_names,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_choices=num_choices,
        qconfig_dict=qconfig_dict,
        target=target,
    )


def prepare_qat(
    model,
    input_names=None,
    batch_size=1,
    sequence_length=128,
    num_choices=-1,
    qconfig_name=None,
    qconfig_dict=None,
    target=None,
):
    if qconfig_name and qconfig_dict:
        raise ValueError("either specify a qconfig name or a qconfig dict, but not both")
    if qconfig_name is None and qconfig_dict is None:
        if target is None:
            raise ValueError("a qconfig name, a qconfig dict or a target need to be specified")
        else:
            print(f"Using default qconfig dict for target {target.__name__}")
            qconfig_dict = target.default_qat_qconfig_dict()
    if qconfig_name:
        qconfig_dict = create_qconfig(qconfig_name, mode="qat")

    if target:
        target.validate_qat_qconfig_dict(qconfig_dict)

    return _prepare(
        model,
        True,
        prepare_qat_fx,
        input_names=input_names,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_choices=num_choices,
        qconfig_dict=qconfig_dict,
        target=target,
    )


def quantize(prepared_model):
    """Quantizes a prepared model."""
    prepared_model.eval()

    convert_custom_config_dict = {"preserved_attributes": ["config", "dummy_inputs"]}
    model_int8 = convert_fx(prepared_model, convert_custom_config_dict=convert_custom_config_dict)

    broadcast_attention_mask(model_int8)
    broadcast_nonorm_bias(model_int8)
    return model_int8
