import copy
import operator

import torch
from torch.quantization.quantize_fx import (
    convert_fx,
    prepare_fx,
    prepare_qat_fx,
)
from transformers.modeling_fx_utils import (
    _generate_random_input_shape,
    symbolic_trace,
    transform_to_dynamic_input,
)

from .quantization_config import create_qconfig
from .quantization_transformations import (
    broadcast_attention_mask,
    broadcast_nonorm_bias,
    change_attention_mask_value,
    change_truediv_to_mul_when_possible,
    compose_transformations,
)


def _cache_slice_args_for_retracing(gm, dummy_value=2):
    graph = gm.graph
    cached_nodes = {}
    for node in graph.nodes:
        node_args = []
        for arg in node.args:
            new_arg = arg
            if isinstance(arg, tuple):
                new_arg = []
                for sub_arg in arg:
                    if type(sub_arg) is slice:
                        attributes = [sub_arg.start, sub_arg.step, sub_arg.stop]
                        if any((isinstance(a, torch.fx.Node) for a in attributes)):
                            replace_by_dummy = lambda a: dummy_value if isinstance(a, torch.fx.Node) else a
                            cached_nodes[node.name] = copy.deepcopy(node)
                            start = replace_by_dummy(sub_arg.start)
                            step = replace_by_dummy(sub_arg.step)
                            stop = replace_by_dummy(sub_arg.stop)
                            new_arg.append(slice(start, stop, step))
                    else:
                        new_arg.append(arg)

            if isinstance(new_arg, list):
                new_arg = tuple(new_arg)
            node_args.append(new_arg)
        node.args = tuple(node_args)

    graph.lint()
    gm.recompile()
    return cached_nodes


def _restore_cached_nodes(gm, cached_nodes):
    graph = gm.graph
    for node in graph.nodes:
        if node.name in cached_nodes:
            print(f"restoring {node}")
        node = cached_nodes.get(node.name, node)
    graph.lint()
    gm.recompile()
    return gm


def _prepare(
    model,
    training,
    torch_prepare_fn,
    input_names=None,
    num_choices=-1,
    qconfig_dict=None,
    target=None,
):
    """Helper function that prepares the model for either static quantization or QAT."""
    if training:
        model.train()
    else:
        model.eval()

    batch_size, sequence_length = _generate_random_input_shape(model)

    # batch_size = 16
    # sequence_length=128

    traced = symbolic_trace(
        model, input_names=input_names, batch_size=batch_size, sequence_length=sequence_length, num_choices=num_choices
    )

    pre_prepare_transformation = compose_transformations(
        change_attention_mask_value,
        # change_truediv_to_mul_when_possible,
    )
    pre_prepare_transformation(traced)
    prepare_custom_config_dict = {
        "preserved_attributes": ["config", "dummy_inputs", "static_batch_size", "static_sequence_length"]
    }

    prepared_model = torch_prepare_fn(traced, qconfig_dict, prepare_custom_config_dict)

    prepared_model = transform_to_dynamic_input(
        prepared_model,
        dynamic_batch_size=True,
        dynamic_encoder_sequence_length=True,
        dynamic_decoder_sequence_length=True,
    )

    if target:
        prepared_model = target.prepare(prepared_model, qconfig_dict=qconfig_dict)

    return prepared_model


def prepare_static(
    model,
    input_names=None,
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
        num_choices=num_choices,
        qconfig_dict=qconfig_dict,
        target=target,
    )


def prepare_qat(
    model,
    input_names=None,
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
