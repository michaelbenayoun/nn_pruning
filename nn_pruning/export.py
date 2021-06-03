import importlib
import itertools
import json
from pathlib import Path

import torch
from torch.onnx.symbolic_helper import parse_args, cast_pytorch_to_onnx
from torch.onnx.symbolic_registry import register_op

import onnx
import onnxruntime as ort
import onnxruntime.quantization.quant_utils as ort_quant_utils

from transformers import AutoConfig, MODEL_MAPPING

from .inference_model_patcher import optimize_model
from .modules.quantization import prepare_qat

_DEFAULT_OPSET_VERSION = 13
_AVAILABLE_TARGETS = {"onnx"}


# Copied from:
# https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset10.py#L291
def fake_quantize_per_tensor_affine(g, inputs, scale, zero_point, quant_min=-128, quant_max=127):
    if quant_min not in [0, -128] or quant_max not in [127, 255]:
        raise RuntimeError(
            "ONNX defines [0, 255] for quint8 and [-128, 127] for qint8, got [{}, {}]".format(quant_min, quant_max))
    scale = scale.float().data  # Avoid exporter generating double type
    zero_point_dtype = torch.int8 if quant_min == -128 else torch.uint8
    zero_point = torch.tensor(zero_point, dtype=zero_point_dtype)  # ONNX requires zero_point to be tensor
    return g.op("DequantizeLinear", g.op("QuantizeLinear", inputs, scale, zero_point), scale, zero_point)


# Copied from:
# https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset13.py#L131
@parse_args('v', 'v', 'v', 'i', 'i', 'i')
def fake_quantize_per_channel_affine(g, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127):
    if quant_min not in [0, -128] or quant_max not in [127, 255]:
        raise RuntimeError(
            "ONNX defines [0, 255] for quint8 and [-128, 127] for qint8, got [{}, {}]".format(quant_min, quant_max))

    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx['Byte'])
    else:
        zero_point = g.op("Cast", zero_point, to_i=sym_help.cast_pytorch_to_onnx['Char'])
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis),
        scale, zero_point, axis_i=axis)


def _insert_identity_node_if_output_is_fakequantize(graph):
    output_names = set([output.name for output in graph.output])
    for idx, node in enumerate(graph.node):
        node_outputs = set([output_name for output_name in node.output])
        if node.op_type == "DequantizeLinear" and (output_names & node_outputs):
            output_name = (output_names & node_outputs).pop()
            identity_node = onnx.helper.make_node(
                "Identity",
                name=f"Identity_{output_name}",
                inputs=[node.name],
                outputs=[output_name]
            )
            # graph.node.remove(node)
            # graph.node.append(identity_node)
            for output in node.output:
                if output == output_name:
                    node.output.remove(output)

            # output_value_info = onnx.helper.make_tensor_value_info(
            #     identity_node.name,
            #     onnx.TensorProto.FLOAT,
            #     shape=[]
            # )

            # node.output.append(identity_node.name)
            # graph.output.pop()
            # graph.output.append(output_value_info)

def _prepare_qat_model_for_onnx(model):

    """
        Prepares a model with FakeQuantization nodes for the export to a quantized version in the
        ONNX format.
        To do that, the fake quantization nodes are disabled for the weights (as it is already
        handled by ONNX), and the all the observers are disabled as ONNX traces the model (and thus
        performs a forward pass) for the export.
    """
    to_visit = [('', model, None)]
    while to_visit:
        (name, module, parent) = to_visit.pop()
        named_children = list(module.named_children())
        if isinstance(module, torch.quantization.FakeQuantize):
            module.disable_observer()
            if (not isinstance(parent, torch.quantization.QuantStub)) and "quant" in name:
                module.disable_fake_quant()

        if named_children:
            names, children = zip(*named_children)
            to_visit += list(zip(names, children, itertools.cycle([module])))


def _register_fake_quantization_ops_to_onnx_registry(opset_version=_DEFAULT_OPSET_VERSION):
    """ Helper function to register symbolic fake quantization op if opset version is not high enough. """
    if opset_version < 10:
        register_op("fake_quantize_per_tensor_affine", fake_quantize_per_tensor_affine, "", opset_version)
    if opset_version < 13:
        register_op("fake_quantize_per_channel_affine", fake_quantize_per_channel_affine, "", opset_version)


def _load_model(model_dir):
    """ Helper function that infers the model class from the configuration, and loads the model. """
    config = AutoConfig.from_pretrained(model_dir)
    if not config.architectures or len(config.architectures) > 1:
        raise ValueError("")
    module_name = MODEL_MAPPING[config.__class__].__module__
    module = importlib.import_module(module_name)
    model_class = getattr(module, config.architectures[0])
    model = model_class.from_pretrained(model_dir)
    return model


# Copied from:
# https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py#L30
def ort_optimize_model(model_path: Path, optimization_level=None):
    '''
        Generate model that applies graph optimization (constant folding,etc.)
        parameter model_path: path to the original onnx model
        return: optimized onnx model
    '''
    opt_model_path = ort_quant_utils.generate_identified_filename(model_path, "-opt")
    sess_option = ort.SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    if optimization_level is None:
        optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_option.graph_optimization_level = optimization_level
    _ = ort.InferenceSession(model_path.as_posix(), sess_option, providers=['CPUExecutionProvider'])
    optimized_model = onnx.load(opt_model_path.as_posix())
    return optimized_model


# Copied from:
# https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/quantize.py#L294
def quantize_qat(
    model_input: Path,
    model_output: Path,
    op_types_to_quantize=[],
    per_channel=False,
    reduce_range=False,
    activation_type=ort_quant_utils.QuantType.QUInt8,
    weight_type=ort_quant_utils.QuantType.QUInt8,
    nodes_to_quantize=[],
    nodes_to_exclude=[],
    use_external_data_format=False,
    quantization_mode="integer_ops",
):
    '''
        Given a quantize-aware traning onnx model, create a quantized onnx model and save it into a file
    :param model_input: file path of model to quantize
    :param model_output: file path of quantized model
    :param op_types_to_quantize: specify the types of operators to quantize, like ['Conv'] to quantize Conv only. It quantizes all supported operators by default
    :param per_channel: quantize weights per channel
    :param reduce_range: quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode
    :param activation_type: quantization data type of activation
    :param nodes_to_quantize:
        List of nodes names to quantize. When this list is not None only the nodes in this list
        are quantized.
        example:
        [
            'Conv__224',
            'Conv__252'
        ]
    :param nodes_to_exclude:
        List of nodes names to exclude. The nodes in this list will be excluded from quantization
        when it is not None.
    :parma use_external_data_format: option used for large size (>2GB) model. Set to False by default.
    '''

    mode_mapping = {
        "integer_ops": ort_quant_utils.QuantizationMode.IntegerOps,
        "linear_ops": ort_quant_utils.QuantizationMode.QLinearOps,
    }
    if quantization_mode not in mode_mapping:
        raise ValueError(f'quantization_mode must either be "integer_ops" or "linear_ops", here: {quantization_mode}')
    mode = mode_mapping.get(quantization_mode)


    optimized_model = ort_optimize_model(Path(model_input), ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    # TODO: solve this issue.
    # _insert_identity_node_if_output_is_fakequantize(optimized_model.graph)

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(ort.quantization.registry.IntegerOpsRegistry.keys())

    quantizer = ort.quantization.onnx_quantizer.ONNXQuantizer(
        optimized_model,
        per_channel,
        reduce_range,
        mode,
        False,  #static
        weight_type,
        activation_type,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize)

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
    # optimize the original model
    optimized_model = ort_optimize_model(model_output, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)


def _export_to_onnx(
    model,
    saving_path,
    batch_size=-1,
    sequence_length=-1,
    quantization=None,
    opset_version=_DEFAULT_OPSET_VERSION,
    quantization_mode="integer_ops"
):
    # TODO: handle the input_names, the dynamic_axes, etc.
    if quantization is not None and quantization not in ["dynamic", "qat"]:
        raise ValueError(f'quantization must be either set to "dynamic" or to "qat", here: {quantization}.')

    if isinstance(model, torch.fx.GraphModule):
        inputs = model.dummy_inputs
    else:
        batch_size = max(batch_size, 1)
        sequence_length = max(sequence_length, 1)
        dummy_input = torch.randint(10, [batch_size, sequence_length])
        inputs = {
            "input_ids": dummy_input,
            "attention_mask": dummy_input,
            "token_type_ids": dummy_input
        }
    output_names = ["logits"]
    # if model.__class__ in MODEL_FOR_QUESTION_ANSWERING.values():
    #     output_names = ["start_logits", "end_logits"]

    if quantization == "qat":
        _prepare_qat_model_for_onnx(model)
        _register_fake_quantization_ops_to_onnx_registry(opset_version)

    torch.onnx.export(
        model,
        tuple(inputs.values()),
        saving_path,
        input_names=tuple(inputs.keys()),
        output_names=output_names,
        opset_version=opset_version
    )

    if quantization is not None:
        quantized_model_path = saving_path.parent.joinpath(f"{quantization}_{saving_path.stem}.onnx")
    if quantization == "dynamic":
        ort.quantization.quantize_dynamic(saving_path, quantized_model_path)
        saving_path.unlink()

    if quantization == "qat":
        quantize_qat(
            saving_path,
            quantized_model_path,
            op_types_to_quantize=[
                "MatMul",
                "Add",
                "Mul",
                "Relu",
                "Reshape",
                "Transpose",
                "Softmax",
                "Gather"
            ],
            quantization_mode=quantization_mode,
        )
        print(f"saved to {quantized_model_path}")
        saving_path.unlink()


def export_model_for_inference(
        model_dir,
        target_name,
        target="onnx",
        opset_version=_DEFAULT_OPSET_VERSION
):
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError("model_dir must be a directory.")
    elif not model_dir.joinpath("sparse_args.json").exists():
        raise FileNotFoundError(f"could not find sparse_args.json in {model_dir}.")
    elif target not in _AVAILABLE_TARGETS:
        raise ValueError(f"supported targets: {_AVAILABLE_TARGETS}, {target} was provided.")

    with open(model_dir.joinpath("sparse_args.json")) as fp:
        sparse_args = json.load(fp)

    model = _load_model(model_dir)
    optimize_model(model)

    if sparse_args["qat"]:
        qconfig_name = sparse_args["qconfig"]
        prepare_qat(model, qconfig_name=qconfig_name)
        model.load_state_dict(model_dir.joinpath("pytorch_model.bin"), strict=False)

    if quantization == "dynmic" and sparse_args["qat"]:
        print("")
        quantization == "qat"
    elif dynamic_quantization:
        quantization = "dynamic"
    elif sparse_args["qat"]:
        quantization = "qat"
    else:
        quantization = None

    if target == "onnx":
        _export_to_onnx(model, saving_path, batch_size=batch_size, sequence_length=sequence_length, quantization=quantization, opset_version=opset_version)
    else:
        raise NotImplementedError("only exporting to ONNX is currently supported.")
