from abc import ABC, abstractmethod
from functools import partial

from torch.fx import GraphModule

from .quantization_config import create_qconfig
from .quantization_transformations import (
    compose_transformations,
    insert_fake_quantize_node_between_matmul_and_bias_in_linear,
    append_noop_node_to_fake_quantize_nodes_before_output,
    wrap_matmul_with_fake_quantize_nodes,
)


class Target(ABC):
    @classmethod
    @abstractmethod
    def prepare(cls, gm: GraphModule, qconfig_dict=None) -> GraphModule:
        pass

    @property
    @abstractmethod
    def default_qconfig_dict(cls):
        pass

    @classmethod
    def validate_qconfig_dict(cls, qconfig_dict):
        return True

    @classmethod
    def validate_qat_qconfig_dict(cls, qconfig_dict):
        return True


class ONNXTarget(Target):
    @classmethod
    def prepare(cls, gm: GraphModule, qconfig_dict=None) -> GraphModule:
        transformation = compose_transformations(
            insert_fake_quantize_node_between_matmul_and_bias_in_linear,
            append_noop_node_to_fake_quantize_nodes_before_output,
            partial(wrap_matmul_with_fake_quantize_nodes, qconfig_dict=qconfig_dict)
        )
        return transformation(gm)

    @classmethod
    def default_qconfig_dict(cls):
        return create_qconfig("onnx", "static")

    @classmethod
    def default_qat_qconfig_dict(cls):
        return create_qconfig("onnx", "qat")
