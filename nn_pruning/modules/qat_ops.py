import functools
import inspect

import torch
from torch import nn
from torch.quantization import QConfig, QuantStub, DeQuantStub

import transformers

from .masked_nn import MaskedLinear
from .qat_modules import QATMaskedLinear


_DEFAULT_BLACKLIST = {
    nn.Linear,
    MaskedLinear,
    QATMaskedLinear,
    nn.LayerNorm,
    "get_extended_attention_mask",
}


def _find_proper_op_name(initial_name, module):
    i = 0
    name = f"{initial_name}_{i}"
    while hasattr(module, name):
        name = f"{initial_name}_{i}"
        i += 1
    return name


def _num_op_in_module(op_name, module):
    i = 0
    name = f"{op_name}_{i}"
    while hasattr(module, name):
        i += 1
        name = f"{op_name}_{i}"
    return i


def _mark_submodules(model):
    """ Helper function that marks model submodules as submodules. """
    for name, module in model.named_children():
        module.is_submodule = True
        _mark_submodules(module)


def _mark_as_patched(model):
    """ Helper function that marks a model and its submodules as patched. """
    for name, module in model.named_children():
        if hasattr(module, "is_patched") and module.is_patched:
            print(f"{name} is already patched")
        else:
            module.is_patched = True
            _mark_as_patched(module)


def _find_module_in_frame(frame):
    if "self" in frame.f_locals and isinstance(frame.f_locals["self"], nn.Module):
        return frame.f_locals["self"]
    for v in frame.f_locals:
        if isinstance(v, nn.Module):
            return v
    return None


def _get_model_submodules_ids(model):
    to_visit = [model]
    submodules = set()
    while to_visit:
        module = to_visit.pop()
        submodules.add(id(module))
        to_visit += list(module.children())
    return submodules


def _is_blacklisted(frame, module, blacklist):
    blacklisted_functions = set([entry for entry in blacklist if isinstance(entry, str)])
    blacklisted_modules = tuple((entry for entry in blacklist if not isinstance(entry, str)))
    if frame.f_code.co_name in blacklisted_functions or isinstance(module, blacklisted_modules):
        return True
    return False


def _broadcast_input(a, b):
    rank_a = len(a.size())
    rank_b = len(b.size())
    rank = max(rank_a, rank_b)
    shape_a = [1] * (rank - rank_a) + list(a.size())
    shape_b = [1] * (rank - rank_b) + list(b.size())
    shape = [max(x, y) for x, y in zip(shape_a, shape_b)]
    return torch.broadcast_to(a, shape), torch.broadcast_to(b, shape)


def wrap_float_op(op_name, blacklist=None, supports_broadcasting=True):
    float_op = getattr(torch.Tensor, op_name)
    functional_op_name = f"{op_name}_functional"
    functional_op_name_index = f"{functional_op_name}_index"

    if blacklist is None:
        blacklist = tuple()
    blacklist = list(blacklist)
    blacklist += [torch.quantization.observer.ObserverBase]
    blacklist = tuple(blacklist)

    @functools.wraps(float_op)
    def wrapped_op(self, other):
        frame = inspect.currentframe()
        calling_frame = frame.f_back
        mod = calling_frame.f_locals.get("self", None)
        if mod is None or _is_blacklisted(calling_frame, mod, blacklist):
            op = getattr(self, op_name)
            return op(other)
        if not (hasattr(mod, "is_patched") and mod.is_patched):
            attr_name = _find_proper_op_name(functional_op_name, mod)
            setattr(mod, attr_name, torch.nn.quantized.FloatFunctional())

        index = getattr(mod, functional_op_name_index, 0)
        functional_op = getattr(mod, f"{functional_op_name}_{index}")
        updated_index = (index + 1) % _num_op_in_module(functional_op_name, mod)
        setattr(mod, functional_op_name_index, updated_index)

        function_name = op_name
        if not isinstance(other, torch.Tensor):
            function_name = f"{op_name}_scalar"
        function = getattr(functional_op, function_name)

        x, y = self, other
        if not supports_broadcasting:
            x, y = _broadcast_input(x, y)

        return function(x, x)

    return wrapped_op


def wrap_op_in_module_or_class(op_name, module_or_class=None, qconfig: QConfig = None, blacklist=None):
    float_op = getattr(module_or_class, op_name)
    qat_op_name = f"qat_{op_name}"
    qat_op_name_index = f"{qat_op_name}_index"

    if blacklist is None:
        blacklist = tuple()
    blacklist = tuple(blacklist)

    class QATOp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.float_op = float_op
            self.qconfig = qconfig
            if qconfig:
                self.dequant = DeQuantStub()
                self.activation_post_process = QuantStub(qconfig)

        def _dequant_input(self, input):
            if isinstance(input, torch.Tensor) and input.dtype in [torch.qint8, torch.quint8]:
                return self.dequant(input)
            return input

        def forward(self, *args, **kwargs):
            args = list(map(self._dequant_input, args))
            output = self.float_op(*args, **kwargs)
            if self.qconfig:
                output = self.activation_post_process(output)
            return output

    @functools.wraps(float_op)
    def wrapped_op(*args, **kwargs):
        print(op_name)
        frame = inspect.currentframe()
        calling_frame = frame.f_back
        mod = _find_module_in_frame(calling_frame)
        if mod is None:
            return float_op(*args, **kwargs)
        if not getattr(mod, "is_submodule", False):
            # Happens for module not defined as submodules.
            # Going two frame back because:
            # mod.forward -> mod._call_impl -> parent.forward
            calling_frame = calling_frame.f_back.f_back
            parent_mod = _find_module_in_frame(calling_frame)
            # TODO: understand what happens with torch.onnx.export
            if parent_mod is mod:  # Happens during torch.onnx.export for some reason.
                calling_frame = calling_frame.f_back
                mod = _find_module_in_frame(calling_frame)
            else:
                mod = parent_mod
        # If it's module from a non patched model, use regular op.
        if not getattr(mod, "is_submodule", False):
            mod = None
        if mod is None or _is_blacklisted(calling_frame, mod, blacklist):
            return float_op(*args, **kwargs)
        if not (hasattr(mod, "is_patched") and mod.is_patched):
            attr_name = _find_proper_op_name(qat_op_name, mod)
            setattr(mod, attr_name, QATOp())

        index = getattr(mod, qat_op_name_index, 0)
        qat_op = getattr(mod, f"{qat_op_name}_{index}")
        updated_index = (index + 1) % _num_op_in_module(qat_op_name, mod)
        setattr(mod, qat_op_name_index, updated_index)

        return qat_op(*args, **kwargs)

    return wrapped_op

# TODO: refactor the monkey patch functions
wrap_torch_op = functools.partial(wrap_op_in_module_or_class, module_or_class=torch)
wrap_torch_nn_functional_op = functools.partial(wrap_op_in_module_or_class, module_or_class=torch.nn.functional)
wrap_transformer_modeling_utils_op = functools.partial(wrap_op_in_module_or_class, module_or_class=transformers.modeling_utils.ModuleUtilsMixin)


def _monkey_patch_float_op(float_op_name, blacklist=None):
    supports_broadcasting = True
    if isinstance(float_op_name, (tuple, list)):
        float_op_name, supports_broadcasting = float_op_name
    if not hasattr(torch.Tensor, float_op_name):
        print(f"{float_op_name} not found in torch.Tensor attributes, skipping monkey patching.")
        return
    new_op = wrap_float_op(float_op_name, blacklist=blacklist, supports_broadcasting=supports_broadcasting)
    setattr(torch.Tensor, f"__{float_op_name}__", new_op)
    setattr(torch.Tensor, f"__i{float_op_name}__", new_op)  # inplace counterpart


def _monkey_patch_torch_op(op_name, qconfig=None, blacklist=None):
    if (not hasattr(torch, op_name)) and (not hasattr(torch.nn.functional, op_name)):
        print(f"{op_name} not found in torch and torch.nn.functional modules, skipping monkey patching.")
        return
    if hasattr(torch, op_name):
        setattr(torch, op_name, wrap_torch_op(op_name, qconfig=qconfig, blacklist=blacklist))
    if hasattr(torch.nn.functional, op_name):
        setattr(
            torch.nn.functional,
            op_name,
            wrap_torch_nn_functional_op(op_name, qconfig=qconfig, blacklist=blacklist)
        )

def _monkey_patch_transformers_op(op_name, qconfig=None, blacklist=None):
    if hasattr(transformers.modeling_utils.ModuleUtilsMixin, op_name):
        setattr(transformers.modeling_utils.ModuleUtilsMixin, op_name, wrap_transformer_modeling_utils_op(op_name))


def monkey_op_patcher(
    float_ops=(("add", False), "mul"),
    torch_ops=("matmul", "softmax", "relu", "tanh"),
    transformers_ops=("get_extended_attention_mask", ),
    qconfig: QConfig = None,
    blacklist=None,
):
    if blacklist is None:
        blacklist = _DEFAULT_BLACKLIST
    else:
        blacklist = set(blacklist)
    if isinstance(float_ops, str):
        float_ops = (float_ops, )
    if isinstance(torch_ops, str):
        torch_ops = (torch_ops, )
    if isinstance(transformers_ops, str):
        transformers_ops = (transformers_ops, )
    for op in float_ops:
        _monkey_patch_float_op(op, blacklist=blacklist)
    for op in torch_ops:
        _monkey_patch_torch_op(op, qconfig=qconfig, blacklist=blacklist)
    for op in transformers_ops:
        _monkey_patch_transformers_op(op, qconfig=qconfig, blacklist=blacklist)


def qat_op_patcher(model, qconfig: QConfig = None, blacklist=None):
    _mark_submodules(model)
    monkey_op_patcher(qconfig=qconfig, blacklist=blacklist)
    model(**model.dummy_inputs)
    _mark_as_patched(model)
