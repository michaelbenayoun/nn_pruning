import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional
from torch.quantization import (
    QConfig,
    QuantStub,
    DeQuantStub,
    FakeQuantize,
    PlaceholderObserver,
    MinMaxObserver,
)
from torch.quantization.quantization_mappings import get_default_qat_module_mappings

from ..model_patcher import ModelPatcher
from ..training_patcher import ReplacementModule
from .masked_nn import MaskedLinear
from .nonorm import Layer2NoNorm, NoNorm


_DEFAULT_EMBEDDING_QAT_CONFIG = QConfig(
    # activation=FakeQuantize.with_args(observer=PlaceholderObserver),
    activation=FakeQuantize.with_args(
        observer=MinMaxObserver,
        qscheme=torch.per_tensor_symmetric
    ),
    weight=FakeQuantize.with_args(
        observer=MinMaxObserver,
        qscheme=torch.per_tensor_affine,
        dtype=torch.quint8,
    )
)


class QATEmbedding(nn.Module):
    def __init__(self, embedding, qconfig: QConfig = None, **kwargs):
        super().__init__()
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx

        self.weight = nn.Parameter(embedding.weight.detach())

        if qconfig and qconfig.weight() not in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
            self.qconfig = _DEFAULT_EMBEDDING_QAT_CONFIG
        else:
            self.qconfig = qconfig
        if qconfig:
            self.weight_fake_quant = self.qconfig.weight()

    def forward(self, input: torch.Tensor):
        weight = self.weight_fake_quant(self.weight) if self.qconfig else self.weight
        return weight[input]

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**self.__dict__)

    @classmethod
    def from_float(cls, embedding):
        return cls(embedding, qconfig=embedding.qconfig)


class QATMaskedLinear(MaskedLinear):
    def __init__(self, masked_linear, qconfig: QConfig = None, **kwargs):
        super(ReplacementModule, self).__init__()

        def is_attribute(attribute_name):
            attribute = getattr(masked_linear, attribute_name)
            is_special = attribute_name.startswith("__")
            is_callable = callable(attribute) and not isinstance(attribute, nn.Module)
            return (not is_special) and (not is_callable)

        attributes = [name for name in dir(masked_linear) if is_attribute(name)]
        for attribute_name in attributes:
            setattr(self, attribute_name, getattr(masked_linear, attribute_name))

        self.qconfig = qconfig
        if qconfig:
            # Respecting naming convention to match nn.qat.Linear.
            self.weight_fake_quant = qconfig.weight()
            self.activation_post_process = QuantStub(qconfig)

    def forward(self, input):
        masked_weights, bias = self.get_masked_weights_bias()

        if self.qconfig:
            masked_weights = self.weight_fake_quant(masked_weights)

        output = F.linear(input, masked_weights, bias)

        if self.qconfig:
            output = self.activation_post_process(output)

        return output

    @classmethod
    def from_float(cls, masked_linear):
        return cls(masked_linear, qconfig=masked_linear.qconfig)


class QATGelu(nn.Module):
    def __init__(self, qconfig: QConfig = None, **kwargs):
        super().__init__()
        self.qconfig = qconfig
        if qconfig:
            self.quant_mul = QuantStub(qconfig)
            self.quant_erf = QuantStub(qconfig)
            self.quant_mul_x = QuantStub(qconfig)
            self.quant_out = QuantStub(qconfig)

    def forward(self, x):

        y = x / 1.4142135623730951

        if self.qconfig:
            y = self.quant_mul(y)

        y = torch.erf(y)

        if self.qconfig:
            y = self.quant_erf(y)

        y += 1.0

        if self.qconfig:
            x = self.quant_mul_x(0.5 * x)

        res = x * y

        if self.qconfig:
            res = self.quant_out(res)

        return res


class QATFastGelu(nn.Module):
    def __init__(self, qconfig: QConfig = None, **kwargs):
        super().__init__()

        self.coeff1 = 0.044715
        self.coeff2 = 0.7978845608

        if qconfig:
            self.quant_square = QuantStub(qconfig)
            self.quant_coeff1 = QuantStub(qconfig)
            self.quant_coeff2 = QuantStub(qconfig)
            self.quant_mul = QuantStub(qconfig)
            self.quant_prod = QuantStub(qconfig)
            self.quant_tanh = QuantStub(qconfig)
            self.quant_output = QuantStub(qconfig)

    def forward(self, x):
        # Initial computation:
        # 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

        square = x * x

        if self.qconfig:
            square = self.quant_square(square)

        square = square * self.coeff1

        if self.qconfig:
            square = self.quant_coeff1(square)

        square = square + 1

        y = x * self.coeff2

        if self.qconfig:
            y = self.quant_coeff2(y)

        prod = y * square

        if self.qconfig:
            prod = self.quant_prod(prod)

        prod = torch.tanh(prod)

        if self.qconfig:
            prod = self.quant_tanh(prod)

        prod = prod + 1

        z = 0.5 * x

        if self.qconfig:
            z = self.quant_mul(z)

        output = z * prod

        if self.qconfig:
            output = self.quant_output(output)

        return output


class QATNoNorm(nn.Module):
    def __init__(self, nonorm, qconfig: QConfig = None, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(nonorm.weight.detach())
        self.bias = nn.Parameter(nonorm.bias.detach())

        self.qconfig = qconfig
        if qconfig:
            self.add = FloatFunctional()
            self.mul = FloatFunctional()
            self.weight_fake_quant = self.weight_quant().to(nonorm.weight.device)
            self.bias_fake_quant = self.weight_quant().to(nonorm.weight.device)

    def forward(self, batch):
        weight = self.weight_fake_quant(self.weight) if self.qconfig else self.weight
        bias = self.bias_fake_quant(self.bias) if self.qconfig else self.bias
        output = self.add.add(self.mul.mul(batch, weight), bias)
        return output

    @classmethod
    def from_float(cls, nonorm):
        return cls(nonorm, qconfig=nonorm.qconfig)


class QATLayer2NoNorm(nn.Module):
    # There are two ways to specify how the module will move progressively from a LayerNorm to a NoNorm
    # If you give a non-None schedule_callback, steps and start_delta won"t be used.
    # It must be a function that returns a dictionary containing at least two keys:
    #  - mix : moving from 1.0 to 0.0 , it is the lerp factor between LayerNorm and NoNorm: 1.0 -> LayerNorm, 0.0 -> NoNorm
    #  - delta : moving from 0.99 to 1.0 for example, it is the accumulator exponential decay,
    #   the higher the longer the period it smooth the mean/variance accumulator
    # If you don"t specify a schedule_callback, each call to forward will count as a step, and in "steps" steps
    # it will move to a LayerNorm to a NoNorm

    def __init__(
        self,
        layerNorm,
        qconfig: QConfig = None,
        steps: int = 5000,
        start_delta: float = 0.99,
        schedule_callback=None,
        **kwargs
    ):
        super().__init__()
        self.normalized_shape = layerNorm.normalized_shape
        self.eps = layerNorm.eps
        self.elementwise_affine = layerNorm.elementwise_affine
        assert(self.elementwise_affine)
        self.weight = nn.Parameter(layerNorm.weight.detach().clone())
        self.bias = nn.Parameter(layerNorm.bias.detach().clone())
        # Accumulators are for mean and std, and accumulator normalization factor
        self.schedule_callback = schedule_callback

        if self.schedule_callback is None:
            self.steps = steps
            self.delta = start_delta
            self.final_delta = 1.0
            self.mix = 1.0
            self.delta_step = (self.final_delta - self.delta) / self.steps
            self.mix_step = 1 / self.steps
        else:
            self.steps = None
            self.delta_step = None
            self.mix_step = None
            self.delta = None
            self.final_delta = None
            self.mix = None

        self.register_buffer("accumulator", torch.zeros(3, device=layerNorm.weight.device))

        self.qconfig = qconfig
        if qconfig:
            self.weight_fake_quant = qconfig.weight()
            self.bias_fake_quant = qconfig.weight()
            self.add = FloatFunctional()
            self.mul = FloatFunctional()

    def forward(self, batch):
        accumulator = self.accumulator.clone()

        if self.schedule_callback is not None:
            d = self.schedule_callback()
            mix = d["mix"]
            delta = d["delta"]
        else:
            if self.training:
                mix = self.mix
                delta = self.delta
            else:
                mix = 0
                delta = 1.0

        if mix == 0 and delta == 1.0:
            batch_mean = accumulator[0] / accumulator[2]
            batch_var = accumulator[1] / accumulator[2]
        else:
            batch_mean = batch.mean(-1, keepdim=True)
            batch_var = batch.var(-1, unbiased=False, keepdim=True)

            if self.training:
                one = torch.tensor(1.0, device=batch_var.device)
                new_acc = torch.stack([batch_mean.mean(), batch_var.mean(), one])
                accumulator = torch.lerp(new_acc, accumulator, delta)

            batch_mean = torch.lerp(accumulator[0] / accumulator[2], batch_mean, mix)
            batch_var = torch.lerp(accumulator[1] / accumulator[2], batch_var, mix)

        ret = (batch - batch_mean) / (batch_var + self.eps).sqrt()
        if self.qconfig:
            weight = self.weight_fake_quant(self.weight)
            bias = self.bias_fake_quant(self.bias)
        else:
            weight = self.weight
            bias = self.bias

        if self.qconfig:
            ret = self.add(self.mul(ret, weight), bias)

        ret = ret * weight + bias

        if self.training:
            self.accumulator = accumulator.detach()
            if self.schedule_callback is None:
                self.mix = max(0.0, self.mix - self.mix_step)
                self.delta = min(self.final_delta, self.delta + self.delta_step)

        return ret

    def compile(self):
        accumulator = self.accumulator
        mean = accumulator[0] / accumulator[2]
        var = accumulator[1] / accumulator[2]

        inv_var = 1.0 / (var + self.eps).sqrt()

        weight = self.weight * inv_var
        bias = - mean * inv_var * self.weight + self.bias

        return NoNorm(weight.detach().clone(), bias.detach().clone())

# class QATPatcher(ModelPatcher):
#     qat_layers = {
#         Layer2NoNorm: QuantizedLayer2NoNorm,
#         NoNorm: QATNoNorm,
#         MaskedLinear: QuantizedMaskedLinear,
#     }
#
#     def __init__(self, qconfig, layers_to_patch=None, layers_not_to_patch=None, **kwargs):
#         super().__init__(all_match=True)
#         self.qconfig = qconfig
#         self.kwargs = kwargs
#         if layers_to_patch is None:
#             layers_to_patch = QATPatcher.qat_layers.keys()
#         elif not isinstance(layers_to_patch, (list, tuple)):
#             layers_to_patch = [layers_to_patch]
#         self.layers_to_patch = set(layers_to_patch)
#         if layers_not_to_patch is None:
#             layers_not_to_patch = []
#         elif not isinstance(layers_not_to_patch, (list, tuple)):
#             layers_not_to_patch = [layers_not_to_patch]
#         self.layers_not_to_patch = set(layers_not_to_patch)
#         self.patchable_layers = tuple(self.layers_to_patch - self.layers_not_to_patch)
#
#     def is_patchable(self, module_name, module, raiseError):
#         return isinstance(module, self.patchable_layers)
#
#     def new_child_module(self, child_module_name, child_module, patch_info):
#         qat_cls = QATPatcher.qat_layers[type(child_module)]
#         qat_module = qat_cls(child_module, qconfig=self.qconfig, **self.kwargs)
#         return qat_module
#
#
# def prepare_for_qat(model, qconfig_name: str = None, qconfig: QConfig = None):
#     if qconfig_name and qconfig:
#         raise ValueError("either specify a qconfig name or a qconfig, but not both")
#     if qconfig_name:
#         qconfig = create_qconfig(qconfig_name)
#     model.train()
#     qat_module_patcher = QATPatcher(qconfig)
#     qat_module_patcher.patch(model)
#     qat_op_patcher(model, qconfig=qconfig)
#     model.qconfig = qconfig
#     model = torch.quantization.prepare_qat(model)
#     return model


_QAT_MAPPING = get_default_qat_module_mappings()
_QAT_MAPPING.update({
    nn.Embedding: QATEmbedding,
    NoNorm: QATNoNorm,
    Layer2NoNorm: QATLayer2NoNorm,
    MaskedLinear: QATMaskedLinear,
})
