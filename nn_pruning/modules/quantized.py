import torch
from torch import nn
from torch.quantization.quantization_mappings import get_default_static_quant_module_mappings
from torch.nn.quantized import QFunctional, Linear as QLinear
# from torch.nn.quantized.modules.utils import _quantize_weight

from .qat_modules import QATMaskedLinear, QATEmbedding, QATNoNorm, QATLayer2NoNorm


def _check_module_type(cls, module, allowed_types):
    if not isinstance(allowed_types, (tuple, list, set)):
        allowed_types = (allowed_types, )
    else:
        allowed_types = tuple(allowed_types)
    if not isinstance(module, allowed_types):
        error_message = f"{cls}.from_float module argument must be in {allowed_types} but {type(module)} was provided"
        raise ValueError(error_message)


def _quantize_weight(float_wt, observer):
    wt_scale, wt_zp = observer.calculate_qparams()
    if observer.qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]:
        qweight = torch.quantize_per_tensor(
            float_wt,
            float(wt_scale), int(wt_zp), observer.dtype)
    elif observer.qscheme in [torch.per_channel_symmetric, torch.per_channel_affine]:
        wt_axis = observer.ch_axis
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.double), wt_zp.to(torch.int64), wt_axis, observer.dtype)
    elif observer.qscheme in [torch.per_channel_affine_float_qparams]:
        qweight = torch.quantize_per_channel(
            float_wt,
            wt_scale.to(torch.float), wt_zp.to(torch.float), observer.ch_axis, observer.dtype)
    else:
        raise ValueError("Unexpected qscheme " + observer.qscheme)
    return qweight


class QuantizedEmbedding(nn.Module):
    def __init__(self, qweight):
        super().__init__()
        assert qweight.dtype in [torch.qint8, torch.quint8]
        self.qweight = qweight

    def forward(self, input):
        return self.qweight[input]

    @classmethod
    def from_float(cls, mod):
        _check_module_type(cls, mod, QATEmbedding)

        # weight_scale, weight_zp = mod.weight_fake_quant.calculate_qparams()
        # qweight = torch.quantize_per_tensor(
        #     mod.weight,
        #     float(weight_scale),
        #     int(weight_zp),
        #     dtype=mod.weight_fake_quant.dtype
        # )

        qweight = _quantize_weight(mod.weight, mod.weight_fake_quant)
        qembedding = cls(qweight)

        return qembedding


class QuantizedNoNorm(nn.Module):
    def __init__(self, qweight, qbias, qadd=None, qmul=None):
        super().__init__()
        assert qweight.dtype in [torch.qint8, torch.quint8]
        assert qbias.dtype in [torch.qint8, torch.quint8]
        self.qweight = qweight
        self.qbias = qbias
        self.qadd = qadd if qadd else QFunctional()
        self.qmul = qmul if qmul else QFunctional()

    def forward(self, batch):
        return self.qadd.add(
            self.qmul.mul(self.qweight, batch),
            torch.broadcast_to(self.qbias, batch.size())  # Add operands must have the same size.
        )

    @classmethod
    def from_float(cls, mod):
        _check_module_type(cls, mod, (QATNoNorm, QATLayer2NoNorm))

        if isinstance(mod, QATLayer2NoNorm):
            nonorm = mod.compile()
            weight = nonorm.weight
            bias = nonorm.bias
        else:
            weight = mod.weight
            bias = mod.bias

        weight_post_process = mod.weight_fake_quant
        bias_post_process = mod.bias_fake_quant

        weight_post_process(weight)
        weight_scale, weight_zp = weight_post_process.calculate_qparams()
        qweight = torch.quantize_per_tensor(
            mod.weight.float(),
            float(weight_scale),
            int(weight_zp),
            torch.quint8
        )

        bias_post_process(bias)
        bias_scale, bias_zp = bias_post_process.calculate_qparams()
        qbias = torch.quantize_per_tensor(
            mod.bias.float(),
            float(bias_scale),
            int(bias_zp),
            torch.quint8
        )

        qnonorm = cls(qweight, qbias, qadd=mod.add, qbias=mod.mul)

        return qnonorm


_QUANTIZE_MAPPING = get_default_static_quant_module_mappings()
_QUANTIZE_MAPPING.update({
    QATEmbedding: QuantizedEmbedding,
    QATMaskedLinear: QLinear,
    QATNoNorm: QuantizedNoNorm,
    QATLayer2NoNorm: QuantizedNoNorm
})


def quantize(model):
    model.eval()
    model = torch.quantization.convert(
        model,
        mapping=_QUANTIZE_MAPPING,
        inplace=False,
        remove_qconfig=False,
    )
    return model
