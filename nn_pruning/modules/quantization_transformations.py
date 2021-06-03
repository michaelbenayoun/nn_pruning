import operator
import functools
from typing import Callable

import torch
from torch.fx import GraphModule
from torch.quantization import QuantStub, FakeQuantize
from torch.quantization.quantize import register_activation_post_process_hook


def _add_fake_quant_to_module(module, fake_quant_name, qconfig, device=None):
    module.add_module(fake_quant_name, QuantStub(qconfig))
    stub = getattr(module, fake_quant_name)
    fake_quant = stub.qconfig.activation()
    if device:
        fake_quant = fake_quant.to(device)
    stub.add_module(
        "activation_post_process",
        fake_quant,
    )
    register_activation_post_process_hook(stub)


def _remove_nodes_matching_pattern_in_range(gm, start, end, patterns, named_modules, lint_and_recompile=True):
    graph = gm.graph
    n = start
    nodes_to_erase = []
    while n != end:
        if _match(n, patterns, named_modules):
            input_nodes = [arg for arg in n.args if isinstance(arg, torch.fx.Node)]
            if len(input_nodes) != 1:
                # TODO: choose the right kind of error.
                raise RuntimeError("cannot remove node that does not have exactly one input node")
            input_node = input_nodes[0]
            n.replace_all_uses_with(input_node)
            # print(named_modules[n.target])
            # print(named_modules[input_node.target])
            graph.erase_node(n)
            nodes_to_erase.append(n)
        n = n.next

    # for node in nodes_to_erase:
    #     graph.erase_node(node)

    if lint_and_recompile:
        graph.lint()
        gm.recompile()
    return gm


def change_truediv_to_mul_when_possible(gm: GraphModule):
    """Transforms truediv nodes by multiplication of the inverse when the denominator is static."""
    graph = gm.graph
    for node in graph.nodes:
        if node.op == "call_function" and node.target == operator.truediv:
            x, y = node.args
            if not isinstance(y, torch.fx.Node):
                node.target = operator.mul
                node.args = (x, 1 / y)
    graph.lint()
    gm.recompile()
    return gm


def insert_fake_quantize_node_between_matmul_and_bias_in_linear(gm: GraphModule) -> GraphModule:
    """
    Inserts a FakeQuantize node between the matrix multiplication and the addition of the bias in
    linear layers. This is useful when exporting to ONNX as linear layers are transformed to:
        torch.nn.linear(x) => x @ Wt + b
    """
    graph = gm.graph
    named_modules = dict(gm.named_modules())
    for node in graph.nodes:
        if node.op == 'call_module' and isinstance(named_modules.get(node.target, None), torch.nn.Linear):
            linear = named_modules[node.target]
            setattr(
                linear,
                "weight",
                torch.nn.Parameter(linear.weight.permute(1, 0))
            )
            _add_fake_quant_to_module(
                linear, "matmul_fake_quant", linear.qconfig, linear.weight.device
            )

            with graph.inserting_after(node):
                weight = graph.get_attr(f"{node.target}.weight")
                bias = graph.get_attr(f"{node.target}.bias")

            with graph.inserting_after(weight):
                fake_quant_weight = graph.call_module(
                    f"{node.target}.weight_fake_quant",
                    args=(weight, )
                )

            with graph.inserting_after(fake_quant_weight):
                matmul_node = graph.call_function(
                    torch.matmul,
                    args=(node.args[0], fake_quant_weight),
                )

            with graph.inserting_after(matmul_node):
                fake_quant_matmul = graph.call_module(
                    f"{node.target}.matmul_fake_quant",
                    args=(matmul_node, )
                )

            with graph.inserting_after(fake_quant_matmul):
                output_node = graph.call_function(
                    torch.add,
                    args=(fake_quant_matmul, bias),
                )

            node.replace_all_uses_with(output_node)
            graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return gm


def wrap_matmul_with_fake_quantize_nodes(gm: GraphModule, qconfig_dict) -> GraphModule:
    """ Wraps matrix multplication with Fake Quantize nodes. """
    graph = gm.graph
    named_modules = dict(gm.named_modules())
    # TODO: handle the case in which some specific quantization scheme was passed for torch.matmul.
    qconfig = qconfig_dict[""]
    for node in graph.nodes:
        if node.op == "call_function" and node.target == torch.matmul:
            inputs_and_outputs = {n: "input" for n in node.args}
            inputs_and_outputs.update({n: "output" for n in node.users.keys()})
            for n, type_ in inputs_and_outputs.items():
                if n.op == "call_module" and isinstance(named_modules.get(n.target, None), FakeQuantize):
                    continue

                insertion = graph.inserting_after if type_ == "input" else graph.inserting_before
                with insertion(n):
                    fake_quant_name = f"{n.name}_activation_post_process"
                    _add_fake_quant_to_module(gm, fake_quant_name, qconfig)
                    input_node = n if type_ == "input" else node
                    fake_quant_node = graph.call_module(
                        fake_quant_name,
                        args=(input_node, )
                    )
                    n.replace_all_uses_with(fake_quant_node)
                    fake_quant_node.args = (input_node, )
    graph.lint()
    gm.recompile()
    return gm


def append_noop_node_to_fake_quantize_nodes_before_output(gm: GraphModule) -> GraphModule:
    """
    Appends a NoOp node (currently Add by 0) to Fake Quantized nodes that are outputs.
    This is useful because ONNXRuntime quantization tool does not support the case when Fake
    Quantize nodes are not followed by any other node.
    """
    # TODO: make it cleaner instead of using torch.add(node, 0)
    output_node = list(gm.graph.nodes)[-1]
    args = output_node.args
    named_modules = dict(gm.named_modules())
    if isinstance(args[0], dict):
        args = args[0].values()
    for node in args:
        if node.op == "call_module" and isinstance(named_modules[node.target], FakeQuantize):
            with gm.graph.inserting_after(node):
                identity_node = gm.graph.call_function(torch.add, args=(node, 1e-12))
            node.replace_all_uses_with(identity_node)
            identity_node.args = (node, 1e-12)

    gm.graph.lint()
    gm.recompile()
    return gm


def remove_fake_quantize_nodes_from_attention(gm: GraphModule) -> GraphModule:
    # TODO: write proper attention_pattern (the one with linears)
    attention_pattern = [
        torch.nn.Linear,
        torch.nn.Linear,
        "view",
        "permute",
        torch.nn.Linear,
        "view",
        "permute",
        "view",
        "permute",
        "transpose",
        torch.matmul,
        # Both cases are valid because change_truediv_to_mul_when_possible transformation can be
        # applied.
        (operator.mul, operator.truediv),
        operator.add,
        torch.nn.Softmax,
        torch.matmul,
        "permute",
        "contiguous",
        "view",
    ]
    attention_with_decomposed_linears_pattern = [
        torch.matmul,
        torch.add,
        torch.matmul,
        torch.add,
        "view",
        "permute",
        torch.matmul,
        torch.add,
        "view",
        "permute",
        "view",
        "permute",
        "transpose",
        torch.matmul,
        # Both cases are valid because change_truediv_to_mul_when_possible transformation can be
        # applied.
        (operator.mul, operator.truediv),
        operator.add,
        torch.nn.Softmax,
        torch.matmul,
        "permute",
        "contiguous",
        "view",
    ]
    ignore_pattern = [QuantStub, FakeQuantize, torch.nn.Dropout]
    matches = match_pattern(gm, attention_pattern, ignore=ignore_pattern)
    if not matches:
        matches = match_pattern(gm, attention_with_decomposed_linears_pattern, ignore=ignore_pattern)

    named_modules = dict(gm.named_modules())
    pattern = [QuantStub, FakeQuantize]
    for (start, end) in matches:
        _remove_nodes_matching_pattern_in_range(
            gm, start, end, pattern, named_modules, lint_and_recompile=False
        )

    gm.graph.lint()
    gm.recompile()

    return gm


def change_attention_mask_value(gm: GraphModule, initial_value: int = -10000, final_value: int = -20) -> GraphModule:
    """Changes the attention mask initial value (default is -10000) to some smaller value."""
    graph = gm.graph
    for node in graph.nodes:
        if node.target in [torch.mul, operator.mul] and initial_value in node.args:
            new_args = []
            for arg in node.args:
                if arg == initial_value:
                    new_args.append(final_value)
                else:
                    new_args.append(arg)
            node.args = tuple(new_args)
            break

    graph.lint()
    gm.recompile()
    return gm


def broadcast_attention_mask(gm: GraphModule) -> GraphModule:
    """
    Broadcasts attention_mask to match the shape of attention_scores as broadcasting is not
    supported for quantized add.
    """
    graph = gm.graph
    attention_mask = None
    for node in graph.nodes:
        if (
            node.target in [torch.mul, torch.ops.quantized.mul, operator.mul]
            and len(node.args) > 1
            and node.args[1] < 0
        ):
            attention_mask = node
            break
    if attention_mask is None:
        print("Could not find attention_mask to broadcast.")
        return
    for node in graph.nodes:
        if node.target == torch.ops.quantized.add and attention_mask in node.args:
            attention_scores, mask, *rest = node.args
            with graph.inserting_before(node):
                broadcasted_mask = graph.call_function(
                    torch.broadcast_to, args=(mask, graph.call_method("size", args=(attention_scores, )))
                )
                new_args = []
                for arg in node.args:
                    if arg is mask:
                        new_args.append(broadcasted_mask)
                    else:
                        new_args.append(arg)
                node.args = tuple(new_args)
    graph.lint()
    gm.recompile()
    return gm


def broadcast_nonorm_bias(gm: GraphModule) -> GraphModule:
    """
    Broadcasts the NoNorm bias to match the shape of the scaled input as broadcasting is not
    supported for quantized add.
    """
    graph = gm.graph
    for node in graph.nodes:
        if node.target == torch.ops.quantized.mul:
            users = list(node.users.keys())
            if len(users) == 1 and users[0].target == torch.ops.quantized.add:
                add_node = users[0]
                with graph.inserting_before(add_node):
                    broadcasted_bias = graph.call_function(
                        torch.broadcast_to,
                        args=(add_node.args[1], graph.call_method("size", args=(add_node.args[0],))),
                    )
                    new_args = add_node.args[:1] + (broadcasted_bias,) + add_node.args[2:]
                    add_node.args = tuple(new_args)

    graph.lint()
    gm.recompile()
    return gm


def _match(node, node_patterns, named_modules):
    if not isinstance(node_patterns, (list, tuple)):
        node_patterns = [node_patterns]
    return any(
        any((
            callable(pattern) and node.op == "call_function" and node.target == pattern,
            isinstance(pattern, str) and node.op == "call_method" and node.target == pattern,
            isinstance(pattern, type) and node.op == "call_module" and isinstance(named_modules[node.target], pattern),
        ))
        for pattern in node_patterns
    )


def match_pattern(gm, pattern, ignore=None):
    if ignore is None:
        ignore = []
    graph = gm.graph
    named_modules = dict(gm.named_modules())
    num_nodes_in_pattern = len(pattern)
    matches = []
    for node in graph.nodes:
        idx = 0
        n = node
        is_a_match = True
        while idx < num_nodes_in_pattern:
            p = pattern[idx]
            if n.op not in ["call_function", "call_method", "call_module"] or _match(n, ignore, named_modules):
                if n.next:
                    n = n.next
                    continue
                else:
                    break
            if _match(n, p, named_modules):
                if n.next:
                    idx += 1
                    n = n.next
                    continue
                else:
                    break
            else:
                is_a_match = False
                break

        if is_a_match and idx == num_nodes_in_pattern:
            matches.append((node, n))

    return matches


def compose_transformations(*args: Callable[[GraphModule], GraphModule]) -> GraphModule:
    return functools.reduce(lambda f, g: lambda gm: f(g(gm)), args, lambda x: x)
