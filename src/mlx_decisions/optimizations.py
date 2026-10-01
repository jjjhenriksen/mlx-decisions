"""Small auditable MLX adaptations; no copied model implementation.

Head narrowing adapts mlxfast's last-position/evaluation-only prefill seam.
Fused gate/up adapts its row-concatenation idea to dense affine MLX modules.
Both preserve stored quantization codes; neither re-quantizes the model.
"""

import copy

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.models.activations import swiglu
from mlx_lm.models.cache import ArraysCache, KVCache


class SelectedHead:
    """Only project the 52 letter rows, not the 248,320-token vocabulary.

    Conditional softmax over candidate logits equals re-normalized full-vocab
    log probabilities: the shared logsumexp cancels. These are NOT full-vocab
    probabilities or a legal-mass estimate.
    """

    def __init__(self, head, token_ids):
        ids = mx.array(token_ids, dtype=mx.int32)
        self.weight = mx.contiguous(head.weight[ids])
        self.bias = None if getattr(head, "bias", None) is None else mx.contiguous(head.bias[ids])
        self.quantized = isinstance(head, nn.QuantizedLinear)
        if self.quantized:
            self.scales = mx.contiguous(head.scales[ids])
            self.biases = None if head.biases is None else mx.contiguous(head.biases[ids])
            self.group_size, self.bits, self.mode = head.group_size, head.bits, head.mode
        mx.eval(self.weight)
        if self.quantized:
            mx.eval(self.scales, self.biases)

    def __call__(self, hidden):
        if self.quantized:
            result = mx.quantized_matmul(
                hidden,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        else:
            result = hidden @ self.weight.T
        return result if self.bias is None else result + self.bias


def fork_cache(cache, batch_size=1):
    """Copy BOTH attention K/V and recurrent conv/SSM state.

    Only unpadded, equal-offset branches are supported. Do not trim or treat
    Qwen3.5 recurrent state as a token-indexed KV cache. All mutable array
    leaves are copied, including when batch_size=1.
    """
    result = []
    for layer in cache:
        if type(layer) not in (KVCache, ArraysCache):
            raise TypeError(f"unsupported cache class: {type(layer).__name__}")
        clone = copy.copy(layer)
        clone.state = tree_map(
            lambda x: (
                mx.repeat(x, batch_size, axis=0)
                if isinstance(x, mx.array) and batch_size > 1
                else mx.array(x)
                if isinstance(x, mx.array)
                else x
            ),
            layer.state,
        )
        result.append(clone)
    return result


def evaluate_cache(cache):
    mx.eval([layer.state for layer in cache])


class FusedGateUp(nn.Module):
    """Optional dense SwiGLU gate/up fusion. Preserve the down projection."""

    def __init__(self, original):
        super().__init__()
        gate, up = original.gate_proj, original.up_proj
        if type(gate) is not type(up) or type(gate) not in (nn.Linear, nn.QuantizedLinear):
            raise TypeError("gate/up fusion requires matching standard MLX linear modules")
        if gate.weight.shape != up.weight.shape:
            raise ValueError("gate/up shape mismatch")
        self.quantized = isinstance(gate, nn.QuantizedLinear)
        if self.quantized and (gate.bits, gate.group_size, gate.mode) != (
            up.bits,
            up.group_size,
            up.mode,
        ):
            raise ValueError("mixed gate/up quantization is unsupported")
        self.width = gate.weight.shape[0]
        self.weight = mx.concatenate([gate.weight, up.weight], axis=0)
        self.bias = None
        gb, ub = getattr(gate, "bias", None), getattr(up, "bias", None)
        if (gb is None) != (ub is None):
            raise ValueError("mixed gate/up bias is unsupported")
        if gb is not None:
            self.bias = mx.concatenate([gb, ub], axis=0)
        if self.quantized:
            self.scales = mx.concatenate([gate.scales, up.scales], axis=0)
            if (gate.biases is None) != (up.biases is None):
                raise ValueError("mixed quantization biases")
            self.biases = (
                None if gate.biases is None else mx.concatenate([gate.biases, up.biases], axis=0)
            )
            self.group_size, self.bits, self.mode = gate.group_size, gate.bits, gate.mode
        self.down_proj = original.down_proj
        mx.eval(self.parameters())

    def __call__(self, x):
        if self.quantized:
            both = mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        else:
            both = x @ self.weight.T
        if self.bias is not None:
            both = both + self.bias
        return self.down_proj(swiglu(both[..., : self.width], both[..., self.width :]))


def fuse_gate_up(model):
    """Opt-in only: replacement can temporarily increase load-time memory."""
    layers = model.language_model.model.layers
    # Validate the complete surface before making any changes.
    for layer in layers:
        mlp = layer.mlp
        if not all(hasattr(mlp, k) for k in ("gate_proj", "up_proj", "down_proj")):
            raise TypeError("fusion supports dense Qwen3.5 MLPs only")
    for layer in layers:
        layer.mlp = FusedGateUp(layer.mlp)
    return len(layers)
