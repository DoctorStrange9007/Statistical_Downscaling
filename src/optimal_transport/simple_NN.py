"""Lightweight JAX MLP building block for q_0, q_1, ..., q_N models.

This module provides a tiny, dependency-free MLP in pure JAX/JAX NumPy with:
- functional APIs: `init_mlp_params`, `apply_mlp`
- a small OO convenience wrapper: `SimpleMLP`

You can reuse the same MLP for every q_n by choosing appropriate input/output
dimensions (e.g., concatenate variables you need as inputs; set out_dim to
desired size like a scalar, logits vector, etc.).
"""

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Dict

import jax
import jax.numpy as jnp


Array = jax.Array
Params = List[Dict[str, Array]]  # [{"W": (in, out), "b": (out,)}, ...]


def _select_activation(name: str) -> Callable[[Array], Array]:
    """Return activation function by name: 'tanh', 'relu', 'gelu', 'identity'."""
    name = (name or "identity").lower()
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "gelu":
        return jax.nn.gelu
    if name == "identity" or name == "none":
        return lambda x: x
    raise ValueError(f"Unknown activation '{name}'.")


def _glorot_uniform(key: Array, fan_in: int, fan_out: int) -> Array:
    """Glorot uniform initializer compatible with JAX arrays."""
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape=(fan_in, fan_out), minval=-limit, maxval=limit)


@dataclass
class MLPConfig:
    """Configuration for the MLP architecture."""

    hidden_dims: Sequence[int]
    activation: str = "tanh"  # 'tanh' | 'relu' | 'gelu' | 'identity'
    final_activation: str = "identity"


def init_mlp_params(
    key: Array,
    in_dim: int,
    out_dim: int,
    config: MLPConfig,
) -> Params:
    """Initialize parameters for an MLP.

    Args:
        key: PRNGKey for initialization.
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        config: MLPConfig object.

    Returns:
        List of layer parameter dicts. Each dict has:
        - "W": weight matrix of shape (in_features, out_features)
        - "b": bias vector of shape (out_features,)
    """
    keys = jax.random.split(key, num=len(config.hidden_dims) + 1)

    layer_dims: List[Tuple[int, int]] = []
    prev = in_dim
    for h in config.hidden_dims:
        layer_dims.append((prev, h))
        prev = h
    # Final layer
    layer_dims.append((prev, out_dim))

    params: Params = []
    for k, (fan_in, fan_out) in zip(keys, layer_dims):
        W = _glorot_uniform(k, fan_in, fan_out)
        b = jnp.zeros((fan_out,))
        params.append({"W": W, "b": b})
    return params


def apply_mlp(
    params: Params,
    x: Array,
    config: MLPConfig,
) -> Array:
    """Apply MLP to input x.

    Args:
        params: Parameters from `init_mlp_params`.
        x: Input array with shape (..., in_dim).
        config: MLPConfig object.

    Returns:
        Output array with shape (..., out_dim).
    """
    act = _select_activation(config.activation)
    final_act = _select_activation(config.final_activation)

    h = x
    num_layers = len(params)
    for layer_idx, layer in enumerate(params):
        W, b = layer["W"], layer["b"]
        h = h @ W + b  # affine
        is_last = layer_idx == (num_layers - 1)
        if not is_last:
            h = act(h)
        else:
            h = final_act(h)
    return h


class SimpleMLP:
    """Object-oriented wrapper around the functional MLP for convenience."""

    def __init__(self, in_dim: int, out_dim: int, config: MLPConfig):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config

    def init(self, key: Array) -> Params:
        return init_mlp_params(key, self.in_dim, self.out_dim, self.config)

    def apply(self, params: Params, x: Array) -> Array:
        return apply_mlp(params, x, self.config)

    def jit_apply(self) -> Callable[[Params, Array], Array]:
        """Return a jitted apply function specialized to this config."""
        return jax.jit(lambda p, x: apply_mlp(p, x, self.config))


def make_mlp(
    key: Array,
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    activation: str = "tanh",
    final_activation: str = "identity",
) -> Tuple[Params, Callable[[Params, Array], Array]]:
    """Convenience constructor returning (params, apply_fn).

    Example:
        key = jax.random.PRNGKey(0)
        params, apply_fn = make_mlp(key, in_dim=4, out_dim=1, hidden_dims=(64, 64))
        y = apply_fn(params, jnp.ones((3, 4)))  # (3, 1)
    """
    config = MLPConfig(
        hidden_dims=tuple(hidden_dims),
        activation=activation,
        final_activation=final_activation,
    )
    params = init_mlp_params(key, in_dim, out_dim, config)
    apply_fn = jax.jit(lambda p, x: apply_mlp(p, x, config))
    return params, apply_fn


__all__ = [
    "MLPConfig",
    "Params",
    "init_mlp_params",
    "apply_mlp",
    "SimpleMLP",
    "make_mlp",
]
