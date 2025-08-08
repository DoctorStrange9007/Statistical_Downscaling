import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class LSTMLayerJax(nn.Module):
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, S: jax.Array, X: jax.Array) -> jax.Array:
        glorot = nn.initializers.glorot_uniform()
        Uz = self.param("Uz", glorot, (self.input_dim, self.output_dim))
        Ug = self.param("Ug", glorot, (self.input_dim, self.output_dim))
        Ur = self.param("Ur", glorot, (self.input_dim, self.output_dim))
        Uh = self.param("Uh", glorot, (self.input_dim, self.output_dim))

        Wz = self.param("Wz", glorot, (self.output_dim, self.output_dim))
        Wg = self.param("Wg", glorot, (self.output_dim, self.output_dim))
        Wr = self.param("Wr", glorot, (self.output_dim, self.output_dim))
        Wh = self.param("Wh", glorot, (self.output_dim, self.output_dim))

        bz = self.param("bz", nn.initializers.zeros, (self.output_dim,))
        bg = self.param("bg", nn.initializers.zeros, (self.output_dim,))
        br = self.param("br", nn.initializers.zeros, (self.output_dim,))
        bh = self.param("bh", nn.initializers.zeros, (self.output_dim,))

        Z = jnp.tanh(jnp.matmul(X, Uz) + jnp.matmul(S, Wz) + bz)
        G = jnp.tanh(jnp.matmul(X, Ug) + jnp.matmul(S, Wg) + bg)
        R = jnp.tanh(jnp.matmul(X, Ur) + jnp.matmul(S, Wr) + br)
        H = jnp.tanh(jnp.matmul(X, Uh) + jnp.matmul(S * R, Wh) + bh)
        S_new = (1.0 - G) * H + Z * S
        return S_new


class DenseLayerJax(nn.Module):
    input_dim: int
    output_dim: int
    transformation: Optional[str] = None  # 'tanh', 'relu', or None

    @nn.compact
    def __call__(self, X: jax.Array) -> jax.Array:
        W = self.param(
            "W", nn.initializers.glorot_uniform(), (self.input_dim, self.output_dim)
        )
        b = self.param("b", nn.initializers.zeros, (self.output_dim,))
        S = jnp.matmul(X, W) + b
        if self.transformation == "tanh":
            S = jnp.tanh(S)
        elif self.transformation == "relu":
            S = nn.relu(S)
        return S


class DGMNetJax(nn.Module):
    input_dim: int  # spatial input dimension d
    layer_width: int
    num_layers: int
    final_trans: Optional[str] = None

    @nn.compact
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        X = jnp.concatenate([t, x], axis=1)
        S = DenseLayerJax(self.input_dim + 1, self.layer_width, transformation="tanh")(
            X
        )
        for _ in range(self.num_layers):
            S = LSTMLayerJax(self.input_dim + 1, self.layer_width)(S, X)
        out = DenseLayerJax(self.layer_width, 1, transformation=self.final_trans)(S)
        return out
