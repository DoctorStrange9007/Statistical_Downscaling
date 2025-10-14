"""Helper script to count parameters of a UNet configuration.

Run this to quickly sanity-check the parameter count for a given UNet setup.
Optionally verify a forward pass shape using `--verify_forward`.
"""

import os
import sys
import argparse
from typing import Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np


def add_repo_root_to_path() -> None:
    """Ensure repository root is at the front of `sys.path`."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def human_count(n: int) -> str:
    """Format an integer count with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


def count_params(pytree) -> int:
    """Count total number of scalar parameters in a JAX pytree."""
    sizes = [int(np.prod(np.array(x.shape))) for x in jax.tree_util.tree_leaves(pytree)]
    return int(sum(sizes))


def build_model_and_params(
    out_channels: int,
    num_channels: Tuple[int, ...],
    downsample_ratio: Tuple[int, ...],
    num_blocks: int,
    noise_embed_dim: int,
    use_attention: bool,
    num_heads: int,
):
    """Construct a UNet and initialize parameters for a fixed input shape.

    Returns a tuple of (model, variables) where `variables` contains a 'params'
    collection suitable for counting.
    """
    try:
        from src.generation.unets import UNet
    except Exception as e:
        raise ImportError("Failed to import UNet from src.generation.unets") from e

    model = UNet(
        out_channels=out_channels,
        num_channels=num_channels,
        downsample_ratio=downsample_ratio,
        num_blocks=num_blocks,
        noise_embed_dim=noise_embed_dim,
        use_attention=use_attention,
        num_heads=num_heads,
        use_position_encoding=False,
        dropout_rate=0.0,
    )

    key = jax.random.PRNGKey(0)
    x = jnp.zeros((1, 192, 1))
    sigma = jnp.ones((1,))
    variables: dict[str, Any] = model.init(
        {"params": key}, x, sigma, cond=None, is_training=False
    )
    return model, variables


def main():
    """Parse args, build UNet, and report parameter count (and optional shape)."""
    add_repo_root_to_path()

    parser = argparse.ArgumentParser(description="Count parameters of UNet (KS config)")
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument(
        "--num_channels",
        type=str,
        default="32,64,128",
        help="Comma-separated channels per stage",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=str,
        default="2,2,2",
        help="Comma-separated ratios per stage (e.g., 2,2,2 for 192→96→48→24)",
    )
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--noise_embed_dim", type=int, default=128)
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument(
        "--verify_forward",
        action="store_true",
        help="If set, run a forward pass to verify output shape",
    )

    args = parser.parse_args()

    num_channels = tuple(int(x) for x in args.num_channels.split(",") if x)
    downsample_ratio = tuple(int(x) for x in args.downsample_ratio.split(",") if x)

    model, variables = build_model_and_params(
        out_channels=args.out_channels,
        num_channels=num_channels,
        downsample_ratio=downsample_ratio,
        num_blocks=args.num_blocks,
        noise_embed_dim=args.noise_embed_dim,
        use_attention=args.use_attention,
        num_heads=args.num_heads,
    )

    params = variables.get("params", {})
    total = count_params(params)

    print("Model:", model.__class__.__name__)
    print("Config:")
    print("  out_channels:", args.out_channels)
    print("  num_channels:", num_channels)
    print("  downsample_ratio:", downsample_ratio)
    print("  num_blocks:", args.num_blocks)
    print("  noise_embed_dim:", args.noise_embed_dim)
    print("  use_attention:", args.use_attention)
    print("  num_heads:", args.num_heads)
    print()
    print(f"Total parameters: {total} ({human_count(total)})")

    if args.verify_forward:
        x = jnp.zeros((2, 192, 1))
        sigma = jnp.full((2,), 1.0)
        y = model.apply(variables, x, sigma, cond=None, is_training=False)
        print("Forward output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
