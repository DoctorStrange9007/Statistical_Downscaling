import os
import sys

import jax
import jax.numpy as jnp

# Ensure repository root on path for absolute-style imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.generation.Heat_PDE import HeatPDESolver, make_heat_settings


if __name__ == "__main__":
    d = 10
    T = 1.0
    sigma = 0.2
    rng = jax.random.PRNGKey(0)
    a = jax.random.normal(rng, (d,)) * 0.5

    settings = make_heat_settings(d=d, T=T, sampling_stages=50)
    solver = HeatPDESolver(a_vec=a, sigma=sigma, settings=settings, rng_key=rng)
    solver.train()

    # Evaluate error at t=0, 0.5, 1.0 on random batch
    rng, kx = jax.random.split(rng)
    B = 256
    t_eval_0 = jnp.ones((B, 1)) * 0.0
    t_eval_1 = jnp.ones((B, 1)) * 0.5
    t_eval_2 = jnp.ones((B, 1)) * 1.0
    x_eval = jax.random.uniform(
        kx, (B, d), minval=-1.0, maxval=1.0
    )  # only trained on this interval
    pred_0 = solver.V(solver.params, t_eval_0, x_eval)
    pred_1 = solver.V(solver.params, t_eval_1, x_eval)
    pred_2 = solver.V(solver.params, t_eval_2, x_eval)
    gt_0 = solver.exact_solution(t_eval_0, x_eval)
    gt_1 = solver.exact_solution(t_eval_1, x_eval)
    gt_2 = solver.exact_solution(t_eval_2, x_eval)
    mae_0 = jnp.mean(jnp.abs(pred_0 - gt_0))
    rmse_0 = jnp.sqrt(jnp.mean((pred_0 - gt_0) ** 2))
    mae_1 = jnp.mean(jnp.abs(pred_1 - gt_1))
    rmse_1 = jnp.sqrt(jnp.mean((pred_1 - gt_1) ** 2))
    mae_2 = jnp.mean(jnp.abs(pred_2 - gt_2))
    rmse_2 = jnp.sqrt(jnp.mean((pred_2 - gt_2) ** 2))
    print(
        f"Evaluation on d={d}: MAE_0={float(mae_0):.6f}, RMSE_0={float(rmse_0):.6f}, "
        f"MAE_1={float(mae_1):.6f}, RMSE_1={float(rmse_1):.6f}, "
        f"MAE_2={float(mae_2):.6f}, RMSE_2={float(rmse_2):.6f}"
    )
