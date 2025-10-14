import os
import sys
import argparse
from datetime import datetime
import jax
import jax.numpy as jnp

# Ensure repository root on path for absolute-style imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.generation.Other.experiments.Heat_PDE import HeatPDESolver, make_heat_settings

try:
    import wandb  # type: ignore
except Exception:  # wandb is optional
    wandb = None


if __name__ == "__main__":

    os.environ["WANDB"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--sampling_stages", type=int, default=50)
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="statistical-downscaling-heat-experiment",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity", type=str, default=None, help="W&B entity (team or user)"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Optional W&B run name"
    )
    args = parser.parse_args()

    d = args.d
    T = args.T
    sigma = args.sigma

    rng = jax.random.PRNGKey(0)
    a = jax.random.normal(rng, (d,)) * 0.5

    settings = make_heat_settings(d=d, T=T, sampling_stages=args.sampling_stages)
    solver = HeatPDESolver(a_vec=a, sigma=sigma, settings=settings, rng_key=rng)

    env_enable = os.environ.get("WANDB", "").lower() in {"1", "true", "yes", "on"}
    attempt_wandb = bool(args.wandb or env_enable)
    use_wandb = bool(attempt_wandb and (wandb is not None))
    run = None
    if attempt_wandb and wandb is None:
        print(
            "wandb not installed; proceeding without logging. pip install wandb to enable."
        )

    if use_wandb:
        config = {
            "d": d,
            "T": T,
            "sigma": sigma,
            "sampling_stages": settings["pde_solver"]["sampling_stages"],
            "steps_per_sample": settings["pde_solver"]["steps_per_sample"],
            "learning_rate": settings["pde_solver"]["learning_rate"],
            "x_low": settings["pde_solver"]["x_low"],
            "x_high": settings["pde_solver"]["x_high"],
            "x_multiplier": settings["pde_solver"]["x_multiplier"],
            "nSim_interior": settings["pde_solver"]["nSim_interior"],
            "nSim_terminal": settings["pde_solver"]["nSim_terminal"],
            "net_nodes_per_layer": settings.get("pre_trained", {})
            .get("model", {})
            .get("nodes_per_layer", 64),
            "net_num_layers": settings.get("pre_trained", {})
            .get("model", {})
            .get("num_layers", 3),
            "seed": 0,
        }
        run_name = (
            args.run_name
            or f"heat_d{d}_T{T}_sig{sigma}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        run = wandb.init(
            project=args.project, entity=args.entity, name=run_name, config=config
        )
        try:
            print(
                f"W&B logging enabled: project={args.project}, entity={args.entity}, run={run.name}"
            )
            if hasattr(run, "url"):
                print(f"View run: {run.url}")
        except Exception:
            pass

    def log_metrics(payload: dict):
        """Safe metric logger that no-ops if logging is disabled or fails."""
        if use_wandb:
            try:
                wandb.log(payload)
            except Exception:
                pass

    log_cb = log_metrics if use_wandb else None
    solver.train(log_fn=log_cb)

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

    log_metrics(
        {
            "eval/mae_t0": float(mae_0),
            "eval/rmse_t0": float(rmse_0),
            "eval/mae_t0.5": float(mae_1),
            "eval/rmse_t0.5": float(rmse_1),
            "eval/mae_t1.0": float(mae_2),
            "eval/rmse_t1.0": float(rmse_2),
        }
    )

    if run is not None:
        run.finish()
