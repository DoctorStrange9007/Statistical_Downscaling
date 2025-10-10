import yaml
import os
import sys
import argparse
from datetime import datetime
import jax
import jax.numpy as jnp

# Ensure repository root on path when running this file directly by path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.generation.prior import HR_data, HR_prior
from src.generation.Statistical_Downscaling_PDE import (
    StatisticalDownscalingPDESolver,
)
from src.generation import utils_generation as utils

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

if __name__ == "__main__":

    os.environ["WANDB"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="src/generation/settings_generation.yaml"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--project", type=str, default="statistical-downscaling-main-generation"
    )
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    # Master RNG: use general.seed if present; else fixed
    seed = run_sett.get("general", {}).get("seed", 37)
    master_key = jax.random.PRNGKey(seed)
    key_data, key_prior, key_pde, key_sde = jax.random.split(master_key, 4)

    env_enable = os.environ.get("WANDB", "").lower() in {"1", "true", "yes", "on"}
    use_wandb = bool((args.wandb or env_enable) and (wandb is not None))
    run = None
    if (args.wandb or env_enable) and wandb is None:
        print("wandb not installed; proceeding without logging.")

    run_name = args.run_name or ("main_gen_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if use_wandb:
        run = wandb.init(
            project=args.project, entity=args.entity, name=run_name, config=run_sett
        )
        try:
            print(
                f"W&B logging enabled: project={args.project}, entity={args.entity}, run={run.name}"
            )
            if hasattr(run, "url"):
                print(f"View run: {run.url}")
        except Exception:
            pass

    def log_fn(payload: dict):
        """Safe metric logger that no-ops if logging is disabled or fails."""
        if use_wandb:
            try:
                wandb.log(payload)
            except Exception:
                pass

    def log_image(key: str, path: str):
        """Safe image logger to avoid repeating guards and try/except blocks."""
        if use_wandb:
            try:
                wandb.log({key: wandb.Image(path)})
            except Exception:
                pass

    hr_data = HR_data(run_sett, rng_key=key_data)
    samples = hr_data.get_samples()
    utils.plot_samples(samples, run_sett["output_dir"], "samples.png")
    log_image("plots/samples", os.path.join(run_sett["output_dir"], "samples.png"))

    hr_prior = HR_prior(samples, run_sett, rng_key=key_prior)
    hr_prior.train(log_fn=log_fn if use_wandb else None)

    all_msd = {}
    diff_before = samples @ jnp.array(run_sett["pde_solver"]["C"]).T - jnp.array(
        run_sett["pde_solver"]["y_target"]
    )
    all_msd["input_data"] = jnp.mean(jnp.square(diff_before).reshape(-1, 1))

    # def analytical_score(x, t):
    #    return -(x-hr_prior.s(t)*5.0)

    x_1, samples_after = utils.sde_solver_backwards_cond(
        key_sde,
        hr_prior.trained_score,  # analytical_score,
        None,
        hr_prior.g,
        hr_prior.f,
        run_sett["general"]["d"],
        run_sett["general"]["n_samples_generate"],
        run_sett["general"]["T"],
        hr_prior.sigma2,
        hr_prior.s,
        conditional=False,
    )
    all_msd["gen_without_conditioning"] = utils.calculate_msd(samples_after, run_sett)

    utils.plot_samples(
        samples_after,
        run_sett["output_dir"],
        "samples_after_gen_without_conditioning.png",
    )
    log_image(
        "plots/samples_after_gen_without_conditioning",
        os.path.join(
            run_sett["output_dir"], "samples_after_gen_without_conditioning.png"
        ),
    )
    utils.plot_hyperplane(
        samples_after,
        all_msd["gen_without_conditioning"],
        run_sett,
        "samples_2d_after_gen_without_conditioning.png",
        None,
    )
    log_image(
        "plots/samples_2d_after_gen_without_conditioning",
        os.path.join(
            run_sett["output_dir"], "samples_2d_after_gen_without_conditioning.png"
        ),
    )

    # Use subclassed statistical downscaling PDE
    pde_solver = StatisticalDownscalingPDESolver(
        grad_log=hr_prior.trained_score,  # analytical_score,
        samples=samples,
        settings=run_sett,
        rng_key=key_pde,
    )

    lambda_value = (
        jnp.float32(run_sett["pde_solver"]["lambda"])
        if "lambda" in run_sett.get("pde_solver", {})
        else None
    )

    pde_solver.train(
        log_fn=(
            (lambda payload: log_fn({**payload, "lambda": lambda_value}))
            if use_wandb
            else None
        )
    )

    x_1, samples_after = utils.sde_solver_backwards_cond(
        key_sde,
        hr_prior.trained_score,  # analytical_score
        pde_solver.grad_log_h,
        hr_prior.g,
        hr_prior.f,
        run_sett["general"]["d"],
        run_sett["general"]["n_samples_generate"],
        run_sett["general"]["T"],
        hr_prior.sigma2,
        hr_prior.s,
        conditional=True,
    )

    all_msd["gen_with_conditioning"] = utils.calculate_msd(samples_after, run_sett)
    utils.plot_samples(
        samples_after, run_sett["output_dir"], "samples_after_gen_with_conditioning.png"
    )
    log_image(
        "plots/samples_after_gen_with_conditioning",
        os.path.join(run_sett["output_dir"], "samples_after_gen_with_conditioning.png"),
    )
    utils.plot_hyperplane(
        samples_after,
        all_msd["gen_with_conditioning"],
        run_sett,
        "samples_2d_after_gen_with_conditioning.png",
        run_sett["pde_solver"]["lambda"],
    )
    log_image(
        "plots/samples_2d_after_gen_with_conditioning",
        os.path.join(
            run_sett["output_dir"], "samples_2d_after_gen_with_conditioning.png"
        ),
    )

    utils.plot_marginals_1_2_and_joint12(
        samples_after,
        run_sett["output_dir"],
        ["marginal_x1.png", "marginal_x2.png", "joint_x1x2.png"],
        run_sett,
    )
    log_image(
        "plots/marginal_x1",
        os.path.join(run_sett["output_dir"], "marginal_x1.png"),
    )
    log_image(
        "plots/marginal_x2",
        os.path.join(run_sett["output_dir"], "marginal_x2.png"),
    )
    log_image(
        "plots/joint_x1x2",
        os.path.join(run_sett["output_dir"], "joint_x1x2.png"),
    )

    if lambda_value is not None:
        log_fn({"lambda": lambda_value})

    msd = utils.calculate_msd(samples_after, run_sett)
    log_fn({"msd/conditional": float(msd)})
    if lambda_value is not None:
        log_fn({"lambda": lambda_value})

    if run is not None:
        run.finish()
