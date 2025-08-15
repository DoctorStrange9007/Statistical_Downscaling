import yaml
import os
import argparse
from datetime import datetime
import jax
import jax.numpy as jnp
from prior import HR_data, HR_prior
from Statistical_Downscaling_PDE import StatisticalDownscalingPDESolver
import utils_generation as utils

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
        if use_wandb:
            wandb.log(payload)

    hr_data = HR_data(run_sett, rng_key=key_data)
    samples = hr_data.get_samples()
    utils.plot_samples(samples, run_sett["output_dir"], "samples.png")
    if use_wandb:
        try:
            wandb.log(
                {
                    "plots/samples": wandb.Image(
                        os.path.join(run_sett["output_dir"], "samples.png")
                    )
                }
            )
        except Exception:
            pass

    hr_prior = HR_prior(samples, run_sett, rng_key=key_prior)
    hr_prior.train(log_fn=log_fn if use_wandb else None)

    # Use subclassed statistical downscaling PDE
    pde_solver = StatisticalDownscalingPDESolver(
        grad_log=hr_prior.trained_score,
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
        hr_prior.trained_score,
        pde_solver.grad_log_h,
        hr_prior.g,
        hr_prior.f,
        run_sett["general"]["d"],
        run_sett["general"]["n_samples"],
        run_sett["general"]["T"],
        hr_prior.sigma2,
        hr_prior.s,
        conditional=True,
    )  # 2however, doesn't seem to translate to the samples we generate.

    utils.plot_samples(samples_after, run_sett["output_dir"], "samples_after.png")
    if use_wandb:
        try:
            wandb.log(
                {
                    "plots/samples_after": wandb.Image(
                        os.path.join(run_sett["output_dir"], "samples_after.png")
                    ),
                    "lambda": lambda_value,
                }
            )
        except Exception:
            pass

    msd = utils.calculate_msd(samples_after, run_sett)
    if use_wandb:
        try:
            wandb.log(
                {
                    "msd/conditional": float(msd),
                    "lambda": lambda_value,
                }
            )
        except Exception:
            pass

    if run is not None:
        run.finish()
