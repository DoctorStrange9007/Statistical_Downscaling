import yaml
import os
import sys
import argparse
from datetime import datetime
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prior import HR_data, HR_prior
from Statistical_Downscaling_PDE import StatisticalDownscalingPDESolver
import utils_generation as utils

try:
    import wandb  # type: ignore
except Exception:
    wandb = None

if __name__ == "__main__":

    os.environ["WANDB"] = "0"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="src/generation/settings_generation.yaml"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--project", type=str, default="statistical-downscaling-lambda-sweep"
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

    run_name = args.run_name or (
        "lambda_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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

    def log_fn(payload: dict):
        if use_wandb:
            wandb.log(payload)

    hr_prior = HR_prior(samples, run_sett, rng_key=key_prior)
    hr_prior.train(log_fn=log_fn if use_wandb else None)

    all_msd = {}
    diff_before = samples @ jnp.array(run_sett["pde_solver"]["C"]).T - jnp.array(
        run_sett["pde_solver"]["y_target"]
    )
    all_msd["input_data"] = jnp.mean(jnp.square(diff_before).reshape(-1, 1))
    if use_wandb:
        try:
            wandb.log(
                {
                    "msd/input_data": float(
                        all_msd["input_data"]
                    )  # baseline MSD on input
                }
            )
        except Exception:
            pass

    # Deterministic key for the unconditional generation (independent of loop order)
    x_1, samples_after = utils.sde_solver_backwards_cond(
        key_sde,
        hr_prior.trained_score,
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

    utils.plot_samples(
        samples_after,
        run_sett["output_dir"],
        "samples_after_gen_without_conditioning.png",
    )
    all_msd["gen_without_conditioning"] = utils.calculate_msd(samples_after, run_sett)
    if use_wandb:
        try:
            wandb.log(
                {
                    "msd/unconditional": float(
                        all_msd["gen_without_conditioning"]
                    )  # MSD after unconditional gen
                }
            )
        except Exception:
            pass
    utils.plot_hyperplane(
        samples_after,
        all_msd["gen_without_conditioning"],
        run_sett,
        "samples_2d_after_gen_without_conditioning.png",
        None,
    )
    if use_wandb:
        try:
            wandb.log(
                {
                    "plots/uncond_hyperplane": wandb.Image(
                        os.path.join(
                            run_sett["output_dir"],
                            "samples_2d_after_gen_without_conditioning.png",
                        )
                    )
                }
            )
        except Exception:
            pass

    lambdas = list(jnp.linspace(0.1, 2, num=2))
    for lambda_ in lambdas:
        run_sett["pde_solver"]["lambda"] = float(lambda_)

        pde_solver = StatisticalDownscalingPDESolver(
            grad_log=hr_prior.trained_score,
            samples=samples,
            settings=run_sett,
            rng_key=key_pde,
        )
        pde_solver.train(
            log_fn=(
                (lambda payload, lam=lambda_: log_fn({**payload, "lambda": float(lam)}))
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

        # utils.plot_samples(samples_after, run_sett["output_dir"], "samples_after_" + str(lambda_) + "_.png")
        lam_key = f"{float(lambda_):.6g}"
        all_msd[lam_key] = utils.calculate_msd(samples_after, run_sett)
        if use_wandb:
            try:
                wandb.log(
                    {
                        "msd/conditional": float(all_msd[lam_key]),
                        "lambda": float(lambda_),
                    }
                )
                # Also log by-key metric name for quick comparison across runs
                wandb.log({f"msd/by_lambda/{lam_key}": float(all_msd[lam_key])})
            except Exception:
                pass
        utils.plot_hyperplane(
            samples_after,
            all_msd[lam_key],
            run_sett,
            "samples_2d_after_" + lam_key + "_.png",
            lambda_,
        )
        if use_wandb:
            try:
                img_path = os.path.join(
                    run_sett["output_dir"], f"samples_2d_after_{lam_key}_.png"
                )
                wandb.log(
                    {
                        "plots/cond_hyperplane": wandb.Image(img_path),
                        "lambda": float(lambda_),
                    }
                )
            except Exception:
                pass

    utils.output_to_excel_and_plot(all_msd, run_sett)
    if use_wandb:
        try:
            # Log the full MSD dictionary as a table for convenience
            table = wandb.Table(columns=["key", "msd"])
            for k, v in all_msd.items():
                table.add_data(str(k), float(v))
            wandb.log({"summary/msd_table": table})
        except Exception:
            pass
    if run is not None:
        run.finish()
    a = 5
