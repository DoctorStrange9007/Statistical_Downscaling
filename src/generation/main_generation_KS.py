import os
import sys
import jax

try:
    import wandb  # type: ignore
except Exception:  # wandb is optional
    wandb = None  # type: ignore
import jax.numpy as jnp
from clu import metric_writers
from src.generation.wandb import WandbWriter
import yaml
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.generation.Statistical_Downscaling_PDE_KS import (
    KSStatisticalDownscalingPDESolver,
)
from src.generation.denoiser_utils import (
    create_denoiser_model,
    create_diffusion_scheme,
    restore_denoise_fn,
    build_model,
    build_trainer,
    run_training,
)

from src.generation.utils_metrics import (
    calculate_constraint_rmse,
    calculate_kld,
    calculate_sample_variability,
    calculate_melr,
)
from src.generation.data_utils import get_raw_datasets, get_ks_dataset
from src.generation.sampler_utils import (
    sample_unconditional,
    sample_wan_guided,
    sample_pde_guided,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/generation/settings_generation.yaml"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

USE_WANDB = False


def log_fn(payload: dict):
    """Safe metric logger that no-ops if logging is disabled or fails."""
    if USE_WANDB and (wandb is not None):
        try:
            wandb.log(payload)
        except Exception:
            pass


def main():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )  # local machine
    work_dir = os.path.join(
        project_root, "temporary", "main_generation_KS"
    )  # FOR A FRESH RUN empty temporary/
    os.makedirs(work_dir, exist_ok=True)

    u_HFHR, u_LFLR, u_HFLR, x, t = get_raw_datasets()

    u_hfhr_samples = u_HFHR.reshape(-1, int(run_sett["general"]["d"]), 1)
    u_lflr_samples = u_LFLR.reshape(-1, int(run_sett["general"]["d_prime"]), 1)
    DATA_STD = u_hfhr_samples.std()

    denoiser_model = create_denoiser_model()
    diffusion_scheme = create_diffusion_scheme(DATA_STD)

    mode = "sample"

    # Create a unified metric writer once (W&B-backed if enabled)
    use_wandb = bool(USE_WANDB and not os.environ.get("WANDB_DISABLED"))
    base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)
    if use_wandb:
        project = os.environ.get("WANDB_PROJECT", "statistical-downscaling-denoiser-KS")
        run_name = os.environ.get("WANDB_NAME", os.path.basename(work_dir))
        entity = os.environ.get("WANDB_ENTITY")  # optional
        writer_name = run_name if mode == "train" else f"{run_name}-sample"
        writer = WandbWriter(
            base_writer,
            project=project,
            name=writer_name,
            entity=entity,
            config={"work_dir": work_dir, "mode": mode, **run_sett},
        )
    else:
        writer = base_writer
    if mode == "train":
        print("--- Running in Training Mode ---")
        model = build_model(denoiser_model, diffusion_scheme, DATA_STD)
        trainer = build_trainer(model)

        # writer already initialized above

        run_training(
            train_dataloader=get_ks_dataset(
                u_hfhr_samples,
                split="train[:75%]",
                batch_size=run_sett["general"]["batch_size"],
            ),
            trainer=trainer,
            workdir=work_dir,
            total_train_steps=run_sett["general"]["total_train_steps"],
            metric_writer=writer,
            metric_aggregation_steps=run_sett["general"]["metric_aggregation_steps"],
            eval_dataloader=get_ks_dataset(
                u_hfhr_samples,
                split="train[75%:]",
                batch_size=run_sett["general"]["batch_size"],
            ),
            eval_every_steps=run_sett["general"]["eval_every_steps"],
            num_batches_per_eval=run_sett["general"]["num_batches_per_eval"],
            save_interval_steps=run_sett["general"]["save_interval_steps"],
            max_to_keep=run_sett["general"]["max_to_keep"],
        )

        # writer is closed at the end of main()

        denoise_fn = restore_denoise_fn(f"{work_dir}/checkpoints", denoiser_model)
        if run_sett["option"] == "conditional":
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                y_bar=u_lflr_samples[
                    0 : int(run_sett["pde_solver"]["num_models"])
                ],  # can do this for any sample, maybe code this down better such that we can extend this to multiple samples at once
                settings=run_sett,
                denoise_fn=denoise_fn,
                scheme=diffusion_scheme,
                rng_key=jax.random.PRNGKey(int(run_sett["rng_key"])),
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
            pde_params_dir = os.path.join(work_dir, "pde_params")
            pde_solver.save_params(pde_params_dir)

    elif mode == "sample":
        print("--- Running in Sampling-Only Mode ---")
        downsampling_factor = int(run_sett["general"]["d"]) / int(
            run_sett["general"]["d_prime"]
        )  # ensure integer stride
        C_prime = jnp.array(
            [
                [
                    1 if j == downsampling_factor * i else 0
                    for j in range(int(run_sett["general"]["d"]))
                ]
                for i in range(int(run_sett["general"]["d_prime"]))
            ]
        )
        denoise_fn = restore_denoise_fn(f"{work_dir}/checkpoints", denoiser_model)
        if run_sett["option"] == "unconditional":
            ##########################################################
            #### GENERATES UNCONDITIONAL SAMPLES ####

            samples = sample_unconditional(
                diffusion_scheme,
                denoise_fn,
                jax.random.PRNGKey(run_sett["rng_key"]),
                num_samples=int(run_sett["pde_solver"]["num_gen_samples"]),
            )

        elif run_sett["option"] == "wan_conditional":
            ##########################################################
            #### GENERATES CONDITIONAL SAMPLES as per WAN et al. - passed through the gudance_transforms argument ####
            samples = sample_wan_guided(
                diffusion_scheme,
                denoise_fn,
                y_bar=u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
                num_samples=int(run_sett["pde_solver"]["num_gen_samples"]),
            )
        elif run_sett["option"] == "conditional":
            ##########################################################
            #### GENERATES CONDITIONAL SAMPLES as per our paper ####
            # Save PDE params_list for post-training sampling
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                y_bar=u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                settings=run_sett,
                denoise_fn=denoise_fn,
                scheme=diffusion_scheme,
                rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
            )
            pde_params_dir = os.path.join(work_dir, "pde_params")
            pde_solver.load_params(pde_params_dir)
            samples = sample_pde_guided(
                diffusion_scheme,
                denoise_fn,
                pde_solver,
                rng_key=jax.random.PRNGKey(8888),
                samples_per_condition=int(run_sett["pde_solver"]["num_gen_samples"]),
            )

            print(jnp.mean(samples))
            ##########################################################
            #### CALCULATES METRICS ####
            constraint_rmse = calculate_constraint_rmse(
                samples,
                u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                C_prime,
            )
            kld = calculate_kld(samples, u_hfhr_samples, epsilon=1e-10)
            sample_variability = calculate_sample_variability(samples)
            melr_weighted = calculate_melr(
                samples,
                u_hfhr_samples,
                sample_shape=(run_sett["general"]["d"],),
                weighted=True,
                epsilon=1e-10,
            )
            melr_unweighted = calculate_melr(
                samples,
                u_hfhr_samples,
                sample_shape=(run_sett["general"]["d"],),
                weighted=False,
                epsilon=1e-10,
            )
            print("constraint_rmse:", float(constraint_rmse))
            print("kld:", float(kld))
            print("sample_variability:", float(sample_variability))
            print("melr_weighted:", float(melr_weighted))
            print("melr_unweighted:", float(melr_unweighted))
            try:
                writer.write_scalars(
                    scalars={
                        "metrics/constraint_rmse": float(constraint_rmse),
                        "metrics/kld": float(kld),
                        "metrics/sample_variability": float(sample_variability),
                        "metrics/melr_weighted": float(melr_weighted),
                        "metrics/melr_unweighted": float(melr_unweighted),
                    }
                )
            except Exception:
                pass

    # Flush/close the writer once
    try:
        writer.flush()
    except Exception:
        pass
    try:
        writer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
