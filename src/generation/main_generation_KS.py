"""Entry point for KS statistical downscaling: train or sample the model."""

import os
import sys
import jax

import jax.numpy as jnp
from clu import metric_writers
import yaml
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.generation.wandb_adapter import WandbWriter
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

USE_WANDB = True
ALSO_TRAIN_PDE = False
mode = "train"


def main():
    """Run training or sampling depending on `mode`."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    work_dir = os.path.join(project_root, "temporary", "main_generation_KS")
    os.makedirs(work_dir, exist_ok=True)

    u_HFHR, u_LFLR, u_HFLR, x, t = get_raw_datasets(
        file_name=run_sett["general"]["data_file_name"]
    )

    u_hfhr_samples = u_HFHR.reshape(-1, int(run_sett["general"]["d"]), 1)
    u_lflr_samples = u_LFLR.reshape(-1, int(run_sett["general"]["d_prime"]), 1)
    DATA_STD = u_hfhr_samples.std()

    denoiser_model = create_denoiser_model()
    diffusion_scheme = create_diffusion_scheme(DATA_STD)

    # Use env or defaults for project/entity; allow user override
    os.environ.setdefault("WANDB_PROJECT", "statistical-downscaling-main-training")
    os.environ.setdefault("WANDB_ENTITY", "jesse-hoekstra-university-of-oxford")
    use_wandb = bool(USE_WANDB)
    base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)
    writer = base_writer
    if use_wandb:
        project = os.environ.get(
            "WANDB_PROJECT", "statistical-downscaling-main-training"
        )
        run_name = os.environ.get("WANDB_NAME", os.path.basename(work_dir))
        entity = os.environ.get("WANDB_ENTITY")  # optional
        writer_name = run_name if mode == "train" else f"{run_name}-sample"
        writer = WandbWriter(
            base_writer,
            project=project,
            name=writer_name,
            entity=entity,
            config={"work_dir": work_dir, "mode": mode, **run_sett},
            active=True,
        )

    def log_fn(payload: dict):
        try:
            writer.write_scalars(scalars=payload)
        except Exception:
            pass

    if mode == "train":
        print("Running in training mode…")
        model = build_model(denoiser_model, diffusion_scheme, DATA_STD)
        trainer = build_trainer(model)

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

        denoise_fn = restore_denoise_fn(f"{work_dir}/checkpoints", denoiser_model)
        if ALSO_TRAIN_PDE:
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                y_bar=u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
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
                log_fn=(lambda payload: log_fn({**payload, "lambda": lambda_value}))
            )
            pde_params_dir = os.path.join(work_dir, "pde_params")
            pde_solver.save_params(pde_params_dir)

    elif mode == "sample":
        print("Running in sampling-only mode…")
        downsampling_factor = int(run_sett["general"]["d"]) // int(
            run_sett["general"]["d_prime"]
        )
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
            samples = sample_unconditional(
                diffusion_scheme,
                denoise_fn,
                jax.random.PRNGKey(run_sett["rng_key"]),
                num_samples=int(run_sett["pde_solver"]["num_gen_samples"]),
            )

        elif run_sett["option"] == "wan_conditional":
            samples = sample_wan_guided(
                diffusion_scheme,
                denoise_fn,
                y_bar=u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
                num_samples=int(run_sett["pde_solver"]["num_gen_samples"]),
            )
        elif run_sett["option"] == "conditional":
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
                rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
                samples_per_condition=int(run_sett["pde_solver"]["num_gen_samples"]),
            )
            print(jnp.mean(samples))
            constraint_rmse = calculate_constraint_rmse(
                samples,
                u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                C_prime,
            )
            kld = calculate_kld(
                samples, u_hfhr_samples, epsilon=float(run_sett["epsilon"])
            )
            sample_variability = calculate_sample_variability(samples)
            melr_weighted = calculate_melr(
                samples,
                u_hfhr_samples,
                sample_shape=(run_sett["general"]["d"],),
                weighted=True,
                epsilon=float(run_sett["epsilon"]),
            )
            melr_unweighted = calculate_melr(
                samples,
                u_hfhr_samples,
                sample_shape=(run_sett["general"]["d"],),
                weighted=False,
                epsilon=float(run_sett["epsilon"]),
            )

            writer.write_scalar("metrics/constraint_rmse", float(constraint_rmse))
            writer.write_scalar("metrics/kld", float(kld))
            writer.write_scalar("metrics/sample_variability", float(sample_variability))
            writer.write_scalar("metrics/melr_weighted", float(melr_weighted))
            writer.write_scalar("metrics/melr_unweighted", float(melr_unweighted))

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
