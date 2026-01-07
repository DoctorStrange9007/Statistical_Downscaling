"""Entry point for KS statistical downscaling: train or sample the model."""

import os
import sys
import jax

import jax.numpy as jnp
import numpy as np
import h5py
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
    calculate_sample_variability,
    calculate_melr_pooled,
    calculate_kld_pooled,
    calculate_wass1_pooled,
)
from src.generation.data_utils import get_raw_datasets, get_ks_dataset
from src.generation.sampler_utils import (
    sample_unconditional,
    less_memory_sample_wan_guided,
    sample_pde_guided,
    sample_wan_guided,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/generation/settings_generation.yaml"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

USE_WANDB = False
TRAIN_DENOISER = False
TRAIN_PDE = False
CONTINUE_TRAINING = False
mode = "eval"


def save_samples_h5(path, samples, *, y_bar=None, run_settings=None, rng_key=None):
    """Save only the samples to an HDF5 file as dataset 'samples'."""
    arr = np.asarray(
        samples, dtype=np.float64
    )  # All the training was performed in single precision (fp32), while the generation of the data was performed in double precision (fp64).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("samples", data=arr)


def load_samples_h5(path, *, as_jax=True):
    """Load samples from an HDF5 file and return a JAX array by default."""
    with h5py.File(path, "r") as f:
        samples_np = f["samples"][()]
    return jnp.asarray(samples_np) if as_jax else samples_np


def main():
    """Run training or sampling depending on `mode`."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    work_dir = os.path.join(project_root, "temporary", "main_generation_KS")
    os.makedirs(work_dir, exist_ok=True)

    u_HFHR, u_LFLR, u_HFLR, x, t = get_raw_datasets(
        file_name=run_sett["general"]["data_file_name"]
    )

    u_LFLR = u_LFLR[
        :, :, ::2
    ]  # they do this but don't explicitly mention it, note now we need a d_prime of 24

    u_hfhr_samples = u_HFHR.reshape(-1, int(run_sett["general"]["d"]), 1)
    u_lflr_samples = u_LFLR.reshape(-1, int(run_sett["general"]["d_prime"]), 1)
    DATA_STD = u_hfhr_samples.std()

    denoiser_model = create_denoiser_model()
    diffusion_scheme = create_diffusion_scheme(DATA_STD)

    # Use env or defaults for project/entity; allow user override
    project_name = "statistical-downscaling-main-" + mode
    os.environ.setdefault("WANDB_PROJECT", project_name)
    os.environ.setdefault("WANDB_ENTITY", "jesse-hoekstra-university-of-oxford")
    use_wandb = bool(USE_WANDB)
    base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)
    writer = base_writer
    if use_wandb:
        project = os.environ.get("WANDB_PROJECT", project_name)
        run_name = os.environ.get("WANDB_NAME", os.path.basename(work_dir))
        entity = os.environ.get("WANDB_ENTITY")  # optional
        if mode == "train":
            writer_name = run_name
        elif mode == "sample":
            writer_name = f"{run_name}-sample"
        else:
            writer_name = f"{run_name}-eval"
        writer = WandbWriter(
            base_writer,
            project=project,
            name=writer_name,
            entity=entity,
            config={"work_dir": work_dir, "mode": mode, **run_sett},
            active=True,
        )

    def log_fn(payload: dict):
        step = int(payload.get("step", 0)) if isinstance(payload, dict) else 0
        metrics = (
            {k: v for k, v in payload.items() if k != "step"}
            if isinstance(payload, dict)
            else payload
        )
        writer.write_scalars(step=step, scalars=metrics)

    if mode == "train":
        print("Running in training mode…")
        if TRAIN_DENOISER:
            model = build_model(denoiser_model, diffusion_scheme, DATA_STD)
            trainer = build_trainer(model)

            run_training(
                train_dataloader=get_ks_dataset(
                    u_hfhr_samples,
                    split="train[:75%]",
                    batch_size=run_sett["general"]["batch_size"],
                    seed=int(run_sett["rng_key"]),
                ),
                trainer=trainer,
                workdir=work_dir,
                total_train_steps=run_sett["general"]["total_train_steps"],
                metric_writer=writer,
                metric_aggregation_steps=run_sett["general"][
                    "metric_aggregation_steps"
                ],
                eval_dataloader=get_ks_dataset(
                    u_hfhr_samples,
                    split="train[75%:]",
                    batch_size=run_sett["general"]["batch_size"],
                    seed=int(run_sett["rng_key"]),
                ),
                eval_every_steps=run_sett["general"]["eval_every_steps"],
                num_batches_per_eval=run_sett["general"]["num_batches_per_eval"],
                save_interval_steps=run_sett["general"]["save_interval_steps"],
                max_to_keep=run_sett["general"]["max_to_keep"],
            )
        if TRAIN_PDE:
            denoise_fn = restore_denoise_fn(f"{work_dir}/checkpoints", denoiser_model)
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                y_bar=u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                settings=run_sett,
                denoise_fn=denoise_fn,
                scheme=diffusion_scheme,
                rng_key=jax.random.PRNGKey(int(run_sett["rng_key"])),
            )

            lambda_value = jnp.float32(run_sett["pde_solver"]["lambda"])

            pde_solver.train(
                log_fn=(lambda payload: log_fn({**payload, "lambda": lambda_value}))
            )
            pde_params_dir = os.path.join(work_dir, "pde_params")
            pde_solver.save_params(pde_params_dir)
        if CONTINUE_TRAINING:
            denoise_fn = restore_denoise_fn(f"{work_dir}/checkpoints", denoiser_model)
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                y_bar=u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                settings=run_sett,
                denoise_fn=denoise_fn,
                scheme=diffusion_scheme,
                rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
            )
            pde_params_dir = os.path.join(work_dir, "pde_params_continued")
            pde_solver.load_params(pde_params_dir)
            lambda_value = jnp.float32(run_sett["pde_solver"]["lambda"])
            pde_solver.train(
                log_fn=(lambda payload: log_fn({**payload, "lambda": lambda_value}))
            )
            pde_params_dir = os.path.join(work_dir, "pde_params_continued2")
            pde_solver.save_params(pde_params_dir)
    elif mode == "sample":
        jax.config.update(
            "jax_enable_x64", True
        )  # while the generation of the data was performed in double precision (fp64)
        print("Running in sampling-only mode…")

        denoise_fn = restore_denoise_fn(f"{work_dir}/checkpoints", denoiser_model)
        sample_file = os.path.join(work_dir, f"samples_{run_sett['option']}.h5")
        if run_sett["option"] == "unconditional":
            samples = sample_unconditional(
                diffusion_scheme,
                denoise_fn,
                jax.random.PRNGKey(run_sett["rng_key"]),
                num_samples=int(run_sett["pde_solver"]["num_gen_samples"]),
            )
            print(jnp.mean(samples))
            print(samples.std())
            save_samples_h5(sample_file, samples)
        elif run_sett["option"] == "wan_conditional":
            num_models = int(run_sett["pde_solver"]["num_models"])  # C
            samples_per_condition = int(run_sett["pde_solver"]["num_gen_samples"])  # N
            y_bars = u_lflr_samples[:num_models]

            if int(run_sett["pde_solver"]["num_models"]) % 16 != 0:
                samples = sample_wan_guided(
                    diffusion_scheme,
                    denoise_fn,
                    y_bar=y_bars,
                    rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
                    num_samples=samples_per_condition,
                )
            else:
                samples = less_memory_sample_wan_guided(
                    diffusion_scheme,
                    denoise_fn,
                    y_bar=y_bars,
                    rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
                    num_samples=samples_per_condition,
                )
            print(samples.std())
            print(samples.shape)
            save_samples_h5(sample_file, samples)
        elif run_sett["option"] == "conditional":
            pde_solver = KSStatisticalDownscalingPDESolver(
                samples=u_hfhr_samples,
                y_bar=u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
                settings=run_sett,
                denoise_fn=denoise_fn,
                scheme=diffusion_scheme,
                rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
            )
            pde_params_dir = os.path.join(work_dir, "pde_params_continued2")
            pde_solver.load_params(pde_params_dir)
            samples = sample_pde_guided(
                diffusion_scheme,
                denoise_fn,
                pde_solver,
                rng_key=jax.random.PRNGKey(run_sett["rng_key"]),
                samples_per_condition=int(run_sett["pde_solver"]["num_gen_samples"]),
            )
            print(samples.std())
            print(samples.shape)
            save_samples_h5(sample_file, samples)
    elif mode == "eval":
        jax.config.update("jax_enable_x64", True)
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
        print("Running in evaluation mode…")
        sample_file = os.path.join(work_dir, f"samples_{run_sett['option']}.h5")
        samples = load_samples_h5(sample_file, as_jax=True)

        constraint_rmse = calculate_constraint_rmse(
            samples,
            u_lflr_samples[0 : int(run_sett["pde_solver"]["num_models"])],
            C_prime,
        )
        kld = calculate_kld_pooled(
            samples, u_hfhr_samples, epsilon=float(run_sett["epsilon"])
        )
        sample_variability = calculate_sample_variability(samples)
        melr_weighted = calculate_melr_pooled(
            samples,
            u_hfhr_samples,
            sample_shape=(run_sett["general"]["d"],),
            weighted=True,
            epsilon=float(run_sett["epsilon"]),
        )

        # import numpy as np
        # import matplotlib.pyplot as plt
        # Er = np.asarray(jax.device_get(E_ref))
        ##Ep = np.asarray(jax.device_get(E_pred))
        # k = np.arange(1, Ep.shape[0] + 1, dtype=float)
        # plt.figure(figsize=(5.0, 3.6), dpi=150)
        # plt.loglog(k, Er, "k-", label="Reference", linewidth=2.0)
        # plt.loglog(k, Ep, color="tab:red", label="Predicted", linewidth=2.0)
        # plt.xlabel(r"$k$")
        # plt.ylabel(r"$E(k)$")
        # plt.grid(True, which="both", ls="--", alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        melr_unweighted = calculate_melr_pooled(
            samples,
            u_hfhr_samples,
            sample_shape=(run_sett["general"]["d"],),
            weighted=False,
            epsilon=float(run_sett["epsilon"]),
        )

        wass1 = calculate_wass1_pooled(
            samples,
            u_hfhr_samples,
            num_bins=1000,
        )

        print(
            "constraint_rmse: ",
            constraint_rmse,
            "sample_variability: ",
            sample_variability,
            "melr_unweighted: ",
            melr_unweighted,
            "melr_weighted: ",
            melr_weighted,
            "kld: ",
            kld,
            "wass1: ",
            wass1,
        )
        if use_wandb:
            writer.write_scalar("metrics/constraint_rmse", float(constraint_rmse))
            writer.write_scalar("metrics/kld", float(kld))
            writer.write_scalar("metrics/sample_variability", float(sample_variability))
            writer.write_scalar("metrics/melr_weighted", float(melr_weighted))
            writer.write_scalar("metrics/melr_unweighted", float(melr_unweighted))
            writer.write_scalar("metrics/wass1", float(wass1))

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
