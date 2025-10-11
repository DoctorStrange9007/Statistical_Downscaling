import os
import sys
import h5py
import jax
import wandb
import jax.numpy as jnp
import orbax.checkpoint as ocp
from clu import metric_writers
import optax
from swirl_dynamics.lib import diffusion as dfn_lib
import tensorflow as tf
from swirl_dynamics.lib.diffusion import InvertibleSchedule
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics import templates
from swirl_dynamics.projects import probabilistic_diffusion as dfn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.generation.Statistical_Downscaling_PDE_KS import (
    KSStatisticalDownscalingPDESolver,
)
from src.generation.swirl_dynamics_new_guidance.guidance import LinearConstraint
from src.generation.swirl_dynamics_new_sampler.samplers import NewDriftSdeSampler

from src.generation.utils_metrics import calculate_constraint_rmse
from src.generation.utils_metrics import calculate_kld
from src.generation.utils_metrics import calculate_sample_variability
from src.generation.utils_metrics import calculate_melr

import yaml
import argparse

USE_WANDB = False

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/generation/settings_generation.yaml"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)


class WandbWriter:
    pass


def log_fn(payload: dict):
    """Safe metric logger that no-ops if logging is disabled or fails."""
    if USE_WANDB:
        try:
            wandb.log(payload)
        except Exception:
            pass


def get_raw_datasets(file_name="data/ks_trajectories_512.h5", ds_x=4):

    with h5py.File(file_name, "r+") as f1:
        # Trajectories with finite volumes.
        u_LFLR = f1["LFLR"][()]

        # Trajectories with pseudo spectral methods.
        u_HFHR = f1["HFHR"][()]

        t = f1["t"][()]
        x = f1["x"][()]

    u_HFLR = u_HFHR[:, :, ::ds_x]

    return u_HFHR, u_LFLR, u_HFLR, x, t


def get_ks_dataset(u_samples: jnp.ndarray, split: str, batch_size: int):
    """Returns a batched dataset from u_samples with the same interface as get_mnist_dataset.

    Args:
        u_samples: Array of shape (N, 192, 1)
        split: A TFDS-style split string (e.g., 'train[:75%]')
        batch_size: Batch size for training

    Returns:
        A NumPy iterator over batches of {'x': ...}
    """
    # Create tf.data.Dataset from memory
    ds = tf.data.Dataset.from_tensor_slices({"x": u_samples.astype(jnp.float32)})

    # Use take/skip to support basic TFDS-style splits
    total_len = len(u_samples)
    if split == "train":
        pass  # use full dataset
    elif split.startswith("train[:"):
        frac = float(split[len("train[:") : -2]) / 100
        ds = ds.take(int(frac * total_len))
    elif split.startswith("train["):
        frac = float(split[len("train[") : -3]) / 100
        ds = ds.skip(int(frac * total_len))
    else:
        raise ValueError(f"Unsupported split string: {split}")

    # Format as required
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    return ds


def vp_linear_beta_schedule(beta_min: float = 0.1, beta_max: float = 20.0):
    bdiff = beta_max - beta_min
    # forward: σ(t)
    forward = lambda t: jnp.sqrt(jnp.expm1(0.5 * bdiff * t * t + beta_min * t))

    # inverse: t(σ); solve 0.5*bdiff*t^2 + beta_min*t = log(1+σ^2)
    def inverse(sig):
        L = jnp.log1p(jnp.square(sig))
        return (-beta_min + jnp.sqrt(beta_min**2 + 2.0 * bdiff * L)) / bdiff

    return InvertibleSchedule(forward, inverse)


def main():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )  # local machine
    work_dir = os.path.join(
        project_root, "temporary", "main_generation_KS"
    )  # FOR A FRESH RUN empty temporary/
    os.makedirs(work_dir, exist_ok=True)

    u_HFHR, u_LFLR, u_HFLR, x, t = get_raw_datasets()

    u_hfhr_samples = u_HFHR.reshape(-1, 192, 1)
    u_lflr_samples = u_LFLR.reshape(-1, 48, 1)
    # DATA_STD = 1.33
    DATA_STD = (
        u_hfhr_samples.std()
    )  # temporary, draw this from the data directly. Technically this is leaking information from the future.

    denoiser_model = dfn_lib.PreconditionedDenoiserUNet(
        out_channels=1,
        num_channels=(32, 64, 128),
        downsample_ratio=(2, 2, 2),  # 192→96→48→24
        num_blocks=6,  # 6 ResNet blocks per resolution
        noise_embed_dim=128,  # Fourier embedding dim
        use_attention=True,  # attention at lowest res
        num_heads=8,
        use_position_encoding=False,  # if not desired
        dropout_rate=0.0,
    )

    diffusion_scheme = dfn_lib.Diffusion.create_variance_preserving(
        sigma=vp_linear_beta_schedule(),
        data_std=DATA_STD,
    )

    model = dfn.DenoisingModel(
        # `input_shape` must agree with the expected sample shape (without the batch
        # dimension), which in this case is simply the dimensions of a single MNIST
        # sample.
        input_shape=(192, 1),
        denoiser=denoiser_model,
        noise_sampling=dfn_lib.time_uniform_sampling(
            diffusion_scheme,
            clip_min=1e-3,
            uniform_grid=True,
        ),
        noise_weighting=dfn_lib.edm_weighting(data_std=DATA_STD),
    )

    trainer = dfn.DenoisingTrainer(
        model=model,
        rng=jax.random.PRNGKey(888),
        optimizer=optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=0.0,
                    peak_value=1e-3,  # actually 1e-3, but with this data I get NaN values
                    warmup_steps=1_000,
                    decay_steps=990_000,
                    end_value=1e-6,
                ),
            ),
        ),
        # We keep track of an exponential moving average of the model parameters
        # over training steps. This alleviates the "color-shift" problems known to
        # exist in the diffusion models.
        ema_decay=0.95,
    )

    use_wandb = USE_WANDB and not os.environ.get("WANDB_DISABLED")
    if use_wandb:
        project = os.environ.get("WANDB_PROJECT", "statistical-downscaling-denoiser-KS")
        run_name = os.environ.get("WANDB_NAME", os.path.basename(work_dir))
        entity = os.environ.get("WANDB_ENTITY")  # optional
        writer = WandbWriter(
            project=project, name=run_name, entity=entity, config={"work_dir": work_dir}
        )
    else:
        writer = metric_writers.create_default_writer(work_dir, asynchronous=False)

    templates.run_train(
        train_dataloader=get_ks_dataset(
            u_hfhr_samples, split="train[:75%]", batch_size=32  # 512
        ),
        trainer=trainer,
        workdir=work_dir,
        total_train_steps=50,  # 1_000_000,
        metric_writer=writer,
        metric_aggregation_steps=10,  # 1000,
        eval_dataloader=get_ks_dataset(
            u_hfhr_samples, split="train[75%:]", batch_size=32  # 512
        ),
        eval_every_steps=10,  # 1000,
        num_batches_per_eval=1,  # 2,
        callbacks=(
            # This callback displays the training progress in a tqdm bar
            templates.TqdmProgressBar(
                total_train_steps=50,  # 1_000_000,
                train_monitors=("train_loss",),
            ),
            # This callback saves model checkpoint periodically
            templates.TrainStateCheckpoint(
                base_dir=work_dir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=10, max_to_keep=1  # 1000, 5
                ),
            ),
        ),
    )

    trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
        f"{work_dir}/checkpoints", step=None
    )

    denoise_fn = dfn.DenoisingTrainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=denoiser_model
    )

    option = 3
    if option == 1:
        ##########################################################
        #### GENERATES UNCONDITIONAL SAMPLES ####

        num_samples = 4
        sampler = dfn_lib.SdeSampler(
            input_shape=(192, 1),
            integrator=solver_lib.EulerMaruyama(),
            tspan=dfn_lib.exponential_noise_decay(
                diffusion_scheme,
                num_steps=256,  # table 6
                end_sigma=1e-2,  # not defined in paper, end_sigma=sigma(clip_min) chosen
            ),
            scheme=diffusion_scheme,
            denoise_fn=denoise_fn,
            guidance_transforms=(),
            apply_denoise_at_end=True,
            return_full_paths=False,  # Set to `True` if the full sampling paths are needed
        )

        generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
        samples = generate(rng=jax.random.PRNGKey(8888), num_samples=num_samples)

    elif option == 2:
        ##########################################################
        #### GENERATES CONDITIONAL SAMPLES as per WAN et al. - passed through the gudance_transforms argument ####
        num_samples = 4
        C_prime = jnp.array(
            [[1 if j == 4 * i else 0 for j in range(192)] for i in range(48)]
        )
        guidance_transform = LinearConstraint.create(
            C_prime=C_prime,
            y_bar=u_lflr_samples[0:num_samples],  # can do this for any sample
            norm_guide_strength=1.0,
        )

        sampler = dfn_lib.SdeSampler(
            input_shape=(192, 1),
            integrator=solver_lib.EulerMaruyama(),
            tspan=dfn_lib.exponential_noise_decay(
                diffusion_scheme,
                num_steps=256,  # table 6
                end_sigma=1e-2,  # not defined in paper, end_sigma=sigma(clip_min) chosen
            ),
            scheme=diffusion_scheme,
            denoise_fn=denoise_fn,
            guidance_transforms=(guidance_transform,),
            apply_denoise_at_end=True,
            return_full_paths=False,  # Set to `True` if the full sampling paths are needed
        )

        generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
        samples = generate(rng=jax.random.PRNGKey(8888), num_samples=num_samples)

    elif option == 3:
        ##########################################################
        #### GENERATES CONDITIONAL SAMPLES as per our paper ####
        C_prime = jnp.array(
            [[1 if j == 4 * i else 0 for j in range(192)] for i in range(48)]
        )
        num_models = run_sett["pde_solver"]["num_models"]  # aka num_conditions 512
        pde_solver = KSStatisticalDownscalingPDESolver(
            samples=u_hfhr_samples,
            y_bar=u_lflr_samples[
                0:num_models
            ],  # can do this for any sample, maybe code this down better such that we can extend this to multiple samples at once
            settings=run_sett,
            denoise_fn=denoise_fn,
            scheme=diffusion_scheme,
            rng_key=jax.random.PRNGKey(888),
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

        sampler = NewDriftSdeSampler(
            input_shape=(192, 1),
            integrator=solver_lib.EulerMaruyama(),
            tspan=dfn_lib.exponential_noise_decay(
                diffusion_scheme,
                num_steps=256,  # table 6
                end_sigma=1e-2,  # not defined in paper, end_sigma=sigma(clip_min) chosen
            ),
            scheme=diffusion_scheme,
            denoise_fn=denoise_fn,
            guidance_transforms=(),
            guidance_fn=pde_solver.grad_log_h_batched_one_per_model,
            apply_denoise_at_end=True,
            return_full_paths=False,  # Set to `True` if the full sampling paths are needed
        )

        base = jax.random.PRNGKey(8888)
        samples_per_condition = 10  # 128
        keys = jax.random.split(base, samples_per_condition)

        generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
        batched_generate = jax.jit(
            jax.vmap(lambda k: generate(rng=k, num_samples=pde_solver.num_models))
        )
        samples = batched_generate(keys)

        ##########################################################
        #### CALCULATES METRICS ####
        constraint_rmse = calculate_constraint_rmse(
            samples, u_lflr_samples[0:num_models], C_prime
        )
        kld = calculate_kld(samples, u_hfhr_samples, epsilon=1e-10)
        sample_variability = calculate_sample_variability(samples)
        melr_weighted = calculate_melr(
            samples, u_hfhr_samples, sample_shape=(192,), weighted=True, epsilon=1e-10
        )
        melr_unweighted = calculate_melr(
            samples, u_hfhr_samples, sample_shape=(192,), weighted=False, epsilon=1e-10
        )

        a = 5


if __name__ == "__main__":
    main()
