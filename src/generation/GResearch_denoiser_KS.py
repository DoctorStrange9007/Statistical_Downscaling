from clu import metric_writers
import jax
import matplotlib.pyplot as plt
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from swirl_dynamics.lib.diffusion import InvertibleSchedule
from swirl_dynamics import templates
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
import h5py
import jax.numpy as jnp
import os
import numpy as np
from typing import Any, Dict, Iterable, Optional


# Toggle W&B logging here. You can also disable via environment: WANDB_DISABLED=true
USE_WANDB = False


class WandbWriter:
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
    ):
        import wandb  # import locally to avoid hard dependency if not used

        self._wandb = wandb
        # If API key is provided via env var, this will ensure the session is authenticated
        try:
            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                self._wandb.login(key=api_key, relogin=True)
        except Exception:
            pass
        self._run = self._wandb.init(
            project=project, entity=entity, name=name, config=config, reinit=True
        )

    def write_scalars(self, step: int, scalars: Dict[str, float]):
        # Log with original keys so dashboards show e.g. 'train_loss' directly
        payload = {k: float(v) for k, v in scalars.items()}
        self._wandb.log(payload, step=step)

    # Optional methods for compatibility with CLU writer interface
    def write_texts(self, step: int, texts: Dict[str, str]):
        payload = {f"texts/{k}": v for k, v in texts.items()}
        self._wandb.log(payload, step=step)

    def write_images(self, step: int, images: Dict[str, Any]):
        payload = {}
        for k, img in images.items():
            try:
                payload[f"images/{k}"] = self._wandb.Image(img)
            except Exception:
                continue
        if payload:
            self._wandb.log(payload, step=step)

    def write_hparams(self, hparams: Dict[str, Any]):
        # W&B stores hparams in config; update dynamically
        try:
            self._wandb.config.update(hparams, allow_val_change=True)
        except Exception:
            pass

    def write_summaries(self, step: int, *summaries: Iterable[Any]):
        # Fallback no-op; CLU sometimes calls this with generic summaries
        pass

    def flush(self):
        pass

    def close(self):
        try:
            self._wandb.finish()
        except Exception:
            pass


def main():
    DATA_STD = 1.33

    # home = os.path.expanduser("~")   # -> /nfs/home/jesseh
    # work_dir = os.path.join(project_root, "temporary", "denoiser_KS")
    # Set a writable work directory within the project tree
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )  # local machine
    work_dir = os.path.join(project_root, "temporary", "denoiser_KS")
    os.makedirs(work_dir, exist_ok=True)

    def get_raw_datasets(
        file_name="data/KS_finite_volumes_vs_pseudo_spectral.hdf5", ds_x=4
    ):

        with h5py.File(file_name, "r+") as f1:
            # Trajectories with finite volumes.
            u_lr = f1["u_fd"][()]
            # Trajectories with pseudo spectral methods.
            u_hr = f1["u_sp"][()]
            # Time stamps for the trajectories.
            t = f1["t"][()]
            # Grid in which the trajectories are computed. 512 equispaced points with
            # periodic boundary conditions.
            x = f1["x"][()]

        t_ = t
        x_ = jnp.concatenate([x, jnp.array(x[-1] + x[1] - x[0]).reshape((-1,))])[::ds_x]

        u_lr_hf = u_hr[:, :, ::ds_x]
        x_lr_hf = x_[::ds_x]
        u_lr_lf = u_lr

        return u_hr, u_lr_hf, u_lr_lf, x, x_, t, t_

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

    u_hr, u_lr_hf, u_lr_lf, x, x_, t, t_ = get_raw_datasets()

    u_hr_subsampled = u_hr[:, ::12, :]  # sample every 12.5
    u_hr_samples = u_hr_subsampled.reshape(-1, 192, 1)

    # dataset = get_ks_dataset(u_hr, split="train[:75%]", batch_size=64)

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

    def vp_linear_beta_schedule(beta_min: float = 0.1, beta_max: float = 20.0):
        bdiff = beta_max - beta_min
        # forward: σ(t)
        forward = lambda t: jnp.sqrt(jnp.expm1(0.5 * bdiff * t * t + beta_min * t))

        # inverse: t(σ); solve 0.5*bdiff*t^2 + beta_min*t = log(1+σ^2)
        def inverse(sig):
            L = jnp.log1p(jnp.square(sig))
            return (-beta_min + jnp.sqrt(beta_min**2 + 2.0 * bdiff * L)) / bdiff

        return InvertibleSchedule(forward, inverse)

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
            clip_min=10e-3,
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
                    peak_value=10e-4,  # 10e-3, but with this data I get NaN values otherwise
                    warmup_steps=1_000,
                    decay_steps=990_000,
                    end_value=10e-6,
                ),
            ),
        ),
        # We keep track of an exponential moving average of the model parameters
        # over training steps. This alleviates the "color-shift" problems known to
        # exist in the diffusion models.
        ema_decay=0.95,
    )

    # Choose metric writer: W&B if env vars are set, else CLU default writer
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
            u_hr_samples, split="train[:75%]", batch_size=256  # 512
        ),
        trainer=trainer,
        workdir=work_dir,
        total_train_steps=1_000_000,
        metric_writer=writer,
        metric_aggregation_steps=1000,
        eval_dataloader=get_ks_dataset(
            u_hr_samples, split="train[75%:]", batch_size=256  # 512
        ),
        eval_every_steps=1000,
        num_batches_per_eval=2,
        callbacks=(
            # This callback displays the training progress in a tqdm bar
            templates.TqdmProgressBar(
                total_train_steps=1_000_000,
                train_monitors=("train_loss",),
            ),
            # This callback saves model checkpoint periodically
            templates.TrainStateCheckpoint(
                base_dir=work_dir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=1000, max_to_keep=5
                ),
            ),
        ),
    )

    trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
        f"{work_dir}/checkpoints", step=None
    )
    # Construct the inference function
    denoise_fn = dfn.DenoisingTrainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=denoiser_model
    )

    sampler = dfn_lib.SdeSampler(
        input_shape=(192, 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=256,
            end_sigma=10e-2,  # 256 from grid_search section and end_sigma=sigma(clip_min) chosen
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False,  # Set to `True` if the full sampling paths are needed
    )

    generate = jax.jit(sampler.generate, static_argnames=("num_samples",))

    samples = generate(rng=jax.random.PRNGKey(8888), num_samples=4)

    # More informative visualizations
    samples_np = np.squeeze(np.array(samples))  # (N, 192)
    real_np = np.squeeze(np.array(u_hr_samples))  # (M, 192)

    # 1) Overlay a few generated samples as line plots
    num_to_show = min(8, samples_np.shape[0])
    x_idx = np.arange(samples_np.shape[1])

    # 2) Mean and uncertainty band of generated ensemble
    gen_mean = samples_np.mean(axis=0)
    gen_std = samples_np.std(axis=0)

    # 3) Value distribution (marginal over space) vs training data
    real_flat = real_np.reshape(-1)
    gen_flat = samples_np.reshape(-1)

    # 4) Average power spectrum comparison (generated vs real)
    def average_power_spectrum(arr_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr_centered = arr_2d - arr_2d.mean(axis=1, keepdims=True)
        fft_vals = np.fft.rfft(arr_centered, axis=1)
        power = (np.abs(fft_vals) ** 2).mean(axis=0)
        k = np.arange(power.shape[0])  # wavenumber index
        return k, power

    k_gen, p_gen = average_power_spectrum(samples_np)
    # Use at most 2048 real samples for speed
    real_subset = real_np[: min(2048, real_np.shape[0])]
    k_real, p_real = average_power_spectrum(real_subset)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    # Top-left: overlay of generated curves
    ax = axes[0, 0]
    for i in range(num_to_show):
        ax.plot(x_idx, samples_np[i], alpha=0.7)
    ax.set_title("Generated samples (overlay)")
    ax.set_xlabel("Space index")
    ax.set_ylabel("u(x)")

    # Top-right: mean and ±2σ band
    ax = axes[0, 1]
    ax.plot(x_idx, gen_mean, color="C0", label="Gen mean")
    ax.fill_between(
        x_idx,
        gen_mean - 2 * gen_std,
        gen_mean + 2 * gen_std,
        color="C0",
        alpha=0.2,
        label="Gen ±2σ",
    )
    ax.set_title("Generated ensemble mean ± 2σ")
    ax.set_xlabel("Space index")
    ax.set_ylabel("u(x)")
    ax.legend()

    # Bottom-left: value distribution
    ax = axes[1, 0]
    bins = 50
    ax.hist(real_flat, bins=bins, alpha=0.5, density=True, label="Train")
    ax.hist(gen_flat, bins=bins, alpha=0.5, density=True, label="Generated")
    ax.set_title("Value distribution across space")
    ax.set_xlabel("u")
    ax.set_ylabel("Density")
    ax.legend()

    # Bottom-right: power spectrum comparison (log-log)
    ax = axes[1, 1]
    ax.loglog(k_real[1:], p_real[1:] + 1e-12, label="Train")
    ax.loglog(k_gen[1:], p_gen[1:] + 1e-12, label="Generated")
    ax.set_title("Average power spectrum")
    ax.set_xlabel("Wavenumber index k")
    ax.set_ylabel("Power")
    ax.legend()

    plt.tight_layout()
    # Log summary figure to W&B if enabled
    if use_wandb:
        try:
            writer.write_images(step=0, images={"summary": fig})
        except Exception:
            pass
    plt.show()


if __name__ == "__main__":
    main()
