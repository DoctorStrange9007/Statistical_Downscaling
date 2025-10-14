import jax.numpy as jnp
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.projects import probabilistic_diffusion as dfn
import optax
import jax
from swirl_dynamics import templates
import orbax.checkpoint as ocp
import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/generation/settings_generation.yaml"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)


def _as_tuple(value):
    """Accept tuple/list or comma-separated string and return a tuple of ints."""
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return tuple(int(p) for p in parts)
    # fallback for single scalar
    return (int(value),)


def vp_linear_beta_schedule(beta_min: float = None, beta_max: float = None):
    beta_min = float(
        beta_min if beta_min is not None else run_sett["general"]["beta_min"]
    )
    beta_max = float(
        beta_max if beta_max is not None else run_sett["general"]["beta_max"]
    )
    bdiff = beta_max - beta_min
    forward = lambda t: jnp.sqrt(jnp.expm1(0.5 * bdiff * t * t + beta_min * t))

    def inverse(sig):
        L = jnp.log1p(jnp.square(sig))
        return (-beta_min + jnp.sqrt(beta_min**2 + 2.0 * bdiff * L)) / bdiff

    return dfn_lib.InvertibleSchedule(forward, inverse)


def create_denoiser_model():
    return dfn_lib.PreconditionedDenoiserUNet(
        out_channels=int(run_sett["UNET"]["out_channels"]),
        num_channels=_as_tuple(run_sett["UNET"]["num_channels"]),
        downsample_ratio=_as_tuple(run_sett["UNET"]["downsample_ratio"]),
        num_blocks=int(run_sett["UNET"]["num_blocks"]),
        noise_embed_dim=int(run_sett["UNET"]["noise_embed_dim"]),
        use_attention=bool(run_sett["UNET"]["use_attention"]),
        num_heads=int(run_sett["UNET"]["num_heads"]),
        use_position_encoding=bool(run_sett["UNET"]["use_position_encoding"]),
        dropout_rate=float(run_sett["UNET"]["dropout_rate"]),
    )


def create_diffusion_scheme(data_std: float):
    return dfn_lib.Diffusion.create_variance_preserving(
        sigma=vp_linear_beta_schedule(),
        data_std=data_std,
    )


def restore_denoise_fn(checkpoint_dir: str, denoiser_model):
    trained_state = dfn.DenoisingModelTrainState.restore_from_orbax_ckpt(
        checkpoint_dir, step=None
    )
    return dfn.DenoisingTrainer.inference_fn_from_state_dict(
        trained_state, use_ema=True, denoiser=denoiser_model
    )


def build_model(denoiser_model, diffusion_scheme, data_std: float):
    return dfn.DenoisingModel(
        input_shape=(int(run_sett["general"]["d"]), 1),
        denoiser=denoiser_model,
        noise_sampling=dfn_lib.time_uniform_sampling(
            diffusion_scheme,
            clip_min=float(run_sett["general"]["clip_min"]),
            uniform_grid=True,
        ),
        noise_weighting=dfn_lib.edm_weighting(data_std=data_std),
    )


def build_trainer(model):
    return dfn.DenoisingTrainer(
        model=model,
        rng=jax.random.PRNGKey(int(run_sett["rng_key"])),
        optimizer=optax.chain(
            optax.clip_by_global_norm(float(run_sett["optimizer"]["clip_norm"])),
            optax.adam(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=float(run_sett["optimizer"]["init_value"]),
                    peak_value=float(run_sett["optimizer"]["peak_value"]),
                    warmup_steps=int(run_sett["optimizer"]["warmup_steps"]),
                    decay_steps=int(run_sett["optimizer"]["decay_steps"]),
                    end_value=float(run_sett["optimizer"]["end_value"]),
                ),
            ),
        ),
        ema_decay=float(run_sett["general"]["ema_decay"]),
    )


def run_training(
    *,
    train_dataloader,
    trainer,
    workdir: str,
    total_train_steps: int,
    metric_writer,
    metric_aggregation_steps: int,
    eval_dataloader,
    eval_every_steps: int,
    num_batches_per_eval: int,
    save_interval_steps: int = run_sett["general"]["save_interval_steps"],
    max_to_keep: int = run_sett["general"]["max_to_keep"],
):
    """Wrapper around templates.run_train with standard callbacks/checkpointing."""
    return templates.run_train(
        train_dataloader=train_dataloader,
        trainer=trainer,
        workdir=workdir,
        total_train_steps=total_train_steps,
        metric_writer=metric_writer,
        metric_aggregation_steps=metric_aggregation_steps,
        eval_dataloader=eval_dataloader,
        eval_every_steps=eval_every_steps,
        num_batches_per_eval=num_batches_per_eval,
        callbacks=(
            templates.TqdmProgressBar(
                total_train_steps=total_train_steps, train_monitors=("train_loss",)
            ),
            templates.TrainStateCheckpoint(
                base_dir=workdir,
                options=ocp.CheckpointManagerOptions(
                    save_interval_steps=save_interval_steps, max_to_keep=max_to_keep
                ),
            ),
        ),
    )
