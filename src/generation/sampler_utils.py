import jax
import jax.numpy as jnp
from swirl_dynamics.lib import diffusion as dfn_lib
from swirl_dynamics.lib import solvers as solver_lib

from src.generation.swirl_dynamics_new_guidance.guidance import LinearConstraint
from src.generation.swirl_dynamics_new_sampler.samplers import NewDriftSdeSampler
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="src/generation/settings_generation.yaml"
)
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)


def sample_unconditional(
    diffusion_scheme, denoise_fn, rng_key: jax.Array, num_samples: int
):
    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["general"]["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=run_sett["exp_tspan"]["num_steps"],
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        apply_denoise_at_end=True,
        return_full_paths=False,
    )
    generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
    return generate(rng=rng_key, num_samples=num_samples)


def sample_wan_guided(
    diffusion_scheme,
    denoise_fn,
    y_bar: jnp.ndarray,
    rng_key: jax.Array,
    num_samples: int,
):
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
    guidance_transform = LinearConstraint.create(
        C_prime=C_prime,
        y_bar=y_bar,
        norm_guide_strength=run_sett["general"]["norm_guide_strength"],
    )
    sampler = dfn_lib.SdeSampler(
        input_shape=(run_sett["general"]["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(run_sett["exp_tspan"]["num_steps"]),
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(guidance_transform,),
        apply_denoise_at_end=True,
        return_full_paths=False,
    )
    generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
    return generate(rng=rng_key, num_samples=num_samples)


def sample_pde_guided(
    diffusion_scheme,
    denoise_fn,
    pde_solver,
    rng_key: jax.Array,
    samples_per_condition: int,
):
    sampler = NewDriftSdeSampler(
        input_shape=(run_sett["general"]["d"], 1),
        integrator=solver_lib.EulerMaruyama(),
        tspan=dfn_lib.exponential_noise_decay(
            diffusion_scheme,
            num_steps=int(run_sett["exp_tspan"]["num_steps"]),
            end_sigma=float(run_sett["exp_tspan"]["end_sigma"]),
        ),
        scheme=diffusion_scheme,
        denoise_fn=denoise_fn,
        guidance_transforms=(),
        guidance_fn=pde_solver.grad_log_h_batched_one_per_model,
        apply_denoise_at_end=True,
        return_full_paths=False,
    )

    keys = jax.random.split(rng_key, samples_per_condition)
    generate = jax.jit(sampler.generate, static_argnames=("num_samples",))
    batched_generate = jax.jit(
        jax.vmap(lambda k: generate(rng=k, num_samples=pde_solver.num_models))
    )
    return batched_generate(keys)
