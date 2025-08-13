import yaml
import os
import jax
import jax.numpy as jnp
from pre_trained_model import HR_data, HR_prior
from Statistical_Downscaling_PDE import StatisticalDownscalingPDESolver
import part1_utils as utils

if __name__ == "__main__":

    with open("src/generation/part1_settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    # Master RNG: use general.seed if present; else fixed
    seed = run_sett.get("general", {}).get("seed", 37)
    master_key = jax.random.PRNGKey(seed)
    key_data, key_prior, key_pde, key_sde = jax.random.split(master_key, 4)

    hr_data = HR_data(run_sett, rng_key=key_data)
    samples = hr_data.get_samples()
    utils.plot_samples(samples, run_sett["output_dir"], "samples.png")

    hr_prior = HR_prior(samples, run_sett, rng_key=key_prior)
    hr_prior.train()

    # Use subclassed statistical downscaling PDE
    pde_solver = StatisticalDownscalingPDESolver(
        grad_log=hr_prior.trained_score,
        samples=samples,
        settings=run_sett,
        rng_key=key_pde,
    )
    pde_solver.train()

    # Inspect grad_log_h at terminal time T for the original samples,
    # and compare Cx to the solver's stored y
    t_T = jnp.ones((samples.shape[0], 1)) * run_sett["general"]["T"]

    # Also evaluate the network output V(samples, T)
    V_T = pde_solver.V(pde_solver.params, t_T, samples)
    diff = samples @ jnp.array(run_sett["pde_solver"]["C"]).T - jnp.array(
        run_sett["pde_solver"]["y_target"], dtype=jnp.float32
    )
    mae_to_target = jnp.mean(
        jnp.square(V_T - jnp.exp(-jnp.linalg.norm(diff, axis=1, ord=2)).reshape(-1, 1))
    )
    print(
        "Mean |V-h(T,x)|:", float(mae_to_target)
    )  # 1learns the correct h(T,x) values it seems

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
    utils.print_distances(samples, samples_after, run_sett)
    a = 5
