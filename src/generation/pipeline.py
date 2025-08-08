import yaml
import os
import jax
from pre_trained_model import HR_data, HR_prior
from PDE_solver import PDE_solver
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
    # Use JAX-based DGMNetJax internally via PDE_solver
    pde_solver = PDE_solver(hr_prior.trained_score, samples, run_sett, rng_key=key_pde)
    pde_solver.train()

    x_1, samples_after = utils.sde_solver_backwards_cond(
        key_sde,
        hr_prior.trained_score,
        pde_solver.grad_log_h,
        hr_prior.g,
        hr_prior.f,
        run_sett["general"]["d"],
        run_sett["general"]["n_samples"],
        hr_prior.sigma2,
    )

    utils.plot_samples(samples_after, run_sett["output_dir"], "samples_after.png")
    utils.get_summary(samples, samples_after, run_sett)
    a = 5
