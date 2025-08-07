import yaml
import os
import jax
from pre_trained_model import HR_data, HR_prior
from PDE_solver import PDE_solver
import DGM
import part1_utils as utils

if __name__ == "__main__":

    with open("src/generation/part1_settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    hr_data = HR_data(run_sett)
    samples = hr_data.get_samples()
    utils.plot_samples(samples, run_sett["output_dir"], "samples.png")

    hr_prior = HR_prior(samples, run_sett)
    hr_prior.train()
    model = DGM.DGMNet(run_sett)
    # The PDE_solver constructor will automatically wrap hr_prior.trained_score
    pde_solver = PDE_solver(model, hr_prior.trained_score, samples, run_sett)
    pde_solver.train()

    key = jax.random.PRNGKey(37)
    x_1, samples_after = utils.sde_solver_backwards_cond(
        key,
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
