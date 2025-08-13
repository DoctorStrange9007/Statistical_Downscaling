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

    all_msd = {}
    diff_before = samples @ jnp.array(run_sett["pde_solver"]["C"]).T - jnp.array(
        run_sett["pde_solver"]["y_target"]
    )
    all_msd["input_data"] = jnp.mean(jnp.square(diff_before).reshape(-1, 1))

    # Deterministic key for the unconditional generation (independent of loop order)
    x_1, samples_after = utils.sde_solver_backwards_cond(
        key_sde,
        hr_prior.trained_score,
        None,
        hr_prior.g,
        hr_prior.f,
        run_sett["general"]["d"],
        run_sett["general"]["n_samples"],
        run_sett["general"]["T"],
        hr_prior.sigma2,
        hr_prior.s,
        conditional=False,
    )

    # utils.plot_samples(samples_after, run_sett["output_dir"], "samples_after_gen_without_conditioning.png")
    all_msd["gen_without_conditioning"] = utils.calculate_msd(samples_after, run_sett)
    utils.plot_hyperplane(
        samples_after,
        all_msd["gen_without_conditioning"],
        run_sett,
        "samples_2d_after_gen_without_conditioning.png",
        None,
    )

    lambdas = list(jnp.linspace(0.1, 80, num=200))
    for lambda_ in lambdas:
        run_sett["pde_solver"]["lambda"] = float(lambda_)

        pde_solver = StatisticalDownscalingPDESolver(
            grad_log=hr_prior.trained_score,
            samples=samples,
            settings=run_sett,
            rng_key=key_pde,
        )
        pde_solver.train()

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

        # utils.plot_samples(samples_after, run_sett["output_dir"], "samples_after_" + str(lambda_) + "_.png")
        lam_key = f"{float(lambda_):.6g}"
        all_msd[lam_key] = utils.calculate_msd(samples_after, run_sett)
        utils.plot_hyperplane(
            samples_after,
            all_msd[lam_key],
            run_sett,
            "samples_2d_after_" + lam_key + "_.png",
            lambda_,
        )

    utils.output_to_excel_and_plot(all_msd, run_sett)
    a = 5
