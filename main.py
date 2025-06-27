import os
import numpy as np
import random
import yaml
from src.model import LinearMDP
from src.model import GaussianModel
from src.model import GaussianKernel
from src.utils import plot_theta_pairs


# Set the global random seed
seed = 42
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    with open("settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    param_model = GaussianModel(run_sett)
    trans_kernel = GaussianKernel(run_sett)

    # Create LinearMDP
    mdp = LinearMDP(run_sett, param_model=param_model, trans_kernel=trans_kernel)

    lr_rate = run_sett["lr_rate"]
    mdp.gd(lr=lr_rate, save_results=False)

    print("Creating theta convergence plots...")
    plot_theta_pairs(
        run_sett["output_dir"],
        true_theta=run_sett["models"]["LinearMDP"]["true_theta"],
        save_plot=False,
    )
