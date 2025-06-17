import os
import numpy as np
import random
import yaml
from src.model import LinearMDP
from src.model import GaussianModel
from src.model import GaussianKernel


# Set the global random seed
seed = 42
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    with open("settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    param_model = GaussianModel()
    trans_kernel = GaussianKernel()

    # Dummy toy data: (y, y', x, x') tuples
    trajectory = GaussianKernel().trajectory

    # Create LinearMDP
    mdp = LinearMDP(param_models=param_model, trans_kernels=trans_kernel)
    # One gradient step
    mdp.sgd_step(trajectory, lr=0.01)

    # Print updated theta_0
    print("Updated theta_0:", mdp.transitions[0].theta)
