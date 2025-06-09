import os
import numpy as np
import random
import yaml
from src.model import LinearMDP

# Set the global random seed
seed = 42
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    with open("settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    LinearMDP_obj = LinearMDP(run_sett)
    print(LinearMDP_obj.result)
