import os
import numpy as np
import random
import yaml
import argparse
from datetime import datetime
from src.optimal_transport.model_OT import LinearMDP
from src.optimal_transport.model_OT import GaussianModel
from src.optimal_transport.model_OT import GaussianKernel
from src.optimal_transport.utils_OT import plot_theta_pairs
import pandas as pd

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# Set the global random seed
seed = 42
np.random.seed(seed)
random.seed(seed)

if __name__ == "__main__":
    os.environ["WANDB"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="settings.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--project", type=str, default=None, help="Override project name"
    )
    parser.add_argument("--entity", type=str, default=None, help="Override entity")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])

    # Optional W&B
    env_enable = os.environ.get("WANDB", "").lower() in {"1", "true", "yes", "on"}
    project = args.project or run_sett.get("wandb", {}).get(
        "project", "statistical-downscaling-ot"
    )
    entity = args.entity or run_sett.get("wandb", {}).get("entity", None)
    run_name = args.run_name or run_sett.get("wandb", {}).get(
        "run_name", "ot_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    use_wandb = bool((args.wandb or env_enable) and (wandb is not None))
    run = None
    if (args.wandb or env_enable) and wandb is None:
        print("wandb not installed; proceeding without logging.")
    if use_wandb:
        run = wandb.init(project=project, entity=entity, name=run_name, config=run_sett)
        try:
            print(
                f"W&B logging enabled: project={project}, entity={entity}, run={run.name}"
            )
            if hasattr(run, "url"):
                print(f"View run: {run.url}")
        except Exception:
            pass

    def log_fn(payload: dict):
        if use_wandb:
            wandb.log(payload)

    param_model = GaussianModel(run_sett)
    trans_kernel = GaussianKernel(run_sett)

    # Create LinearMDP
    mdp = LinearMDP(run_sett, param_model=param_model, trans_kernel=trans_kernel)

    l_rates = run_sett["l_rates"]
    mdp.gd(l_rates=l_rates, save_results=True, log_fn=log_fn if use_wandb else None)

    print("Creating theta convergence plots...")
    plot_theta_pairs(
        run_sett["output_dir"],
        true_theta=run_sett["models"]["LinearMDP"]["true_theta"],
        save_plot=True,
    )
    if use_wandb:
        plot_path = os.path.join(run_sett["output_dir"], "theta_pairs_convergence.png")
        if os.path.exists(plot_path):
            try:
                wandb.log({"plots/theta_pairs": wandb.Image(plot_path)})
            except Exception:
                pass
    if use_wandb:
        try:
            # Log the full MSD dictionary as a table for convenience
            df = pd.DataFrame(mdp.theta_history).sort_values("gd_step")
            wandb.log({"summary/theta_history": wandb.Table(dataframe=df)})
        except Exception:
            pass

    if run is not None:
        run.finish()
