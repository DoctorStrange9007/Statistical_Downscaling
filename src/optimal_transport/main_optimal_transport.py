import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.optimal_transport.alg1_OT import (
    PolicyGradient,
    NormalizingFlowModel,
    TrueDataModel,
)
import argparse
import yaml
from src.optimal_transport.utils_distance_metrics import (
    calculate_kld_OT,
    calculate_wass1_OT,
    cost_function_yz_OT,
    cost_function_yyprime_OT,
    plot_comparison,
)
import jax
from clu import metric_writers
from src.generation.wandb_adapter import WandbWriter


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/optimal_transport/settings.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)

USE_WANDB = False
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
work_dir = os.path.join(project_root, "main_optimal_transport")
os.makedirs(work_dir, exist_ok=True)
use_wandb = bool(USE_WANDB)
base_writer = metric_writers.create_default_writer(work_dir, asynchronous=False)
writer = base_writer
if use_wandb:
    project = os.environ.get("WANDB_PROJECT", "optimal-transport")
    run_name = os.environ.get("WANDB_NAME", os.path.basename(work_dir))
    entity = os.environ.get("WANDB_ENTITY")  # optional
    writer_name = f"{run_name}-OT"
    writer = WandbWriter(
        base_writer,
        project=project,
        name=writer_name,
        entity=entity,
        config={"work_dir": work_dir, **run_sett},
        active=True,
    )


def main():
    true_data_model = TrueDataModel(run_sett)
    normalizing_flow_model = NormalizingFlowModel(run_sett)
    policy_gradient = PolicyGradient(
        run_sett,
        normalizing_flow_model=normalizing_flow_model,
        true_data_model=true_data_model,
    )

    key_master = jax.random.PRNGKey(int(run_sett["seed"]))
    for _ in range(int(run_sett["num_iterations"])):
        key_step = jax.random.fold_in(key_master, _)
        metrics_key = jax.random.fold_in(key_master, 1000000 + _)
        policy_gradient.update_params(key_step)
        (kld_OT, kld_OT_prime) = calculate_kld_OT(
            policy_gradient, true_data_model, metrics_key
        )
        (wass1_OT, wass1_OT_prime) = calculate_wass1_OT(
            policy_gradient, true_data_model, metrics_key
        )
        cost_function_yz = cost_function_yz_OT(
            policy_gradient, true_data_model, metrics_key
        )
        cost_function_yyprime = cost_function_yyprime_OT(policy_gradient, metrics_key)
        if use_wandb:
            # Log all metrics at the current iteration step to create one row per loop
            writer.write_scalars(
                step=int(_),
                scalars={
                    "metrics/kld_OT": float(kld_OT),
                    "metrics/kld_OT_prime": float(kld_OT_prime),
                    "metrics/wass1_OT": float(wass1_OT),
                    "metrics/wass1_OT_prime": float(wass1_OT_prime),
                    "metrics/cost_function_yz": float(cost_function_yz),
                    "metrics/cost_function_yyprime": float(cost_function_yyprime),
                },
            )

    plot_comparison(
        n=1,
        dims=2,
        policy_gradient=policy_gradient,
        true_data_model=true_data_model,
        run_sett=run_sett,
        writer=writer,
    )
    # Flush/close the writer once
    try:
        writer.flush()
    except Exception:
        pass
    try:
        writer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
