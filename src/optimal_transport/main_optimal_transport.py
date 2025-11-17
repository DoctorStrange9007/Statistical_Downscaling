import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.optimal_transport.alg1_OT import (
    PolicyGradient,
    NormalizedFlowModel,
    TrueDataModel,
)
import argparse
import yaml
from src.optimal_transport.utils_distance_metrics import (
    calculate_kld_OT,
    calculate_wass1_OT,
)
import jax


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="src/optimal_transport/settings.yaml")
args = parser.parse_args()
with open(args.config, "r") as f:
    run_sett = yaml.safe_load(f)


def main():
    true_data_model = TrueDataModel(run_sett)
    normalized_flow_model = NormalizedFlowModel(run_sett)
    policy_gradient = PolicyGradient(
        run_sett,
        normalized_flow_model=normalized_flow_model,
        true_data_model=true_data_model,
    )

    key_master = jax.random.PRNGKey(int(run_sett["seed"]))
    for _ in range(int(run_sett["num_iterations"])):
        key_step = jax.random.fold_in(key_master, _)
        policy_gradient.update_params(key_step)
        print(f"Iteration {_} completed")

    print(policy_gradient.normalized_flow_model.param_models)
    print("Calculating KLD OT...")
    (kld_OT, kld_OT_prime) = calculate_kld_OT(policy_gradient, true_data_model)
    print("kld_OT: ", kld_OT)
    print("kld_OT_prime: ", kld_OT_prime)
    print("Calculating Wass1 OT...")
    (wass1_OT, wass1_OT_prime) = calculate_wass1_OT(policy_gradient, true_data_model)
    print("wass1_OT: ", wass1_OT)
    print("wass1_OT_prime: ", wass1_OT_prime)


if __name__ == "__main__":
    main()
