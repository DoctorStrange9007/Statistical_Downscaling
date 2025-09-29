import os, sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax_cfd.base as cfd
import jax_cfd.spectral as spectral
import jax
from jax_cfd.base import funcutils
from data.jax_cfd_new_base.initial_conditions import (
    kuramoto_sivashinsky_initial_conditions,
)
from data.jax_cfd_new_base.equations import kuramoto_sivashinsky
import matplotlib.pyplot as plt
import jax.numpy as jnp
from data.saving_utils import save_trajectories


def plot_trajectory(sampled_trajectory, domain, total_run_time, warmup_time):
    print(f"Simulation complete. Final trajectory shape: {sampled_trajectory.shape}")

    # The total time of the *plotted* data is 4000 units.
    plotted_time = total_run_time - warmup_time

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(
        sampled_trajectory,
        extent=[domain[0], domain[1], 0, plotted_time],
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
    )
    ax.set_title("Kuramoto-Sivashinsky Spacetime Plot")
    ax.set_xlabel("Spatial Domain (x)")
    ax.set_ylabel("Time (t)")
    fig.colorbar(im, label="u(t, x)", ax=ax)
    plt.tight_layout()
    plt.show()


check_one_trajectory_spectral = False
check_one_trajectory_fvm = False
obtain_data_file = True

L = 64.0
domain = (0, L)
total_run_time = 4025.0  #
warmup_time = 25.0
sampling_interval = 12.5
output_filename = "data/ks_trajectories_512.h5"
num_trajectories = 512


def main():
    if check_one_trajectory_spectral:
        dt = 0.0025
        N = 192
        grid = cfd.grids.Grid(shape=(N,), domain=(domain,))
        total_steps = int(total_run_time / dt)
        warmup_steps = int(warmup_time / dt)
        steps_per_sample = int(sampling_interval / dt)

        step_fn = spectral.time_stepping.crank_nicolson_rk4(
            spectral.equations.KuramotoSivashinsky(grid=grid), dt
        )

        key = jax.random.PRNGKey(42)
        # initial_velocity_real = create_initial_conditions(grid, key, n_c=30)
        initial_velocity_real = kuramoto_sivashinsky_initial_conditions(
            grid, key, n_c=30
        )
        u_hat0 = jnp.fft.rfft(initial_velocity_real.data)

        trajectory_fn = cfd.funcutils.trajectory(step_fn, total_steps)
        _, full_trajectory = trajectory_fn(u_hat0)

        trajectory_after_warmup = full_trajectory[warmup_steps:]
        sampled_trajectory = trajectory_after_warmup[::steps_per_sample]

        u_trajectory_real = jax.vmap(jnp.fft.irfft)(sampled_trajectory)
        plot_trajectory(u_trajectory_real, domain, total_run_time, warmup_time)
    elif check_one_trajectory_fvm:
        dt = 0.02
        N = 48
        grid = cfd.grids.Grid(shape=(N,), domain=(domain,))
        total_steps = int(total_run_time / dt)
        warmup_steps = int(warmup_time / dt)
        steps_per_sample = int(sampling_interval / dt)
        key = jax.random.PRNGKey(42)
        initial_data = kuramoto_sivashinsky_initial_conditions(grid, key, n_c=30)
        ks_step_fn = kuramoto_sivashinsky(
            nu=-1.0,
            gamma=1.0,
            dt=dt,
            grid=grid,
        )
        trajectory_fn = jax.jit(funcutils.trajectory(ks_step_fn, steps=total_steps))

        print("Starting simulation... (JIT compilation may take a moment)")
        # Call with the correct variable `u_initial`
        final_state, full_trajectory = trajectory_fn(initial_data)

        # 6. Process and Display Results
        # Note: full_trajectory is now a PyTree (a GridVariable). We access its .data attribute.
        trajectory_after_warmup = full_trajectory.data[warmup_steps:]
        sampled_trajectory = trajectory_after_warmup[::steps_per_sample]

        print("\nSimulation complete!")
        print(f"Shape of final state data: {final_state.data.shape}")
        print(f"Shape of the full trajectory data: {full_trajectory.data.shape}")
        print(f"Shape of the sampled trajectory: {sampled_trajectory.shape}")

        plot_trajectory(sampled_trajectory, domain, total_run_time, warmup_time)
    elif obtain_data_file:
        save_trajectories(
            output_filename,
            domain,
            total_run_time,
            warmup_time,
            sampling_interval,
            num_trajectories,
        )


if __name__ == "__main__":
    main()
