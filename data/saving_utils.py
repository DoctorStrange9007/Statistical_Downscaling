import jax
import jax.numpy as jnp
from jax_cfd.base import funcutils
from jax_cfd.base import grids
import jax_cfd.spectral as spectral
import h5py
import numpy as np
from tqdm import tqdm  # For a progress bar
from data.jax_cfd_new_base.initial_conditions import (
    kuramoto_sivashinsky_initial_conditions,
)
from data.jax_cfd_new_base.equations import kuramoto_sivashinsky


def save_trajectories(
    output_filename,
    domain,
    total_run_time,
    warmup_time,
    sampling_interval,
    num_trajectories,
):
    # --- Generate High-Fidelity High-Resolution (HFHR) Spectral Data ---
    print("Generating HFHR Spectral data...")
    dt_hfhr = 0.0025
    N_hfhr = 192
    grid_hfhr = grids.Grid(shape=(N_hfhr,), domain=(domain,))

    total_steps_hfhr = int(total_run_time / dt_hfhr)
    warmup_steps_hfhr = int(warmup_time / dt_hfhr)
    steps_per_sample_hfhr = int(sampling_interval / dt_hfhr)

    step_fn_hfhr = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.KuramotoSivashinsky(grid=grid_hfhr), dt_hfhr
    )

    trajectory_fn_hfhr = jax.jit(funcutils.trajectory(step_fn_hfhr, total_steps_hfhr))

    all_trajectories_hfhr = []
    for i in tqdm(range(num_trajectories), desc="HFHR Trajectories"):
        key = jax.random.PRNGKey(i)
        initial_cond = kuramoto_sivashinsky_initial_conditions(grid_hfhr, key, n_c=30)
        u_hat0 = jnp.fft.rfft(initial_cond.data)

        _, full_trajectory_hat = trajectory_fn_hfhr(u_hat0)

        trajectory_after_warmup = full_trajectory_hat[warmup_steps_hfhr:]
        sampled_trajectory_hat = trajectory_after_warmup[::steps_per_sample_hfhr]

        u_trajectory_real = jax.vmap(jnp.fft.irfft)(sampled_trajectory_hat)
        all_trajectories_hfhr.append(u_trajectory_real)

    # --- Generate Low-Fidelity Low-Resolution (LFLR) FVM Data ---
    print("\nGenerating LFLR FVM data...")
    dt_lflr = 0.02
    N_lflr = 48
    grid_lflr = grids.Grid(shape=(N_lflr,), domain=(domain,))

    total_steps_lflr = int(total_run_time / dt_lflr)
    warmup_steps_lflr = int(warmup_time / dt_lflr)
    steps_per_sample_lflr = int(sampling_interval / dt_lflr)

    ks_step_fn_lflr = kuramoto_sivashinsky(
        nu=-1.0, gamma=1.0, dt=dt_lflr, grid=grid_lflr
    )

    trajectory_fn_lflr = jax.jit(
        funcutils.trajectory(ks_step_fn_lflr, steps=total_steps_lflr)
    )

    all_trajectories_lflr = []
    for i in tqdm(range(num_trajectories), desc="LFLR Trajectories"):
        key = jax.random.PRNGKey(i)
        initial_data = kuramoto_sivashinsky_initial_conditions(grid_lflr, key, n_c=30)

        _, full_trajectory = trajectory_fn_lflr(initial_data)

        trajectory_after_warmup = full_trajectory.data[warmup_steps_lflr:]
        sampled_trajectory = trajectory_after_warmup[::steps_per_sample_lflr]
        all_trajectories_lflr.append(sampled_trajectory)

    # --- Save all data to a single HDF5 file ---
    print(f"\nSaving all data to {output_filename}...")
    with h5py.File(output_filename, "w") as f:
        # Save trajectory data
        f.create_dataset("HFHR", data=np.stack(all_trajectories_hfhr))
        f.create_dataset("LFLR", data=np.stack(all_trajectories_lflr))

        # Calculate and save the time and space arrays
        # We use the high-resolution grid and sampling for these, as is standard.
        num_samples = len(all_trajectories_hfhr[0])
        t_stamps = np.linspace(warmup_time, total_run_time, num_samples, endpoint=False)
        x_grid = grid_hfhr.axes()[0]

        f.create_dataset("t", data=t_stamps)
        f.create_dataset("x", data=x_grid)

    print("Data generation and saving complete.")
