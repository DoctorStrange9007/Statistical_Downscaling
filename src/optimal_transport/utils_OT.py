"""help functions"""

import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_theta_pairs(output_dir: str, true_theta: list, save_plot: bool = True):
    """
    Plot theta parameters in pairs over gradient descent steps.

    Creates plots where each plot shows theta_0, theta_1, ..., theta_N.

    Parameters
    ----------
    output_dir : str
        Directory containing the theta_gradient_descent_history.xlsx file.
    true_theta : list
        List of true theta values for comparison.
    save_plot : bool, optional
        Whether to save the plot to a file (default: True).
    """
    # Load the data
    excel_path = os.path.join(output_dir, "theta_gradient_descent_history.xlsx")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"File not found: {excel_path}")

    df = pd.read_excel(excel_path)

    # Get the number of theta parameters
    theta_cols = [col for col in df.columns if col.startswith("theta_")]

    # Group theta columns by time step
    theta_by_time = {}
    for col in theta_cols:
        parts = col.split("_")
        time_step = int(parts[1])
        if time_step not in theta_by_time:
            theta_by_time[time_step] = []
        theta_by_time[time_step].append(col)

    # Create subplots - one for each time step
    n_time_steps = len(theta_by_time)
    fig, axes = plt.subplots(1, n_time_steps, figsize=(6 * n_time_steps, 5))
    if n_time_steps == 1:
        axes = [axes]

    # Plot each time step's theta parameters
    for i, (time_step, cols) in enumerate(sorted(theta_by_time.items())):
        for col in cols:
            # Extract component index for legend
            component = int(col.split("_")[2])
            axes[i].plot(
                df["gd_step"],
                df[col],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"$\\hat{{\\theta}}_{{{time_step},{component}}}$",
            )

            # Add horizontal line for true value if available
            if time_step < len(true_theta) and component < len(true_theta[time_step]):
                if isinstance(true_theta[time_step][component], str):
                    true_val = eval(
                        true_theta[time_step][component]
                    )  # Convert string to float
                else:
                    true_val = true_theta[time_step][component]
                axes[i].axhline(
                    y=true_val,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"$\\theta_{{{time_step},{component}}}$: {true_val:.1f}",
                )

        axes[i].set_xlabel("Gradient Descent Step")
        axes[i].set_ylabel("$\\theta$ Value")
        axes[i].set_title(f"Time Step {time_step} Parameters")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].legend()

    plt.tight_layout()

    if save_plot:
        plot_path = os.path.join(output_dir, "theta_pairs_convergence.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {plot_path}")

    plt.show()


def plot_theta_periodically(theta_history: list, output_dir: str, true_theta: list):
    """
    Plot the current theta convergence periodically and save the plot.

    Parameters
    ----------
    theta_history : list
        List of dictionaries containing theta history data.
    output_dir : str
        Directory to save the plot.
    true_theta : list
        List of true theta values for comparison.
    """
    if len(theta_history) == 0:
        print("No results to plot.")
        return

    df = pd.DataFrame(theta_history)

    # Get theta columns
    theta_cols = [col for col in df.columns if col.startswith("theta_")]

    # Group by time step
    theta_by_time = {}
    for col in theta_cols:
        parts = col.split("_")
        time_step = int(parts[1])
        if time_step not in theta_by_time:
            theta_by_time[time_step] = []
        theta_by_time[time_step].append(col)

    # Create subplots
    n_time_steps = len(theta_by_time)
    fig, axes = plt.subplots(1, n_time_steps, figsize=(6 * n_time_steps, 5))
    if n_time_steps == 1:
        axes = [axes]

    # Plot each time step
    for i, (time_step, cols) in enumerate(sorted(theta_by_time.items())):
        for col in cols:
            # Extract component index for legend
            component = int(col.split("_")[2])
            axes[i].plot(
                df["gd_step"],
                df[col],
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=f"$\\hat{{\\theta}}_{{{time_step},{component}}}$",
            )

            # Add horizontal line for true value if available
            if (
                true_theta
                and time_step < len(true_theta)
                and component < len(true_theta[time_step])
            ):
                if isinstance(true_theta[time_step][component], str):
                    true_val = eval(true_theta[time_step][component])
                else:
                    true_val = true_theta[time_step][component]
                axes[i].axhline(
                    y=true_val,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"$\\theta_{{{time_step},{component}}}$: {true_val:.1f}",
                )

        axes[i].set_xlabel("Gradient Descent Step")
        axes[i].set_ylabel("$\\theta$ Value")
        axes[i].set_title(f"Time Step {time_step} Parameters")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].legend()

    plt.tight_layout()
    plt.suptitle("Theta Convergence (Intermediate Results)")

    # Save and show plot
    plot_path = os.path.join(output_dir, "theta_convergence_intermediate.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
