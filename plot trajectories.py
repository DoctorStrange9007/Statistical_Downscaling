import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def phi_n_objects(ynm1, ynm1_prime):
    """
    Create multivariate normal distributions for transition dynamics.
    
    This function creates the mixture components for the transition from
    (y_{n-1}, y'_{n-1}) to (y_n, y'_n). Each component is a multivariate
    normal distribution with mean depending on the previous states.

    Parameters
    ----------
    ynm1 : np.ndarray
        Previous state y_{n-1}.
    ynm1_prime : np.ndarray
        Previous state y'_{n-1}.

    Returns
    -------
    tuple
        Tuple containing:
        - components_yn: List of multivariate normal distributions for y_n
        - components_yn_prime: List of multivariate normal distributions for y'_n
        - components_yn_yn_prime: List of joint multivariate normal distributions
    """
    d = 3
    # Add mean reversion: multiply by factor < 1 to pull back toward zero
    A = [0.1 ** (i + 1) * np.eye(d) for i in range(d)]  # Changed from (i+1) to 0.8*(i+1)
    # B = 0.1*(i+1) * np.eye(d)
    B = [np.zeros((d, d)) for i in range(d)]
    # C = 0.1*(i+1) * np.eye(d)
    C = [np.zeros((d, d)) for i in range(d)]
    D = [0.1 ** (i + 1) * np.eye(d) for i in range(d)]  # Changed from (i+1) to 0.8*(i+1)

    # Reduce noise variance to make trajectories diverge less
    Sigma = 0.1 * np.eye(2 * d)  # Changed from np.eye(2 * d) to 0.1 * np.eye(2 * d)

    components_yn = [multivariate_normal(mean=A[i] @ ynm1 + B[i] @ ynm1_prime, cov=Sigma[:d, :d]) for i in range(d)]  # type: ignore
    components_yn_prime = [multivariate_normal(mean=C[i] @ ynm1 + D[i] @ ynm1_prime, cov=Sigma[d:, d:]) for i in range(d)]  # type: ignore
    components_yn_yn_prime = [multivariate_normal(mean=np.concatenate([A[i] @ ynm1 + B[i] @ ynm1_prime, C[i] @ ynm1 + D[i] @ ynm1_prime]), cov=Sigma) for i in range(d)]  # type: ignore
    
    return components_yn, components_yn_prime, components_yn_yn_prime

def phi_0_objects():
    """
    Create multivariate normal distributions for initial states.
    
    This function creates the mixture components for the initial distribution
    of (y_0, y'_0). Each component is a multivariate normal distribution
    with fixed means.

    Returns
    -------
    tuple
        Tuple containing:
        - components_y0: List of multivariate normal distributions for y_0
        - components_y0_prime: List of multivariate normal distributions for y'_0
        - components_y0_y0_prime: List of joint multivariate normal distributions
    """
    d = 3
    A = [0.1 ** (i + 1) * np.ones(d) for i in range(d)]
    B = [0.1 ** (i + 1) * np.ones(d) for i in range(d)]

    # Reduce noise variance to make trajectories diverge less
    Sigma = 0.1 * np.eye(d)  # Changed from np.eye(d) to 0.1 * np.eye(d)

    components_y0 = [multivariate_normal(mean=A[i], cov=Sigma) for i in range(d)]  # type: ignore
    components_y0_prime = [multivariate_normal(mean=B[i], cov=Sigma) for i in range(d)]  # type: ignore
    components_y0_y0_prime = [multivariate_normal(mean=np.concatenate([A[i], B[i]]), cov=0.1 * np.eye(2 * d)) for i in range(d)]  # type: ignore

    return components_y0, components_y0_prime, components_y0_y0_prime

def get_trajectories(k):
    """
    Generate sample trajectories for visualization.
    
    This function creates trajectories by sampling from the mixture model
    components. It always selects component 0 for demonstration purposes.

    Parameters
    ----------
    k : int
        Number of time steps (trajectory will have k+1 points).

    Returns
    -------
    np.ndarray
        Trajectory array of shape (k+1, 2, d) containing:
        - First dimension: time steps (0 to k)
        - Second dimension: [y, y'] states
        - Third dimension: d-dimensional state space
    """
    d = 3
    trajectories = np.zeros((k + 1, 2, d))
    for i in range(k + 1):
        if i == 0:
            y0, y0_prime = phi_0_objects()[0][0].rvs(), phi_0_objects()[1][0].rvs()
            trajectories[i, 0, :] = y0
            trajectories[i, 1, :] = y0_prime
        else:
            y_prev = trajectories[i - 1, 0, :]
            y_prev_prime = trajectories[i - 1, 1, :]
            y, y_prime = phi_n_objects(ynm1=y_prev, ynm1_prime=y_prev_prime)[0][0].rvs(), phi_n_objects(ynm1=y_prev, ynm1_prime=y_prev_prime)[1][0].rvs()
            trajectories[i, 0, :] = y
            trajectories[i, 1, :] = y_prime

    return trajectories

def plot_trajectories(trajectories, title="Trajectories", save_path=None):
    """
    Plot the trajectories for both y and y' over time.
    
    Parameters
    ----------
    trajectories : np.ndarray
        Array of shape (k+1, 2, d) containing trajectory data.
    title : str, optional
        Title for the plot (default: "Trajectories").
    save_path : str, optional
        Path to save the plot (default: None).
    """
    k = trajectories.shape[0] - 1
    d = trajectories.shape[2]
    time_steps = np.arange(k + 1)
    
    # Calculate squared Euclidean distance between y and y' at each time step
    distances = []
    for t in range(k + 1):
        yn = trajectories[t, 0, :]  # y at time t
        yn_prime = trajectories[t, 1, :]  # y' at time t
        distance = (1 / 2) * np.sum((yn - yn_prime) ** 2)
        distances.append(distance)
    
    # Calculate average distance for legend
    avg_distance = np.mean(distances)
    
    colors = ['blue', 'red']
    labels = ['y', "y'"]
    
    # Handle single dimension case separately
    if d == 1:
        fig, ax_single = plt.subplots(1, 1, figsize=(10, 3))
        
        # Plot both y and y' for this dimension
        for i in range(2):
            ax_single.plot(time_steps, trajectories[:, i, 0], 
                    color=colors[i], label=labels[i], marker='o', markersize=3)
        
        ax_single.set_ylabel('Dimension 1')
        ax_single.legend()
        ax_single.grid(True, alpha=0.3)
        ax_single.set_xlabel('Time Step')
        
    else:
        # Create subplots for multiple dimensions
        fig, axes = plt.subplots(d, 1, figsize=(10, 3*d), sharex=True, squeeze=False)
        axes = axes.flatten()  # Flatten to 1D array
        
        for dim in range(d):
            # Plot both y and y' for this dimension
            for i in range(2):
                axes[dim].plot(time_steps, trajectories[:, i, dim], 
                        color=colors[i], label=labels[i], marker='o', markersize=3)
            
            axes[dim].set_ylabel(f'Dimension {dim+1}')
            axes[dim].legend()
            axes[dim].grid(True, alpha=0.3)
            
            if dim == d - 1:  # Only add x-label to bottom subplot
                axes[dim].set_xlabel('Time Step')
    
    # Add distance information to title
    title_with_distance = f"{title} (Avg Distance: {avg_distance:.4f})"
    plt.suptitle(title_with_distance)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print distance statistics
    print(f"Distance Statistics:")
    print(f"  Average distance: {avg_distance:.6f}")
    print(f"  Min distance: {np.min(distances):.6f}")
    print(f"  Max distance: {np.max(distances):.6f}")
    print(f"  Std distance: {np.std(distances):.6f}")
    
    return distances

def plot_trajectories_2d(trajectories, title="2D Trajectories", save_path=None):
    """
    Plot the trajectories in 2D space for each time step.
    
    Parameters
    ----------
    trajectories : np.ndarray
        Array of shape (k+1, 2, d) containing trajectory data.
    title : str, optional
        Title for the plot (default: "2D Trajectories").
    save_path : str, optional
        Path to save the plot (default: None).
    """
    if trajectories.shape[2] != 2:
        raise ValueError("This function requires 2D trajectories (d=2)")
    
    k = trajectories.shape[0] - 1
    
    # Calculate squared Euclidean distance between y and y' at each time step
    distances = []
    for t in range(k + 1):
        yn = trajectories[t, 0, :]  # y at time t
        yn_prime = trajectories[t, 1, :]  # y' at time t
        distance = (1 / 2) * np.sum((yn - yn_prime) ** 2)
        distances.append(distance)
    
    # Calculate average distance for legend
    avg_distance = np.mean(distances)
    
    plt.figure(figsize=(10, 8))
    
    # Plot trajectories
    plt.plot(trajectories[:, 0, 0], trajectories[:, 0, 1], 
            'b-o', label='y', markersize=4, linewidth=2)
    plt.plot(trajectories[:, 1, 0], trajectories[:, 1, 1], 
            'r-s', label="y'", markersize=4, linewidth=2)
    
    # Mark start and end points
    plt.plot(trajectories[0, 0, 0], trajectories[0, 0, 1], 'go', 
            markersize=8, label='Start (y)', markeredgecolor='black')
    plt.plot(trajectories[0, 1, 0], trajectories[0, 1, 1], 'gs', 
            markersize=8, label="Start (y')", markeredgecolor='black')
    plt.plot(trajectories[-1, 0, 0], trajectories[-1, 0, 1], 'ko', 
            markersize=8, label='End (y)', markeredgecolor='white')
    plt.plot(trajectories[-1, 1, 0], trajectories[-1, 1, 1], 'ks', 
            markersize=8, label="End (y')", markeredgecolor='white')
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Add distance information to title
    title_with_distance = f"{title} (Avg Distance: {avg_distance:.4f})"
    plt.title(title_with_distance)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print distance statistics
    print(f"Distance Statistics:")
    print(f"  Average distance: {avg_distance:.6f}")
    print(f"  Min distance: {np.min(distances):.6f}")
    print(f"  Max distance: {np.max(distances):.6f}")
    print(f"  Std distance: {np.std(distances):.6f}")
    
    return distances

trajectories = get_trajectories(k=100)
plot_trajectories(trajectories, title="Trajectories", save_path=None)
