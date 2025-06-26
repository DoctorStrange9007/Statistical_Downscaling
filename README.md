# Statistical Downscaling Project

A Python implementation of statistical downscaling. This project implements a linear Markov Decision Process (MDP) framework and uses mixture of Gaussians for the transition kernels and model. This could be extended to include various other classes/examples. 

## Installation

### Prerequisites
- Python 3.9.6 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Statistical_Downscaling
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv env
   ```

3. **Activate the environment**
   ```bash
   # On macOS/Linux:
   source env/bin/activate
   
   # On Windows:
   env\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
Statistical_Downscaling/
├── main.py                 # Main execution script
├── settings.yaml           # Configuration file
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── model.py            # Core MDP and model implementations
│   └── utils.py            # Utility functions
├── input/                  # Input data directory
├── output/                 # Output results directory
└── scripts/                # Utility scripts
    └── pre_commit.sh       # Code formatting script
```

## Configuration

The project uses `settings.yaml` for configuration:

```yaml
models:
  LinearMDP:
    n_sim: 30              # Number of Monte Carlo samples
    beta: 5                # KL divergence regularization weight
    d: 2                   # State space dimension
    N: 2                   # Number of time steps {0,1,...,N} so N+1
    true_theta:            # Target mixture weights
      - [1, 0]
      - [1, 0]
      - [1, 0]
    init_theta:            # Initial mixture weights
      - [0.5, 0.5]
      - [0.5, 0.5]
      - [0.5, 0.5]
    nr_gd_steps: 100       # Maximum gradient descent iterations
```

## Usage

### Basic Execution

Run the main optimization:

```bash
python main.py
```

### Custom Configuration

Modify `settings.yaml` to experiment with different parameters:

- **Learning rate**: Adjust `lr_rate` for optimization speed vs stability
- **Regularization**: Modify `beta` to control KL divergence weight
- **Model complexity**: Change `d` for different state space dimensions
- **Convergence**: Adjust `nr_gd_steps` and convergence tolerances

## Key Components

### LinearMDP Class
- Implements the main optimization framework for the Linear MDP assumption
- Handles gradient computation and parameter updates
- Tracks convergence for each time step

### GaussianModel Class
- Parameterized mixture of Gaussians
- Learnable mixture weights via softmax parameterization
- Methods for sampling and probability computation

### GaussianKernel Class
- Target transition kernel (ground truth)
- Fixed mixture weights defining the target distribution
- Used for KL divergence computation

## Development

### Code Formatting

The project includes automated code formatting:

1. **Setup pre-commit hook**:
   ```bash
   # Add to .git/hooks/pre-commit:
   #!/bin/sh
   source scripts/pre_commit.sh
   ```

2. **Make executable**:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

### Adding New Features

1. **Model Extensions**: Add new transition kernels or parameter models in `src/model.py`
2. **Utilities**: Add helper functions in `src/utils.py`

## Troubleshooting

### Common Issues

1. **Convergence Problems**:
   - Reduce learning rate in `settings.yaml`
   - Increase `beta` for stronger regularization
   - Check initial parameter values

2. **Numerical Instability**:
   - Verify input data scaling
   - Check for NaN/inf values in gradients
   - Adjust epsilon values in log computations

3. **Memory Issues**:
   - Reduce `n_sim` (Monte Carlo samples)
   - Decrease trajectory length `N`
   - Decrease state dimensions `d`
