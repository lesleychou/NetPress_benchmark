# NetPress Benchmark

## Overview
NetPress Benchmark is an automated benchmarking tool that provides testing environments for three distinct applications: Malt, Routing, and Kubernetes (K8s).

## Prerequisites
- Conda package manager
- Python environment

## Installation

1. Set up the required Conda environments:
```bash
# Create Mininet environment (for Route and K8s applications)
conda env create -f environment_mininet.yml

# Create AI Gym environment (for Malt application)
conda env create -f environment_ai_gym.yml
```

2. Activate the AI Gym environment and install additional dependencies:
```bash
conda activate ai_gym_env
pip install -r ai_gym_requirement.txt
```

## Quick Start

To run the Malt application benchmark:
```bash
cd experiments
./run_app_malt.sh
```

## Detailed Application Guides

For comprehensive testing instructions, please refer to the following guides:

- [Malt Application Guide](../app-malt/README.md)
- [Routing Application Guide](../app-route/README.md)
- [Kubernetes Application Guide](../app-k8s/README.md)

## Project Structure
- `experiments/` - Contains benchmark execution scripts
- `environment_mininet.yml` - Conda environment configuration for Route and K8s applications
- `environment_ai_gym.yml` - Conda environment configuration for Malt application
- `ai_gym_requirement.txt` - Additional Python dependencies for the AI Gym environment