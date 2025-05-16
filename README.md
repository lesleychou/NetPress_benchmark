# Auto Benchmark

## Installation

Install conda environments
```bash
conda env create -f environment_mininet.yml
conda env create -f environment_ai_gym.yml
conda activate ai_gym_env
pip install -r ai_gym_requirement.txt
```
The `ai_gym_env` is for malt application, and the `mininet` encironment can be used for route and k8s application.

## Quick running script for three applications
```
# example for app-CP
cd experiments
./run_app_malt.sh
```

## Code structure 
To test fully with app-CP, please refer to [this guide](../app-malt/README.md).

To test fully with app-Routing, please refer to [this guide](../app-route/README.md).

To test fully with app-K8s, please refer to [this guide](../app-k8s/README.md).