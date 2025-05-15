# Auto Benchmark

## Installation

Install conda environments
```bash
conda env create -f environment_mininet.yml
conda env create -f environment_ai_gym.yml
```
The `ai_gym_env` is for malt application, and the `mininet` encironment can be used for route and k8s application.

## Code structure 
Each application has three parts
1. Dynamic benchmark generation. 
```text
Input: args to control [query, answer] dataset complexitiy
Output: a benchmark dataset to evaluate LLM agents
```
If you want to test on it, you can refer to [this guide](../app-malt/README.md)
2. Network environment to interact with LLM agents
```text
Input: agent's answer for each query
Output: results metric "correctness, safety, latency"
```
If you want to test on it, you can refer to [this guide](../app-route/README.md)
3. LLM agents (can be separate files depending on agent usage)
```text
Input: queries from the benchmark
Output: result summary in `.jsonl` format for the current agent
```
If you want to test on it, you can refer to [this guide](../app-k8s/README.md)