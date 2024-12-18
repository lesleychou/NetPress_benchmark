# Auto Benchmark

## Code structure 
Each application has three parts
1. Dynamic benchmark generation. 
```text
Input: args to control [query, answer] dataset complexitiy
Output: a benchmark dataset to evaluate LLM agents
```

2. Network environment to interact with LLM agents
```text
Input: agent's answer for each query
Output: results metric "correctness, safety, latency"
```

3. LLM agents (can be separate files depending on agent usage)
```text
Input: queries from the benchmark
Output: result summary in `.jsonl` format for the current agent
```