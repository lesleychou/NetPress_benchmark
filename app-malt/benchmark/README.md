# Capacity planning in datacenter
## Run the application
Example usage.
```python
python main.py --llm_agent_type AzureGPT4Agent --num_queries 10 --complexity_level level1 --output_dir logs/llm_agents --output_file gpt4o.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl
```

## Code structure
`dy_query_generation.py` generate the dynamic benchmark datset. Input: number of queries per cataorgory, complexity level; Output: benchmark_data.jsonl

`solid_step_helper.py` contains all the functions that help dynamically generating ground truth for new queries.

`error_check.py` check if LLM generated answer satisify all the safety constraints.

`llm_model.py` includes all the LLM agents and their prompt used.

`malt_env.py` is the application simulator. It takes the LLM generated answer, run it in the enviroment and return the evaluated results.

`main.py` is the end-to-end controller. It first generate a new set of benchmark, second run the LLM agent, third analyze the results.

## LLM usage
### Azure GPT
Obtain GPT resources and endpoints

### Google Gemini
Obtain the gemini API key: https://ai.google.dev/gemini-api/docs/api-key



