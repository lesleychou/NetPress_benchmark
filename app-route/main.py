import argparse
from test_function import combined_error_test, run_full_test, run

# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="Qwen/Qwen2.5-72B-Instruct", help='Choose the LLM agent')
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', type=str, default=['level1'], choices=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/nemo_benchmark/app-route", help='Directory to save output JSONL file')
    parser.add_argument('--max_iteration', type=int, default=20, help='Choose maximum trials for a query')
    parser.add_argument('--full_test', type=int, default=0, choices=[0, 1], help='Enable full test if set to 1')
    return parser.parse_args()

# Call the appropriate function based on the full_test argument
if __name__ == "__main__":
    args = parse_args()
    if args.full_test == 1:
        run_full_test(args)
    else:
        run(args)
    
    # Run the combined error test function
    combined_error_test(args)
