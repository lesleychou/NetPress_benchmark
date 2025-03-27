import argparse
import os
from test_function import combined_error_test, run_full_test, run, static_benchmark_run
from file_utils import prepare_file, initialize_json_file, summarize_results, error_classification, plot_metrics_from_json, delete_result_folder, plot_combined_error_metrics, plot_metrics
# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="GPT-Agent", help='Choose the LLM agent')#"GPT-Agent"
    parser.add_argument('--num_queries', type=int, default=5, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', type=str, default=['level1'], choices=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/jiajun_benchmark/app-route", help='Directory to save output JSONL file')
    parser.add_argument('--max_iteration', type=int, default=15, help='Choose maximum trials for a query')
    parser.add_argument('--full_test', type=int, default=1, choices=[0, 1], help='Enable full test if set to 1')
    parser.add_argument('--vllm', type=int, default=0, choices=[0, 1], help='Enable vllm if set to 1')
    parser.add_argument('--static', type=int, default=1, choices=[0, 1], help='If set to 1, use static benchmark')
    parser.add_argument('--static_benchmark_generation', type=int, default=1, choices=[0, 1], help='If set to 1, generate static benchmark')
    return parser.parse_args()

# Call the appropriate function based on the full_test argument
if __name__ == "__main__":
    args = parse_args()
    if args.full_test == 1 and args.static == 0:
        run_full_test(args)    
        combined_error_test(args)
    elif args.full_test ==1 and args.static == 1:
        print("Running static benchmark")
        static_benchmark_run(args)
    else:
        run(args)


