import argparse
import os
from test_function import combined_error_test, run_full_test, run, static_benchmark_run, static_benchmark_run_modify
from file_utils import plot_results, process_results
from datetime import datetime
# from error_function import generate_config
from multiprocessing import Process
from test_function import run_benchmark_parallel, static_benchmark_run_modify
from advanced_error_function import generate_config, process_single_error


# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="ReAct_Agent", help='Choose the LLM agent')#"GPT-Agent"
    parser.add_argument('--num_queries', type=int, default=1, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', type=str, default=['level1'], choices=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/nemo_benchmark/app-route", help='Directory to save output JSONL file')
    parser.add_argument('--max_iteration', type=int, default=10, help='Choose maximum trials for a query')
    parser.add_argument('--full_test', type=int, default=1, choices=[0, 1], help='Enable full test if set to 1')
    parser.add_argument('--vllm', type=int, default=1, choices=[0, 1], help='Enable vllm if set to 1')
    parser.add_argument('--static', type=int, default=1, choices=[0, 1], help='If set to 1, use static benchmark')
    parser.add_argument('--static_benchmark_generation', type=int, default=0, choices=[0, 1], help='If set to 1, generate static benchmark')
    parser.add_argument('--agent_test', type=int, default=1, choices=[0, 1], help='If set to 1, run agent test')
    parser.add_argument('--prompt_type', type=str, default="base", help='Path to the error configuration file')
    parser.add_argument('--parallel', type=int, default=1, choices=[0, 1], help='If set to 1, run in parallel')
    return parser.parse_args()

# Call the appropriate function based on the full_test argument
if __name__ == "__main__":
    # args = parse_args()
    # if args.full_test == 1 and args.static == 0 and args.agent_test == 0:
    #     run_full_test(args)    
    #     combined_error_test(args)
    # elif args.full_test ==1 and args.static == 1 and args.agent_test == 0:
    #     print("Running static benchmark")
    #     static_benchmark_run(args)
    # elif args.agent_test == 1:
    #     # for i in range(3):
    #     #     if i == 0:
    #     #         save_result_path = args.root_dir = os.path.join(args.root_dir, 'result',args.llm_agent_type,"agenttest", datetime.now().strftime("%Y%m%d-%H%M%S"))
    #     #         os.makedirs(save_result_path, exist_ok=True) 
    #     #         args.prompt_type = "base"
    #     #         args.static_benchmark_generation = 1
    #     #         static_benchmark_run(args)
    #     #     if i == 1:
    #     #         args.root_dir = save_result_path
    #     #         args.prompt_type = "cot"
    #     #         args.static_benchmark_generation = 0
    #     #         static_benchmark_run(args)
    #     #     if i == 2:
    #     #         args.root_dir = save_result_path
    #     #         args.prompt_type = "few_shot_basic"
    #     #         args.static_benchmark_generation = 0
    #     #         static_benchmark_run(args)
    #     save_result_path = args.root_dir = "/home/ubuntu/jiajun_benchmark/app-route/result/GPT-Agent/agenttest/20250401-050934"
    #     args.prompt_type = "cot"
    #     args.static_benchmark_generation = 0
    #     args.llm_agent_type = "Qwen/Qwen2.5-72B-Instruct"
    #     static_benchmark_run(args)
    #     args.prompt_type = "few_shot_basic"
    #     static_benchmark_run(args)
    #     # process_results(save_result_path)
        
    #     # plot_results(save_result_path,5)
    #     # plot_results(save_result_path,10)
    #     # plot_results(save_result_path,30)
    #     # plot_results(save_result_path,50)
    # else:
    #     run(args)
    args = parse_args()
    # if args.parallel == 1:
    #     start_time = datetime.now()
    #     run_benchmark_parallel(args)
    #     end_time = datetime.now()
    #     duration = end_time - start_time
    #     print(f"Benchmark completed in {duration}")
    args.root_dir = "/home/ubuntu/nemo_benchmark/app-route/result/GPT-Agent/agenttest/111"    
    static_benchmark_run_modify(args)

    # Create a directory to save results
    # save_result_path = os.path.join(args.root_dir, 'result', args.llm_agent_type, "agenttest", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # os.makedirs(save_result_path, exist_ok=True)
    # args.root_dir = save_result_path
    # generate_config(os.path.join(save_result_path, "error_config.json"), num_errors_per_type=args.num_queries)
    # static_benchmark_run_modify(args)