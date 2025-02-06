import numpy as np
import pandas as pd
import jsonlines
import random
import os
import time
import matplotlib.pyplot as plt
import json
from solid_step_helper import get_node_value_ranges, getGraphData, \
solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, solid_step_list_child_nodes, solid_step_update_node_value, solid_step_rank_child_nodes
from dy_query_generation import QueryGenerator
from malt_env import BenchmarkEvaluator
import argparse


# define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default='AzureGPT4Agent', help='Choose the LLM agent', choices=['AzureGPT4Agent', 'GoogleGeminiAgent', 'Qwen/QwQ-32B-Preview', 'meta-llama/Meta-Llama-3.1-70B-Instruct'])
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', nargs='+', default=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--output_dir', type=str, default='logs/llm_agents', help='Directory to save output JSONL file')
    parser.add_argument('--output_file', type=str, default='gpt4o.jsonl', help='Name of the output JSONL file')
    parser.add_argument('--dynamic_benchmark_path', type=str, default='data/benchmark_malt.jsonl', help='Path to save dynamic dataset')
    return parser.parse_args()

# anexample of how to use main.py with input args
# Example usage:
# python main.py --llm_agent_type AzureGPT4Agent --num_queries 2 --complexity_level level1 --output_dir logs/llm_agents --output_file gpt4o.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl

def main(args):

    benchmark_config = {
        'llm_agent_type': args.llm_agent_type,
        'num_queries': args.num_queries,
        'complexity_level': args.complexity_level,
        'output_dir': args.output_dir,
        'output_file': args.output_file,
        'dynamic_benchmark_path': args.dynamic_benchmark_path
    }

    # create the output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create the output file if it does not exist
    output_path = os.path.join(args.output_dir, args.output_file)
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            pass

    # dynamically generate a set of new queries
    query_generator = QueryGenerator()
    query_generator.generate_queries(num_each_type=benchmark_config['num_queries'], complexity_level=benchmark_config['complexity_level'])
    benchmark_path = benchmark_config['dynamic_benchmark_path']
    query_generator.save_queries_to_file(benchmark_path)

    # Load the evaluator
    evaluator = BenchmarkEvaluator(graph_data=query_generator.malt_real_graph)

    # the format is {"messages": [{"question": "XXX."}, {"answer": "YYY"}]}
    benchmark_data = []
    with jsonlines.open(benchmark_path) as reader:
        for obj in reader:
            benchmark_data.append(obj['messages'])
    
    # for each object in the benchmark list, get the question and answer
    for obj in benchmark_data:
        # obj is a list of dictionaries, load question, answer, task_label from it
        for item in obj:
            if 'question' in item:
                current_query = item['question']
            elif 'answer' in item:
                golden_answer = item['answer']
            elif 'task_label' in item:
                task_label = item['task_label']
            
        ret, ground_truth_ret, verifier_results, verifier_error, query_run_latency, ret_graph_copy = evaluator.userQuery(current_query, golden_answer, llm_agent_type=benchmark_config['llm_agent_type'])
        evaluator.ground_truth_check(current_query, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, verifier_error,query_run_latency, output_path)

        # have to sleep for Gemini API quota
        if benchmark_config['llm_agent_type'] == 'GoogleGeminiAgent':
            time.sleep(5)

    # Analyze the results
    # load the data from output_path
    results = []
    with jsonlines.open(output_path) as reader:
        for obj in reader:
            results.append(obj)

    # group the results by task label
    grouped_results = {}
    for result in results:
        task_label = result["Label"]
        if task_label not in grouped_results:
            grouped_results[task_label] = []
        grouped_results[task_label].append(result)

    task_labels = list(grouped_results.keys())
    avg_latencies = [sum(result["Result-Latency"] for result in grouped_results[task_label]) / len(grouped_results[task_label]) for task_label in task_labels]

     # plot the average query run latency for each task label
    plt.figure(figsize=(10, 6))
    plt.bar(task_labels, avg_latencies, color='skyblue')
    plt.xlabel('Task Label')
    plt.ylabel('Average Query Run Latency (seconds)')
    plt.title('Average Query Run Latency by Task Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'figs/average_latency_{args.llm_agent_type}.png'), dpi=300)
    plt.show()

    # plot the pass rate of correctness for each task label
    correctness_pass_rates = [sum(1 for result in grouped_results[task_label] if result["Result-Correctness"] == "Pass") / len(grouped_results[task_label]) * 100 for task_label in task_labels]

    plt.figure(figsize=(12, 6))
    plt.bar(task_labels, correctness_pass_rates, color='green')
    plt.xlabel('Task Label')
    plt.ylabel('Correctness Pass Rate (%)')
    plt.title('Correctness Pass Rate by Task Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'figs/correctness_pass_rate_{args.llm_agent_type}.png'), dpi=300)
    plt.show()

    # plot the pass rate of safety for each task label
    safety_pass_rates = [sum(1 for result in grouped_results[task_label] if result["Result-Safety"] == "Pass") / len(grouped_results[task_label]) * 100 for task_label in task_labels]

    plt.figure(figsize=(12, 6))
    plt.bar(task_labels, safety_pass_rates, color='orange')
    plt.xlabel('Task Label')
    plt.ylabel('Safety Pass Rate (%)')
    plt.title('Safety Pass Rate by Task Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'figs/safety_pass_rate_{args.llm_agent_type}.png'), dpi=300)
    plt.show()


# run the main function
if __name__ == "__main__":
    args = parse_args()
    main(args)