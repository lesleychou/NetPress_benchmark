import numpy as np
import pandas as pd
import jsonlines
import random
import os
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
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', nargs='+', default=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--output_dir', type=str, default='logs/llm_agents', help='Directory to save output JSONL file')
    parser.add_argument('--output_file', type=str, default='gpt4o.jsonl', help='Name of the output JSONL file')
    parser.add_argument('--dynamic_benchmark_path', type=str, default='data/benchmark_malt.jsonl', help='Path to save dynamic dataset')
    return parser.parse_args()

# anexample of how to use main.py with input args
# python main.py --num_queries 10 --complexity_level level1 level2 --output_dir logs/llm_agents --output_file gpt4o.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl

def main(args):

    benchmark_config = {
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
            
        ret, ground_truth_ret, verifier_results, query_run_latency, ret_graph_copy = evaluator.userQuery(current_query, golden_answer)
        evaluator.ground_truth_check(current_query, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, query_run_latency, output_path)

    # TODO: Analyze the results



# run the main function
if __name__ == "__main__":
    args = parse_args()
    main(args)