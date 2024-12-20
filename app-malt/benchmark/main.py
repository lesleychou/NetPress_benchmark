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

# output the evaluation results to a jsonl file
OUTPUT_JSONL_DIR = 'logs/llm_agents'
OUTPUT_JSONL_FILE = 'gpt4.jsonl'

def main():
    # create 'output.jsonl' file if it does not exist: 'logs/malt-benchmark/test.jsonl'
    # create the directory if it does not exist
    if not os.path.exists(OUTPUT_JSONL_DIR):
        os.makedirs(OUTPUT_JSONL_DIR)

    # create the file if it does not exist
    output_path = os.path.join(OUTPUT_JSONL_DIR, OUTPUT_JSONL_FILE)
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            pass

    # dynamically generate a set of new queries
    query_generator = QueryGenerator()
    query_generator.generate_queries(num_each_type=2)
    dynamic_dataset_path = 'data/benchmark_malt.jsonl'
    query_generator.save_queries_to_file(dynamic_dataset_path)

    # Load the evaluator
    evaluator = BenchmarkEvaluator()

    # the format is {"messages": [{"question": "XXX."}, {"answer": "YYY"}]}
    benchmark_data = []
    with jsonlines.open(dynamic_dataset_path) as reader:
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



# run the main function
if __name__ == "__main__":
    main()