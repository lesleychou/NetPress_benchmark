import numpy as np
import pandas as pd
import jsonlines
import random
import os
import time
import matplotlib.pyplot as plt
import json

import argparse
from scipy import stats
import math

# anexample of how to use main.py with input args
# Example usage:
# python main.py --llm_agent_type AzureGPT4Agent --num_queries 2 --complexity_level level1 --output_dir logs/llm_agents --output_file gpt4o.jsonl --dynamic_benchmark_path data/benchmark_malt.jsonl

def main():

    input_path = "logs/AzureGPT4Agent_few_shot_semantic/new_gpt4o_few_shot_semantic.jsonl"

    # Analyze the results
    # load the data from input_path (fixed variable name)
    results = []
    with jsonlines.open(input_path) as reader:
        for obj in reader:
            results.append(obj)

    # group the results by task label
    grouped_results = {}
    for result in results:
        task_label = result["Label"]
        if task_label not in grouped_results:
            grouped_results[task_label] = []
        grouped_results[task_label].append(result)

    sample_sizes = [10, 30, 50]
    
    for sample_size in sample_sizes:
        print(f"\n=== STATISTICS FOR FIRST {sample_size} SAMPLES PER LABEL ===")
        
        # Calculate stats for each task label
        task_labels = list(grouped_results.keys())
        
        # Correctness stats
        print("\nCorrectness Pass Rates:")
        correctness_pass_rates = []
        correctness_error_margins = []
        
        for task_label in task_labels:
            # Get first N samples for this task (or all if less than N)
            samples = grouped_results[task_label][:sample_size]
            n = len(samples)
            
            # Calculate pass rate
            binary_outcomes = [1 if result["Result-Correctness"] == "Pass" else 0 for result in samples]
            pass_rate = sum(binary_outcomes) / n * 100
            
            # Calculate SEM and 95% CI
            sem = stats.sem(binary_outcomes, ddof=0) * 100
            error_margin = 1.96 * sem
            
            correctness_pass_rates.append(pass_rate)
            correctness_error_margins.append(error_margin)
            
            print(f"{task_label}: {pass_rate:.2f}% ±{error_margin:.2f}% (n={n})")
        
        # Print average
        avg_pass_rate = np.mean(correctness_pass_rates)
        avg_error_margin = np.mean(correctness_error_margins)
        print(f"Average Correctness: {avg_pass_rate:.2f}% ±{avg_error_margin:.2f}%")

# run the main function
if __name__ == "__main__":
    main()