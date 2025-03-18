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

    input_path = ["logs/AzureGPT4Agent_few_shot_semantic/new_gpt4o_few_shot_semantic.jsonl", 
                  "logs/AzureGPT4Agent_cot/new_gpt4o_cot.jsonl",
                  "logs/Qwen2.5-72B-Instruct_cot/new_qwen_cot_50.jsonl"]
    
    # Load data from each file separately
    all_results = []
    for path in input_path:
        results = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                results.append(obj)
        
        # Group the results by task label
        grouped_results = {}
        for result in results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': path,
            'name': os.path.basename(path).split('.')[0],
            'grouped_results': grouped_results
        })

    sample_sizes = [10, 20, 50]
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Initialize lists to store data for plotting
    for i, sample_size in enumerate(sample_sizes):
        print(f"\n=== STATISTICS FOR FIRST {sample_size} SAMPLES PER LABEL ===")
        
        safety_data = []
        correctness_data = []
        
        for result_set in all_results:
            grouped_results = result_set['grouped_results']
            name = result_set['name']
            
            # Calculate stats for each task label
            task_labels = list(grouped_results.keys())
            
            # Correctness stats
            print(f"\nCorrectness Pass Rates for {name}:")
            
            # Calculate overall statistics for correctness
            all_binary_outcomes = []
            for task_label in task_labels:
                samples = grouped_results[task_label][:sample_size]
                binary_outcomes = [1 if result["Result-Correctness"] == "Pass" else 0 for result in samples]
                all_binary_outcomes.extend(binary_outcomes)
            
            # Calculate overall pass rate and error margin for correctness
            total_samples = len(all_binary_outcomes)
            correctness_pass_rate = (sum(all_binary_outcomes) / total_samples) * 100
            correctness_sem = stats.sem(all_binary_outcomes, ddof=0) * 100
            correctness_error_margin = 1.96 * correctness_sem
            
            print(f"Overall Correctness: {correctness_pass_rate:.2f}% ±{correctness_error_margin:.2f}% (n={total_samples})")

            # Safety stats
            print(f"\nSafety Pass Rates for {name}:")
            
            # Calculate overall safety statistics
            all_binary_outcomes = []
            for task_label in task_labels:
                samples = grouped_results[task_label][:sample_size]
                binary_outcomes = [1 if result["Result-Safety"] == "Pass" else 0 for result in samples]
                all_binary_outcomes.extend(binary_outcomes)
            
            total_samples = len(all_binary_outcomes)
            safety_pass_rate = (sum(all_binary_outcomes) / total_samples) * 100
            safety_sem = stats.sem(all_binary_outcomes, ddof=0) * 100
            safety_error_margin = 1.96 * safety_sem
            
            print(f"Overall Safety: {safety_pass_rate:.2f}% ±{safety_error_margin:.2f}% (n={total_samples})")
            
            # Store data for plotting
            safety_data.append({
                'name': name,
                'pass_rate': safety_pass_rate,
                'error_margin': safety_error_margin
            })
            
            correctness_data.append({
                'name': name,
                'pass_rate': correctness_pass_rate,
                'error_margin': correctness_error_margin
            })
        
        # Plot the scatter points with error bars for this sample size
        ax = axes[i]
        
        for j in range(len(safety_data)):
            s_data = safety_data[j]
            c_data = correctness_data[j]
            
            ax.errorbar(
                s_data['pass_rate'], 
                c_data['pass_rate'],
                xerr=s_data['error_margin'],
                yerr=c_data['error_margin'],
                fmt='o',
                capsize=5,
                label=s_data['name']
            )
        
        ax.set_xlabel('Safety Pass Rate (%)')
        ax.set_ylabel('Correctness Pass Rate (%)')
        ax.set_title(f'N = {sample_size}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Set reasonable axis limits (optional)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('figs/safety_vs_correctness.png', dpi=300)
    plt.show()

# run the main function
if __name__ == "__main__":
    main()