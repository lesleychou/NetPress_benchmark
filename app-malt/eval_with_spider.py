import numpy as np
import pandas as pd
import jsonlines
import random
import os
import time
import matplotlib.pyplot as plt
import json
import matplotlib.patches as mpatches

import argparse
from scipy import stats
import math

# Example usage:
# python eval_with_spider_charts.py --sampling_method random

def main():
    parser = argparse.ArgumentParser(description='Generate spider charts for semantic error detection performance')
    parser.add_argument('--sampling_method', type=str, choices=['first', 'random'], default='first',
                        help='Method to sample: "first" takes first N samples, "random" takes random N samples')
    args = parser.parse_args()

    input_path = ["logs/AzureGPT4Agent_few_shot_semantic/new_gpt4o_few_shot_semantic.jsonl", 
                  "logs/AzureGPT4Agent_few_shot_semantic/extra_50_gpt4o_few_shot_semantic.jsonl",
                  "logs/AzureGPT4Agent_cot/new_gpt4o_cot.jsonl",
                  "logs/AzureGPT4Agent_cot/extra_50_gpt4o_cot.jsonl",
                  "logs/Qwen2.5-72B-Instruct_cot/new_qwen_cot_50.jsonl",
                  "logs/Qwen2.5-72B-Instruct_cot/extra_50_qwen_cot.jsonl",
                  "logs/Qwen2.5-72B-Instruct_few_shot_semantic/new_qwen_few_shot_semantic_50.jsonl",
                  "logs/Qwen2.5-72B-Instruct_few_shot_semantic/extra_50_qwen_few_shot_semantic.jsonl"]
    
    # Define which files to merge
    cot_merge_files = [
        "logs/AzureGPT4Agent_cot/new_gpt4o_cot.jsonl",
        "logs/AzureGPT4Agent_cot/extra_50_gpt4o_cot.jsonl"
    ]
    cot_merge_name = "gpt4o_cot_combined"

    few_shot_merge_files = [
        "logs/AzureGPT4Agent_few_shot_semantic/new_gpt4o_few_shot_semantic.jsonl",
        "logs/AzureGPT4Agent_few_shot_semantic/extra_50_gpt4o_few_shot_semantic.jsonl"
    ]
    few_shot_merge_name = "gpt4o_few_shot_combined"
    
    # Add Qwen merge file definitions
    qwen_cot_merge_files = [
        "logs/Qwen2.5-72B-Instruct_cot/new_qwen_cot_50.jsonl",
        "logs/Qwen2.5-72B-Instruct_cot/extra_50_qwen_cot.jsonl"
    ]
    qwen_cot_merge_name = "qwen_cot_combined"
    
    qwen_few_shot_merge_files = [
        "logs/Qwen2.5-72B-Instruct_few_shot_semantic/new_qwen_few_shot_semantic_50.jsonl",
        "logs/Qwen2.5-72B-Instruct_few_shot_semantic/extra_50_qwen_few_shot_semantic.jsonl"
    ]
    qwen_few_shot_merge_name = "qwen_few_shot_combined"
    
    # Load data from each file
    file_results = {}
    for path in input_path:
        print(f"Processing file: {path}")
        results = []
        with jsonlines.open(path) as reader:
            for obj in reader:
                results.append(obj)
        file_results[path] = results
    
    # Process results, merging specified files
    all_results = []
    processed_files = set()
    
    # Handle the CoT merged files
    if set(cot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in cot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(cot_merge_files),
            'name': cot_merge_name,
            'grouped_results': grouped_results
        })

    # Handle the Few Shot merged files
    if set(few_shot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in few_shot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(few_shot_merge_files),
            'name': few_shot_merge_name,
            'grouped_results': grouped_results
        })
        
    # Handle the Qwen CoT merged files
    if set(qwen_cot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in qwen_cot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(qwen_cot_merge_files),
            'name': qwen_cot_merge_name,
            'grouped_results': grouped_results
        })

    # Handle the Qwen Few Shot merged files
    if set(qwen_few_shot_merge_files).issubset(set(input_path)):
        merged_results = []
        for merge_file in qwen_few_shot_merge_files:
            merged_results.extend(file_results[merge_file])
            processed_files.add(merge_file)
        
        # Group the merged results by task label
        grouped_results = {}
        for result in merged_results:
            task_label = result["Label"]
            if task_label not in grouped_results:
                grouped_results[task_label] = []
            grouped_results[task_label].append(result)
        
        all_results.append({
            'path': '+'.join(qwen_few_shot_merge_files),
            'name': qwen_few_shot_merge_name,
            'grouped_results': grouped_results
        })
    
    # Then process any remaining files
    for path in input_path:
        if path not in processed_files:
            results = file_results[path]
            
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

    # Use a fixed sample size for the spider charts
    sample_size = 100
    
    # Set up data structures for spider charts
    agent_names = [result_set['name'] for result_set in all_results]
    # Get all task labels excluding the first one (which is typically 'None' or similar)
    all_task_labels = set()
    for result_set in all_results:
        for label in result_set['grouped_results'].keys():
            if label != 'None':  # Skip the 'None' label if it exists
                all_task_labels.add(label)
    task_labels = sorted(list(all_task_labels))
    
    # Remove "capacity planning" from task labels if it exists
    if "capacity planning, " in task_labels:
        task_labels.remove("capacity planning, ")

    # Prepare data for spider charts
    correctness_data = {agent: {} for agent in agent_names}
    safety_data = {agent: {} for agent in agent_names}
    
    # Calculate stats for each agent and task label
    for i, result_set in enumerate(all_results):
        agent_name = result_set['name']
        grouped_results = result_set['grouped_results']
        
        for task_label in task_labels:
            if task_label in grouped_results:
                # Sample based on the specified method
                if args.sampling_method == 'first':
                    samples = grouped_results[task_label][:sample_size]
                else:  # random sampling
                    all_samples = grouped_results[task_label]
                    # If we have enough samples, take a random sample, otherwise take all
                    if len(all_samples) > sample_size:
                        samples = random.sample(all_samples, sample_size)
                    else:
                        samples = all_samples
                
                # Calculate correctness pass rate
                correctness_binary = [1 if result["Result-Correctness"] == "Pass" else 0 for result in samples]
                correctness_pass_rate = (sum(correctness_binary) / len(correctness_binary)) * 100
                correctness_data[agent_name][task_label] = correctness_pass_rate
                
                # Calculate safety pass rate
                safety_binary = [1 if result["Result-Safety"] == "Pass" else 0 for result in samples]
                safety_pass_rate = (sum(safety_binary) / len(safety_binary)) * 100
                safety_data[agent_name][task_label] = safety_pass_rate
            else:
                # If no data for this task label, set to 0
                correctness_data[agent_name][task_label] = 0
                safety_data[agent_name][task_label] = 0
    
    # Create spider charts
    create_spider_chart(correctness_data, task_labels, "Correctness Pass Rate (%)", 
                        f"figs/correctness_spider_{args.sampling_method}_sampling.png")
    create_spider_chart(safety_data, task_labels, "Safety Pass Rate (%)", 
                        f"figs/safety_spider_{args.sampling_method}_sampling.png")

    print(f"Spider charts saved to figs/correctness_spider_{args.sampling_method}_sampling.png and figs/safety_spider_{args.sampling_method}_sampling.png")

def create_spider_chart(data, categories, title, output_path):
    """Create a polygon-style spider chart with the given data."""
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))
    
    # Set the category labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Set y-limits
    ax.set_ylim(0, 100)
    
    # Draw y-axis grid lines
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=10)
    
    # Set color cycle for different agents
    colors = plt.cm.tab10.colors
    
    # Remove the circular grid and spines
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    
    # Draw polygon grid lines
    grid_values = [20, 40, 60, 80, 100]
    for grid_val in grid_values:
        # Plot polygon for each grid level
        polygon_points = [(a, grid_val) for a in angles]
        ax.plot([p[0] for p in polygon_points], [p[1] for p in polygon_points], 
                '-', color='gray', alpha=0.2, linewidth=1)
    
    # Draw axis lines
    for i in range(N):
        ax.plot([angles[i], angles[i]], [0, 100], color='gray', linestyle='-', linewidth=1)
    
    # Plot each agent
    legend_patches = []
    for i, (agent_name, agent_data) in enumerate(data.items()):
        # Order values according to categories
        values = [agent_data.get(cat, 0) for cat in categories]
        values += values[:1]  # Close the loop
        
        # Plot the agent's data
        color = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, linestyle='-', color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
        # Add to legend
        legend_patches.append(mpatches.Patch(color=color, label=agent_name))
    
    # Add legend
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title(title, size=15, y=1.1)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

# run the main function
if __name__ == "__main__":
    main()
