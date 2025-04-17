import json
import traceback
from dotenv import load_dotenv
import openai
import pandas as pd
from collections import Counter
from prototxt_parser.prototxt import parse
import os
from solid_step_helper import clean_up_llm_output_func
import networkx as nx
import jsonlines
import json
import re
import time
import sys
import numpy as np
import copy

"""
python prepare_data_for_finetune.py data/static_benchmark_malt.jsonl data/fine_tune_test.json
"""

def convert_update_node_value(match_obj):
    """Convert solid_step_update_node_value function calls to direct implementation"""
    # Extract the parameters from the function call
    params_str = match_obj.group(1)
    params = [p.strip() for p in params_str.split(',')]
    
    if len(params) < 3:
        return match_obj.group(0)  # Not enough parameters, return original
    
    graph_data_var = params[0]
    node_name = params[1].strip("'\"")
    new_value = params[2]
    
    # Create the direct implementation
    implementation = f"""
    # Find the node ID by name
    child_node_id = None
    for node in {graph_data_var}.nodes:
        if {graph_data_var}.nodes[node].get('name') == {params[1]}:
            child_node_id = node
            break
    
    # Check if node is found
    if child_node_id is None:
        print(f"Node with name {params[1]} not found.")
        return {graph_data_var}
    
    # Check if the node is of type EK_PORT and update its physical_capacity_bps attribute
    if 'EK_PORT' in {graph_data_var}.nodes[child_node_id]['type']:
        {graph_data_var}.nodes[child_node_id]['physical_capacity_bps'] = {new_value}
    """
    
    return implementation.strip()

def convert_add_node_to_graph(match_obj):
    """Convert solid_step_add_node_to_graph function calls to direct implementation"""
    # Extract the parameters from the function call
    params_str = match_obj.group(1)
    params = [p.strip() for p in params_str.split(',', 2)]
    
    if len(params) < 2:
        return match_obj.group(0)  # Not enough parameters, return original
    
    graph_data_var = params[0]
    
    # Try to parse the new_node parameter
    new_node_param = params[1]
    parent_node_param = params[2] if len(params) > 2 else "None"
    
    # Create the direct implementation
    implementation = f"""
    # Create a new unique node ID
    new_node_id = len({graph_data_var}.nodes) + 1
    
    # Extract node details from parameters
    new_node = {new_node_param}
    
    # Add physical_capacity_bps for EK_PORT type
    node_attrs = {{'name': new_node['name'], 'type': new_node['type']}}
    if 'EK_PORT' in new_node['type']:
        node_attrs['physical_capacity_bps'] = 1000
    
    # Add the new node to the graph
    {graph_data_var}.add_node(new_node_id, **node_attrs)
    
    # If a parent node is specified, add an edge between parent and the new node
    parent_node_name = {parent_node_param}
    if parent_node_name:
        parent_node_id = None
        for node in {graph_data_var}.nodes:
            if {graph_data_var}.nodes[node].get('name') == parent_node_name:
                parent_node_id = node
                break
        if parent_node_id:
            {graph_data_var}.add_edge(parent_node_id, new_node_id, type='RK_CONTAINS')
    """
    
    return implementation.strip()

def convert_remove_node_from_graph(match_obj):
    """Convert solid_step_remove_node_from_graph function calls to direct implementation"""
    # Extract the parameters from the function call
    params_str = match_obj.group(1)
    params = [p.strip() for p in params_str.split(',')]
    
    if len(params) < 2:
        return match_obj.group(0)  # Not enough parameters, return original
    
    graph_data_var = params[0]
    node_name_param = params[1]
    
    # Create the direct implementation
    implementation = f"""
    # Find the node ID by name
    node_id = None
    for node in {graph_data_var}.nodes:
        if {graph_data_var}.nodes[node].get('name') == {node_name_param}:
            node_id = node
            break
    
    if node_id is None:
        print(f"Node with name {{node_name_param}} not found.")
        return {graph_data_var}
    
    # Remove the node and its edges from the graph
    {graph_data_var}.remove_node(node_id)
    """
    
    return implementation.strip()

def convert_counting_query(match_obj):
    """Convert solid_step_counting_query function calls to direct implementation"""
    # Extract the parameters from the function call
    params_str = match_obj.group(1)
    params = [p.strip() for p in params_str.split(',', 2)]
    
    if len(params) < 2:
        return match_obj.group(0)  # Not enough parameters, return original
    
    graph_data_var = params[0]
    node1_param = params[1]
    node2_param = params[2] if len(params) > 2 else "None"
    
    # Create the direct implementation based on number of parameters
    if node2_param == "None":
        implementation = f"""
        # Count the total number of node1 in the graph
        total_count = 0
        node1_type = {node1_param}['type']
        for node in {graph_data_var}.nodes(data=True):
            if node1_type in node[1]['type']:
                total_count += 1
        result = total_count
        """
    else:
        implementation = f"""
        # Find the target node1
        target_node1 = None
        for node in {graph_data_var}.nodes:
            if {graph_data_var}.nodes[node].get('name') == {node1_param}['name']:
                target_node1 = node
                break
        
        if target_node1 is None:
            print(f"Node1 not found")
            return 0
        
        # Use BFS to count all node2 contained within node1
        node2_count = 0
        queue = [target_node1]
        visited = set()
        
        while queue:
            current_node = queue.pop(0)
            if current_node in visited:
                continue
            visited.add(current_node)
            for edge in {graph_data_var}.out_edges(current_node, data=True):
                if edge[2]['type'] == 'RK_CONTAINS':
                    destination_node = edge[1]
                    if {node2_param}['type'] in {graph_data_var}.nodes[destination_node]['type']:
                        node2_count += 1
                    queue.append(destination_node)
        
        result = node2_count
        """
    
    return implementation.strip()

def convert_list_child_nodes(match_obj):
    """Convert solid_step_list_child_nodes function calls to direct implementation"""
    # Extract the parameters from the function call
    params_str = match_obj.group(1)
    params = [p.strip() for p in params_str.split(',')]
    
    if len(params) < 2:
        return match_obj.group(0)  # Not enough parameters, return original
    
    graph_data_var = params[0]
    parent_node_param = params[1]
    
    # Create the direct implementation
    implementation = f"""
    # Find the parent node ID by name
    parent_node_id = None
    for node in {graph_data_var}.nodes:
        if {graph_data_var}.nodes[node].get('name') == {parent_node_param}['name']:
            parent_node_id = node
            break
    
    if parent_node_id is None:
        print(f"Parent node not found.")
        return []
    
    # Get all child nodes
    child_nodes = []
    for edge in {graph_data_var}.out_edges(parent_node_id, data=True):
        if edge[2]['type'] == 'RK_CONTAINS':
            child_nodes.append({graph_data_var}.nodes[edge[1]])
    
    # Only return the name of the child nodes
    child_nodes_name = [node['name'] for node in child_nodes]
    
    result = child_nodes_name
    """
    
    return implementation.strip()

def convert_rank_child_nodes(match_obj):
    """Convert solid_step_rank_child_nodes function calls to direct implementation"""
    # Extract the parameters from the function call
    params_str = match_obj.group(1)
    params = [p.strip() for p in params_str.split(',')]
    
    if len(params) < 2:
        return match_obj.group(0)  # Not enough parameters, return original
    
    graph_data_var = params[0]
    parent_node_name_param = params[1]
    
    # Create the direct implementation
    implementation = f"""
    # Find the parent node ID by name
    parent_node_id = None
    for node in {graph_data_var}.nodes:
        if {graph_data_var}.nodes[node].get('name') == {parent_node_name_param}:
            parent_node_id = node
            break
    
    if parent_node_id is None:
        print(f"Parent node not found.")
        return []
    
    # Initialize a list to store child nodes and their total physical capacity
    child_nodes_capacity = []
    
    # Find all child nodes and calculate their total physical capacity
    for edge in {graph_data_var}.out_edges(parent_node_id, data=True):
        if edge[2]['type'] == 'RK_CONTAINS':
            child_node = edge[1]
            total_physical_capacity_bps = 0
            if 'EK_PORT' in {graph_data_var}.nodes[child_node]['type']:
                total_physical_capacity_bps += {graph_data_var}.nodes[child_node].get('physical_capacity_bps', 0)
            for child_edge in {graph_data_var}.out_edges(child_node, data=True):
                if child_edge[2]['type'] == 'RK_CONTAINS':
                    grandchild_node = child_edge[1]
                    if 'EK_PORT' in {graph_data_var}.nodes[grandchild_node]['type']:
                        total_physical_capacity_bps += {graph_data_var}.nodes[grandchild_node].get('physical_capacity_bps', 0)
            child_nodes_capacity.append(({graph_data_var}.nodes[child_node], total_physical_capacity_bps))
    
    # Sort the child nodes by their total physical capacity in descending order
    child_nodes_capacity.sort(key=lambda x: x[1], reverse=True)
    
    # Return only the sorted child nodes name and each of total physical capacity
    sorted_child_nodes_names = [(node['name'], capacity) for node, capacity in child_nodes_capacity]
    
    result = sorted_child_nodes_names
    """
    
    return implementation.strip()

def convert_solid_step_functions(answer_code):
    """
    Convert all solid_step_* function calls to direct implementations
    """
    # Dictionary mapping function patterns to their converters
    converters = {
        r'solid_step_update_node_value\((.*?)\)': convert_update_node_value,
        r'solid_step_add_node_to_graph\((.*?)\)': convert_add_node_to_graph,
        r'solid_step_remove_node_from_graph\((.*?)\)': convert_remove_node_from_graph,
        r'solid_step_counting_query\((.*?)\)': convert_counting_query,
        r'solid_step_list_child_nodes\((.*?)\)': convert_list_child_nodes,
        r'solid_step_rank_child_nodes\((.*?)\)': convert_rank_child_nodes
    }
    
    modified_code = answer_code
    
    # Apply each converter
    for pattern, converter in converters.items():
        # Find all occurrences of the function call
        matches = re.finditer(pattern, modified_code)
        
        # Process each match from end to start to avoid index issues
        matches = list(matches)
        for match in reversed(matches):
            # Get the converted implementation
            implementation = converter(match)
            
            # Replace the function call with the implementation
            modified_code = modified_code[:match.start()] + implementation + modified_code[match.end():]
    
    return modified_code

def process_example(example):
    """Process a single example to convert solid_step functions"""
    # Check if example is in the messages format
    if "messages" in example:
        messages = example["messages"]
        # Find question and answer in messages
        question = None
        answer = None
        task_label = None
        
        for message in messages:
            if "question" in message:
                question = message["question"]
            elif "answer" in message:
                answer = message["answer"]
            elif "task_label" in message:
                task_label = message["task_label"]
        
        if not answer or 'ground_truth_process_graph' not in answer:
            return example
        
        # Extract the function body
        match = re.search(r'ground_truth_process_graph\(.*?\):(.*?)(?=return return_object|$)', answer, re.DOTALL)
        
        if not match:
            return example
        
        function_body = match.group(1)
        
        # Convert solid_step functions in the function body
        converted_body = convert_solid_step_functions(function_body)
        
        # Reconstruct the answer with converted function body
        new_answer = answer.replace(function_body, converted_body)
        
        # Replace ground_truth_process_graph with process_graph
        new_answer = new_answer.replace('ground_truth_process_graph', 'process_graph')
        
        # Create a new example with the converted answer
        new_example = copy.deepcopy(example)
        new_example["messages"] = []
        
        if question:
            new_example["messages"].append({"question": question})
        if new_answer:
            new_example["messages"].append({"answer": new_answer})
        if task_label:
            new_example["messages"].append({"task_label": task_label})
        
        return new_example
    else:
        # For direct question/answer format
        question = example.get('question', '')
        answer = example.get('answer', '')
        
        # Skip if no answer or not ground_truth_process_graph found
        if not answer or 'ground_truth_process_graph' not in answer:
            return example
        
        # Extract the function body
        match = re.search(r'ground_truth_process_graph\(.*?\):(.*?)(?=return return_object|$)', answer, re.DOTALL)
        
        if not match:
            return example
        
        function_body = match.group(1)
        
        # Convert solid_step functions in the function body
        converted_body = convert_solid_step_functions(function_body)
        
        # Reconstruct the answer with converted function body
        new_answer = answer.replace(function_body, converted_body)
        
        # Replace ground_truth_process_graph with process_graph
        new_answer = new_answer.replace('ground_truth_process_graph', 'process_graph')
        
        # Create a new example with the converted answer
        new_example = copy.deepcopy(example)
        new_example['answer'] = new_answer
        
        return new_example

def process_jsonlines_file(input_file, output_file):
    """Process a JSONL file containing examples"""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            example = json.loads(line)
            processed_example = process_example(example)
            f_out.write(json.dumps(processed_example) + '\n')

def process_json_file(input_file, output_file):
    """Process a JSON file containing examples"""
    with open(input_file, 'r') as f_in:
        data = json.load(f_in)
    
    if isinstance(data, list):
        processed_data = [process_example(example) for example in data]
    else:
        processed_data = process_example(data)
    
    with open(output_file, 'w') as f_out:
        json.dump(processed_data, f_out, indent=2)

if __name__ == "__main__":    
    if len(sys.argv) < 3:
        print("Usage: python convert_solid_step_examples.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Determine file type by extension
    if input_file.endswith('.jsonl'):
        process_jsonlines_file(input_file, output_file)
    else:
        process_json_file(input_file, output_file)
    
    print(f"Processed examples saved to {output_file}")
