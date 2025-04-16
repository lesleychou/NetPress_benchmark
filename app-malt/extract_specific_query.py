import jsonlines
import json
from collections import defaultdict
import os

path = "data/sampled_50_benchmark_malt_qwen.jsonl"

# Function to extract data from each level and subcategory
def extract_benchmark_data(input_path, items_per_category=30):
    # Dictionary to track items per category and sub-category
    category_items = defaultdict(list)
    category_counts = defaultdict(int)
    all_results = []
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found!")
        return [], {}
    
    # Read the JSONL file and collect items by category
    with jsonlines.open(input_path) as reader:
        for item in reader:
            # Debug print to see what's in the file
            print(f"Processing item: {item.keys()}")
            
            label = item.get("Label", "")
            
            # Skip empty labels
            if not label:
                continue
                
            # Parse the label to get level and subcategory
            parts = label.split(", ")
            if len(parts) >= 2:
                main_category = parts[0]  # e.g., "capacity planning"
                level = parts[1]  # e.g., "level-1", "level-2"
                
                # Create a category key that includes level and subcategory if available
                if len(parts) > 2:
                    subcategory = parts[2]
                    category_key = f"{main_category}, {level}, {subcategory}"
                else:
                    category_key = f"{main_category}, {level}"
                
                # Check if we need more of this category
                if category_counts[category_key] < items_per_category:
                    # Add the original item to our collection
                    category_items[category_key].append(item)
                    category_counts[category_key] += 1
                    print(f"Added item to category: {category_key}, count: {category_counts[category_key]}")
    
    # Combine all category items into one list
    for category, items in category_items.items():
        all_results.extend(items)
    
    return all_results, dict(category_counts)

# Extract the data
print(f"Attempting to read from: {path}")
benchmark_data, category_stats = extract_benchmark_data(path, items_per_category=30)

# Write the output to a new JSONL file
output_path = "data/sampled_30_benchmark_malt_qwen.jsonl"
with jsonlines.open(output_path, "w") as writer:
    for item in benchmark_data:
        writer.write(item)

# Print statistics
print(f"Extracted {len(benchmark_data)} total items to {output_path}")
print("Items per category:")
for category, count in category_stats.items():
    print(f"  - {category}: {count} items")

"""
example input
{"Query": "Remove ju1.s4.s2c1 from the graph. Rank direct child nodes of ju1.s4 in the updated graph based on physical_capacity_bps attribute. Return a list of tuple, each tuple has node name and its total physical capacity.", "Label": "capacity planning, level-2, remove-rank", "Result-Correctness": "Pass", "Result-Safety": "Pass", "GT-Result-Safety": "Pass", "Result-Latency": 5.123737812042236, "Ground truth code": "def ground_truth_process_graph(graph_data):\n                                    child_node_name = 'ju1.s4.s2c1'\n                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)\n                                    parent_node_name = 'ju1.s4'\n                                    ranked_child_nodes = solid_step_rank_child_nodes(graph_data, parent_node_name)\n                                    return_object = {'type': 'list', 'data': ranked_child_nodes}\n                                    return return_object", "LLM code": "def process_graph(graph_data):\n            # Create a copy of the graph before modifications\n            graph_copy = copy.deepcopy(graph_data)\n            \n            # Remove the specified node\n            node_to_remove = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4.s2c1':\n                    node_to_remove = node[0]\n                    break\n            \n            if node_to_remove:\n                graph_copy.remove_node(node_to_remove)\n            \n            # Find direct child nodes of ju1.s4\n            parent_node = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4':\n                    parent_node = node[0]\n                    break\n            \n            child_nodes = []\n            if parent_node:\n                for successor in graph_copy.successors(parent_node):\n                    child_nodes.append(successor)\n            \n            # Calculate total physical capacity for each child node\n            def calculate_total_capacity(node):\n                total_capacity = 0\n                for successor in nx.descendants(graph_copy, node):\n                    if 'EK_PORT' in graph_copy.nodes[successor]['type']:\n                        total_capacity += graph_copy.nodes[successor].get('physical_capacity_bps', 0)\n                return total_capacity\n            \n            child_capacities = []\n            for child in child_nodes:\n                child_name = graph_copy.nodes[child]['name']\n                total_capacity = calculate_total_capacity(child)\n                child_capacities.append((child_name, total_capacity))\n            \n            # Sort the child nodes based on total physical capacity\n            child_capacities.sort(key=lambda x: x[1], reverse=True)\n            \n            # Return the result as a list of tuples\n            return_object = {\n                'type': 'list',\n                'data': child_capacities\n            }\n            \n            return return_object", "Ground truth exec": [["ju1.s4.s1c1", 16000000000.0], ["ju1.s4.s1c2", 16000000000.0], ["ju1.s4.s1c3", 16000000000.0], ["ju1.s4.s1c5", 16000000000.0], ["ju1.s4.s1c6", 16000000000.0], ["ju1.s4.s1c7", 16000000000.0], ["ju1.s4.s1c8", 16000000000.0], ["ju1.s4.s2c2", 16000000000.0], ["ju1.s4.s2c3", 16000000000.0], ["ju1.s4.s2c4", 16000000000.0], ["ju1.s4.s2c5", 16000000000.0], ["ju1.s4.s2c6", 16000000000.0], ["ju1.s4.s2c8", 16000000000.0], ["ju1.s4.s1c4", 15000000000.0], ["ju1.s4.s2c7", 15000000000.0]], "LLM code exec": [["ju1.s4.s1c1", 16000000000.0], ["ju1.s4.s1c2", 16000000000.0], ["ju1.s4.s1c3", 16000000000.0], ["ju1.s4.s1c5", 16000000000.0], ["ju1.s4.s1c6", 16000000000.0], ["ju1.s4.s1c7", 16000000000.0], ["ju1.s4.s1c8", 16000000000.0], ["ju1.s4.s2c2", 16000000000.0], ["ju1.s4.s2c3", 16000000000.0], ["ju1.s4.s2c4", 16000000000.0], ["ju1.s4.s2c5", 16000000000.0], ["ju1.s4.s2c6", 16000000000.0], ["ju1.s4.s2c8", 16000000000.0], ["ju1.s4.s1c4", 15000000000.0], ["ju1.s4.s2c7", 15000000000.0]]}
"""


