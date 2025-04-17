import jsonlines
import json
from collections import defaultdict

path = "logs/AzureGPT4Agent_few_shot_semantic/gpt4o_fewshot_semantic_50.jsonl"

# Function to extract data and create fine-tuning format
def extract_finetune_data(input_path):
    # Dictionary to track items per sub-category
    subcategory_items = defaultdict(list)
    subcategory_counts = defaultdict(int)
    all_results = []
    global_count = 0
    
    # First pass: Read the JSONL file and identify all sub-categories
    with jsonlines.open(input_path) as reader:
        for item in reader:
            label = item.get("Label", "")
            
            # Check if this is a capacity planning, level-2 item
            if label.startswith("capacity planning, level-2"):
                # Extract the sub-category (everything after "capacity planning, level-2")
                parts = label.split(", ")
                if len(parts) > 2:
                    subcategory = parts[2]  # Get the third part of the label
                    
                    # Check if this item passes and we need more of this subcategory
                    if (item.get("Result-Correctness") == "Pass" and 
                        subcategory_counts[subcategory] < 10):
                        
                        # Create the conversation format
                        conversation_item = {
                            "id": f"identity_{global_count}",
                            "conversations": [
                                {
                                    "from": "user",
                                    "value": item.get("Query", "")
                                },
                                {
                                    "from": "assistant",
                                    "value": item.get("LLM code", "")
                                }
                            ]
                        }
                        
                        # Add to the subcategory items
                        subcategory_items[subcategory].append(conversation_item)
                        subcategory_counts[subcategory] += 1
                        global_count += 1
    
    # Combine all subcategory items into one list
    for subcategory, items in subcategory_items.items():
        all_results.extend(items)
    
    return all_results, dict(subcategory_counts)

# Extract the data
finetune_data, subcategory_stats = extract_finetune_data(path)

# Write the output to a file
output_path = "capacity_planning_finetune_data.json"
with open(output_path, "w") as f:
    json.dump(finetune_data, f, indent=2)

# Print statistics
print(f"Extracted {len(finetune_data)} total items for fine-tuning to {output_path}")
print("Items per subcategory:")
for subcategory, count in subcategory_stats.items():
    print(f"  - {subcategory}: {count} items")

"""
example input
{"Query": "Remove ju1.s4.s2c1 from the graph. Rank direct child nodes of ju1.s4 in the updated graph based on physical_capacity_bps attribute. Return a list of tuple, each tuple has node name and its total physical capacity.", "Label": "capacity planning, level-2, remove-rank", "Result-Correctness": "Pass", "Result-Safety": "Pass", "GT-Result-Safety": "Pass", "Result-Latency": 5.123737812042236, "Ground truth code": "def ground_truth_process_graph(graph_data):\n                                    child_node_name = 'ju1.s4.s2c1'\n                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)\n                                    parent_node_name = 'ju1.s4'\n                                    ranked_child_nodes = solid_step_rank_child_nodes(graph_data, parent_node_name)\n                                    return_object = {'type': 'list', 'data': ranked_child_nodes}\n                                    return return_object", "LLM code": "def process_graph(graph_data):\n            # Create a copy of the graph before modifications\n            graph_copy = copy.deepcopy(graph_data)\n            \n            # Remove the specified node\n            node_to_remove = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4.s2c1':\n                    node_to_remove = node[0]\n                    break\n            \n            if node_to_remove:\n                graph_copy.remove_node(node_to_remove)\n            \n            # Find direct child nodes of ju1.s4\n            parent_node = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4':\n                    parent_node = node[0]\n                    break\n            \n            child_nodes = []\n            if parent_node:\n                for successor in graph_copy.successors(parent_node):\n                    child_nodes.append(successor)\n            \n            # Calculate total physical capacity for each child node\n            def calculate_total_capacity(node):\n                total_capacity = 0\n                for successor in nx.descendants(graph_copy, node):\n                    if 'EK_PORT' in graph_copy.nodes[successor]['type']:\n                        total_capacity += graph_copy.nodes[successor].get('physical_capacity_bps', 0)\n                return total_capacity\n            \n            child_capacities = []\n            for child in child_nodes:\n                child_name = graph_copy.nodes[child]['name']\n                total_capacity = calculate_total_capacity(child)\n                child_capacities.append((child_name, total_capacity))\n            \n            # Sort the child nodes based on total physical capacity\n            child_capacities.sort(key=lambda x: x[1], reverse=True)\n            \n            # Return the result as a list of tuples\n            return_object = {\n                'type': 'list',\n                'data': child_capacities\n            }\n            \n            return return_object", "Ground truth exec": [["ju1.s4.s1c1", 16000000000.0], ["ju1.s4.s1c2", 16000000000.0], ["ju1.s4.s1c3", 16000000000.0], ["ju1.s4.s1c5", 16000000000.0], ["ju1.s4.s1c6", 16000000000.0], ["ju1.s4.s1c7", 16000000000.0], ["ju1.s4.s1c8", 16000000000.0], ["ju1.s4.s2c2", 16000000000.0], ["ju1.s4.s2c3", 16000000000.0], ["ju1.s4.s2c4", 16000000000.0], ["ju1.s4.s2c5", 16000000000.0], ["ju1.s4.s2c6", 16000000000.0], ["ju1.s4.s2c8", 16000000000.0], ["ju1.s4.s1c4", 15000000000.0], ["ju1.s4.s2c7", 15000000000.0]], "LLM code exec": [["ju1.s4.s1c1", 16000000000.0], ["ju1.s4.s1c2", 16000000000.0], ["ju1.s4.s1c3", 16000000000.0], ["ju1.s4.s1c5", 16000000000.0], ["ju1.s4.s1c6", 16000000000.0], ["ju1.s4.s1c7", 16000000000.0], ["ju1.s4.s1c8", 16000000000.0], ["ju1.s4.s2c2", 16000000000.0], ["ju1.s4.s2c3", 16000000000.0], ["ju1.s4.s2c4", 16000000000.0], ["ju1.s4.s2c5", 16000000000.0], ["ju1.s4.s2c6", 16000000000.0], ["ju1.s4.s2c8", 16000000000.0], ["ju1.s4.s1c4", 15000000000.0], ["ju1.s4.s2c7", 15000000000.0]]}
"""

"""
example output

[
    {
      "id": "identity_0",
      "conversations": [
        {
          "from": "user",
          "value": "Remove ju1.s4.s2c1 from the graph. Rank direct child nodes of ju1.s4 in the updated graph based on physical_capacity_bps attribute. Return a list of tuple, each tuple has node name and its total physical capacity."
        },
        {
          "from": "assistant",
          "value": "def process_graph(graph_data):\n            # Create a copy of the graph before modifications\n            graph_copy = copy.deepcopy(graph_data)\n            \n            # Remove the specified node\n            node_to_remove = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4.s2c1':\n                    node_to_remove = node[0]\n                    break\n            \n            if node_to_remove:\n                graph_copy.remove_node(node_to_remove)\n            \n            # Find direct child nodes of ju1.s4\n            parent_node = None\n            for node in graph_copy.nodes(data=True):\n                if node[1]['name'] == 'ju1.s4':\n                    parent_node = node[0]\n                    break\n            \n            child_nodes = []\n            if parent_node:\n                for successor in graph_copy.successors(parent_node):\n                    child_nodes.append(successor)\n            \n            # Calculate total physical capacity for each child node\n            def calculate_total_capacity(node):\n                total_capacity = 0\n                for successor in nx.descendants(graph_copy, node):\n                    if 'EK_PORT' in graph_copy.nodes[successor]['type']:\n                        total_capacity += graph_copy.nodes[successor].get('physical_capacity_bps', 0)\n                return total_capacity\n            \n            child_capacities = []\n            for child in child_nodes:\n                child_name = graph_copy.nodes[child]['name']\n                total_capacity = calculate_total_capacity(child)\n                child_capacities.append((child_name, total_capacity))\n            \n            # Sort the child nodes based on total physical capacity\n            child_capacities.sort(key=lambda x: x[1], reverse=True)\n            \n            # Return the result as a list of tuples\n            return_object = {\n                'type': 'list',\n                'data': child_capacities\n            }\n            \n            return return_object"
        }
      ]
    },
]

"""

