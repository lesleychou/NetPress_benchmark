import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import json
from helper import get_node_value_ranges, getGraphData, solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, solid_step_list_child_nodes, solid_step_update_node_value

# plot a networkx graph, that each node represent an entity and each edge represent a relationship
# JUPITER contains SPINEBLOCK
# AGG_BLOCK contains PACKET_SWITCH
# CHASSIS contains CONTROL_POINT
# CONTROL_POINT contains PACKET_SWITCH
# RACK contains CHASSIS
# PACKET_SWITCH contains PORT
# SPINEBLOCK contains PACKET_SWITCH
# CONTROL_DOMAIN contains CONTROL_POINT
# CHASSIS contains PACKET_SWITCH
# JUPITER contains SUPERBLOCK
# SUPERBLOCK contains AGG_BLOCK
# def plot_abstraction_graph(relationships, value_dict, attr_dict=None, attr_value_example_dict=None):
#     G = nx.DiGraph()
#     for relationship in relationships:
#         source, target = relationship.split(' contains ')
#         G.add_edge(source, target)
#     for node, value in value_dict.items():
#         G.nodes[node]['value'] = value
#     # add attributes to the nodes
#     for node in G.nodes:
#         if node in attr_dict:
#             G.nodes[node][attr_dict[node]] = attr_value_example_dict[attr_dict[node]]

#     # print(G.nodes(data=True))
#     return G

# relationships = ['JUPITER contains SPINE_BLOCK', 
#                  'AGG_BLOCK contains PACKET_SWITCH', 
#                  'CHASSIS contains CONTROL_POINT', 
#                  'CONTROL_POINT contains PACKET_SWITCH', 
#                  'RACK contains CHASSIS', 
#                  'PACKET_SWITCH contains PORT', 
#                  'SPINE_BLOCK contains PACKET_SWITCH', 
#                  'CONTROL_DOMAIN contains CONTROL_POINT', 
#                  'CHASSIS contains PACKET_SWITCH', 
#                  'JUPITER contains SUPER_BLOCK', 
#                  'SUPER_BLOCK contains AGG_BLOCK']

# node_value_example_dict = {
#     'JUPITER': 'ju1',
#     'SPINE_BLOCK': 'ju1.s1',
#     'AGG_BLOCK': 'ju1.a1.m1',
#     'PACKET_SWITCH': 'ju1.a1.m1.s2c1',
#     'CHASSIS': 'ju1.a1.m1.s2c1',
#     'CONTROL_POINT': 'ju1.a1.m1.s2c1',
#     'RACK': 'ju1.a1.m1rack',
#     'PORT': 'ju1.a1.m1.s2c1.p1',
#     'CONTROL_DOMAIN': 'ju1.a1.dom',
#     'SUPER_BLOCK': 'ju1.a1',
# }

# attr_dict = {'PACKET_SWITCH': 'switch_loc',
#              'PORT': 'physical_capacity_mbps'}

# attr_value_example_dict = { 'switch_loc': 1,
#                             'physical_capacity_mbps': 1000 }

# malt_abstraction_graph = plot_abstraction_graph(relationships, node_value_example_dict, attr_dict, attr_value_example_dict)


# # operation_type = ['list', 'add', 'remove', 'update', 'rank', 'sum', 'average', 'traverse', 'cluster with louvain communities algorithm']
_, malt_real_graph = getGraphData()

# Level-1 query: only involve one operation. 
def genarate_level_1_query(node_value_ranges, operation_type='add'):
    """
    Generate template for a level-1 query based on some pre-defined constraints on the input.
    """
    if operation_type=='add':
        # Constraints for "add": the first node should be a new child node that does not exist in the node_value_ranges yet, 
        # the second node should be an existing parent node.
        # based on the relationships and node_value_example_dict, we can generate a random child node
        child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
        # the child_node_name should be a string start with 'new_' existing child node type, and a random number
        child_node_name = f"new_{child_node}_{random.randint(1, 100)}"
        # the parent node name should be an existing parent node name
        parent_node_name = random.choice(node_value_ranges[parent_node])

        template = f"Add {child_node_name} to {parent_node_name}. Return a graph."

        # genrate ground truth of the query based on solid_step_add_node_to_graph function
        new_node = {'name': child_node_name, 'type': child_node}
        # ground truth is a python function in string, that name and input is ground_truth_process_graph(graph_data). 
        # it use existing solid steps and replace all the variable names with the corresponding values in the template
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                new_node = {new_node}
                                parent_node_name = '{parent_node_name}'
                                graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                                return graph_data"""
        return template, ground_truth, new_node
    
    elif operation_type=='remove':
        # Constraints for "remove": the node should be a child node in MALT graph.
        # based on the relationships and node_value_example_dict, we can generate a random child node
        child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        # the child_node_name should be an existing child node name
        child_node_name = random.choice(node_value_ranges[child_node])

        template = f"Remove {child_node_name} from the graph. Return a graph."

        # genrate ground truth of the query based on solid_step_remove_node_from_graph function
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                child_node_name = '{child_node_name}'
                                graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)
                                return graph_data"""
        return template, ground_truth, child_node_name
    
    elif operation_type=='count':
        # Constraints for "count": the first node should be a parent node in MALT graph, the second node should be a child node type in MALT graph.
        # based on the relationships and node_value_example_dict, we can generate a random parent node and child node
        parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
        child_node_type = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        # the parent node name should be an existing parent node name
        parent_node_name = random.choice(node_value_ranges[parent_node])

        template = f"Count the {child_node_type} in the {parent_node_name}. Return only the count number."

        # genrate ground truth of the query based on solid_step_counting_query function
        node1 = {'type': parent_node, 'name': parent_node_name}
        node2 = {'type': child_node_type, 'name': None}
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                node1 = {node1}
                                node2 = {node2}
                                count = solid_step_counting_query(graph_data, node1, node2)
                                return count"""
        return template, ground_truth, None
    
    elif operation_type=='list':
        # Constraints for "list": the node should be a parent node in MALT graph.
        # based on the relationships and node_value_example_dict, we can generate a random parent node
        parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN', 'EK_RACK', 'EK_PACKET_SWITCH'])
        # the parent node name should be an existing parent node name
        parent_node_name = random.choice(node_value_ranges[parent_node])

        template = f"List all the child nodes of {parent_node_name}. Return a list of child nodes."

        # genrate ground truth of the query based on solid_step_list_child_nodes function
        node = {'type': parent_node, 'name': parent_node_name}
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                node = {node}
                                child_nodes = solid_step_list_child_nodes(graph_data, node)
                                return child_nodes"""
        return template, ground_truth, None
    
    elif operation_type=='update':
        # Constraints for "update": the node should be a child node in MALT graph.
        # based on the relationships and node_value_example_dict, we can generate a random child node
        child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        # the child_node_name should be an existing child node name
        child_node_name = random.choice(node_value_ranges[child_node])
        # the new value should be a random value
        new_value = random.randint(1, 100)

        template = f"Update the value of {child_node_name} to {new_value}. Return a graph."

        # genrate ground truth of the query based on solid_step_update_node_value function
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                child_node_name = '{child_node_name}'
                                new_value = {new_value}
                                graph_data = solid_step_update_node_value(graph_data, child_node_name, new_value)
                                return graph_data"""
        return template, ground_truth, child_node_name


def genarate_level_2_query_sequential(node_value_ranges, operation_type_1='add', operation_type_2='count'):
    """
    Level-2 query: two operations, control sequence is sequential.
    """
    if operation_type_1=='add' and operation_type_2=='count':
        # Constraints for "add" and "count": the first node should be a new child node that does not exist in the node_value_ranges yet, 
        # the second node should be an existing parent node.
        # based on the relationships and node_value_example_dict, we can generate a random child node
        child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
        # the child_node_name should be a string start with 'new_' existing child node type, and a random number
        child_node_name = f"new_{child_node}_{random.randint(1, 100)}"
        # the parent node name should be an existing parent node name
        parent_node_name = random.choice(node_value_ranges[parent_node])

        template = f"Add {child_node_name} to {parent_node_name}. Count the {child_node} in {parent_node_name} in the updated graph. Return only the count number."

        # genrate ground truth of the query based on solid_step_add_node_to_graph and solid_step_counting_query functions
        new_node = {'name': child_node_name, 'type': child_node}
        # ground truth is a python function in string, that name and input is ground_truth_process_graph(graph_data). 
        # it use existing solid steps and replace all the variable names with the corresponding values in the template
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                new_node = {new_node}
                                parent_node_name = '{parent_node_name}'
                                graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                                node1 = {{"type": "{parent_node}", "name": "{parent_node_name}"}}
                                node2 = {{"type": "{child_node}", "name": None}}
                                count = solid_step_counting_query(graph_data, node1, node2)
                                return count"""
        return template, ground_truth, new_node
    
    elif operation_type_1=='remove' and operation_type_2=='count':
        # Constraints for "remove" and "count": the node should be a child node in MALT graph.
        # based on the relationships and node_value_example_dict, we can generate a random child node
        child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
        # the child_node_name should be an existing child node name
        child_node_name = random.choice(node_value_ranges[child_node])
        # the parent node name should be an existing parent node name
        parent_node_name = random.choice(node_value_ranges[parent_node])

        template = f"Remove {child_node_name} from the graph. Count the {child_node} in {parent_node_name} in the updated graph. Return only the count number."

        # genrate ground truth of the query based on solid_step_remove_node_from_graph and solid_step_counting_query functions
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                child_node_name = '{child_node_name}'
                                graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)
                                node1 = {{"type": "{parent_node}", "name": "{parent_node_name}"}}
                                node2 = {{"type": "{child_node}", "name": None}}
                                count = solid_step_counting_query(graph_data, node1, node2)
                                return count"""
        return template, ground_truth, child_node_name


# # run the ground truth function to verify the correctness
# node_value_ranges = get_node_value_ranges(malt_real_graph, 'data/node_value_ranges.json')
# query, ground_truth, new_node = genarate_level_2_query_sequential(node_value_ranges, operation_type_1='remove', operation_type_2='count')
# print(query, ground_truth)
# exec(ground_truth)
# new_malt_graph = eval("ground_truth_process_graph(malt_real_graph)")
# print(new_malt_graph)


# def genarate_level_2_query_for_loop(node_value_ranges, operation_type_1='add', operation_type_2='count'):
#     """
#     Level-2 query: two operations, control sequence is for-loop.
#     For each parent node in the graph, add a new child node to it. Count the total number of child nodes in the updated graph. Return the counts.
def genarate_level_2_query_for_loop(node_value_ranges, operation_type_1='add', operation_type_2='count'):
    """
    Level-2 query: two operations, control sequence is for-loop.
    For each parent node in the graph, add a new child node to it. Count the total number of child nodes in the updated graph. Return the counts.
    """
    if operation_type_1 == 'add' and operation_type_2 == 'count':
        parent_node_type = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
        child_node_type = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        parent_node_names = node_value_ranges[parent_node_type]

        template = f"For each {parent_node_type}, add a new {child_node_type} to it. Count the total number of {child_node_type} in the updated graph. Return the counts."
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                total_count = 0
                                for parent_node_name in {parent_node_names}:
                                    new_node = {{"name": f"new_{child_node_type}_{{random.randint(1, 100)}}", "type": "{child_node_type}"}}
                                    node1 = {{"type": "{parent_node_type}", "name": parent_node_name}}
                                    node2 = {{"type": "{child_node_type}", "name": None}}
                                    graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                                    count = solid_step_counting_query(graph_data, node1, node2)
                                    total_count += count
                                return total_count"""
        return template, ground_truth, None
    
    # TODO: need further debug
    elif operation_type_1 == 'remove' and operation_type_2 == 'count':
        child_node_type = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
        parent_node_type = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
        parent_node_names = node_value_ranges['EK_AGG_BLOCK']

        template = f"For each {parent_node_type}, remove a {child_node_type} from it. Count the total number of {child_node_type} in the updated graph. Return the counts."
        ground_truth = f"""def ground_truth_process_graph(graph_data):
                                total_count = 0
                                for parent_node_name in {parent_node_names}:
                                    node1 = {{"type": "EK_AGG_BLOCK", "name": parent_node_name}}
                                    node2 = {{"type": "{child_node_type}", "name": None}}
                                    graph_data = solid_step_remove_node_from_graph(graph_data, parent_node_name)
                                    count = solid_step_counting_query(graph_data, node1, node2)
                                    total_count += count
                                return total_count"""
        return template, ground_truth, None


# # run the ground truth function to verify the correctness
# node_value_ranges = get_node_value_ranges(malt_real_graph, 'data/node_value_ranges.json')
# query, ground_truth, _  = genarate_level_2_query_for_loop(node_value_ranges, operation_type_1='remove', operation_type_2='count')
# print(query, ground_truth)
# exec(ground_truth)
# new_malt_graph = eval("ground_truth_process_graph(malt_real_graph)")
# print(new_malt_graph)
    

# run genarate_level_1_query 10 times and write the query and ground_truth to a jsonl file, with the format of {"messages": [{"question": }, {"answer": }, {"task": "capacity planning"}]}
queries = []
NUM_EACH_TYPE = 5
node_value_ranges = get_node_value_ranges(malt_real_graph, 'data/node_value_ranges.json')
# run the ground truth function to verify the correctness
query, ground_truth, new_node = genarate_level_1_query(node_value_ranges, operation_type='update')
print(query, ground_truth, new_node)
exec(ground_truth)
new_malt_graph = eval("ground_truth_process_graph(malt_real_graph)")
print(new_malt_graph)

for _ in range(NUM_EACH_TYPE):
    query, ground_truth, new_node = genarate_level_1_query(node_value_ranges, operation_type='update')
    queries.append({
        "messages": [
            {"question": query},
            {"answer": ground_truth},
            {"task": "capacity planning, level-1, update"}
        ]
    })

# for _ in range(NUM_EACH_TYPE):
#     query, ground_truth, new_node = genarate_level_1_query(node_value_ranges, operation_type='list')
#     queries.append({
#         "messages": [
#             {"question": query},
#             {"answer": ground_truth},
#             {"task": "capacity planning, level-1, list"}
#         ]
#     })

# for _ in range(NUM_EACH_TYPE):
#     query, ground_truth, new_node = genarate_level_1_query(node_value_ranges, operation_type='add')
#     queries.append({
#         "messages": [
#             {"question": query},
#             {"answer": ground_truth},
#             {"task": "capacity planning, level-1, add"}
#         ]
#     })
# for _ in range(NUM_EACH_TYPE):
#     query, ground_truth, new_node = genarate_level_1_query(node_value_ranges, operation_type='count')
#     queries.append({
#         "messages": [
#             {"question": query},
#             {"answer": ground_truth},
#             {"task": "capacity planning, level-1, count"}
#         ]
#     })
# for _ in range(NUM_EACH_TYPE):
#     query, ground_truth, new_node = genarate_level_1_query(node_value_ranges, operation_type='remove')
#     queries.append({
#         "messages": [
#             {"question": query},
#             {"answer": ground_truth},
#             {"task": "capacity planning, level-1, remove"}
#         ]
#     })

# for _ in range(NUM_EACH_TYPE):
#     query, ground_truth, new_node = genarate_level_2_query_sequential(node_value_ranges, operation_type_1='add', operation_type_2='count')
#     queries.append({
#         "messages": [
#             {"question": query},
#             {"answer": ground_truth},
#             {"task": "capacity planning, level-2 sequential, add then count"}
#         ]
#     })

# for _ in range(NUM_EACH_TYPE):
#     query, ground_truth, new_node = genarate_level_2_query_sequential(node_value_ranges, operation_type_1='remove', operation_type_2='count')
#     queries.append({
#         "messages": [
#             {"question": query},
#             {"answer": ground_truth},
#             {"task": "capacity planning, level-2 sequential, remove then count"}
#         ]
#     })

# for _ in range(NUM_EACH_TYPE):
#     query, ground_truth, new_node = genarate_level_2_query_for_loop(node_value_ranges, operation_type_1='add', operation_type_2='count')
#     queries.append({
#         "messages": [
#             {"question": query},
#             {"answer": ground_truth},
#             {"task": "capacity planning, level-2 for-loop, add then count"}
#         ]
    # })

with open('data/benchmark_level_1.jsonl', 'w') as f:
    for item in queries:
        f.write(json.dumps(item) + "\n")



# Level-2.1 query: involve two operations. Sequential relations.
# Add _ to _. Count _ in the updated graph.
# Level-2.2 query: involve two operations. If-else relations. For-loop relations.
# Level-3 query: involve three operations.