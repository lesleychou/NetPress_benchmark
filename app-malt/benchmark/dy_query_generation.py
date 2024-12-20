import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import json
from solid_step_helper import get_node_value_ranges, getGraphData, \
solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, solid_step_list_child_nodes, solid_step_update_node_value, solid_step_rank_child_nodes

class QueryGenerator:
    def __init__(self,):
        _, self.malt_real_graph = getGraphData()
        node_value_ranges_path = 'data/node_value_ranges.json'
        self.node_value_ranges = get_node_value_ranges(self.malt_real_graph, node_value_ranges_path)
        self.queries = []

    def generate_level_1_query(self, operation_type='add'):
        if operation_type == 'add':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_name = f"new_{child_node}_{random.randint(1, 100)}"
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Add new node with name {child_node_name} type {child_node}, to {parent_node_name}. Return a graph."
            new_node = {'name': child_node_name, 'type': child_node}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                        new_node = {new_node}
                        parent_node_name = '{parent_node_name}'
                        graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)
                        return_object = {{'type': 'graph', 'data': graph_data}}
                        return return_object"""
            return template, ground_truth, new_node

        elif operation_type == 'remove':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            child_node_name = random.choice(self.node_value_ranges[child_node])

            template = f"Remove {child_node_name} from the graph. Return a graph."
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    child_node_name = '{child_node_name}'
                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)
                                    return_object = {{'type': 'graph', 'data': graph_data}}
                                    return return_object"""
            return template, ground_truth, child_node_name

        elif operation_type == 'count':
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            child_node_type = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Count the {child_node_type} in the {parent_node_name}. Return only the count number."
            node1 = {'type': parent_node, 'name': parent_node_name}
            node2 = {'type': child_node_type, 'name': None}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    node1 = {node1}
                                    node2 = {node2}
                                    count = solid_step_counting_query(graph_data, node1, node2)
                                    return_object = {{'type': 'text', 'data': count}}
                                    return return_object"""
            return template, ground_truth, None

        elif operation_type == 'list':
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN', 'EK_RACK', 'EK_PACKET_SWITCH'])
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"List all the child nodes of {parent_node_name}. Return a list of child nodes."
            node = {'type': parent_node, 'name': parent_node_name}
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                        node = {node}
                        child_nodes = solid_step_list_child_nodes(graph_data, node)
                        return_object = {{'type': 'list', 'data': child_nodes}}
                        return return_object"""
            return template, ground_truth, None

        elif operation_type == 'update':
            child_node = random.choice(['EK_PACKET_SWITCH', 'EK_PORT'])
            child_node_name = random.choice(self.node_value_ranges[child_node])
            new_value = random.randint(1, 100)

            template = f"Update the value of {child_node_name} to {new_value}. Return a graph."
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                    child_node_name = '{child_node_name}'
                                    new_value = {new_value}
                                    graph_data = solid_step_update_node_value(graph_data, child_node_name, new_value)
                                    return_object = {{'type': 'graph', 'data': graph_data}}
                                    return return_object"""
            return template, ground_truth, child_node_name

        elif operation_type == 'rank':
            parent_node = random.choice(['EK_AGG_BLOCK', 'EK_CONTROL_DOMAIN'])
            parent_node_name = random.choice(self.node_value_ranges[parent_node])

            template = f"Rank all child nodes of {parent_node} type {parent_node_name} based on physical_capacity_bps attribute. Return a list of child nodes name."
            ground_truth = f"""def ground_truth_process_graph(graph_data):
                                parent_node_name = '{parent_node_name}'
                                ranked_child_nodes = solid_step_rank_child_nodes(graph_data, parent_node_name)
                                return_object = {{'type': 'list', 'data': ranked_child_nodes}}
                                return return_object"""
            return template, ground_truth, None

    def generate_queries(self, num_each_type=3):
        for _ in range(num_each_type):
            query, ground_truth, new_node = self.generate_level_1_query(operation_type='update')
            self.queries.append({
                "messages": [
                    {"question": query},
                    {"answer": ground_truth},
                    {"task_label": "capacity planning, level-1, update"}
                ]
            })

        for _ in range(num_each_type):
            query, ground_truth, new_node = self.generate_level_1_query(operation_type='add')
            self.queries.append({
                "messages": [
                    {"question": query},
                    {"answer": ground_truth},
                    {"task_label": "capacity planning, level-1, add"}
                ]
            })

        for _ in range(num_each_type):
            query, ground_truth, new_node = self.generate_level_1_query(operation_type='count')
            self.queries.append({
                "messages": [
                    {"question": query},
                    {"answer": ground_truth},
                    {"task_label": "capacity planning, level-1, count"}
                ]
            })

        for _ in range(num_each_type):
            query, ground_truth, new_node = self.generate_level_1_query(operation_type='remove')
            self.queries.append({
                "messages": [
                    {"question": query},
                    {"answer": ground_truth},
                    {"task_label": "capacity planning, level-1, remove"}
                ]
            })

    def save_queries_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for item in self.queries:
                f.write(json.dumps(item) + "\n")

# Usage
# query_generator = QueryGenerator()
# query_generator.generate_queries()
# query_generator.save_queries_to_file('data/benchmark_level_1.jsonl')
