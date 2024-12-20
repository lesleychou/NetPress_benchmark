import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import json
from solid_step_helper import get_node_value_ranges, getGraphData, \
solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, solid_step_list_child_nodes, solid_step_update_node_value, solid_step_rank_child_nodes
from dy_query_generation import QueryGenerator


def main():
    query_generator = QueryGenerator()
    query_generator.generate_queries(num_each_type=2)
    dynamic_dataset_path = 'data/benchmark_malt.jsonl'
    query_generator.save_queries_to_file(dynamic_dataset_path)

# run the main function
if __name__ == "__main__":
    main()