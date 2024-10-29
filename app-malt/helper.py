import json
import traceback
from dotenv import load_dotenv
import openai
import pandas as pd
from prototxt_parser.prototxt import parse
from collections import Counter
import os
import networkx as nx
import jsonlines
import random
from networkx.readwrite import json_graph
from langchain.callbacks import get_openai_callback
import json
import re
import time
import sys
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt
from azure.core.credentials import AzureKeyCredential

# Load environ variables from .env, will not override existing environ variables
load_dotenv()

def getGraphData():
    input_string = open("data/malt-example-final.textproto.txt").read()
    parsed_dict = parse(input_string)

    # Load MALT data
    G = nx.DiGraph()

    # Insert all the entities as nodes
    for entity in parsed_dict['entity']:
        # Check if the node exists
        if entity['id']['name'] not in G.nodes:
            G.add_node(entity['id']['name'], type=[entity['id']['kind']], name=entity['id']['name'])
        else:
            G.nodes[entity['id']['name']]['type'].append(entity['id']['kind'])
        # Add all the attributes
        for key, value in entity.items():
            if key == 'id':
                continue
            for k, v in value.items():
                G.nodes[entity['id']['name']][k] = v

    # Insert all the relations as edges
    for relation in parsed_dict['relationship']:
        G.add_edge(relation['a']['name'], relation['z']['name'], type=relation['kind'])

    rawData = json_graph.node_link_data(G)

    return rawData, G

def node_attributes_are_equal(node1_attrs, node2_attrs):
    # Check if both nodes have the exact same set of attributes
    if set(node1_attrs.keys()) != set(node2_attrs.keys()):
        return False

    # Check if all attribute values are equal
    for attr_name, attr_value in node1_attrs.items():
        if attr_value != node2_attrs[attr_name]:
            return False

    return True

def clean_up_llm_output_func(answer):
    '''
    Extract only the def process_graph() funtion from the output of LLM
    :param answer: output of LLM
    :return: cleaned function
    '''
    start = answer.find("def process_graph")
    end = -1
    index = 0
    for _ in range(2):  # change the number 2 to any 'n' to find the nth occurrence
        end = answer.find("```", index)
        index = end + 1
    clean_code = answer[start:end].strip()
    return clean_code

def check_list_equal(lst1, lst2):
    if lst1 and isinstance(lst1[0], list):
        return Counter(json.dumps(i) for i in lst1) == Counter(json.dumps(i) for i in lst2)
    else:
        return Counter(lst1) == Counter(lst2)


def clean_up_output_graph_data(ret):
    if isinstance(ret['data'], nx.Graph):
        # Create a nx.graph copy, so I can compare two nx.graph later directly
        ret_graph_copy = ret['data']
        jsonGraph = nx.node_link_data(ret['data'])
        ret['data'] = jsonGraph

    else:  # Convert the jsonGraph back to nx.graph, to check if they are identical later
        ret_graph_copy = json_graph.node_link_graph(ret['data'])

    return ret_graph_copy