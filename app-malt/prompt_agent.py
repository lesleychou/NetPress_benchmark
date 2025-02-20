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
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain 
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)


class BasePromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        prompt_prefix = """
        Generate the Python code needed to process the network graph to answer the user question or request. The network graph data is stored as a networkx graph object, the Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data and returns a single object return_object. The input argument graph_data will be a networkx graph object with nodes and edges.
        The graph is directed and each node has a 'name' attribute to represent itself.
        Each node has a 'type' attribute, in the format of EK_TYPE. 'type' must be a list, can include ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN'].
        Each node can have other attributes depending on its type.
        Each directed edge also has a 'type' attribute, include RK_CONTAINS, RK_CONTROL.
        You should check relationship based on edge, check name based on node attribute. 
        Nodes has hierarchy: CHASSIS contains PACKET_SWITCH, JUPITER contains SUPERBLOCK, SUPERBLOCK contains AGG_BLOCK, AGG_BLOCK contains PACKET_SWITCH, PACKET_SWITCH contains PORT
        Adding new nodes need to consider attributes of the new node. Also consider adding edges based on their relationship with existing nodes. 
        The name to add on each layer can be inferred from new node name string.
        When adding new nodes, you should also add edges based on their relationship with existing nodes. 
        Each PORT node has an attribute 'physical_capacity_bps'. For example, a PORT node name is ju1.a1.m1.s2c1.p3. 
        When you add a new packet switch, should add a new port to it and use the default physical_capacity_bps as 1000.
        When calculating capacity of a node, you need to sum the physical_capacity_bps on the PORT of each hierarchy contains in this node.
        When creating a new graph, need to filter nodes and edges with attributes from the original graph. 
        When update a graph, always create a graph copy, do not modify the input graph. 
        To find node based on type, check the name and type list. For example, [node[0] == 'ju1.a1.m1.s2c1' and 'EK_PACKET_SWITCH' in node[1]['type']].

        Do not use multi-layer function. The output format should only return one object. The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table' or 'graph'.
        The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. If the output type is 'list' then the 'data' key should contain a list of items.
        If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.If the output type is 'graph' then the 'data' key should be a graph json "graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)".
        node.startswith will not work for the node name. you have to check the node name with the node['name'].

        Context: When the user requests to make changes to the graph, it is generally appropriate to return the graph. 
        In the Python code you generate, you should process the networkx graph object to produce the needed output.

        Remember, your reply should always start with string "\nAnswer:\n", and you should generate a function called "def process_graph".
        All of your output should only contain the defined function without example usages, no additional text, and display in a Python code block.
        Do not include any package import in your answer.
        """
        return prompt_prefix    


class CoTPromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        cot_prompt_prefix = """
        Please think of the problem step by step based on the following instructions:

        Generate the Python code needed to process the network graph to answer the user question or request. The network graph data is stored as a networkx graph object, the Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data and returns a single object return_object. The input argument graph_data will be a networkx graph object with nodes and edges.
        
        Steps to consider:
        1. First, understand what the input graph structure looks like:
           - The graph is directed with nodes having 'name' and 'type' attributes
           - Node types include EK_SUPERBLOCK, EK_CHASSIS, EK_RACK, etc.
           - Edges have types like RK_CONTAINS, RK_CONTROL
        
        2. Consider the hierarchy relationships:
           - CHASSIS contains PACKET_SWITCH
           - JUPITER contains SUPERBLOCK
           - SUPERBLOCK contains AGG_BLOCK
           - AGG_BLOCK contains PACKET_SWITCH
           - PACKET_SWITCH contains PORT

        3. When modifying the graph:
           - Create a copy of the graph before modifications
           - Consider all required attributes for new nodes
           - Add appropriate edges based on relationships
           - For new packet switches, add ports with default capacity 1000
        
        4. For capacity calculations:
           - Sum the physical_capacity_bps of PORTs in the hierarchy
           - Consider all contained nodes at each level

        5. Format the output appropriately:
           - Return a JSON object with 'type' and 'data' keys
           - Types can be: 'text', 'list', 'table', or 'graph'
           - Format data according to the specified type

        """
        return cot_prompt_prefix


class FewShotPromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        examples = [
                    {
                        "question": "Update the physical capacity value of ju1.s3.s2c2.p8 to 60. Return a graph.",
                        "answer": """def process_graph(graph_data):\n                                    child_node_name = 'ju1.s3.s2c2.p8'\n                                    new_value = 60\n                                    graph_data = solid_step_update_node_value(graph_data, child_node_name, new_value)\n                                    return_object = {'type': 'graph', 'data': graph_data}\n                                    return return_object""",
                    },
                    {
                        "question": "Add new node with name new_EK_PACKET_SWITCH_32 type EK_PACKET_SWITCH, to ju1.a4.m4. Return a graph.",
                        "answer": """def process_graph(graph_data):\n                        new_node = {'name': 'new_EK_PACKET_SWITCH_32', 'type': 'EK_PACKET_SWITCH'}\n                        parent_node_name = 'ju1.a4.m4'\n                        graph_data = solid_step_add_node_to_graph(graph_data, new_node, parent_node_name)\n                        return_object = {'type': 'graph', 'data': graph_data}\n                        return return_object""",
                    },
                    {
                        "question": "Count the EK_PACKET_SWITCH in the ju1.s3.dom. Return only the count number.",
                        "answer": """def process_graph(graph_data):\n                                    node1 = {'type': 'EK_CONTROL_DOMAIN', 'name': 'ju1.s3.dom'}\n                                    node2 = {'type': 'EK_PACKET_SWITCH', 'name': None}\n                                    count = solid_step_counting_query(graph_data, node1, node2)\n                                    return_object = {'type': 'text', 'data': count}\n                                    return return_object""",
                    },
                    {
                        "question": "Remove ju1.a1.m4.s3c6.p1 from the graph. Return a graph.",
                        "answer": """def process_graph(graph_data):\n                                    child_node_name = 'ju1.a1.m4.s3c6.p1'\n                                    graph_data = solid_step_remove_node_from_graph(graph_data, child_node_name)\n                                    return_object = {'type': 'graph', 'data': graph_data}\n                                    return return_object""",
                    },
                ]
        few_shot_prompt_prefix = """
        """