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
from langchain.callbacks import get_openai_callback
import json
import re
import time
import sys
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain 
from langchain.callbacks import get_openai_callback
# For GPT3.5 or GPT4
from langchain.chat_models import AzureChatOpenAI

credential = DefaultAzureCredential()

#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ztn-oai-fc.openai.azure.com/"

# Load environ variables from .env, will not override existing environ variables
load_dotenv()
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')

EACH_PROMPT_RUN_TIME = 1

# gpt-4o
# 2024-08-01-preview

class MaltAgent_GPT:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_type=OPENAI_API_TYPE,
            openai_api_base=OPENAI_API_BASE,
            openai_api_version="2024-08-01-preview",
            deployment_name='gpt-4o',
            model_name='gpt-4o',
            temperature=0.0,
            max_tokens=4000,
            )

        prefix = """
        Generate the Python code needed to process the network graph to answer the user question or request. The network graph data is stored as a networkx graph object, the Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data and returns a single object return_object. The input argument graph_data will be a networkx graph object with nodes and edges.
        The graph is directed and each node has a 'name' attribute to represent itself.
        Each node has a 'type' attribute, in the format of EK_TYPE. 'type' must be a list, can include ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN'].
        Each node can have other attributes depending on its type.
        Each directed edge also has a 'type' attribute, include RK_CONTAINS, RK_CONTROL.
        You should check relationship based on edge, check name based on node attribute. 
        Adding new nodes need to consider all hierarchy. For example, adding a new switch requires adding it to the corresponding jupiter, aggregation block, and domain. 
        The name to add on each layer can be inferred from new node name string.
        When adding new nodes, you should also add edges based on their relationship with existing nodes. 
        Each PORT node has an attribute 'physical_capacity_bps'. For example, a PORT node name is ju1.a1.m1.s2c1.p3. 
        When calculating capacity of a node, you need to sum the physical_capacity_bps on the PORT of each hierarchy contains in this node.
        Hierarchy: CHASSIS contains PACKET_SWITCH, JUPITER contains SUPERBLOCK, SUPERBLOCK contains AGG_BLOCK, AGG_BLOCK contains PACKET_SWITCH, PACKET_SWITCH contains PORT
        When creating a new graph, need to filter nodes and edges with attributes from the original graph. 
        When update a graph, always create a graph copy, do not modify the input graph. 
        packet switch nodes also have switch location attribute 'switch_loc' in node attribute 'packet_switch_attr'. 
        To find node based on type, check the name and type list. For example, [node[0] == 'ju1.a1.m1.s2c1' and 'EK_PACKET_SWITCH' in node[1]['type']].


        The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table' or 'graph'.
        The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. If the output type is 'list' then the 'data' key should contain a list of items.
        If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.If the output type is 'graph' then the 'data' key should contain a JSON object that can be rendered using D3.js.

        Context: When the user requests to make changes to the graph, it is generally appropriate to return the graph. 
        In the Python code you generate, you should process the networkx graph object to produce the needed output.

        Remember, your reply should always start with string "\nAnswer:\n".
        All of your output should only contain the defined function without example usages, no additional text, and display in a Python code block.
        Do not include any package import in your answer.
        """

        suffix = """Begin! Remember to ensure that you generate valid Python code in the following format:

        Answer:
        ```python
        ${{Code that will answer the user question or request}}
        ```
        Question: {input}
        """

        prompt = PromptTemplate(
            input_variables=["input"],
            template=prefix + suffix
        )

        self.pyGraphNetExplorer = LLMChain(llm=self.llm, prompt=prompt)

    def call_agent(self, query):
        print("Calling model")
        answer = self.pyGraphNetExplorer.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        print(code)
        return code