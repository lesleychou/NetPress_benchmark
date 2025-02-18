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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login
from vllm import LLM, SamplingParams

# Login huggingface
login(token="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")
# Load environ variables from .env, will not override existing environ variables
load_dotenv()

# For Google Gemini
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# For Azure OpenAI GPT4
from azure.identity import AzureCliCredential
from langchain.chat_models import AzureChatOpenAI
credential = AzureCliCredential()
#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ztn-oai-fc.openai.azure.com/"


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

prompt_suffix = """Begin! Remember to ensure that you generate valid Python code in the following format:

        Answer:
        ```python
        ${{Code that will answer the user question or request}}
        ```
        Question: {input}
        """



class GoogleGeminiAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        prompt = PromptTemplate(
            input_variables=["input"],
            template=prompt_prefix + prompt_suffix
        )

        self.pyGraphNetExplorer = LLMChain(llm=self.llm, prompt=prompt)

    def call_agent(self, query):
        print("Calling Google Gemini")
        answer = self.pyGraphNetExplorer.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        print("code:", code)
        return code


class AzureGPT4Agent:
    def __init__(self):
        # gpt-4o
        # 2024-08-01-preview
        self.llm = AzureChatOpenAI(
            openai_api_type="azure_ad",
            openai_api_version="2024-08-01-preview",
            deployment_name='gpt-4o',
            model_name='gpt-4o',
            temperature=0.0,
            max_tokens=4000,
            )

        prompt = PromptTemplate(
            input_variables=["input"],
            template=prompt_prefix + prompt_suffix
        )

        self.pyGraphNetExplorer = LLMChain(llm=self.llm, prompt=prompt)

    def call_agent(self, query):
        print("Calling GPT-4o")
        answer = self.pyGraphNetExplorer.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
    
class LlamaModel:
    def __init__(self):
        self.model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            device_map=self.device,
            cache_dir="/home/ubuntu"
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=self.quantization_config,
            cache_dir="/home/ubuntu"
        )
        self.prompt = prompt_prefix + prompt_suffix

    def call_agent(self, query):
        print("Calling Llama")
        prompt_text = self.prompt + query + " Please do not repeat the prompt text in your response, only give the format output."
        prompt_text = prompt_text.strip()
        print("prompt_text:", prompt_text)
        
        # Tokenize the prompt and get the input IDs
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_input_ids = prompt_tokens["input_ids"]
        start_index = prompt_input_ids.shape[-1]
        
        # Generate the output
        generated_ids = self.llm.generate(
            **prompt_tokens,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1
        )
        
        # Remove the prompt part from the generated output
        generation_output = generated_ids[0][start_index:]
        answer = self.tokenizer.decode(generation_output, skip_special_tokens=True)
        
        print("llm answer:", answer)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code

class QwQModel:
    def __init__(self):
        self.model_name = "Qwen/QwQ-32B-Preview"
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            device_map=self.device,
            cache_dir="/home/ubuntu"
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=self.quantization_config,
            cache_dir="/home/ubuntu"
        )
        self.prompt = prompt_prefix + prompt_suffix

    def call_agent(self, query):
        print("Calling QwQ")
        prompt_text = self.prompt + query + " Please do not repeat the prompt text in your response, only give the format output."
        prompt_text = prompt_text.strip()
        print("prompt_text:", prompt_text)
        
        # Tokenize the prompt and get the input IDs
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_input_ids = prompt_tokens["input_ids"]
        start_index = prompt_input_ids.shape[-1]
        
        # Generate the output
        generated_ids = self.llm.generate(
            **prompt_tokens,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1
        )
        
        # Remove the prompt part from the generated output
        generation_output = generated_ids[0][start_index:]
        answer = self.tokenizer.decode(generation_output, skip_special_tokens=True)
        
        print("llm answer:", answer)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
    
class QwenModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(
	        model=self.model_name,
            device=self.device,
	        quantization="gptq"  # Enable GPTQ 4-bit loading
	    )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )
        self.prompt = prompt_prefix + prompt_suffix

    def call_agent(self, query):
        print("Calling Qwen")
        prompt_text = self.prompt + query 
        print("prompt_text:", prompt_text)
        
        result = self.llm.generate([prompt_text], sampling_params=self.sampling_params)
        answer = result[0].outputs[0].text
  
        print("llm answer:", answer)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code

    
class Phi4Model:
    def __init__(self):
        self.model_name = "phi-4/Phi-4-70B-Instruct"
        self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            device_map=self.device,
            cache_dir="/home/ubuntu"
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=self.quantization_config,
            cache_dir="/home/ubuntu"
        )
        self.prompt = prompt_prefix + prompt_suffix

    def call_agent(self, query):
        print("Calling Phi4")
        prompt_text = self.prompt + query + " Please do not repeat the prompt text in your response, only give the format output."
        prompt_text = prompt_text.strip()
        print("prompt_text:", prompt_text)
        
        # Tokenize the prompt and get the input IDs
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_input_ids = prompt_tokens["input_ids"]
        start_index = prompt_input_ids.shape[-1]
        
        # Generate the output
        generated_ids = self.llm.generate(
            **prompt_tokens,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1
        )
        
        # Remove the prompt part from the generated output
        generation_output = generated_ids[0][start_index:]
        answer = self.tokenizer.decode(generation_output, skip_special_tokens=True)
        
        print("llm answer:", answer)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
