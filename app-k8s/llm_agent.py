import json
import traceback
from dotenv import load_dotenv
import openai
import pandas as pd
from collections import Counter
from prototxt_parser.prototxt import parse
import os
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

class LLMAgent:
    def __init__(self, llm_agent_type):
        # Call the output code from LLM agents file
        if llm_agent_type == "AzureGPT4Agent":
            self.llm_agent = AzureGPT4Agent()
        elif llm_agent_type == "GoogleGeminiAgent":
            self.llm_agent = GoogleGeminiAgent()
        elif llm_agent_type == "Qwen/QwQ-32B-Preview":
            self.llm_agent = QwQModel()
        elif llm_agent_type == "meta-llama/Meta-Llama-3.1-70B-Instruct":
            self.llm_agent = LlamaModel()
        elif llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
            self.llm_agent = QwenModel()
        elif llm_agent_type == "phi-4/Phi-4-70B-Instruct":
            self.llm_agent = Phi4Model()
        self.prompt = """
        We have a Kubernetes cluster with four nodes and are using Cilium to enforce security policies. The specific nodes and their configurations can be checked by inspecting the cluster, you can use commands to check it. The security rules are as follows: only allow the gateway to access node-prod on port 80/TCP, only allow the payment service to access node-db on port 5432/TCP, and deny the development environment from accessing the production environment. 
        Your task is to inspect the current Cilium policies and verify if they meet these specified rules. You are allowed to give one command at a time to check the policies or node accessibility. After each command, I will provide the result along with the output of any previous commands you made. Based on this, you should identify and fix misconfigurations step by step. Do not create any additional policies beyond those specified.
        """

    def generate_prompt(self, command, result):
        query = f"{self.prompt}\nCommand: {command}\nResult: {result}"
        return self.llm_agent.call_agent(query)

class GoogleGeminiAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.pyGraphNetExplorer = LLMChain(llm=self.llm, prompt=PromptTemplate(
            input_variables=["input"],
            template="{input}"
        ))

    def call_agent(self, prompt):
        print("Calling Google Gemini")
        answer = self.pyGraphNetExplorer.run(prompt)
        print("model returned")
        print("answer:", answer)
        return answer

class AzureGPT4Agent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_type="azure_ad",
            openai_api_version="2024-08-01-preview",
            deployment_name='gpt-4o',
            model_name='gpt-4o',
            temperature=0.0,
            max_tokens=4000,
        )
        self.pyGraphNetExplorer = LLMChain(llm=self.llm, prompt=PromptTemplate(
            input_variables=["input"],
            template="{input}"
        ))

    def call_agent(self, prompt):
        print("Calling GPT-4o")
        answer = self.pyGraphNetExplorer.run(prompt)
        print("model returned")
        print("answer:", answer)
        return answer

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

    def call_agent(self, prompt):
        print("Calling Llama")
        prompt_text = prompt + " Please do not repeat the prompt text in your response, only give the format output."
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
        return answer

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

    def call_agent(self, prompt):
        print("Calling QwQ")
        prompt_text = prompt + " Please do not repeat the prompt text in your response, only give the format output."
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
        return answer

class QwenModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-72B-Instruct"
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

    def call_agent(self, prompt):
        print("Calling Qwen")
        prompt_text = prompt + " Please do not repeat the prompt text in your response, only give the format output."
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
        return answer

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

    def call_agent(self, prompt):
        print("Calling Phi4")
        prompt_text = prompt + " Please do not repeat the prompt text in your response, only give the format output."
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
        return answer
