import json
import traceback
from dotenv import load_dotenv
import openai

from collections import Counter

import os
import networkx as nx

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
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent, FewShot_Semantic_PromptAgent

# Login huggingface
login(token="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")
# Load environ variables from .env, will not override existing environ variables
load_dotenv()

# # For Google Gemini
# import getpass
# from langchain_google_genai import ChatGoogleGenerativeAI
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY")

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# For Azure OpenAI GPT4
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.chat_models import AzureChatOpenAI
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ztn-oai-sweden.openai.azure.com/"

import re

def extract_command(text: str) -> str:
    """
    Extract the content between the first pair of triple backticks (```) in the given text and remove all newline characters.

    Args:
        text (str): The input string containing the content.

    Returns:
        str: The content between the triple backticks with newline characters removed. If no match is found, returns an empty string.
    """
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

class LLMAgent:
    def __init__(self, llm_agent_type, prompt_type="base"):
        # Call the output code from LLM agents file
        if llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
            self.llm_agent = QwenModel(prompt_type=prompt_type)
        if llm_agent_type == "GPT-4o":
            self.llm_agent = AzureGPT4Agent(prompt_type=prompt_type)

class AzureGPT4Agent:
    def __init__(self, prompt_type="base"):
        self.llm = AzureChatOpenAI(
            openai_api_version="2024-08-01-preview",
            deployment_name='ztn-sweden-gpt-4o',
            model_name='ztn-sweden-gpt-4o',
            temperature=0.0,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        
        if prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif prompt_type == "few_shot_semantic":
            print("few_shot_semantic")
            self.prompt_agent = FewShot_Semantic_PromptAgent()

    def call_agent(self, txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            connectivitity_status = txt_file.read()
        
        max_length = 127000  
        if len(connectivitity_status) > max_length:
            connectivitity_status = connectivitity_status[:max_length]

        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivitity_status)
            input_data = {"input": connectivitity_status}
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            input_data = {"input": connectivitity_status}
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivitity_status}
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivitity_status}
        print("prompt:", prompt)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(input_data)
        response = extract_command(response)
        return response

class QwenModel:
    def __init__(self, prompt_type="base"):
        # 模型初始化...
        self.prompt_type = prompt_type
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

        if prompt_type == "base":
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = ZeroShot_CoT_PromptAgent()

    def call_agent(self, txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            connectivitity_status = txt_file.read()
        
        max_length = 127000  
        if len(connectivitity_status) > max_length:
            connectivitity_status = connectivitity_status[:max_length]

        # 生成提示（根据类型调整）
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivitity_status)
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        print("prompt:", prompt)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_input_ids = prompt_tokens["input_ids"]
        start_index = prompt_input_ids.shape[-1]
        
        # Generate the output
        generated_ids = self.llm.generate(
            **prompt_tokens,
            max_new_tokens=4000,
            do_sample=True,
            temperature=0.1
        )
        # Remove the prompt part from the generated output
        generation_output = generated_ids[0][start_index:]
        answer = self.tokenizer.decode(generation_output, skip_special_tokens=True)
        print("llm answer:", answer)
        print("model returned")
        answer = extract_command(answer)
        print("extracted command:", answer)
        print("model returned")
        return answer
    
if __name__ == "__main__":
    text = """```json
                 {
                      "command": "kubectl patch networkpolicy shippingservice -p '{\"spec\":{\"ingress\":[{\"from\":[{\"podSelector\":{\"matchLabels\":{\"app\":\"frontend\"}}}],\"ports\":[{\"port\":50051,\"protocol\":\"TCP\"}]}]}}'"
      }
       ```"""
    output = extract_command(text)
    print(output)