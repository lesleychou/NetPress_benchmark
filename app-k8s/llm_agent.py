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

# def prompt_generation(txt_file_path):
#     with open(txt_file_path, 'r') as txt_file:
#         file_content = txt_file.read()

#     prompt = f"""
#     We have a Google microservices architecture as shown in the diagram. The services and their communication relationships are as follows:
#     - **User** and **loadgenerator** can access the **frontend** service via HTTP.
#     - **frontend** communicates with the following services: **checkout**, **ad**, **recommendation**, **productcatalog**, **cart**, **shipping**, **currency**, **payment**, and **email**.
#     - **checkout** further communicates with **payment**, **shipping**, **email**, and **currency**.
#     - **recommendation** communicates with **productcatalog**.
#     - **cart** communicates with the **Redis cache** for storing cart data.

#     Your task is to inspect the current network policies and verify if they meet the described communication patterns. You are allowed to provide one command at a time to check the connectivity or node accessibility. After each command, I will provide the result along with the output of any previous commands you made. Based on this information, you should identify and fix any misconfigurations step by step. Do not create any additional policies beyond those implied by the architecture.

#     Your response should be in JSON format with the following structure, wrapped between triple backticks (```):
#     ```
#     {{
#         "command": "kubectl get services"
#     }}
#     ```
#     Use `kubectl patch` instead of `kubectl edit networkpolicy`. Since the entire JSON string is wrapped in single quotes (`'`), there is no need to escape internal double quotes (`"`). Just use double quotes as usual for JSON keys and values without extra escaping.
#     The command must be directly executable on the command line. The command string should be enclosed in double quotes without any escaping (i.e., do not include backslashes before the double quotes).
#     You can replace the command value with any valid kubectl command that you believe will help diagnose the issue. I will then provide the output of your command, and you can proceed with the next command based on that output.
#     {file_content}
#     """
#     return prompt
def prompt_generation(txt_file_path):
    with open(txt_file_path, 'r') as txt_file:
        file_content = txt_file.read()

    prompt = f"""
    We have a Google microservices architecture as shown in the diagram. The services and their communication relationships are as follows:
    - **User** and **loadgenerator** can access the **frontend** service via HTTP.
    - **frontend** communicates with the following services: **checkout**, **ad**, **recommendation**, **productcatalog**, **cart**, **shipping**, **currency**, **payment**, and **email**.
    - **checkout** further communicates with **payment**, **shipping**, **email**, and **currency**.
    - **recommendation** communicates with **productcatalog**.
    - **cart** communicates with the **Redis cache** for storing cart data.

    Your task is to inspect the current network policies and verify if they meet the described communication patterns.  
    Provide **one command at a time** to check connectivity or node accessibility.  
    I will return the output of your command along with previous outputs.  
    Use this information to **identify and fix misconfigurations step-by-step**.  
    **Do not create policies beyond those implied by the architecture.**  

    **Response format:**  
    Put the command **directly** between triple backticks without any JSON structure.  
    - Use `kubectl patch` instead of `kubectl edit networkpolicy`.  
    - Example:
    ```
    kubectl get services
    ```

    Replace the above command with any valid `kubectl` command that helps diagnose the issue.  
    I will provide the output, and you can proceed with the next command.  

    {file_content}
    """
    return prompt

import re
import re

def extract_command(text: str) -> str:
    """
    Extract the content between the first pair of triple backticks (```) in the given text.

    Args:
        text (str): The input string containing the content.

    Returns:
        str: The content between the triple backticks. If no match is found, returns an empty string.
    """
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else ""

# def extract_command(text):
#     # Find the position of the first triple backticks (```).
#     start_block = text.find("```")
#     if start_block == -1:
#         return None

#     # Find the position of the second triple backticks (```) after the first.
#     end_block = text.find("```", start_block + 3)
#     if end_block == -1:
#         return None

#     # Extract the content between the two triple backticks.
#     block_content = text[start_block + 3 : end_block]
    
#     # In the extracted block, locate the '"command":' key.
#     key_index = block_content.find('"command":')
#     if key_index == -1:
#         return None

#     # Find the first double quote after '"command":', which marks the start of the command string.
#     start_quote = block_content.find('"', key_index + len('"command":'))
#     if start_quote == -1:
#         return None
    
#     # Search backward from the end of the block to find the last double quote, marking the end of the command string.
#     end_quote = block_content.rfind('"')
#     if end_quote == -1 or end_quote <= start_quote:
#         return None

#     # Extract and return the command string between the identified starting and ending double quotes.
#     command = block_content[start_quote + 1 : end_quote]
#     return command


class LLMAgent:
    def __init__(self, llm_agent_type):
        # Call the output code from LLM agents file
        if llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
            self.llm_agent = QwenModel()
        if llm_agent_type == "GPT-4o":
            self.llm_agent = AzureGPT4Agent()

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

    def call_agent(self, txt_file_path):
        print("Calling GPT-4o with prompt type:", self.prompt_type)
        
        prompt = prompt_generation(txt_file_path)


        answer = self.llm.invoke(prompt).content
        
        
        print("model returned")
        print("llm answer:", answer)
        answer = extract_command(answer)
        print("extracted command:", answer)
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

    def call_agent(self, txt_file_path):
        prompt = prompt_generation(txt_file_path)
        # Tokenize the prompt and get the input IDs
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
        answer = extract_command(answer)
        print("extracted command:", answer)
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

if __name__ == "__main__":
    text = """```json
                 {
                      "command": "kubectl patch networkpolicy shippingservice -p '{\"spec\":{\"ingress\":[{\"from\":[{\"podSelector\":{\"matchLabels\":{\"app\":\"frontend\"}}}],\"ports\":[{\"port\":50051,\"protocol\":\"TCP\"}]}]}}'"
      }
       ```"""
    output = extract_command(text)
    print(output)