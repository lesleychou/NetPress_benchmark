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
from prompt_agent import BasePromptAgent, CoTPromptAgent, FewShotPromptAgent

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


prompt_suffix = """Begin! Remember to ensure that you generate valid Python code in the following format:

        Answer:
        ```python
        ${{Code that will answer the user question or request}}
        ```
        Question: {input}
        """

class GoogleGeminiAgent:
    def __init__(self, prompt_type="base"):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.prompt_type = prompt_type
        # Select prompt agent based on type
        if self.prompt_type == "cot":
            prompt_agent = CoTPromptAgent()
        elif self.prompt_type == "few_shot":
            prompt_agent = FewShotPromptAgent()
        else:
            prompt_agent = BasePromptAgent()

        prompt = PromptTemplate(
            input_variables=["input"],
            template=prompt_agent.prompt_prefix + prompt_suffix
        )
        print("prompt:", prompt)
        self.pyGraphNetExplorer = LLMChain(llm=self.llm, prompt=prompt)

    def call_agent(self, query):
        print("Calling Google Gemini with prompt type:", self.prompt_type)
        answer = self.pyGraphNetExplorer.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        print("code:", code)
        return code


class AzureGPT4Agent:
    def __init__(self, prompt_type="base"):
        self.llm = AzureChatOpenAI(
            openai_api_type="azure_ad",
            openai_api_version="2024-08-01-preview",
            deployment_name='ztn-sweden-gpt-4o',
            model_name='ztn-sweden-gpt-4o',
            temperature=0.0,
            max_tokens=4000,
        )

        self.prompt_type = prompt_type
        # Select prompt agent based on type
        if self.prompt_type == "cot":
            prompt_agent = CoTPromptAgent()
        elif self.prompt_type == "few_shot":
            prompt_agent = FewShotPromptAgent()
        else:
            prompt_agent = BasePromptAgent()

        prompt = PromptTemplate(
            input_variables=["input"],
            template=prompt_agent.prompt_prefix + prompt_suffix
        )
        print("prompt:", prompt)

        self.pyGraphNetExplorer = LLMChain(llm=self.llm, prompt=prompt)

    def call_agent(self, query):
        print("Calling GPT-4o with prompt type:", self.prompt_type)
        answer = self.pyGraphNetExplorer.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
   
class QwenModel:
    def __init__(self, prompt_type="base"):
        self.model_name = "Qwen2.5-72B-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(
            model=self.model_name,
            device=self.device,
            quantization="gptq"
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )

        self.prompt_type = prompt_type
        # Select prompt agent based on type
        if self.prompt_type == "cot":
            prompt_agent = CoTPromptAgent()
        elif self.prompt_type == "few_shot":
            prompt_agent = FewShotPromptAgent()
        else:
            prompt_agent = BasePromptAgent()

        self.prompt = prompt_agent.prompt_prefix + prompt_suffix
        print("prompt:", self.prompt)


    def call_agent(self, query):
        print("Calling Qwen with prompt type:", self.prompt_type)
        prompt_text = self.prompt + query 
        print("prompt_text:", prompt_text)
        
        result = self.llm.generate([prompt_text], sampling_params=self.sampling_params)
        answer = result[0].outputs[0].text
  
        print("llm answer:", answer)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code

 
class LlamaModel:
    def __init__(self, prompt_type="base"):
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

        # Select prompt agent based on type
        if prompt_type == "cot":
            prompt_agent = CoTPromptAgent()
        elif prompt_type == "few_shot":
            prompt_agent = FewShotPromptAgent()
        else:
            prompt_agent = BasePromptAgent()

        self.prompt = prompt_agent.prompt_prefix + prompt_suffix

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
