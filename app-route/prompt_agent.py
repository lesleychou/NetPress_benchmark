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

from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)


def _generate_prompt(file_content, log_content):
    """
    Generates a JSON format prompt for fixing a Mininet network issue.

    Args:
        file_content (str): The content of the file containing previous actions and feedback.
        log_content (str): The latest feedback from the Mininet.

    Returns:
        str: A JSON-formatted string with 'machine' and 'command' keys.
    """
    prompt = (
        """There is a mininet network, but there are some kinds of problems in the router r0, 
        so it cannot function well and PingAll() fails at some nodes. You need to fix it.
        I highly recommend you to use some commands to know the information of the router and 
        the network to know the cause of the problem. But if you think the information is enough 
        and you know the reason causing the problem, you have to give commands to fix it.
        You need to give the output in JSON format, which contains the machine and its command.
        Then I will give you the latest PingAll() feedback from the network, and also your 
        previous actions to the network and the actions' feedback to let you know more information.
        """
        + "Here are the previous actions and their feedbacks:\n"
        + file_content
        + "This is the latest feedback from the mininet:\n"
        + log_content
        + "Please only give me the JSON format output, with key 'machine' and 'command' and their value. You can only give one command at a time and don't include 'sudo', and you are not allowed to use vtysh command."
    )
    return prompt
EXAMPLE_LIST = [
    {
        "question": r'mismatch_summary": "Mismatch: frontend → currencyservice:7000 (Expected: True, Actual: False)\nMismatch: checkoutservice → currencyservice:7000 (Expected: True, Actual: False)',
        "answer": r"""kubectl get networkpolicy frontend -o yaml,
kubectl get networkpolicy currencyservice -o yaml
kubectl patch networkpolicy currencyservice -p $'
spec:
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    - podSelector:
        matchLabels:
          app: checkoutservice
'"""
    },
    {
        "question": r'mismatch_summary": "Mismatch: cartservice → productcatalogservice:3550 (Expected: False, Actual: True)',
        "answer": r"""kubectl get networkpolicy cartservice -o yaml,
kubectl get networkpolicy productcatalogservice -o yaml
kubectl patch networkpolicy productcatalogservice -p $'
spec:
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    - podSelector:
        matchLabels:
          app: checkoutservice
    - podSelector:
        matchLabels:
          app: recommendationservice
  ports:
    - port: 3550
      protocol: TCP
'"""
    },
    {
        "question": r'Mismatch: frontend → adservice:9555 (Expected: True, Actual: False)\nMismatch: frontend → cartservice:7070 (Expected: True, Actual: False)\nMismatch: frontend → checkoutservice:5050 (Expected: True, Actual: False)\nMismatch: frontend → currencyservice:7000 (Expected: True, Actual: False)\nMismatch: frontend → productcatalogservice:3550 (Expected: True, Actual: False)\nMismatch: frontend → recommendationservice:8080 (Expected: True, Actual: False)\nMismatch: frontend → shippingservice:50051 (Expected: True, Actual: False)',
        "answer": r"""kubectl get networkpolicy frontend -o yaml,
kubectl patch networkpolicy frontend --type merge -p $'
spec:
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: adservice
    - podSelector:
        matchLabels:
          app: cartservice
    - podSelector:
        matchLabels:
          app: checkoutservice
    - podSelector:
        matchLabels:
          app: currencyservice
    - podSelector:
        matchLabels:
          app: productcatalogservice
    - podSelector:
        matchLabels:
          app: recommendationservice
    - podSelector:
        matchLabels:
          app: shippingservice
  ports:
    - port: 9555
    - port: 7070
    - port: 5050
    - port: 7000
    - port: 3550
    - port: 8080
    - port: 50051
'"""
    }
]


class BasePromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        """
        Generates a JSON format prompt for fixing a Mininet network issue.

        Args:
            file_content (str): The content of the file containing previous actions and feedback.
            log_content (str): The latest feedback from the Mininet.

        Returns:
            str: A JSON-formatted string with 'machine' and 'command' keys.
        """
        prompt = (
            """In a Mininet network, the router r0 is experiencing issues, causing PingAll() to fail at some nodes. To diagnose and fix the problem, you will use network debugging commands to gather information about r0 and the network topology. Based on your findings, you will then execute the necessary commands to restore connectivity.
            You need to give the output in JSON format, which contains the machine and its command.
            Then I will give you the latest PingAll() feedback from the network, and also your 
            previous actions to the network and the actions' feedback to let you know more information.
            """
            
        )
        return prompt


class ZeroShot_CoT_PromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        cot_prompt_prefix = """
        We have a Google microservices architecture as shown in the diagram, but there are connectivity issues between some nodes. Your task is to diagnose and fix the problem.
        1. Steps to follow:
        - Some nodes should only allow one-way access by restricting ingress and egress traffic.
        - The connectivity rules are as follows: 
            - **User** and **loadgenerator** can access the **frontend** service via HTTP.
            - **frontend** communicates with the following services: **checkout**, **ad**, **recommendation**, **productcatalog**, **cart**, **shipping**, **currency**, **payment**, and **email**.
            - **checkout** further communicates with **payment**, **shipping**, **email**, and **currency**.
            - **recommendation** communicates with **productcatalog**.
            - **cart** communicates with the **Redis cache** for storing cart data.
        2. Analyze Connectivity Issues
        - Infer potential problems based on the current connectivity status.
        - If two nodes cannot communicate, the issue likely lies in the ingress and egress policies managing their connection.
        3. Inspect Network Policies
        - Identify the policies controlling ingress and egress for the affected nodes. Analyze the policy definitions to determine whether they are incorrectly configured.
        - If one policy seems correct, check the corresponding policy on the other side.
        4. Determine the Fix and Apply Changes
        - Based on your analysis, determine the most effective fix.
        - Instead of using 'kubectl edit networkpolicy', use 'kubectl patch' to make necessary modifications
        5. Provide Fix Commands
        - Format only one command.
        - The command should be placed directly between triple backticks, without any JSON structure.
        - Example:
        ```
        kubectl get services
        ```
        """
        return cot_prompt_prefix


class FewShot_Basic_PromptAgent(ZeroShot_CoT_PromptAgent):
    def __init__(self):
        super().__init__()
        self.examples = EXAMPLE_LIST
        self.cot_prompt_prefix = super().generate_prompt()
    
    def get_few_shot_prompt(self):
        example_prompt = PromptTemplate(
            input_variables=["question", "answer"], 
            template="Question: {question}\nAnswer: {answer}"
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix=self.cot_prompt_prefix + "Here are some example question-answer pairs:\n",
            suffix="This is current connectivity status:\n{input}\nYou can proceed with the next command.",
            input_variables=["input"]  
        )
        return few_shot_prompt


# TODO: add few shot examples for knn
class FewShot_Semantic_PromptAgent(ZeroShot_CoT_PromptAgent):
    def __init__(self):
        self.examples = EXAMPLE_LIST
        self.cot_prompt_prefix = super().generate_prompt()

    def get_few_shot_prompt(self, query):
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            self.examples,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            embeddings,
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            Chroma,
            # This is the number of examples to produce.
            k=1)

        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="Question: {question}\nAnswer: {answer}"
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=example_selector.select_examples({"question": query}),
            example_prompt=example_prompt,
            prefix=self.cot_prompt_prefix + "Here are some example question-answer pairs:\n",
            suffix="prompt_suffix",
            input_variables=["input"]
        )
        return few_shot_prompt


# class FewShot_KNN_PromptAgent(ZeroShot_CoT_PromptAgent):
#     def __init__(self):
#         super().__init__()  # Initialize the parent class
#         self.prompt_prefix = self.generate_prompt()

#     def generate_prompt(self):
#         few_shot_prompt_prefix = super().generate_prompt() + str(examples)
#         return few_shot_prompt_prefix