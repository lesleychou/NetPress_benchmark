import json
import traceback
from dotenv import load_dotenv
import openai
# import pandas as pd
from collections import Counter
# from prototxt_parser.prototxt import parse
import os
import networkx as nx
# import jsonlines
import json
import re
import time
import sys
import numpy as np
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain 
import warnings
from langchain._api import LangChainDeprecationWarning
# from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)


EXAMPLE_LIST = [
    {
        "question": r'mismatch_summary": "Mismatch: frontend → currencyservice:7000 (Expected: True, Actual: False)',
        "answer": r"""kubectl get networkpolicy frontend -o yaml,
kubectl get networkpolicy currencyservice -o yaml
kubectl patch networkpolicy currencyservice --type=merge -p $'
spec:
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
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
        # old_prompt = """
        # We have a Google microservices architecture, the services and the desired communication relationships are as follows:
        # - **User** and **loadgenerator** can access the **frontend** service via HTTP.
        # - **frontend** communicates with the following services: **checkout**, **ad**, **recommendation**, **productcatalog**, **cart**, **shipping**, **currency**, **payment**, and **email**.
        # - **checkout** further communicates with **payment**, **shipping**, **email**, and **currency**.
        # - **recommendation** communicates with **productcatalog**.
        # - **cart** communicates with the **Redis cache** for storing cart data.

        # Your task is to inspect the current network policies and verify if they meet the described communication patterns.  
        # Provide **one command at a time** to check connectivity or node accessibility.  
        # Each time, I will give you the pevious commands and their corresponding outputs. Also the current connectivity status will be provided., I will give you the mismacthes between the expected and actual connectivity status.
        # You should use this information to identify and fix misconfigurations step-by-step.  

        # **Response format:**  
        # Put the command **directly** between triple backticks.  
        # You should use `kubectl patch` instead of `kubectl edit networkpolicy`.  

        # You are not allowed to see the logs of the pods and Kubernetes events, and you are not allowed to use 'kubectl exec' and .
        # Restriction: You are not allowed to see the logs of the pods and Kubernetes events, and you are not allowed to use 'kubectl exec' or to see the logs of the pods and Kubernetes events.
        # Also you should not change the existing correct network policies, you should maintain the originally correct connectivity status.
        # """   

        prompt = """
        You need to behave like a network engineer who can find the root cause of network policy deployment issues and fix them in the microservices architecture.
        Our microservices architecture contains following services and desired communication relationships:
        - **User** and **loadgenerator** can access the **frontend** service via HTTP.
        - **frontend** communicates with the following services: **checkout**, **ad**, **recommendation**, **productcatalog**, **cart**, **shipping**, **currency**, **payment**, and **email**.
        - **checkout** further communicates with **payment**, **shipping**, **email**, and **currency**.
        - **recommendation** communicates with **productcatalog**.
        - **cart** communicates with the **Redis cache** for storing cart data.

        Your task is to inspect the current network policies and verify if they meet the described communication patterns. If there are any mismatches, you should fix them.

        How the interaction works:
        - Provide **one command at a time** to check connectivity or node accessibility.
        - Each time, I will give you the previous commands and their corresponding outputs.
        - I will also provide the current connectivity status, including any mismatches between the expected and actual connectivity status.
        - Use this information to identify and fix misconfigurations step-by-step.

        **Response format:**
        Put the command **directly** between triple backticks.
        You should use `kubectl patch` instead of `kubectl edit networkpolicy`.

        Important notes:
        - You are not allowed to see the logs of the pods and Kubernetes events.
        - You are not allowed to use 'kubectl exec'.
        - Your new command should not change the existing correct network policies if not necessary; Please maintain the originally correct connectivity status.
        """
        return prompt


class ZeroShot_CoT_PromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        cot_prompt_prefix = """
        You need to behave like a network engineer who can find the root cause of network policy deployment issues and fix them in the microservices architecture.
        Our microservices architecture contains following services and desired communication relationships:
        - **User** and **loadgenerator** can access the **frontend** service via HTTP.
        - **frontend** communicates with the following services: **checkout**, **ad**, **recommendation**, **productcatalog**, **cart**, **shipping**, **currency**, **payment**, and **email**.
        - **checkout** further communicates with **payment**, **shipping**, **email**, and **currency**.
        - **recommendation** communicates with **productcatalog**.
        - **cart** communicates with the **Redis cache** for storing cart data.

        Your task is to inspect the current network policies and verify if they meet the described communication patterns. If there are any mismatches, you should fix them.

        How the interaction works:
        - Provide **one command at a time** to check connectivity or node accessibility.
        - Each time, I will give you the previous commands and their corresponding outputs.
        - I will also provide the current connectivity status, including any mismatches between the expected and actual connectivity status.
        - Use this information to identify and fix misconfigurations step-by-step.

        **Response format:**
        Put the command **directly** between triple backticks.
        You should use `kubectl patch` instead of `kubectl edit networkpolicy`.
        You should not include bash in the command, and you should not use <namespace> you should use the namespace of the service.

        Important notes:
        - You are not allowed to see the logs of the pods and Kubernetes events.
        - You are not allowed to use 'kubectl exec'.
        - Your new command should not change the existing correct network policies if not necessary; Please maintain the originally correct connectivity status.

        Please think step by step and provide your output.
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

class ReAct_PromptAgent(BasePromptAgent):
    def __init__(self):
        self.base_prompt_prefix = BasePromptAgent.generate_prompt(self)
        # Now set prompt_prefix manually instead of through super().__init__()
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        react_prompt_prefix = """ 
                                Answer the following question as best you can. Please use a tool if you need to.
                                In the last iteration, please output the action you want to execute.
                                """
        react_prompt = react_prompt_prefix + self.base_prompt_prefix

        return react_prompt

# class FewShot_KNN_PromptAgent(ZeroShot_CoT_PromptAgent):
#     def __init__(self):
#         super().__init__()  # Initialize the parent class
#         self.prompt_prefix = self.generate_prompt()

#     def generate_prompt(self):
#         few_shot_prompt_prefix = super().generate_prompt() + str(examples)
#         return few_shot_prompt_prefix