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
from langchain.vectorstores import Chroma
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

EXAMPLE_LIST = [
    {
        "question": r"""
    p29_h1 -> p29_h2 X X p29_r0 
    p29_h2 -> p29_h1 X X p29_r0 
    p29_h3 -> X X p29_h4 p29_r0 
    p29_h4 -> X X p29_h3 p29_r0 
    p29_r0 -> p29_h1 p29_h2 p29_h3 p29_h4 
    *** Results: 40% dropped (12/20 received)
        """,
        "answer": r"""
        machine: p29_r0 
        command: sysctl net.ipv4.ip_forward
        machine: p29_r0 
        command: sysctl -w net.ipv4.ip_forward=1
'"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 X X X 
    p29_h2 -> p29_h1 X X X 
    p29_h3 -> X X p29_h4 p29_r0
    p29_h4 -> X X p29_h3 p29_r0 
    p29_r0 -> X X p29_h3 p29_h4 
    *** Results: 60% dropped (8/20 received)
""",
        "answer": r"""
        machine: p29_r0
        command: ip link show
        machine: p29_r0
        command: ip link set dev p29_r0-eth1 up
'"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 X X p29_r0 
    p29_h2 -> h1 X X p29_r0 
    p29_h3 -> X X p29_h4 X 
    p29_h4 -> X X p29_h3 X 
    p29_r0 -> p29_h1 p29_h2 X X 
    *** Results: 60% dropped (8/20 received)
        """,
        "answer": r"""
        machine: p29_r0
        command: iptables -L -v --line-numbers
        machine: p29_r0
        command: iptables -D INPUT 1
        machine: p29_r0
        command: iptables -D OUTPUT 1

'"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 X X X X X 
    p29_h2 -> p29_h1 X X X X X 
    p29_h3 -> X X p29_h4 p29_h5 p29_h6 p29_r0 
    p29_h4 -> X X p29_h3 p29_h5 p29_h6 p29_r0 
    p29_h5 -> X X p29_h3 p29_h4 p29_h6 p29_r0 
    p29_h6 -> X X p29_h3 p29_h4 p29_h5 p29_r0 
    p29_r0 -> X X p29_h3 p29_h4 p29_h5 p29_h6
    *** Results: 47% dropped (22/42 received)
""",
        "answer": r"""
    machine: p29_r0
    command: ip route
    machine: p29_r0
    command: ip route del 192.168.1.0/24 dev p29_r0-eth2
    machine: p29_r0
    command: ip route add 192.168.1.0/24 dev p29_r0-eth1
"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 p29_h3 p29_h4 X X X X p29_r0 
    p29_h2 -> p29_h1 p29_h3 p29_h4 X X X X p29_r0 
    p29_h3 -> p29_h1 p29_h2 p29_h4 X X X X p29_r0 
    p29_h4 -> p29_h1 p29_h2 p29_h3 X X X X p29_r0 
    p29_h5 -> X X X X p29_h6 p29_h7 p29_h8 X 
    p29_h6 -> X X X X p29_h5 p29_h7 p29_h8 X 
    p29_h7 -> X X X X p29_h5 p29_h6 p29_h8 X 
    p29_h8 -> X X X X p29_h5 p29_h6 p29_h7 X 
    p29_r0 -> p29_h1 p29_h2 p29_h3 p29_h4 X X X X 
    *** Results: 55% dropped (32/72 received)
""",
        "answer": r"""
    machine: p29_r0
    command: p29_r0 ip addr show dev p29_r0-eth2
    machine: p29_r0
    command: ip addr add 192.168.2.1/24 dev p29_r0-eth2
"""
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

        # old_prompt = """
        # You need to behave like a network engineer who find the root cause of the network issue, and fix it in a routing application.
        # There is a mininet network, but there exists problems in the router r0, 
        # so the network is not fully connected which means some nodes cannot ping other nodes successfully. 
        # You need to fix it to make pingall result is all true.
        # I recommend you to use some commands to know the information of the router and 
        # the network so that you can know the cause of the problem. But if you think the information is enough 
        # and you know the reason causing the problem, you have to give commands to fix it.
        # And when you give the command, you should be careful because your command cannot cause the original connected edge to be disconnected.
        # You need to give the output in JSON format, which contains the machine and its command.
        # Please only give me the JSON format output, with key 'machine' and 'command' and their value. 
        # You can only give one command at a time because I can only execute one command.
        # You should be careful that the router's name may not be r0, but we may use prefix to identify the router's name, like p29_r0. And also the same for the host's name and the interface's name, so it can be p29_h1, p29_h2, p29_r0-eth1, p29_r0-eth2, etc. However, the prefix may not be p29, it can be other names like p30, p31, etc.
        # Also please don't include 'sudo', and you are not allowed to use vtysh command, also you can not use ping command because the ping result is already given to you.
        # Then I will give you the latest PingAll() feedback from the network, and also your 
        # previous actions to the network and the actions' feedback to let you know more information.
        # """

        prompt = """
        You need to behave like a network engineer who finds the root cause of network issues and fixes them in a routing application.
        
        There is a Mininet network with problems in the router r0, causing the network to be partially disconnected. Some nodes cannot successfully ping other nodes. Your task is to fix these issues so that the pingall result shows all connections are successful.
        
        I recommend using diagnostic commands to gather information about the router and network to identify the cause of the problem. Once you have sufficient information and understand the root cause, provide commands to fix the issue.
        
        When implementing your solution, be careful not to disrupt existing connected edges - your commands should not cause previously working connections to break.
        
        Please provide your output in JSON format with the keys 'machine' and 'command'. You can only issue one command at a time as I can only execute commands sequentially.
        
        Important notes:
        - The router's name may not be exactly 'r0'. It may have a prefix (like 'p29_r0').
        - The same applies to host names and interface names (e.g., 'p29_h1', 'p29_h2', 'p29_r0-eth1', 'p29_r0-eth2'). 
        - The prefix could be anything ('p29', 'p30', 'p31', etc.).
        - Do not include 'sudo' in your commands.
        - You are not permitted to use the 'vtysh' command.
        - Do not use ping commands as the ping results are already provided to you.
        
        I will provide you with the latest PingAll() feedback from the network along with your previous actions and their results to help you diagnose the problem.
        """
        return prompt


class ZeroShot_CoT_PromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        cot_prompt_prefix = """
        You need to behave like a network engineer who finds the root cause of network issues and fixes them in a routing application.
        
        There is a Mininet network with problems in the router r0, causing the network to be partially disconnected. Some nodes cannot successfully ping other nodes. Your task is to fix these issues so that the pingall result shows all connections are successful.
        
        I recommend using diagnostic commands to gather information about the router and network to identify the cause of the problem. Once you have sufficient information and understand the root cause, provide commands to fix the issue.
        
        When implementing your solution, be careful not to disrupt existing connected edges - your commands should not cause previously working connections to break.
        
        Please provide your output in JSON format with the keys 'machine' and 'command'. You can only issue one command at a time as I can only execute commands sequentially.
        
        Important notes:
        - The router's name may not be exactly 'r0'. It may have a prefix (like 'p29_r0').
        - The same applies to host names and interface names (e.g., 'p29_h1', 'p29_h2', 'p29_r0-eth1', 'p29_r0-eth2'). 
        - The prefix could be anything ('p29', 'p30', 'p31', etc.).
        - Do not include 'sudo' in your commands.
        - You are not permitted to use the 'vtysh' command.
        - Do not use ping commands as the ping results are already provided to you.
        
        I will provide you with the latest PingAll() feedback from the network along with your previous actions and their results to help you diagnose the problem.

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


# class FewShot_KNN_PromptAgent(ZeroShot_CoT_PromptAgent):
#     def __init__(self):
#         super().__init__()  # Initialize the parent class
#         self.prompt_prefix = self.generate_prompt()

#     def generate_prompt(self):
#         few_shot_prompt_prefix = super().generate_prompt() + str(examples)
#         return few_shot_prompt_prefix