import json
import traceback
from dotenv import load_dotenv
import openai
import copy
import pandas as pd
from prototxt_parser.prototxt import parse
from collections import Counter
import os
from solid_step_helper import getGraphData, clean_up_llm_output_func, check_list_equal, node_attributes_are_equal, clean_up_output_graph_data, \
    solid_step_add_node_to_graph, solid_step_counting_query, solid_step_remove_node_from_graph, solid_step_list_child_nodes, solid_step_update_node_value, solid_step_rank_child_nodes
import networkx as nx
import jsonlines
import random
from networkx.readwrite import json_graph
import json
import re
import time
import sys
import numpy as np
from llm_model import AzureGPT4Agent, GoogleGeminiAgent
from error_check import SafetyChecker


# output the evaluation results to a jsonl file
OUTPUT_JSONL_DIR = 'logs/llm_agents'
OUTPUT_JSONL_FILE = 'gpt4.jsonl'


class BenchmarkEvaluator:
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def userQuery(self, current_query, golden_answer, llm_agent_type="AzureGPT4Agent"):
        # for each prompt in the prompt_list, append it as the value of {'query': prompt}
        print("Query: ", current_query)

        G = self.graph_data
        
        # Call the output code from LLM agents file
        start_time = time.time()
        if llm_agent_type == "AzureGPT4Agent":
            llm_agent = AzureGPT4Agent()
        elif llm_agent_type == "GoogleGeminiAgent":
            llm_agent = GoogleGeminiAgent()
        llm_answer = llm_agent.call_agent(current_query)

        try:
            exec(llm_answer)
            ret = eval("process_graph(G)")
        except Exception:
            ret = {'type': "error", 'data': traceback.format_exc()}
        
        query_run_latency = time.time() - start_time

        # if the type of ret is string, turn it into a json object
        if isinstance(ret, str):
            ret = json.loads(ret)
        
        ret_graph_copy = None

        if ret['type'] == 'graph':
            ret_graph_copy = clean_up_output_graph_data(ret)
            verifier = SafetyChecker(ret_graph=ret_graph_copy, ret_list=None)
            verifier_results, verifier_error = verifier.evaluate_all()
        else:
            verifier_results = True
            verifier_error = ""
        print("Verifier results: ", verifier_results, verifier_error)

        # Where we get the golden answer (ground truth) code for each query
        goldenAnswerCode = golden_answer

        # ground truth answer should already be checked to ensure it can run successfully
        exec(goldenAnswerCode)
        ground_truth_ret = eval("ground_truth_process_graph(G)")
        # if the type of ground_truth_ret is string, turn it into a json object
        if isinstance(ground_truth_ret, str):
            ground_truth_ret = json.loads(ground_truth_ret)
        
        print("LLM answer: ", llm_answer)
        print("LLM code result: ", ret)
        print("Ground truth code: ", goldenAnswerCode)
        print("Ground truth result: ", ground_truth_ret)

        ground_truth_ret['reply'] = goldenAnswerCode
        ret['reply'] = llm_answer

        print("=========Current query process is done!=========")

        return ret, ground_truth_ret, verifier_results, query_run_latency, ret_graph_copy

    def ground_truth_check(self, requestData, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, query_run_latency, output_path):

        # Ground truth comparision between the LLM output (ret) and the golden answer (ground_truth_ret)
        # check type "text", "list", "table", "graph" separately.
        if ground_truth_ret['type'] == 'text':
            # if ret['data'] type is int, turn it into string
            if isinstance(ret['data'], int):
                ret['data'] = str(ret['data'])
            if isinstance(ground_truth_ret['data'], int):
                ground_truth_ret['data'] = str(ground_truth_ret['data'])

            if ground_truth_ret['data'] == ret['data']:
                self.result_log_correct(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)
            else:
                self.result_log_wrong(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)

        elif ground_truth_ret['type'] == 'list':
            # Use Counter to check if two lists contain the same items, including duplicate items.
            if check_list_equal(ground_truth_ret['data'], ret['data']):
                self.result_log_correct(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)
            else:
                self.result_log_wrong(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)

        elif ground_truth_ret['type'] == 'table':
            if ground_truth_ret['data'] == ret['data']:
                self.result_log_correct(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)
            else:
                self.result_log_wrong(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)

        elif ground_truth_ret['type'] == 'graph':
            # Undirected graphs will be converted to a directed graph
            # with two directed edges for each undirected edge.
            ground_truth_graph = nx.Graph(ground_truth_ret['data'])
            # TODO: fix ret_graph_copy reference possible error, when it's not created.
            ret_graph = nx.Graph(ret_graph_copy)

            # Check if two graphs are identical, no weights considered
            if nx.is_isomorphic(ground_truth_graph, ret_graph, node_match=node_attributes_are_equal):
                self.result_log_correct(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)
            else:
                self.result_log_wrong(requestData, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path)

    def result_log_wrong(self, current_query, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path):
        result_object = {
            "Query": current_query,
            "Label": task_label,
            "Result-Correctness": "Fail",
            "Result-Safety": "Pass" if verifier_results else "Fail",
            "Result-Latency": query_run_latency,
            "Ground truth code": ground_truth_ret['reply'],
            "LLM code": ret['reply']
        }
        if ground_truth_ret['type'] == 'graph':
            result_object["Error"] = "Two graphs are not identical."
        else:
            result_object["Ground truth exec"] = ground_truth_ret['data']
            result_object["LLM code exec"] = ret['data']
            result_object["Error"] = {
                "Ground truth": ground_truth_ret['data'],
                "Model output": ret['data']
            }

        # Save result_object into a JsonLine file
        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(result_object)
        
        return None

    def result_log_correct(self, current_query, task_label, verifier_results, query_run_latency, ground_truth_ret, ret, output_path):
        result_object = {
            "Query": current_query,
            "Label": task_label,
            "Result-Correctness": "Pass",
            "Result-Safety": "Pass" if verifier_results else "Fail",
            "Result-Latency": query_run_latency,
            "Ground truth code": ground_truth_ret['reply'],
            "LLM code": ret['reply']
        }
        if ground_truth_ret['type'] != 'graph':
            result_object["Ground truth exec"] = ground_truth_ret['data']
            result_object["LLM code exec"] = ret['data']
        
        # Save result_object into a JsonLine file
        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(result_object)
        
        return None
    
# # example usage for the class
# if __name__ == "__main__":
#     evaluator = BenchmarkEvaluator()
#     query = "What is the capital of France?"
#     golden_answer = """
#                     def ground_truth_process_graph(G):
#                         return {'type': 'text', 'data': 'Paris'}
#                     """
#     ret, ground_truth_ret, verifier_results, query_run_latency, ret_graph_copy = evaluator.userQuery(query, golden_answer)
#     evaluator.ground_truth_check(query, "capital_question", ret, ground_truth_ret, ret_graph_copy, verifier_results, query_run_latency, os.path.join(OUTPUT_JSONL_DIR, OUTPUT_JSONL_FILE))

