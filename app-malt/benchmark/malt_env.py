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
from llm_model import AzureGPT4Agent, GoogleGeminiAgent, QwQModel, LlamaModel
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
        elif llm_agent_type == "Qwen/QwQ-32B-Preview":
            llm_agent = QwQModel()
        elif llm_agent_type == "meta-llama/Meta-Llama-3.1-70B-Instruct":
            llm_agent = LlamaModel()
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

        # TODO: the safety checker should be always called, even if the output is not a graph
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

        # Add the verifier error to the ground truth result
        if ground_truth_ret['type'] == 'graph':
            ground_truth_ret_graph_copy = ground_truth_ret['data']
            gt_verifier = SafetyChecker(ret_graph=ground_truth_ret_graph_copy, ret_list=None)
            gt_verifier_results, gt_verifier_error = gt_verifier.evaluate_all()
        else:
            gt_verifier_results = True
            gt_verifier_error = ""
        print("Ground truth verifier results: ", gt_verifier_results, gt_verifier_error)
        

        print("LLM answer: ", llm_answer)
        print("Ground truth code: ", goldenAnswerCode)
        if ret['type'] != 'graph':
            print("LLM code result: ", ret)
            print("Ground truth result: ", ground_truth_ret)

        ground_truth_ret['reply'] = goldenAnswerCode
        ret['reply'] = llm_answer

        print("=========Current query process is done!=========")

        return ret, ground_truth_ret, verifier_results, verifier_error, query_run_latency, ret_graph_copy

    def ground_truth_check(self, requestData, task_label, ret, ground_truth_ret, ret_graph_copy, verifier_results, verifier_error, query_run_latency, output_path):
        # Helper function to log results and avoid code duplication
        def log_result(is_correct):
            log_func = self.result_log_correct if is_correct else self.result_log_wrong
            log_func(requestData, task_label, verifier_results, verifier_error, 
                    query_run_latency, ground_truth_ret, ret, output_path)

        # Convert numeric data to strings for text type
        if ground_truth_ret['type'] == 'text':
            for r in (ret, ground_truth_ret):
                if isinstance(r['data'], int):
                    r['data'] = str(r['data'])
        
        # Define comparison strategies for different types
        comparison_strategies = {
            'text': lambda: ground_truth_ret['data'] == ret['data'],
            'list': lambda: check_list_equal(ground_truth_ret['data'], ret['data']),
            'table': lambda: ground_truth_ret['data'] == ret['data'],
            'graph': lambda: nx.is_isomorphic(
                nx.Graph(ground_truth_ret['data']), 
                nx.Graph(ret_graph_copy), 
                node_match=node_attributes_are_equal
            )
        }

        # Get the appropriate comparison strategy and execute it
        compare_func = comparison_strategies.get(ground_truth_ret['type'])
        if compare_func:
            is_correct = compare_func()
            log_result(is_correct)

    def result_log_wrong(self, current_query, task_label, verifier_results, verifier_error, query_run_latency, ground_truth_ret, ret, output_path):
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

        # Add verifier error details if verification failed
        if not verifier_results:
            result_object["Verifier-Error"] = verifier_error

        # Save result_object into a JsonLine file
        with jsonlines.open(output_path, mode='a') as writer:
            writer.write(result_object)
        
        return None

    def result_log_correct(self, current_query, task_label, verifier_results, verifier_error, query_run_latency, ground_truth_ret, ret, output_path):
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
        
        # Add verifier error details if verification failed
        if not verifier_results:
            result_object["Verifier-Error"] = verifier_error
        
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

