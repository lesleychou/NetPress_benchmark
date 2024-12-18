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
from llm_model import MaltAgent_GPT

# output the evaluation results to a jsonl file
OUTPUT_JSONL_DIR = 'logs/gpt_agents'
OUTPUT_JSONL_FILE = 'gpt4.jsonl'


def userQuery(current_query, golden_answer, output_path):
    # for each prompt in the prompt_list, append it as the value of {'query': prompt}
    print("Query: ", current_query)
    requestData = {'query': current_query}

    # import pdb; pdb.set_trace()
    prompt_accu = 0
    _, G = getGraphData()
    
    # TODO: call the output code from LLM agents file
    llm_agent = MaltAgent_GPT()
    llm_answer = llm_agent.call_agent(current_query)

    try:
        exec(llm_answer)
        ret = eval("process_graph(G)")
    except Exception:
        raise Exception("Error in running the LLM generated code")

    # if the type of ret is string, turn it into a json object
    if isinstance(ret, str):
        ret = json.loads(ret)

    if ret['type'] == 'graph':
        ret_graph_copy = clean_up_output_graph_data(ret)

    # Where we get the golden answer (ground truth) code for each query
    goldenAnswerCode = golden_answer

    # ground truth answer should already be checked to ensure it can run successfully
    exec(goldenAnswerCode)
    ground_truth_ret = eval("ground_truth_process_graph(G)")
    # if the type of ground_truth_ret is string, turn it into a json object
    if isinstance(ground_truth_ret, str):
        ground_truth_ret = json.loads(ground_truth_ret)
    
    print("Ground truth: ", ground_truth_ret)

    ground_truth_ret['reply'] = goldenAnswerCode
    ret['reply'] = llm_answer

    # Ground truth comparision between the LLM output (ret) and the golden answer (ground_truth_ret)
    # check type "text", "list", "table", "graph" separately.
    if ground_truth_ret['type'] == 'text':
        # if ret['data'] type is int, turn it into string
        if isinstance(ret['data'], int):
            ret['data'] = str(ret['data'])
        if isinstance(ground_truth_ret['data'], int):
            ground_truth_ret['data'] = str(ground_truth_ret['data'])

        if ground_truth_ret['data'] == ret['data']:
            prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, output_path)
        else:
            ground_truth_check_debug(requestData, ground_truth_ret, ret, output_path)

    elif ground_truth_ret['type'] == 'list':
        # Use Counter to check if two lists contain the same items, including duplicate items.
        if check_list_equal(ground_truth_ret['data'], ret['data']):
            prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, output_path)
        else:
            ground_truth_check_debug(requestData, ground_truth_ret, ret, output_path)

    elif ground_truth_ret['type'] == 'table':
        if ground_truth_ret['data'] == ret['data']:
            prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, output_path)
        else:
            ground_truth_check_debug(requestData, ground_truth_ret, ret, output_path)

    elif ground_truth_ret['type'] == 'graph':
        # Undirected graphs will be converted to a directed graph
        # with two directed edges for each undirected edge.
        ground_truth_graph = nx.Graph(ground_truth_ret['data'])
        # TODO: fix ret_graph_copy reference possible error, when it's not created.
        ret_graph = nx.Graph(ret_graph_copy)

        # Check if two graphs are identical, no weights considered
        if nx.is_isomorphic(ground_truth_graph, ret_graph, node_match=node_attributes_are_equal):
            prompt_accu = ground_truth_check_accu(prompt_accu, requestData, ground_truth_ret, ret, output_path)
        else:
            ground_truth_check_debug(requestData, ground_truth_ret, ret, output_path)


    print("=========Current query process is done!=========")

    return ret


def ground_truth_check_debug(requestData, ground_truth_ret, ret, output_path):
    print("Fail the test, and here is more info: ")
    if ground_truth_ret['type'] == 'graph':
        print("Two graph are not identical.")
    else:
        print("ground truth: ", ground_truth_ret['data'])
        print("model output: ", ret['data'])

    # Save requestData, code, ground_truth_ret['data'] into a JsonLine file
    with jsonlines.open(output_path, mode='a') as writer:
        writer.write(requestData)
        writer.write({"Result": "Fail"})
        writer.write({"Ground truth code": ground_truth_ret['reply']})
        writer.write({"LLM code": ret['reply']})
        if ground_truth_ret['type'] != 'graph':
            writer.write({"Ground truth exec": ground_truth_ret['data']})
            writer.write({"LLM code exec": ret['data']})
    return None

def ground_truth_check_accu(count, requestData, ground_truth_ret, ret, output_path):
    print("Pass the test!")
    count += 1
    # Save requestData, code, ground_truth_ret['data'] into a JsonLine file
    with jsonlines.open(output_path, mode='a') as writer:
        writer.write(requestData)
        writer.write({"Result": "Pass"})
        writer.write({"Ground truth code": ground_truth_ret['reply']})
        writer.write({"LLM code": ret['reply']})
        if ground_truth_ret['type'] != 'graph':
            writer.write({"Ground truth exec": ground_truth_ret['data']})
            writer.write({"LLM code exec": ret['data']})
    return count

def main():
    # create 'output.jsonl' file if it does not exist: 'logs/malt-benchmark/test.jsonl'
    # create the directory if it does not exist
    if not os.path.exists(OUTPUT_JSONL_DIR):
        os.makedirs(OUTPUT_JSONL_DIR)

    # create the file if it does not exist
    output_path = os.path.join(OUTPUT_JSONL_DIR, OUTPUT_JSONL_FILE)
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            pass
    
    # Load all objects from the benchmark dataset into a benchmark list, each object have "question" and "answer" keys
    benchmark_filename = 'data/benchmark_level_1.jsonl'

    # the format is {"messages": [{"question": "XXX."}, {"answer": "YYY"}]}
    benchmark_data = []
    with jsonlines.open(benchmark_filename) as reader:
        for obj in reader:
            benchmark_data.append(obj['messages'])
    
    # for each object in the benchmark list, get the question and answer
    for obj in benchmark_data:
        for item in obj:
            if 'question' in item:
                current_query = item['question']
            # get the answer of the question
            if 'answer' in item:
                golden_answer = item['answer']
        userQuery(current_query, golden_answer, output_path)


if __name__=="__main__":
    main()
