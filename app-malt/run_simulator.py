import json
import traceback
from dotenv import load_dotenv
import openai
import copy
import pandas as pd
from collections import Counter
from prototxt_parser.prototxt import parse
import os
from helper import getGraphData, clean_up_llm_output_func, check_list_equal, node_attributes_are_equal, clean_up_output_graph_data
from error_check import MyChecker
import networkx as nx
import jsonlines
import random
from networkx.readwrite import json_graph
import json
import re
import time
import sys
import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_attempt


# output the evaluation results to a jsonl file
OUTPUT_JSONL_DIR = 'logs/malt-benchmark'
OUTPUT_JSONL_FILE = 'test.jsonl'


def userQuery(current_query, golden_answer, output_path):
    # for each prompt in the prompt_list, append it as the value of {'query': prompt}
    print("Query: ", current_query)
    requestData = {'query': current_query}

    # import pdb; pdb.set_trace()
    prompt_accu = 0
    _, G = getGraphData()
    
    # TODO: @Jiajun, please get the code from the LLM model
    # This line is just to simulate, if the LLM output code is correct
    llm_answer = "def process_graph(graph_data):\n    packet_switch_node = None\n    for node in graph_data.nodes(data=True):\n        if node[0] == 'ju1.a1.m1.s2c1' and 'EK_PACKET_SWITCH' in node[1]['type']:\n            packet_switch_node = node[0]\n            break\n    if packet_switch_node is None:\n        return {'type': 'text', 'data': 'No packet switch with the given name found.'}\n    \n    ports = []\n    for edge in graph_data.edges(data=True):\n        if edge[0] == packet_switch_node and 'RK_CONTAINS' in edge[2]['type']:\n            destination_node = graph_data.nodes[edge[1]]\n            if 'EK_PORT' in destination_node['type']:\n                ports.append(edge[1])\n    \n    return {'type': 'list', 'data': ports}"

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
    benchmark_filename = 'golden_answer_generator/benchmark_data/app_malt_dataset.jsonl'
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

        # TODO: @Jiajun, once you connect llm_answer to real llm_output, you can remove the following break
        break
    

if __name__=="__main__":
    main()
