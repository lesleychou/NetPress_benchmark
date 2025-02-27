import os
import subprocess
from inject_errors import inject_errors_into_policies
from deploy_policies import deploy_policies
import argparse
from correctness_check import correctness_check
from correct_policy import copy_yaml_to_new_folder
from llm_agent import LLMAgent
import json
from datetime import datetime
from file_util import file_write

# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="GPT-4o", help='Choose the LLM agent')#"Qwen/Qwen2.5-72B-Instruct"
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', type=str, default=['level1'], choices=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/jiajun_benchmark/app-k8s", help='Directory to save output JSONL file')
    parser.add_argument('--microservice_dir', type=str, default="/home/ubuntu/microservices-demo", help='Directory to google microservice demo')
    parser.add_argument('--max_iteration', type=int, default=20, help='Choose maximum trials for a query')
    parser.add_argument('--full_test', type=int, default=0, choices=[0, 1], help='Enable full test if set to 1')
    return parser.parse_args()

expected_results = {
    "frontend": {
        "adservice:9555": True,
        "cartservice:7070": True,
        "checkoutservice:5050": True,
        "currencyservice:7000": True,
        "productcatalogservice:3550": True,
        "recommendationservice:8080": True,
        "shippingservice:50051": True,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "adservice": {
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "cartservice": {
        "adservice:9555": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": True
    },
    "checkoutservice": {
        "adservice:9555": False,
        "cartservice:7070": True,
        "currencyservice:7000": True,
        "productcatalogservice:3550": True,
        "recommendationservice:8080": False,
        "shippingservice:50051": True,
        "emailservice:5000": True,
        "paymentservice:50051": True,
        "redis-cart:6379": False
    },
    "currencyservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "productcatalogservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "recommendationservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": True,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "shippingservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "emailservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "paymentservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "redis-cart:6379": False
    },
    "redis-cart": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False
    },
    "loadgenerator": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    }
}

def run_workflow(args):
    llm = LLMAgent(llm_agent_type=args.llm_agent_type)
    policy_names = [ "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice", "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend", "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice", "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice" ]
    pod_names = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "loadgenerator", "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart", "shippingservice"]

    # Create result directory and timestamped subdirectory
    result_dir = os.path.join(args.root_dir, "result", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(result_dir, exist_ok=True)
    error_types = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]

    for error_type in error_types:
        for i in range(3): 
            copy_yaml_to_new_folder(args.microservice_dir, args.root_dir)

            # Create a JSON file
            json_file_path = os.path.join(result_dir, f"{error_type}_result_{i}.json")
            with open(json_file_path, 'w') as json_file:
                pass  

            # Create a TXT file
            txt_file_path = os.path.join(result_dir, f"{error_type}_result_{i}.txt")
            with open(txt_file_path, 'w') as txt_file:
                pass  

            # Step 3: Inject errors into policies
            inject_errors_into_policies(policy_names, args.root_dir, args.complexity_level, error_type)
            
            # Step 4: Deploy policies
            deploy_policies(policy_names,args.root_dir)

            # Interaction with LLM
            output = "None"
            llm_command = "None"
            mismatch_summary = {}

            for j in range(10):
                if j == 0:
                    pass
                else:
                    file_write(llm_command, output, mismatch_summary, json_file_path, txt_file_path)
                print(f"Running LLM iteration {j+1}...")

                # Use LLM to generate command line code
                llm_command = llm.llm_agent.call_agent(txt_file_path)
                print(f"Generated LLM command: {llm_command}")

                # Check if llm_command is None
                if llm_command is None:
                    print("Error: llm_command is None")
                    continue

                # Use subprocess to execute the command line code
                try:
                    output = subprocess.run(llm_command, shell=True, check=True, text=True, capture_output=True, timeout=10).stdout
                except subprocess.TimeoutExpired:
                    print(f"Command timed out after 60 seconds")
                    output = "Command timed out"
                except subprocess.CalledProcessError as e:
                    print(f"Command failed:\n{e.stderr}")
                    output = e.stderr
                all_match, mismatch_summary = correctness_check(expected_results)
                if all_match:
                    print(f"Success in iteration {j+1}")
                    file_write(llm_command, output, mismatch_summary, json_file_path, txt_file_path)
                    break

if __name__ == "__main__":
    run_workflow(args=parse_args())
