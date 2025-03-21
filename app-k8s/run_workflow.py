import os
import subprocess
from deploy_policies import deploy_policies
import argparse
from correctness_check import correctness_check, create_debug_container
from correct_policy import copy_yaml_to_new_folder
from llm_agent import LLMAgent
import json
from datetime import datetime
from file_util import file_write, summary_tests, plot_metrics, plot_summary_results
from inject_errors import inject_config_errors_into_policies, generate_config
import shutil
import time

# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="Qwen/Qwen2.5-72B-Instruct", choices=["Qwen/Qwen2.5-72B-Instruct", "GPT-4o"], help='Choose the LLM agent')#"Qwen/Qwen2.5-72B-Instruct"，"GPT-4o"
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', type=str, default=['level1'], choices=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/jiajun_benchmark/app-k8s", help='Directory to save output JSONL file')
    parser.add_argument('--microservice_dir', type=str, default="/home/ubuntu/microservices-demo", help='Directory to google microservice demo')
    parser.add_argument('--max_iteration', type=int, default=10, help='Choose maximum trials for a query')
    parser.add_argument('--full_test', type=int, default=1, choices=[0, 1], help='Enable full test if set to 1')
    parser.add_argument('--error_config', type=int, default=1, choices=[0, 1], help='Choose whether to use the pregenerated config')
    parser.add_argument('--config_gen', type=int, default=1, help='Choose whether to generate new config')
    parser.add_argument('--prompt_type', type=str, default="few_shot_semantic", choices=["base", "cot", "few_shot_basic", "few_shot_semantic"], help='Choose the prompt type')
    parser.add_argument('--agent_test', type=int, default=1, choices=[0, 1], help='Choose whether to run the agent test')
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
def deploy_k8s_cluster(skaffold_config_path: str):
    """
    Deletes the existing kind cluster, creates a new one, and deploys an application using skaffold.
    
    :param skaffold_config_path: Path to the skaffold configuration directory.
    """
    try:
        print("Deleting existing kind cluster...")
        subprocess.run(["kind", "delete", "cluster"], check=True)
        
        print("Creating a new kind cluster...")
        subprocess.run(["kind", "create", "cluster"], check=True)
        
        print("Deploying application using skaffold...")
        subprocess.run(["skaffold", "run"], cwd=skaffold_config_path, check=True)
        
        print("Deployment completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def run_config_error(args):
    llm = LLMAgent(llm_agent_type=args.llm_agent_type,  prompt_type=args.prompt_type)
    policy_names = [ "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice", "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend", "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice", "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice" ]
    pod_names = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "loadgenerator", "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart", "shippingservice"]
    # Create result directory and timestamped subdirectory
    if args.agent_test == 0:
        result_dir = os.path.join(args.root_dir, "result", args.llm_agent_type, datetime.now().strftime("%Y%m%d_%H%M%S"))
    else:
        result_dir = os.path.join(args.root_dir, args.prompt_type)
    
    os.makedirs(result_dir, exist_ok=True)
    if args.config_gen == 1:
        generate_config(args.root_dir, policy_names, args.num_queries)

    # Read error configuration
    error_config_path = os.path.join(args.root_dir, "error_config.json")
    with open(error_config_path, 'r') as error_config_file:
        error_config = json.load(error_config_file)

    total_error_num = len(error_config["details"])
    print(f"Total number of errors: {total_error_num}")

    txt_file_path = os.path.join(args.root_dir, "test.txt")

    debug_container_mapping = {}
    for pod_name in pod_names:
        debug_container_name = create_debug_container(pod_name)
        if debug_container_name:
            debug_container_mapping[pod_name] = debug_container_name
    print(debug_container_mapping)

    for i, error in enumerate(error_config["details"]):
        policies_to_inject = error.get("policies_to_inject", [])  
        inject_error_num = error.get("inject_error_num", [])  
        error_detail = error.get("error_detail", [])  
        
        print(f"Error {i+1}:")
        print(f"  Policies to inject: {policies_to_inject}")
        print(f"  Inject error numbers: {inject_error_num}")
        print(f"  Error details: {error_detail}")

        copy_yaml_to_new_folder(args.microservice_dir, args.root_dir)

        # Combine error_detail into a single string
        error_detail_str = "+".join([detail["type"] for detail in error_detail])

        # Create a JSON file
        json_file_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.json")
        with open(json_file_path, 'w') as json_file:
            pass  

        # Create a TXT file
        txt_file_path = os.path.join(result_dir, f"{error_detail_str}_result_{i}.txt")
        with open(txt_file_path, 'w') as txt_file:
            pass  

        # Step 3: Inject errors into policies
        inject_config_errors_into_policies(policy_names, args.root_dir, inject_error_num, policies_to_inject, error_detail)
        
        # Step 4: Deploy policies
        deploy_policies(policy_names, args.root_dir)

        # Interaction with LLM
        output = "None"
        llm_command = "None"
        mismatch_summary = {}

        for k in range(args.max_iteration):
            if k == 0:
                pass
            else:
                file_write(llm_command, output, mismatch_summary, json_file_path, txt_file_path)
            print(f"Running LLM iteration {k+1}...")
            if args.llm_agent_type == "GPT-4o":
                time.sleep(3)
            # Use LLM to generate command line code
            llm_command = llm.llm_agent.call_agent(txt_file_path)
            print(f"Generated LLM command: {llm_command}")

            # Check if llm_command is None
            if llm_command is None:
                print("Error: llm_command is None")
                continue

            # Use subprocess to execute the command line code
            try:
                output = subprocess.run(llm_command,shell=True,executable='/bin/bash',check=True,text=True,capture_output=True,timeout=10).stdout
            except subprocess.TimeoutExpired:
                print(f"Command timed out after 60 seconds")
                output = "Command timed out"
            except subprocess.CalledProcessError as e:
                print(f"Command failed:\n{e.stderr}")
                output = e.stderr
            all_match, mismatch_summary = correctness_check(expected_results, debug_container_mapping)
            if all_match:
                print(f"Success in iteration {k+1}")
                file_write(llm_command, output, mismatch_summary, json_file_path, txt_file_path)
                break
        
    summary_tests(result_dir)
    plot_metrics(result_dir)

def run_agent_test(args):
    args.root_dir = os.path.join(args.root_dir, "result", args.llm_agent_type,"agent_test", datetime.now().strftime("%Y%m%d_%H%M%S"))
    for i in range(4):
        if i == 0:
            deploy_k8s_cluster("/home/ubuntu/microservices-demo")
            args.prompt_type = "base"
            args.generate_config = 1
            run_config_error(args)
        elif i == 1:
            deploy_k8s_cluster("/home/ubuntu/microservices-demo")
            args.generate_config = 0
            args.prompt_type = "cot"
            run_config_error(args)
        elif i == 2:
            deploy_k8s_cluster("/home/ubuntu/microservices-demo")
            args.prompt_type = "few_shot_basic"
            args.generate_config = 0
            run_config_error(args)
        elif i == 3: 
            deploy_k8s_cluster("/home/ubuntu/microservices-demo")   
            args.prompt_type = "few_shot_semantic"
            args.generate_config = 0
            run_config_error(args)

    policies_dir = os.path.join(args.root_dir, "policies")
    if os.path.exists(policies_dir):
        shutil.rmtree(policies_dir)
    
    plot_summary_results(args.root_dir)

if __name__ == "__main__":
# starttime = datetime.now()
    # run_config_error(args=parse_args())
    # endtime = datetime.now()
    # print(f"Total time: {endtime - starttime}")
    starttime = datetime.now()
    args = parse_args()
    if args.agent_test == 1:
        run_agent_test(args=parse_args())
    else:
        run_config_error(args=parse_args())
    endtime = datetime.now()
    # print(f"Total time: {endtime - starttime}")
    # deploy_k8s_cluster("/home/ubuntu/microservices-demo")
