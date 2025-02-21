import os
import subprocess
from generate_policies import generate_yaml_files
from inject_errors import inject_errors_into_policies
from deploy_policies import deploy_policies
import argparse
from correctness_check import correctness_check
# from llm_agent import LLMAgent

# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="Qwen/Qwen2.5-72B-Instruct", help='Choose the LLM agent')
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', type=str, default=['level1'], choices=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/microservices-demo", help='Directory to save output JSONL file')
    parser.add_argument('--max_iteration', type=int, default=20, help='Choose maximum trials for a query')
    parser.add_argument('--full_test', type=int, default=0, choices=[0, 1], help='Enable full test if set to 1')
    return parser.parse_args()

def run_workflow(args):
    # llm = LLMAgent(llm_agent_type=args.llm_agent_type)
    pod_names = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend", "loadgenerator", "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart", "shippingservice"]
    expected_results = {
        "payment": {"http://database-service:5432": True},  
        "analytics": {
            "http://database-service:80": False, 
            "http://payment-service:80": False
        },
        "gateway": {"http://payment-service:80": True}
    } 
    for i in range(1): 

        # Step 1: Deploy pods to Kubernetes
        print("Deploying pods...")
        pod_yamls_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pod_deployments")
        for pod_yaml in os.listdir(pod_yamls_dir):
            if pod_yaml.endswith(".yaml"):
                try:
                    result = subprocess.run(["kubectl", "apply", "-f", os.path.join(pod_yamls_dir, pod_yaml)], check=True, text=True, capture_output=True)
                    print(f"Deployed {pod_yaml}:\n{result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to deploy {pod_yaml}:\n{e.stderr}")

        # Step 3: Inject errors into policies
        print("Injecting errors into policies...")
        inject_errors_into_policies(args.root_dir, args.complexity_level)
        
        # Step 4: Deploy policies
        print("Deploying policies...")
        deploy_policies(pod_names,args.root_dir)

        # Interaction with LLM
        result = "None"
        llm_command = "None"
        for j in range(1):
            print(f"Running LLM iteration {j+1}...")

            # prompt = llm.generate_prompt(llm_command, result)
            # # Use LLM to generate command line code
            # llm_command = llm.llm_agent.call_agent(prompt)
            # print(f"LLM generated command: {llm_command}")

            # # Use subprocess to execute the command line code
            try:
                result = subprocess.run(llm_command, shell=True, check=True, text=True, capture_output=True).stdout
                print(f"Command output:\n{result}")
            except subprocess.CalledProcessError as e:
                print(f"Command failed:\n{e.stderr}")
            if correctness_check(expected_results):
                print(f"Success in iteration {j+1}")
                break

if __name__ == "__main__":
    run_workflow(args=parse_args())