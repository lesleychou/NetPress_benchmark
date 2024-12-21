from mininet.net import Mininet
from mininet.log import info
from llm_model import LLMModel  
from mininet.log import setLogLevel, info, lg
from llm_model import LLMModel
from mininet_logger import MininetLogger
from file_utils import prepare_file, initialize_json_file, summarize_results, error_classification
from error_function import inject_errors
from topology import generate_subnets, NetworkTopo
import argparse
import random


# Define a configuration for the benchmark
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Configuration")
    parser.add_argument('--llm_agent_type', type=str, default="Qwen/Qwen2.5-72B-Instruct", help='Choose the LLM agent', choices=["meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"])
    parser.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate for each type')
    parser.add_argument('--complexity_level', nargs='+', default=['level1', 'level2'], help='Complexity level of queries to generate')
    parser.add_argument('--root_dir', type=str, default="/home/ubuntu/nemo_benchmark/app-route", help='Directory to save output JSONL file')
    parser.add_argument('--max_iteration', type=int, default=20, help='Choose maximum trials for a query')
    return parser.parse_args()


def run(args):
    # Instantiate LLM test taker
    llm_model = LLMModel(model=args.llm_agent_type)

    for i in range(args.num_queries):

        # Dynamically generate subnets and errors
        if 'level1' in args.complexity_level:
            error_number = 2
            num_hosts = num_switches = random.randint(5, 10)
        elif 'level2' in args.complexity_level:
            error_number = 3
            num_hosts = num_switches = random.randint(10, 20)
        subnets = generate_subnets(num_hosts, num_switches)

        # Instantiate Mininet topo
        topo = NetworkTopo(num_hosts=num_hosts, num_switches=num_switches, subnets=subnets)
        net = Mininet(topo=topo, waitConnected=True)

        # Start Mininet
        net.start()

        # Enable IP forwarding on the router
        router = net.get('r0')
        info(router.cmd('sysctl -w net.ipv4.ip_forward=1'))

        # Inject errors
        errors = inject_errors(router, subnets, error_number)
        print(errors)
        Mininet_log = MininetLogger()

        # Create file to store result
        result_file_path = args.root_dir + '/result.txt'
        json_path = args.root_dir + f'/result/result_{i+1}.json'
        prepare_file(result_file_path)
        initialize_json_file(json_path)

        # Let LLM interact with Mininet
        iter = 0
        while iter < args.max_iteration:
            # Set up logging
            Mininet_log.setup_logger()

            # # Execute LLM command
            if iter != 0:
                lg.output(f"Machine: {machine}")
                lg.output(f"Command: {commands}")
                try:
                # # Try executing the command
                    lg.output(net[machine].cmd(commands))
                except Exception as e:
                    # Handle the exception, log the error, and continue
                    lg.output(f"Error occurred while executing command on {machine}: {e}") 
                    
            # Pinging all hosts in the network
            net.pingAll()

            # Read log file content
            log_content = Mininet_log.get_log_content()

            # Get LLM response
            machine, commands = llm_model.model.predict(log_content, result_file_path, json_path)
            
            # # Read log content, if successful then breaks
            if Mininet_log.read_log_content(log_content, iter):
                break

            iter += 1

        net.stop()
        error_classification(errors, json_path)
    summarize_results(args.root_dir+'/result', args.root_dir+'/final_result.json')

# Call the run function to run the test
if __name__ == "__main__":
    args = parse_args()
    run(args)
