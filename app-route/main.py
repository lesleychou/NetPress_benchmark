
from mininet.net import Mininet
from mininet.log import info
import logging
from llm_model import LLMModel  
from mininet.log import setLogLevel, info, lg
from llm_model import LLMModel
from mininet_logger import MininetLogger
from file_utils import prepare_file
from error_function import inject_errors
from topology import generate_subnets, NetworkTopo
import time
import sys
from mininet.log import setLogLevel


# Define the root path (you should set this to the correct path for your project)
root_path = "/home/ubuntu/nemo_benchmark/app-route/result" 

def run(test_taker, max_iter, num_hosts, num_switches):
    # Dynamically generate subnets
    subnets = generate_subnets(num_hosts, num_switches)

    # Instantiate Mininet topo
    topo = NetworkTopo(num_hosts=num_hosts, num_switches=num_switches, subnets=subnets)
    net = Mininet(topo=topo, waitConnected=True)

    # Instantiate LLM test taker
    llm_model = LLMModel(model=test_taker)

    # Start Mininet
    net.start()

    # Enable IP forwarding on the router
    router = net.get('r0')
    info(router.cmd('sysctl -w net.ipv4.ip_forward=1'))

    # Inject errors
    inject_errors(router, subnets, error_number=1)
    Mininet_log = MininetLogger()

    # Create file to store result
    result_file_path = root_path + '_result.txt'
    safety_file_path = root_path + '_safety.txt'
    prepare_file(result_file_path)
    prepare_file(safety_file_path)

    # Let LLM interact with Mininet
    iter = 0
    while iter < max_iter:
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
        machine, commands = llm_model.model.predict(log_content, result_file_path, safety_file_path)
        
        # # Read log content, if successful then breaks
        if Mininet_log.read_log_content(log_content, iter):
            break

        iter += 1

    # net.stop()

# Call the run function to run the test
if __name__ == "__main__":
    max_iter = 10
    num_hosts = 4 
    num_switches = 4
    run("Qwen/Qwen2.5-72B-Instruct", max_iter, num_hosts, num_switches)# "meta-llama/Meta-Llama-3.1-70B-Instruct"

