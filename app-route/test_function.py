from mininet.net import Mininet
from mininet.log import info
from llm_model import LLMModel  
from mininet.log import setLogLevel, info, lg
from llm_model import LLMModel
from mininet_logger import MininetLogger
from file_utils import prepare_file, initialize_json_file, static_summarize_results, summarize_results, error_classification, plot_metrics_from_json, delete_result_folder, plot_combined_error_metrics, plot_metrics, static_plot_metrics
from error_function import inject_errors
from topology import generate_subnets, NetworkTopo, initialize_network
from fast_ping import fastpingall
from safety_check import safety_check, handler
import argparse
import random
import signal
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
from error_function import process_single_error, generate_config
import subprocess




def run(args):
    # Instantiate LLM test taker
    llm_model = LLMModel(model=args.llm_agent_type, vllm=args.vllm)

    for i in range(args.num_queries):
        # Dynamically generate subnets and errors
        if 'level1' in args.complexity_level:
            error_number = 2
            num_hosts = num_switches = random.randint(5, 10)
        elif 'level2' in args.complexity_level:
            error_number = 3
            num_hosts = num_switches = random.randint(8, 12)
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

        # Start logging
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

            # Execute LLM command
            if iter != 0:
                lg.output(f"Machine: {machine}")
                lg.output(f"Command: {commands}")
                
                if safety_check(commands):
                    try:
                        # Set the signal handler and a 100-second alarm
                        signal.signal(signal.SIGALRM, handler)
                        signal.alarm(100)
                        
                        # Try executing the command
                        lg.output(net[machine].cmd(commands))
                        
                        # Disable the alarm after successful execution
                        signal.alarm(0)
                    except TimeoutError as te:
                        lg.output(f"Timeout occurred while executing command on {machine}: {te}")
                    except Exception as e:
                        # Handle the exception, log the error, and continue
                        lg.output(f"Error occurred while executing command on {machine}: {e}") 
                    
            # Pinging all hosts in the network
            fastpingall(net)

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
    plot_metrics_from_json(args.root_dir+'/final_result.json', args.root_dir + '/final_result.png')

def run_full_test(args):
    # Instantiate LLM test taker
    llm_model = LLMModel(model=args.llm_agent_type, vllm=args.vllm)
    args.root_dir = os.path.join(args.root_dir, 'result',args.llm_agent_type, datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Define error types
    error_types = ['disable_routing', 'disable_interface', 'remove_ip', 'drop_traffic_to_from_subnet', 'wrong_routing_table']

    for error_type in error_types:
        for i in range(args.num_queries):
            # Dynamically generate subnets and errors
            num_hosts_per_subnet = random.randint(2, 4)
            num_switches = random.randint(2, 4)
            subnets = generate_subnets(num_switches)
        
            # Instantiate Mininet topo
            topo = NetworkTopo(num_hosts_per_subnet, num_switches, subnets=subnets)
            net = Mininet(topo=topo, waitConnected=True)
            
            # Start Mininet
            net.start()
            
            # Enable IP forwarding on the router
            router = net.get('r0')
            info(router.cmd('sysctl -w net.ipv4.ip_forward=1'))
            
            # Inject errors
            errors = inject_errors(router, subnets, 1, errortype=error_type)
            print(errors)
            
            # Start logging
            Mininet_log = MininetLogger()
            
            # Create directory and file to store result
            result_dir = os.path.join(args.root_dir, 'result', error_type)
            os.makedirs(result_dir, exist_ok=True)
            result_file_path = os.path.join(result_dir, f'result_{i+1}.txt')
            json_path = os.path.join(result_dir, f'result_{i+1}.json')
            prepare_file(result_file_path)
            initialize_json_file(json_path)
            
            # Let LLM interact with Mininet
            iter = 0
            while iter < args.max_iteration:
                # Set up logging
                Mininet_log.setup_logger()
                
                # Execute LLM command
                if iter != 0:
                    lg.output(f"Machine: {machine}")
                    lg.output(f"Command: {commands}")
                    
                    if safety_check(commands):
                        try:
                            # Set the signal handler and a 100-second alarm
                            signal.signal(signal.SIGALRM, handler)
                            signal.alarm(100)
                            
                            # Try executing the command
                            lg.output(net[machine].cmd(commands))
                            
                            # Disable the alarm after successful execution
                            signal.alarm(0)
                        except TimeoutError as te:
                            lg.output(f"Timeout occurred while executing command on {machine}: {te}")
                        except Exception as e:
                            # Handle the exception, log the error, and continue
                            lg.output(f"Error occurred while executing command on {machine}: {e}") 
                            
                # Pinging all hosts in the network
                fastpingall(net)
                
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
    
        summarize_results(os.path.join(args.root_dir, 'result', error_type), os.path.join(args.root_dir, 'result', error_type, f'{error_type}_result.json'))


def combined_error_test(args):
    # Instantiate LLM test taker
    llm_model = LLMModel(model=args.llm_agent_type)
    print(args.root_dir)    
    # args.root_dir = os.path.join(args.root_dir, 'result',args.llm_agent_type)
    print(args.root_dir)
    # Define error types
    error_types = ['disable_routing', 'disable_interface', 'remove_ip', 'drop_traffic_to_from_subnet', 'wrong_routing_table']
    
    # Generate all combinations of two errors from error_types
    error_combinations = [(error_types[i], error_types[j]) for i in range(len(error_types)) for j in range(i+1, len(error_types))]
    
    for i, (error1, error2) in enumerate(error_combinations):
        # Dynamically generate subnets and errors
        num_hosts_per_subnet = random.randint(2, 4)
        num_switches = random.randint(2, 4)
        subnets = generate_subnets(num_switches)

        # Instantiate Mininet topo
        topo = NetworkTopo(num_hosts_per_subnet, num_switches, subnets=subnets)
        net = Mininet(topo=topo, waitConnected=True)
        
        # Start Mininet
        net.start()
        
        # Enable IP forwarding on the router
        router = net.get('r0')
        info(router.cmd('sysctl -w net.ipv4.ip_forward=1'))
        
        # Inject errors
        errors = inject_errors(router, subnets, 1, errortype=error1)
        errors += inject_errors(router, subnets, 1, errortype=error2)
        print(errors)
        
        # Start logging
        Mininet_log = MininetLogger()
        
        # Create directory and file to store result
        result_dir = os.path.join(os.path.join(args.root_dir, 'result', 'combined_test_results'), f'test_{i+1}')
        os.makedirs(result_dir, exist_ok=True)
        result_file_path = os.path.join(result_dir, f'result_{i+1}.txt')
        json_path = os.path.join(result_dir, f'result_{i+1}.json')
        prepare_file(result_file_path)
        initialize_json_file(json_path)
        
        # Let LLM interact with Mininet
        iter = 0
        while iter < args.max_iteration:
            # Set up logging
            Mininet_log.setup_logger()
            
            # Execute LLM command
            if iter != 0:
                lg.output(f"Machine: {machine}")
                lg.output(f"Command: {commands}")
                
                if safety_check(commands):
                    try:
                        # Set the signal handler and a 100-second alarm
                        signal.signal(signal.SIGALRM, handler)
                        signal.alarm(100)
                        
                        # Try executing the command
                        lg.output(net[machine].cmd(commands))
                        
                        # Disable the alarm after successful execution
                        signal.alarm(0)
                    except TimeoutError as te:
                        lg.output(f"Timeout occurred while executing command on {machine}: {te}")
                    except Exception as e:
                        # Handle the exception, log the error, and continue
                        lg.output(f"Error occurred while executing command on {machine}: {e}") 
                        
            # Pinging all hosts in the network
            fastpingall(net)
            
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
    
    plot_combined_error_metrics(args.root_dir, error_combinations)

def static_benchmark_run(args):
    # Run the command with sudo
    subprocess.run(["sudo", "mn", "-c"], check=True)
    file_path = args.root_dir + '/error_config.json'
    if args.static_benchmark_generation == 1:
        generate_config(file_path, args.num_queries)  

    # Instantiate LLM test taker
    llm_model = LLMModel(model=args.llm_agent_type, vllm=args.vllm, prompt_type="base")
    args.root_dir = os.path.join(args.root_dir, 'result',args.llm_agent_type, datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Define error types
    with open(file_path, 'r') as f:
        config = json.load(f)
    queries = config.get("queries", [])

    print(f"Number of queries: {len(queries)}")
    for i, query in enumerate(queries):
        start_time = datetime.now()
        info(f'*** Injecting errors for query {i}\n')
        num_hosts_per_subnet = query.get("num_hosts_per_subnet", 1)
        num_switches = query.get("num_switches")
        errortype = query.get("errortype")
        errordetail = query.get("errordetail")
        errornumber = query.get("errornumber")
        print("error_type:",errortype)
        print("error_number:",errornumber)
        print("error_detail:",errordetail)
        
        subnets, topo, net, router = initialize_network(num_hosts_per_subnet, num_switches)

        
        # Inject errors
        if errornumber == 1:
            process_single_error(router, subnets, errortype, errordetail)
        else:
            if isinstance(errortype, list) and isinstance(errordetail, list) and len(errortype) == errornumber and len(errordetail) == errornumber:
                for et, ed in zip(errortype, errordetail):
                    process_single_error(router, subnets, et, ed)
            else:
                info('*** For multiple error injection, errortype and errordetail must be lists of length equal to errornumber\n')

        # Start logging
        Mininet_log = MininetLogger()
        
        # Create directory and file to store result
        if isinstance(errortype, list):
            errortype = '+'.join(errortype)  

        result_dir = os.path.join(args.root_dir, errortype)

        os.makedirs(result_dir, exist_ok=True)
        result_file_path = os.path.join(result_dir, f'result_{i+1}.txt')
        json_path = os.path.join(result_dir, f'result_{i+1}.json')
        prepare_file(result_file_path)
        initialize_json_file(json_path)
        
        # Let LLM interact with Mininet
        iter = 0
        while iter < args.max_iteration:
            # Set up logging
            Mininet_log.setup_logger()
            
            # Execute LLM command
            if iter != 0:
                lg.output(f"Machine: {machine}")
                lg.output(f"Command: {commands}")
                
                if safety_check(commands):
                    try:
                        # Set the signal handler and a 100-second alarm
                        signal.signal(signal.SIGALRM, handler)
                        signal.alarm(100)
                        
                        # Try executing the command
                        lg.output(net[machine].cmd(commands))
                        
                        # Disable the alarm after successful execution
                        signal.alarm(0)
                    except TimeoutError as te:
                        lg.output(f"Timeout occurred while executing command on {machine}: {te}")
                    except Exception as e:
                        # Handle the exception, log the error, and continue
                        lg.output(f"Error occurred while executing command on {machine}: {e}") 
                        
            # Pinging all hosts in the network
            start_time = datetime.now()
            net.pingAll(timeout=0.01)
            end_time = datetime.now()
            print(f"Time taken for pingAll: {end_time - start_time}")

            # Read log file content
            log_content = Mininet_log.get_log_content()
            print(log_content)
            
            # Get LLM response
            machine, commands = llm_model.model.predict(log_content, result_file_path, json_path)
            print("machine:", machine)
            print("commands:", commands)
            # # Read log content, if successful then breaks
            if Mininet_log.read_log_content(log_content, iter):
                break
            
            iter += 1
            
        net.stop()
        end_time = datetime.now()
        print(f"Time taken for query {i}: {end_time - start_time}")

    for subdir in os.listdir(args.root_dir):
        subdir_path = os.path.join(args.root_dir, subdir)
        if os.path.isdir(subdir_path):
            json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
            static_summarize_results(subdir_path, json_result_path)

    static_plot_metrics(args.root_dir)