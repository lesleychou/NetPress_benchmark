from mininet.net import Mininet
from mininet.log import info
from llm_model import LLMModel  
from mininet.log import setLogLevel, info, lg
from mininet_logger import MininetLogger
from file_utils import prepare_file, initialize_json_file, static_summarize_results, summarize_results, error_classification, plot_metrics_from_json, delete_result_folder, plot_combined_error_metrics, plot_metrics, static_plot_metrics
# from error_function import inject_errors
from topology import generate_subnets, NetworkTopo, initialize_network
from fast_ping import fastpingall, parallelPing
from safety_check import safety_check, handler
from mininet.cli import CLI
import argparse
import random
import signal
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
# from error_function import process_single_error, generate_config
import subprocess
import time
from multiprocessing import Process
from file_utils import process_results, plot_results
from advanced_error_function import generate_config, process_single_error
import shutil

def run(args):
    """
    Run a single benchmark test with the given arguments.

    Args:
        args (argparse.Namespace): The parsed arguments containing configuration.
    """
    # Instantiate the LLM model
    llm_model = LLMModel(model=args.llm_agent_type, vllm=args.vllm)

    for i in range(args.num_queries):
        # Dynamically generate subnets and errors based on complexity level
        if 'level1' in args.complexity_level:
            error_number = 2
            num_hosts = num_switches = random.randint(5, 10)
        elif 'level2' in args.complexity_level:
            error_number = 3
            num_hosts = num_switches = random.randint(8, 12)
        subnets = generate_subnets(num_hosts, num_switches)

        # Instantiate the Mininet topology
        topo = NetworkTopo(num_hosts=num_hosts, num_switches=num_switches, subnets=subnets)
        net = Mininet(topo=topo, waitConnected=True)

        # Start Mininet
        net.start()

        # Enable IP forwarding on the router
        router = net.get('r0')
        info(router.cmd('sysctl -w net.ipv4.ip_forward=1'))

        # Inject errors into the network
        errors = inject_errors(router, subnets, error_number)
        print(errors)

        # Start logging
        Mininet_log = MininetLogger()

        # Create files to store results
        result_file_path = args.root_dir + '/result.txt'
        json_path = args.root_dir + f'/result/result_{i+1}.json'
        prepare_file(result_file_path)
        initialize_json_file(json_path)

        # Let the LLM interact with Mininet
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
                    
            # Ping all hosts in the network
            fastpingall(net)

            # Read log file content
            log_content = Mininet_log.get_log_content()

            # Get LLM response
            machine, commands = llm_model.model.predict(log_content, result_file_path, json_path)
            
            # Check log content, if successful then break
            if Mininet_log.read_log_content(log_content, iter):
                break

            iter += 1

        # Stop the Mininet instance
        net.stop()
        error_classification(errors, json_path)

    # Summarize results and plot metrics
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
    if args.agent_test == 1:
        print ("Running agent test for ", args.prompt_type)
    # Instantiate LLM test taker
    llm_model = LLMModel(model=args.llm_agent_type, vllm=args.vllm, prompt_type=args.prompt_type)
    if args.agent_test == 0:
        args.root_dir = os.path.join(args.root_dir, 'result',args.llm_agent_type, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        if args.llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
            args.root_dir = os.path.join(args.root_dir, args.prompt_type+"_Qwen")
        else:
            args.root_dir = os.path.join(args.root_dir, args.prompt_type+"_GPT")
    # Define error types
    with open(file_path, 'r') as f:
        config = json.load(f)
    queries = config.get("queries", [])

    print(f"Number of queries: {len(queries)}")
    for i, query in enumerate(queries):
        start_time_1 = datetime.now()
        info(f'*** Injecting errors for query {i}\n')
        num_hosts_per_subnet = query.get("num_hosts_per_subnet", 1)
        num_switches = query.get("num_switches")
        errortype = query.get("errortype")
        errordetail = query.get("errordetail")
        errornumber = query.get("errornumber")
        print("error_type:",errortype)
        print("error_number:",errornumber)
        print("error_detail:",errordetail)
        print("num_hosts_per_subnet:",num_hosts_per_subnet)
        print("num_switches:",num_switches)
        # errortype = "drop_traffic_to_from_subnet"
        # errornumber = 1
        # num_hosts_per_subnet = 3
        # num_switches = 3
        # errordetail ={
        #         "subnet": "192.168.3.0/24"
        #     }
        start_time = datetime.now()
        subnets, topo, net, router = initialize_network(num_hosts_per_subnet, num_switches)
        end_time = datetime.now()
        print(f"Time taken for network initialization: {end_time - start_time}")
        
        # Inject errors
        if errornumber == 1:
            process_single_error(router, subnets, errortype, errordetail)
        else:
            if isinstance(errortype, list) and isinstance(errordetail, list) and len(errortype) == errornumber and len(errordetail) == errornumber:
                for et, ed in zip(errortype, errordetail):
                    process_single_error(router, subnets, et, ed)
            else:
                info('*** For multiple error injection, errortype and errordetail must be lists of length equal to errornumber\n')
        
        # Create directory and file to store result
        if isinstance(errortype, list):
            errortype = '+'.join(errortype)  

        # Start logging
        Mininet_log = MininetLogger()

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
            Mininet_log.setup_logger(errortype, log_dir='logs')
            
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
                        print("LLM command executed successfully")
                        # Disable the alarm after successful execution
                        signal.alarm(0)
                    except TimeoutError as te:
                        lg.output(f"Timeout occurred while executing command on {machine}: {te}")
                    except Exception as e:
                        # Handle the exception, log the error, and continue
                        lg.output(f"Error occurred while executing command on {machine}: {e}") 


                        
            # Pinging all hosts in the network
            start_time = datetime.now()
            try:
                net.pingAll(timeout=0.02)
            except Exception as e:
                print(f"Fatal error during pingAll: {e}")
            # fastpingall(net, timeout=5)
            end_time = datetime.now()
            print(f"Time taken for pingAll: {end_time - start_time}")

            # Read log file contents
            log_content = Mininet_log.get_log_content()
            print(log_content)
            
            # Get LLM response
            attempt = 0
            while True:
                attempt += 1
                print(f"Attempt {attempt}: Calling LLM...")
                try:
                    machine, commands = llm_model.model.predict(log_content, result_file_path, json_path)
                    print(f"Generated LLM command: {machine} {commands}")
                    break
                except Exception as e:
                    print(f"Error while generating LLM command: {e}")
                    time.sleep(3)

            # # Read log content, if successful then breaks
            if Mininet_log.read_log_content(log_content, iter):
                break
            
            iter += 1
            
        net.stop()
        end_time_1 = datetime.now()
        print(f"Time taken for query {i}: {end_time_1 - start_time_1}")

    for subdir in os.listdir(args.root_dir):
        subdir_path = os.path.join(args.root_dir, subdir)
        if os.path.isdir(subdir_path):
            json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
            static_summarize_results(subdir_path, json_result_path)

    static_plot_metrics(args.root_dir)


def static_benchmark_run_modify(args):
    """
    Run a separate Mininet instance for each benchmark test.
    Assign a unique root directory for each instance.
    """

    start_time_2 = datetime.now()
    # Get the unique process ID to distinguish between different instances
    unique_id = os.getpid()
    args.root_dir = os.path.join(args.root_dir)
    os.makedirs(args.root_dir, exist_ok=True)

    # Generate or load the error configuration file
    file_path = os.path.join(args.root_dir, 'error_config.json')

    print(f"Process {unique_id}: Running benchmark with prompt type {args.prompt_type}")
    print(file_path)
    # Load the error configuration
    with open(file_path, 'r') as f:
        config = json.load(f)
    queries = config.get("queries", [])

    print(f"Number of queries: {len(queries)}")

    # Initialize the LLM model
    llm_model = LLMModel(model=args.llm_agent_type, vllm=args.vllm, prompt_type=args.prompt_type)
    print("agenttype", args.llm_agent_type)
    if args.llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
        result_path = os.path.join(args.root_dir, args.prompt_type+"_Qwen")
    elif args.llm_agent_type == "all-hands/openhands-lm-32b-v0.1":
        result_path = os.path.join(args.root_dir, args.prompt_type+"Qwen_32B")
    else:      
        result_path = os.path.join(args.root_dir, args.prompt_type+"_GPT")
    for i, query in enumerate(queries[504:], start=505):
        start_time_1 = datetime.now()
        print(f'Process {unique_id}: Injecting errors for query {i}')

        # Extract parameters from the query
        num_hosts_per_subnet = query.get("num_hosts_per_subnet", 1)
        num_switches = query.get("num_switches")
        errortype = query.get("errortype")
        errordetail = query.get("errordetail")
        errornumber = query.get("errornumber")
        # print("error_type:",errortype)
        # print("error_number:",errornumber)
        # print("error_detail:",errordetail)
        # print("num_hosts_per_subnet:",num_hosts_per_subnet)
        # print("num_switches:",num_switches)
        # errortype = "wrong_routing_table"
        # errornumber = 1
        # num_hosts_per_subnet = 2
        # num_switches = 4
        # errordetail = {
        #     "from": "192.168.3.0/24",
        #     "to": "192.168.1.0/24",
        #     "del_interface": "r0-eth3",
        #     "add_interface": "r0-eth1",
        #     "method": 5,
        #     "to_ip": "192.168.1.1"
        # }


        print(f"Process {unique_id}: Initializing Mininet instance")
        start_time = datetime.now()

        # Initialize the network
        subnets, topo, net, router = initialize_network(num_hosts_per_subnet, num_switches, unique_id)

        end_time = datetime.now()
        print(f"Process {unique_id}: Network initialization took {end_time - start_time}")

        # Inject errors into the network
        if errornumber == 1:
            print(f"Process {unique_id}: Injecting single error")
            process_single_error(router, subnets, errortype, errordetail, unique_id)
        else:
            if isinstance(errortype, list) and isinstance(errordetail, list) and len(errortype) == errornumber and len(errordetail) == errornumber:
                for et, ed in zip(errortype, errordetail):
                    process_single_error(router, subnets, et, ed, unique_id)
            else:
                print(f"Process {unique_id}: Error: For multiple error injection, errortype and errordetail must be lists of length equal to errornumber")
                continue
        # CLI(net)   
        if isinstance(errortype, list):
            errortype = '+'.join(errortype)  
        # Create result directory and files
        result_dir = os.path.join(result_path, errortype)
        os.makedirs(result_dir, exist_ok=True)

        result_file_path = os.path.join(result_dir, f'result_{i+1}.txt')
        json_path = os.path.join(result_dir, f'result_{i+1}.json')

        prepare_file(result_file_path)
        initialize_json_file(json_path)

        # LLM interacts with Mininet
        iter = 0
        while iter < args.max_iteration:


            # Execute LLM command
            if iter != 0:

                lg.output(f"Machine: {machine}")
                lg.output(f'{iter} iteration')
                lg.output(f"Command: {commands}")

                if safety_check(commands):
                    try:
                        # Try executing the command
                        command_output = net[machine].cmd(commands)
                        print("LLM command executed successfully")

                    except TimeoutError as te:
                        lg.output(f"Timeout occurred while executing command on {machine}: {te}")
                    except Exception as e:
                        # Handle exceptions, log the error, and continue
                        lg.output(f"Error occurred while executing command on {machine}: {e}")

            # Ping all hosts in the network
            start_time = datetime.now()
            try:
                pingall, loss_percent = parallelPing(net, timeout=0.1)
            except Exception as e:
                print(f"Process {unique_id}: Error during pingAll: {e}")
                if e == "Command execution timed out":
                    break
            end_time = datetime.now()
            print(f"Time taken for pingAll: {end_time - start_time}")
            
            # Read log file content
            if iter != 0:
                log_content = f"Machine: {machine}\n" + f"Command: {commands}\n" + command_output + f"Pingall result: {pingall}\n"
            else:
                log_content = f"Pingall result: {pingall}\n"
            print("log_content: ", log_content)

            # Get LLM response
            attempt = 0
            while True:
                attempt += 1
                print(f"Attempt {attempt}: Calling LLM...")
                try:
                    machine, commands = llm_model.model.predict(log_content, result_file_path, json_path)
                    print(f"Generated LLM command: {machine} {commands}")
                    break
                except Exception as e:
                    print(f"Error while generating LLM command: {e}")
                    time.sleep(3)

            # Check log content, exit loop if successful
            if loss_percent == 0:
                print(f"Query {i}: Success in {iter} iterations")
                break
            end_time = datetime.now()
            print(f"Time taken for LLM response: {end_time - start_time}")
            iter += 1

        # Stop the Mininet instance
        print(f"Process {unique_id}: Stopping Mininet instance")
        net.stop()

        end_time_1 = datetime.now()
        print(f"Process {unique_id}: Time taken for query {i}: {end_time_1 - start_time_1}")

    print(f"Process {unique_id}: Benchmark finished for {args.prompt_type}")



    for subdir in os.listdir(result_path):
        subdir_path = os.path.join(result_path, subdir)
        if os.path.isdir(subdir_path):
            json_result_path = os.path.join(subdir_path, f'{subdir}_result.json')
            static_summarize_results(subdir_path, json_result_path)

    static_plot_metrics(result_path)
    end_time_2 = datetime.now()
    print(f"Process {unique_id}: Total time taken for all queries: {end_time_2 - start_time_2}")


def run_benchmark_parallel(args):
    """
    Run static benchmark tests in parallel using multiple processes.

    Args:
        args (argparse.Namespace): The parsed arguments containing configuration.
    """
    # Clean up any existing Mininet resources
    subprocess.run(["sudo", "mn", "-c"], check=True)

    # Create a directory to save results
    # save_result_path = os.path.join(args.root_dir, 'result', args.llm_agent_type, "agenttest", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # os.makedirs(save_result_path, exist_ok=True)
    save_result_path = "/home/ubuntu/nemo_benchmark/app-route/result/GPT-Agent/agenttest/20250421-182502"
    # Update the root directory in args
    args.root_dir = save_result_path

    # # Generate the error configuration file
    # generate_config(os.path.join(save_result_path, "error_config.json"), num_errors_per_type=args.num_queries)

    # Define a wrapper function to run static benchmarks
    def run_static_benchmark(prompt_type, static_benchmark_generation,llm_agent_type):
        """
        Wrapper function to create an independent args instance per process.
        This ensures no conflicts between parallel processes.
        """
        args_copy = argparse.Namespace(**vars(args))  # Deep copy args to avoid conflicts
        args_copy.prompt_type = prompt_type
        args_copy.llm_agent_type = llm_agent_type
        args_copy.static_benchmark_generation = static_benchmark_generation
        static_benchmark_run_modify(args_copy)

    # Get the list of prompt types from args (comma-separated)
    prompt_types = ["cot", "few_shot_basic"]
    # prompt_types = ["cot"]
    # Create and start processes for each prompt type
    processes = []
    # for prompt_type in prompt_types:
    #     process = Process(target=run_static_benchmark, args=(prompt_type, args.static_benchmark_generation, args.llm_agent_type))
    #     processes.append(process)
    #     process.start()

    process = Process(target=run_static_benchmark, args=("cot", args.static_benchmark_generation,"Qwen/Qwen2.5-72B-Instruct"))
    processes.append(process)
    process.start()
    # Wait for all processes to complete
    for process in processes:
        process.join()


    logs_path = os.path.join(save_result_path, "logs")
    if os.path.exists(logs_path):
        print(f"Deleting logs folder: {logs_path}")
        shutil.rmtree(logs_path)

    # Process the results and generate plots
    process_results(save_result_path)
    plot_results(save_result_path, args.num_queries)

    print(f"✅ Benchmark completed. Results saved to: {save_result_path}")