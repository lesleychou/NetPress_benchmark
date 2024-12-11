
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.cli import CLI
from mininet.log import setLogLevel, info, lg
import random
import re
import os
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from agent_call import llm_interface

# Login huggingface
login(token="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")

# Define the path to store results
root_path = "/home/ubuntu/nemo_benchmark/app-route/result" 

class LinuxRouter( Node ):
    "A Node with IP forwarding enabled."

    # pylint: disable=arguments-differ
    def config( self, **params ):
        super( LinuxRouter, self).config( **params )
        # Enable forwarding on the router
        self.cmd( 'sysctl nesudo python topo.pyt.ipv4.ip_forward=1' )

    def terminate( self ):
        self.cmd( 'sysctl net.ipv4.ip_forward=0' )
        super( LinuxRouter, self ).terminate()


class NetworkTopo(Topo):
    "A LinuxRouter connecting multiple IP subnets"

    def build(self, num_hosts=5, num_switches=5, subnets=None, **_opts):
        if subnets is None:
            # Ensure unique, non-overlapping subnets
            subnets = [
                ('192.168.1.1/24', '192.168.1.100/24'),
                ('172.16.1.1/24', '172.16.1.100/24'),
                ('10.0.1.1/24', '10.0.1.100/24'),
                ('192.168.2.1/24', '192.168.2.100/24'),
                ('172.17.1.1/24', '172.17.1.100/24')
            ]

        # Create the router
        router = self.addNode('r0', cls=LinuxRouter, ip=subnets[0][0])

        # Create switches
        switches = [self.addSwitch(f's{i+1}') for i in range(num_switches)]

        # Create hosts and assign them to unique subnets
        hosts = [self.addHost(f'h{i+1}', 
                              ip=subnets[i % len(subnets)][1], 
                              defaultRoute=f'via {subnets[i % len(subnets)][0].split("/")[0]}') 
                 for i in range(num_hosts)]
        for i in range(num_hosts):
            print(f'via {subnets[i % len(subnets)][0].split("/")[0]}')
        # Link each switch to the router with a unique interface
        for i, switch in enumerate(switches):
            self.addLink(switch, router, intfName2=f'r0-eth{i+1}', 
                         params2={'ip': subnets[i % len(subnets)][0]})

        # Link each host to its corresponding switch
        for i, host in enumerate(hosts):
            self.addLink(host, switches[i % num_switches])


def error_disable_routing(router, subnets):
    # Inject error: Disable IP forwarding (routing) on the router
    info('*** Injecting error: Disabling IP forwarding\n')
    router.cmd('sysctl -w net.ipv4.ip_forward=0')


def error_disable_interface(router, subnets):
    # Inject random errors
    interfaces = [f'r0-eth{i+1}' for i in range(len(subnets))]
    interface_to_disable = random.choice(interfaces)
    info(f'*** Injecting error: Disabling interface {interface_to_disable}\n')
    router.cmd(f'ifconfig {interface_to_disable} down')


def error_remove_ip(router, subnets):
    interfaces = [f'r0-eth{i+1}' for i in range(len(subnets))]
    interface_to_modify = random.choice(interfaces)

    # Remove the IP address assigned to the selected interface
    info(f'*** Injecting error: Removing IP address from interface: {interface_to_modify}\n')
    router.cmd(f'ip addr flush dev {interface_to_modify}')


def error_drop_traffic_to_from_subnet(router, subnets):
    # Drop all traffic to/from one random subnet
    subnet_to_drop = random.choice(subnets)
    subnet_ip = subnet_to_drop[0].split('/')[0]
    info(f'*** Injecting error: Dropping traffic to/from subnet: {subnet_ip}\n')
    router.cmd('iptables -A INPUT -s 192.168.3.0/24 -j DROP')
    router.cmd('iptables -A OUTPUT -d 192.168.3.0/24 -j DROP')

def error_wrong_routing_table(router, subnets):
    # Delete original routing, add wrong routing
    num_subnets = len(subnets)
    selected_indices = random.sample(range(num_subnets), 2)
    router.cmd(f'ip route del {subnets[selected_indices[0]][2]} dev r0-eth{selected_indices[0]+1}')
    router.cmd(f'ip route add {subnets[selected_indices[0]][2]} dev r0-eth{selected_indices[1]+1}')
    

# Complexty control: randomly pick given number error type to inject
def inject_errors(router, subnets, error_number=1):
    error_functions = [
        # error_disable_routing,
        error_disable_interface,
        # error_remove_ip,
        # error_drop_traffic_to_from_subnet
        # error_wrong_routing_table
    ]
    num_errors = min(error_number, len(error_functions))
    errors_to_inject = random.sample(error_functions, num_errors)
    for error in errors_to_inject:
        error(router, subnets)
    return error_functions


def run(test_taker, max_iter):
    "Test Linux router"
    num_hosts = 4
    num_switches = 4
    subnets = []
    base_ip = [192, 168, 1, 1]
    for i in range(num_switches):
        subnet_ip = base_ip.copy()
        subnet_ip[2] += i  # Increment the third octet for each subnet
        subnet = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.{subnet_ip[3]}/24'
        host_ip = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.100/24'
        subnet_address = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.0/24'
        subnets.append((subnet, host_ip, subnet_address))
    # If the path does not exist, create a new one

    # Instantiate Mininet topo
    topo = NetworkTopo(num_hosts=num_hosts, num_switches=num_switches, subnets=subnets)
    net = Mininet(topo=topo, waitConnected=True)

    # Instantiate LLM test taker
    model_kwargs = {
        "torch_dtype": torch.bfloat16,  
        "quantization_config": {"load_in_4bit": True}  
    }
    model = AutoModelForCausalLM.from_pretrained(
        test_taker,
        device_map="auto",  
        **model_kwargs  
    )
    tokenizer = AutoTokenizer.from_pretrained(test_taker)
    
    net.start()

    # Enable IP forwarding on the router
    router = net.get('r0')
    router.cmd('sysctl -w net.ipv4.ip_forward=1')

    # # Display the routing table
    # info('*** Routing Table on Router:\n')
    # info(router.cmd('route'))
    # # Test connectivity
    # info('*** Testing network connectivity\n')
    # net.pingAll()

    errors = str(inject_errors(router, subnets, error_number=1))

    file_path = root_path + '/' + 'Llama' + '/' + 'result.txt'
    if os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("") 
    # # Display the routing tab
    # le for debugging
    # info('*** Routing Table on Router:\n')
    # info(router.cmd('route'))
    # # Test connectivity
    # info('*** Testing network connectivity\n')
    iter = 0
    while iter < max_iter:
        # Clear the log file
        with open('mininet.log', 'w'):
            pass

        # Configure a file handler to save logs to a file
        file_handler = logging.FileHandler('mininet.log')
        file_handler.setLevel(logging.INFO)

        # Only output message content without additional information
        formatter = logging.Formatter('%(message)s')  # Keep only the message
        file_handler.setFormatter(formatter)

        # Add the file handler to the Mininet logger
        lg.addHandler(file_handler)

        # Ensure the logger captures INFO and OUTPUT logs
        lg.setLevel(25)  # Mininet OUTPUT level is 25

        # Enable logging
        setLogLevel('info')


        if iter != 0:
            info(f"Machine: {machine}")
            info(f"Command: {commands}")
            info(net[machine].cmd(commands))
            time.sleep(10)  # wait 10 seconds
        net.pingAll()

        # Read and print log file content
        with open('mininet.log', 'r') as f:
            log_content = f.read()
            print("log_content:")
            print(log_content)
        machine, commands = llm_interface(log_content, model, tokenizer, file_path)
        match = re.search(r'(\d+)%', log_content)  
        if match:
            number = int(match.group(1))  
            if number == 0:
                print(f"Success in {iter} iterations")  
                break
            else:
                print(f"{number}% packet loss.")  
        else:
            print("No '%' found in log content.")

        iter += 1

        # Remove all handlers to prevent duplicate logging
        for handler in lg.handlers[:]:
            lg.removeHandler(handler)


    net.stop()
def debug():
    "Test Linux router"
    num_hosts = 4
    num_switches = 4
    subnets = []
    base_ip = [192, 168, 1, 1]
    for i in range(num_switches):
        subnet_ip = base_ip.copy()
        subnet_ip[2] += i  # Increment the third octet for each subnet
        subnet = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.{subnet_ip[3]}/24'
        host_ip = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.100/24'
        subnet_address = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.0/24'
        subnets.append((subnet, host_ip, subnet_address))

    topo = NetworkTopo(num_hosts=num_hosts, num_switches=num_switches, subnets=subnets)

    net = Mininet(topo=topo, waitConnected=True)
    net.start()
    router = net.get('r0')
    router.cmd('sysctl -w net.ipv4.ip_forward=1')

    # Display the routing table
    info('*** Routing Table on Router:\n')
    info(router.cmd('route'))
    # Test connectivity
    info('*** Testing network connectivity\n')
    net.pingAll()
    error_wrong_routing_table(router, subnets)
    net.pingAll()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    max_iter = 10
    run("Qwen/Qwen2.5-72B-Instruct", max_iter) # "meta-llama/Meta-Llama-3.1-70B-Instruct"
    # debug()