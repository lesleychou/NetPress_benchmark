from mininet.log import setLogLevel, info, lg
import random
import json
import random
from itertools import combinations

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
    subnet_address = subnet_to_drop[2]
    info(f'*** Injecting error: Dropping traffic to/from subnet: {subnet_address}\n')
    router.cmd(f'iptables -A INPUT -s {subnet_address} -j DROP')
    router.cmd(f'iptables -A OUTPUT -d {subnet_address} -j DROP')

def error_wrong_routing_table(router, subnets):
    # Delete original routing, add wrong routing
    num_subnets = len(subnets)
    selected_indices = random.sample(range(num_subnets), 2)
    router.cmd(f'ip route del {subnets[selected_indices[0]][2]} dev r0-eth{selected_indices[0]+1}')
    router.cmd(f'ip route add {subnets[selected_indices[0]][2]} dev r0-eth{selected_indices[1]+1}')
    

# Complexty control: randomly pick given number error type to inject
def inject_errors(router, subnets, error_number=1, errortype=None):
    error_functions = {
        'disable_routing': error_disable_routing,
        'disable_interface': error_disable_interface,
        'remove_ip': error_remove_ip,
        'drop_traffic_to_from_subnet': error_drop_traffic_to_from_subnet,
        'wrong_routing_table': error_wrong_routing_table
    }
    
    if errortype:
        errors_to_inject = [error_functions[errortype]]
    else:
        num_errors = min(error_number, len(error_functions))
        errors_to_inject = random.sample(list(error_functions.values()), num_errors)
    
    for error in errors_to_inject:
        error(router, subnets)
    
    return errors_to_inject

# Generate detailed error information for each error type
def get_detail(error_type, hostnumber):
    if error_type == 'disable_routing':
        return {"action": "Disable IP forwarding"}
    elif error_type == 'disable_interface':
        rand_index = random.randint(1, hostnumber + 1)
        return {"interface": f"r0-eth{rand_index}"}
    elif error_type == 'remove_ip':
        rand_index = random.randint(1, hostnumber + 1)
        return {"interface": f"r0-eth{rand_index}"}
    elif error_type == 'drop_traffic_to_from_subnet':
        rand_index = random.randint(1, hostnumber + 1)
        return {"subnet": f"192.168.{rand_index}.0/24"}
    elif error_type == 'wrong_routing_table':
        # Randomly select two different interface indexes
        indexes = random.sample(range(1, hostnumber + 1), 2)
        from_subnet = f"192.168.{indexes[0]}.0/24"
        to_subnet = f"192.168.{indexes[1]}.0/24"
        del_interface = f"r0-eth{indexes[0]}"
        add_interface = f"r0-eth{indexes[1]}"
        return {"from": from_subnet, "to": to_subnet, "del_interface": del_interface, "add_interface": add_interface}
    else:
        return {}

# Generate a configuration file with a specified number of queries for each error type
def generate_config(filename='error_config.json', num_errors_per_type=5):
    queries = []
    error_types = [
        'disable_routing',
        'disable_interface',
        'remove_ip',
        'drop_traffic_to_from_subnet',
        'wrong_routing_table'
    ]
    
    # Generate queries for each error type
    for et in error_types:
        for _ in range(num_errors_per_type):
            num_hosts_per_subnet = random.randint(2, 4)
            num_switches = random.randint(2, 4)
            detail = get_detail(et, num_switches)
            query = {
                "num_switches": num_switches,
                "num_hosts_per_subnet": num_hosts_per_subnet,
                "errornumber": 1,
                "errortype": et,
                "errordetail": detail
            }
            queries.append(query)
    
    # Generate combined queries
    for et1, et2 in combinations(error_types, 2):
        for _ in range(num_errors_per_type):
            num_hosts_per_subnet = random.randint(2, 4)
            num_switches = random.randint(2, 4)
            detail1 = get_detail(et1, num_switches)
            detail2 = get_detail(et2, num_switches)
            query = {
                "num_switches": num_switches,
                "num_hosts_per_subnet": num_hosts_per_subnet,
                "errornumber": 2,
                "errortype": [et1, et2],
                "errordetail": [detail1, detail2]
            }
            queries.append(query)
    
    config = {"queries": queries}
    
    # Write to file
    with open(filename, 'w') as f:
        f.truncate(0)  # Clear the file content
        json.dump(config, f, indent=4)
    
    print(f"Config file {filename} generated with {len(queries)} queries.")

# Process single error
def process_single_error(router, subnets, errortype, errordetail):
    if errortype == "disable_routing":
        info('*** Injecting error: Disabling IP forwarding\n')
        router.cmd('sysctl -w net.ipv4.ip_forward=0')
        return
    if errortype == "disable_interface":
        if "interface" not in errordetail:
            print("not enough detailed information")
            return
        interface = errordetail["interface"]
        info(f'*** Injecting error: Disabling interface {interface}\n')
        router.cmd(f'ifconfig {interface} down')
        return
    if errortype == "remove_ip":
        if "interface" not in errordetail:
            print("not enough detailed information")
            return
        interface = errordetail["interface"]
        info(f'*** Injecting error: Removing IP address from interface {interface}\n')
        router.cmd(f'ip addr flush dev {interface}')
        return
    if errortype == "drop_traffic_to_from_subnet":
        if "subnet" not in errordetail:
            print("not enough detailed information")
            return
        subnet = errordetail["subnet"]
        info(f'*** Injecting error: Dropping traffic to/from subnet: {subnet}\n')
        router.cmd(f'iptables -A INPUT -s {subnet} -j DROP')
        router.cmd(f'iptables -A OUTPUT -d {subnet} -j DROP')
        return
    if errortype == "wrong_routing_table":
        required_keys = ["from", "to", "del_interface", "add_interface"]
        if not all(key in errordetail for key in required_keys):
            print("not enough detailed information")
            return
        from_subnet = errordetail["from"]
        to_subnet = errordetail["to"]
        del_interface = errordetail["del_interface"]
        add_interface = errordetail["add_interface"]
        info(f'*** Injecting error: Wrong routing table from {from_subnet} (delete via {del_interface}) and add via {add_interface}\n')
        router.cmd(f'ip route del {from_subnet} dev {del_interface}')
        router.cmd(f'ip route add {from_subnet} dev {add_interface}')
        return
    print("not enough detailed information")