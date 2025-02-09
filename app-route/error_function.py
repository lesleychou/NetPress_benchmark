from mininet.log import setLogLevel, info, lg
import random

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