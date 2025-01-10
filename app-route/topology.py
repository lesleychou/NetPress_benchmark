
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.cli import CLI



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

        # Link each switch to the router with a unique interface
        for i, switch in enumerate(switches):
            self.addLink(switch, router, intfName2=f'r0-eth{i+1}', 
                         params2={'ip': subnets[i % len(subnets)][0]})

        # Link each host to its corresponding switch
        for i, host in enumerate(hosts):
            self.addLink(host, switches[i % num_switches])


def generate_subnets(num_hosts, num_switches):
    # Base IP address to start subnet generation
    base_ip = [192, 168, 1, 1]
    subnets = []
    
    # Loop through the number of switches to create subnets
    for i in range(num_switches):
        # Create a new subnet IP by modifying the third octet
        subnet_ip = base_ip.copy()
        subnet_ip[2] += i  # Increment the third octet for each subnet
        
        # Generate subnet information (subnet, host IP, subnet address)
        subnet = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.{subnet_ip[3]}/24'
        host_ip = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.100/24'  # Example host IP within the subnet
        subnet_address = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.0/24'  # Subnet address with .0
        
        # Append the generated subnet details as a tuple to the subnets list
        subnets.append((subnet, host_ip, subnet_address))
    
    # Return the list of generated subnets
    return subnets



