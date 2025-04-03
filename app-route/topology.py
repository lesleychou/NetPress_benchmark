
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.cli import CLI
from mininet.net import Mininet
from mininet.log import info
from mininet.log import setLogLevel, info, lg


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

    def build(self, num_hosts_per_subnet=5, num_switches=5, subnets=None, **_opts):

        # Create the router
        router = self.addNode('r0', cls=LinuxRouter, ip=subnets[0][0])

        # Create switches
        switches = [self.addSwitch(f's{i+1}') for i in range(num_switches)]

        # Link each switch to the router with a unique interface
        for i, switch in enumerate(switches):
            self.addLink(switch, router, intfName2=f'r0-eth{i+1}', 
                         params2={'ip': subnets[i % len(subnets)][0]})

            # Create multiple hosts for each switch, hence each subnet
            for j in range(num_hosts_per_subnet):
                host_ip = f'{subnets[i][0].split(".")[0]}.{subnets[i][0].split(".")[1]}.{subnets[i][0].split(".")[2]}.{100+j}/24'
                host = self.addHost(f'h{i*num_hosts_per_subnet + j + 1}', 
                                    ip=host_ip, 
                                    defaultRoute=f'via {subnets[i][0].split("/")[0]}')
                self.addLink(host, switch)


def generate_subnets(num_switches):
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

def initialize_network(num_hosts_per_subnet, num_switches):
    subnets = generate_subnets(num_switches)
    # Instantiate Mininet topo
    topo = NetworkTopo(num_hosts_per_subnet, num_switches, subnets)
    net = Mininet(topo=topo, waitConnected=True)
    
    # Start Mininet
    net.start()
    
    # Enable IP forwarding on the router
    router = net.get('r0')
    info(router.cmd('sysctl -w net.ipv4.ip_forward=1'))
    return subnets, topo, net, router

if __name__ == '__main__':
    import sys
    import io
    num_hosts_per_subnet = 3
    num_switches = 3
    # subnets = generate_subnets(num_hosts_per_subnet)

    # topo = NetworkTopo(num_hosts_per_subnet=num_hosts_per_subnet,
    #                     num_switches=num_switches,
    #                     subnets=subnets)

    # net = Mininet(topo=topo)
    # net.start()
    subnets, topo, net, router = initialize_network(num_hosts_per_subnet, num_switches)

    CLI(net)
    net.stop()
    import subprocess
    output = subprocess("pingall", stdout=sys.stdout)
    print(output)


    net.stop()