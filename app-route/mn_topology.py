#!/usr/bin/env python

"""
linuxrouter.py: Example network with Linux IP router

This example converts a Node into a router using IP forwarding
already built into Linux.

The example topology creates a router and three IP subnets:

    - 192.168.1.0/24 (r0-eth1, IP: 192.168.1.1)
    - 172.16.0.0/12 (r0-eth2, IP: 172.16.0.1)
    - 10.0.0.0/8 (r0-eth3, IP: 10.0.0.1)

Each subnet consists of a single host connected to
a single switch:

    r0-eth1 - s1-eth1 - h1-eth0 (IP: 192.168.1.100)
    r0-eth2 - s2-eth1 - h2-eth0 (IP: 172.16.0.100)
    r0-eth3 - s3-eth1 - h3-eth0 (IP: 10.0.0.100)

The example relies on default routing entries that are
automatically created for each router interface, as well
as 'defaultRoute' parameters for the host interfaces.

Additional routes may be added to the router or hosts by
executing 'ip route' or 'route' commands on the router or hosts.
"""


from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import Node
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import random

class LinuxRouter( Node ):
    "A Node with IP forwarding enabled."

    # pylint: disable=arguments-differ
    def config( self, **params ):
        super( LinuxRouter, self).config( **params )
        # Enable forwarding on the router
        self.cmd( 'sysctl net.ipv4.ip_forward=1' )

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


def error_disable_interface(router, subnets):
    # Inject random errors
    info('*** Injecting random errors\n')
    interfaces = [f'r0-eth{i+1}' for i in range(len(subnets))]
    interface_to_disable = random.choice(interfaces)
    info(f'*** Disabling interface: {interface_to_disable}\n')
    router.cmd(f'ifconfig {interface_to_disable} down')


def error_remove_ip(router, subnets):
    # Inject random errors
    info('*** Injecting random errors: Removing a random IP address\n')
    interfaces = [f'r0-eth{i+1}' for i in range(len(subnets))]
    interface_to_modify = random.choice(interfaces)

    # Remove the IP address assigned to the selected interface
    info(f'*** Removing IP address from interface: {interface_to_modify}\n')
    router.cmd(f'ip addr flush dev {interface_to_modify}')


def error_drop_traffic_to_from_subnet(router, subnets):
    # Drop all traffic to/from one random subnet
    info('*** Dropping all traffic to/from one random subnet\n')
    subnet_to_drop = random.choice(subnets)
    subnet_ip = subnet_to_drop[0].split('/')[0]
    info(f'*** Dropping traffic to/from subnet: {subnet_ip}\n')
    router.cmd('iptables -A INPUT -s 192.168.3.0/24 -j DROP')
    router.cmd('iptables -A OUTPUT -d 192.168.3.0/24 -j DROP')


def run():
    "Test Linux router"
    num_hosts = 3
    num_switches = 3
    subnets = []
    base_ip = [192, 168, 1, 1]
    for i in range(num_switches):
        subnet_ip = base_ip.copy()
        subnet_ip[2] += i  # Increment the third octet for each subnet
        subnet = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.{subnet_ip[3]}/24'
        host_ip = f'{subnet_ip[0]}.{subnet_ip[1]}.{subnet_ip[2]}.100/24'
        subnets.append((subnet, host_ip))
    print(subnets)

    topo = NetworkTopo(num_hosts=num_hosts, num_switches=num_switches, subnets=subnets)

    net = Mininet(topo=topo, waitConnected=True)
    net.start()

    # Enable IP forwarding on the router
    router = net.get('r0')
    router.cmd('sysctl -w net.ipv4.ip_forward=1')

    # Add static routes for all subnets
    for i in range(len(subnets)):
        for j in range(len(subnets)):
            if i != j:
                router.cmd(f'ip route add {subnets[j][0]} dev r0-eth{i+1}')

    # Display the routing table
    info('*** Routing Table on Router:\n')
    info(router.cmd('route'))
    # Test connectivity
    info('*** Testing network connectivity\n')
    net.pingAll()

    drop_traffic_to_from_subnet(router, subnets)

    # Display the routing table for debugging
    info('*** Routing Table on Router:\n')
    info(router.cmd('route'))
    # Test connectivity
    info('*** Testing network connectivity\n')
    net.pingAll()

    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    run()
