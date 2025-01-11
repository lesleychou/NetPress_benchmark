
from mininet.log import output, error

def _parse_fping(result):
    """Parse fping result.
       result: fping command output
       returns: sent, received"""
    sent = 1
    received = 0
    if '1/1/0' in result or '1/1/1' in result:
        received = 1
    return sent, received

def fastpingall(net, timeout=None):
    """Fast ping all hosts in the network.
       net: Mininet object
       timeout: time to wait for a response, as string
       returns: ploss packet loss percentage"""
    packets = 0
    lost = 0
    ploss = None
    hosts = net.hosts
    output('*** Fast Ping: testing ping reachability\n')
    for node in hosts:
        output('%s -> ' % node.name)
        for dest in hosts:
            if node != dest:
                opts = ''
                if timeout:
                    opts = '-t %s' % timeout
                if dest.intfs:
                    result = node.cmd('fping -c1 %s %s' % (opts, dest.IP()))
                    sent, received = _parse_fping(result)
                else:
                    sent, received = 0, 0
                packets += sent
                if received > sent:
                    error('*** Error: received too many packets')
                    error('%s' % result)
                    node.cmdPrint('route')
                    exit(1)
                lost += sent - received
                output(('%s ' % dest.name) if received else 'X ')
        output('\n')
    if packets > 0:
        ploss = 100.0 * lost / packets
        received = packets - lost
        output("*** Results: %i%% dropped (%d/%d received)\n" %
               (ploss, received, packets))
    else:
        ploss = 0
        output("*** Warning: No packets sent\n")
    return ploss

