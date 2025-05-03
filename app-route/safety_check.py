from mininet.log import setLogLevel, info, lg
# Define a handler for the timeout
def handler(signum, frame):
    raise TimeoutError("Command execution timed out")

def safety_check(commands):
    if commands is None:
        return True
    if 'tcpdump' in commands:
        lg.output("Command containing 'tcpdump' is not allowed.")
        return False
    if "systemctl" in commands:
        lg.output("Command containing 'systemctl' is not allowed.")
        return False
    if "frr" in commands:
        lg.output("Command containing 'frr' is not allowed.")
        return False
    if "ethtool" in commands:
        lg.output("Command containing 'ethtool' is not allowed.")
        return False
    if "ping" in commands:
        lg.output("Command containing 'ping' is not allowed.")
        return False
    return True