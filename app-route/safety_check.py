from mininet.log import setLogLevel, info, lg
# Define a handler for the timeout
def handler(signum, frame):
    raise TimeoutError("Command execution timed out")

def safety_check(commands):
    if 'tcpdump' in commands:
        lg.output("Command containing 'tcpdump' is not allowed.")
        return False
    return True