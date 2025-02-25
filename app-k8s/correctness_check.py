import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_pod_by_prefix(prefix):
    """Find a pod whose name starts with the specified prefix."""
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "--no-headers", "-o", "custom-columns=:metadata.name"],
            capture_output=True, text=True, check=True
        )
        pods = result.stdout.strip().split("\n")
        for pod in pods:
            if pod.startswith(prefix):
                return pod
    except subprocess.CalledProcessError as e:
        print(f"Error listing pods: {e}")
    return None

def wait_for_debug_container(pod_name, container_prefix="debugger-", timeout=5):
    """
    Poll pod information until the debug container (name starts with container_prefix) is in the running state.
    Note: The debug container will appear in the pod's ephemeralContainers.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["kubectl", "get", "pod", pod_name, "-o", "json"],
                capture_output=True, text=True, check=True
            )
            pod_json = json.loads(result.stdout)
            ephemeral_containers = pod_json.get("spec", {}).get("ephemeralContainers", [])
            statuses = pod_json.get("status", {}).get("ephemeralContainerStatuses", [])
            for ec in ephemeral_containers:
                name = ec.get("name", "")
                if name.startswith(container_prefix):
                    # Check if the status is running
                    for status in statuses:
                        if status.get("name") == name:
                            if "running" in status.get("state", {}):
                                return name
        except Exception as e:
            print(f"Error fetching pod info: {e}")
        time.sleep(1)
    return None

def create_debug_container(pod_name, timeout=3):
    """Create a debug container in the specified pod.
    
    Args:
        pod_name: Name of the target pod
        timeout: Timeout for command execution (default: 3 seconds)
        
    Returns:
        Name of the created debug container or None if failed
    """
    # Determine target container based on pod name patterns
    if 'loadgenerator' in pod_name:
        target = "main"  # Verify actual container name for loadgenerator
    elif 'redis-cart' in pod_name:
        target = "redis"  # Container name from pod spec
    else:
        target = "server"  # Default assumption for other services

    # Construct debug command with dynamic target container
    debug_command = [
        "kubectl", "debug", "-it", pod_name,
        "--image=busybox",
        f"--target={target}",  # Dynamically set target container
        "--quiet",  # Suppress verbose output
        "--attach=false",  # Run in detached mode
        "--", "sleep", "infinity"  # Keep container alive
    ]

    try:
        # Execute debug container creation
        subprocess.run(
            debug_command, 
            capture_output=True, 
            text=True, 
            timeout=timeout, 
            check=True
        )

    except subprocess.TimeoutExpired:
        return None
        
    except subprocess.CalledProcessError as e:
        return None

    except Exception as e:
        return None

    # Wait for the debug container to start
    debug_container_name = wait_for_debug_container(pod_name)
    if not debug_container_name:
        print(f"Failed to detect debug container in pod {pod_name}")
    return debug_container_name

def check_connectivity_with_debug(pod_name, debug_container_name, host, port, timeout=1):
    """
    Use the created debug container to execute the `nc` command to check connectivity.
    Command format:
      kubectl exec -it <pod> -c <debug_container_name> -- nc -zv -w <timeout> <host> <port>
    If the output contains "open", the connection is considered successful.
    """
    try:
        nc_command = [
            "kubectl", "exec", "-it", pod_name,
            "-c", debug_container_name,
            "--", "nc", "-zv", "-w", str(timeout), host, str(port)
        ]
        result = subprocess.run(nc_command, capture_output=True, text=True, timeout=timeout+1)
        output = (result.stdout + result.stderr).strip()
        if "open" in output:
            return True
        else:
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout: Command execution exceeded {timeout} seconds.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing nc command: {e}")
        return False

def correctness_check(expected_results):
    """
    Check the connectivity of all pods specified in expected_results to their target services.
    For each pod, create a debug container and use it to execute the `nc` command to check all targets.
    """

    all_match = True
    mismatch_messages = []  # Used to record all mismatch information

    def process_pod(pod_prefix, targets):
        pod_name = find_pod_by_prefix(pod_prefix)
        if not pod_name:
            print(f"Pod {pod_prefix} not found")
            return False, f"Pod {pod_prefix} not found"

        debug_container_name = create_debug_container(pod_name)
        if not debug_container_name:
            print(f"Failed to create debug container for pod {pod_name}")
            return False, f"Failed to create debug container for pod {pod_name}"

        pod_all_match = True
        pod_mismatch_messages = []

        for target, expected in targets.items():
            try:
                host, port = target.split(":")
                port = int(port)
            except ValueError:
                print(f"Invalid target {target}")
                pod_all_match = False
                pod_mismatch_messages.append(f"Invalid target {target}")
                continue

            actual = check_connectivity_with_debug(pod_name, debug_container_name, host, port)
            if actual != expected:
                mismatch_message = f"Mismatch: {pod_prefix} → {target} (Expected: {expected}, Actual: {actual})"
                pod_mismatch_messages.append(mismatch_message)  # Record mismatch information
                pod_all_match = False

        return pod_all_match, "\n".join(pod_mismatch_messages)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pod, pod_prefix, targets): pod_prefix for pod_prefix, targets in expected_results.items()}
        for future in as_completed(futures):
            pod_all_match, pod_mismatch_summary = future.result()
            if not pod_all_match:
                all_match = False
                mismatch_messages.append(pod_mismatch_summary)

    # Combine all mismatch information into a single string
    mismatch_summary = "\n".join(mismatch_messages) if mismatch_messages else "No mismatches found."

    return all_match, mismatch_summary

if __name__ == "__main__":
    expected_results = {
    "frontend": {
        "adservice:9555": True,
        "cartservice:7070": True,
        "checkoutservice:5050": True,
        "currencyservice:7000": True,
        "productcatalogservice:3550": True,
        "recommendationservice:8080": True,
        "shippingservice:50051": True,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "adservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "cartservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": True
    },
    "checkoutservice": {
        "adservice:9555": False,
        "cartservice:7070": True,
        "checkoutservice:5050": False,
        "currencyservice:7000": True,
        "productcatalogservice:3550": True,
        "recommendationservice:8080": False,
        "shippingservice:50051": True,
        "emailservice:5000": True,
        "paymentservice:50051": True,
        "redis-cart:6379": False
    },
    "currencyservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "productcatalogservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "recommendationservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": True,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "shippingservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "emailservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "paymentservice": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "redis-cart": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    },
    "loadgenerator": {
        "adservice:9555": False,
        "cartservice:7070": False,
        "checkoutservice:5050": False,
        "currencyservice:7000": False,
        "productcatalogservice:3550": False,
        "recommendationservice:8080": False,
        "shippingservice:50051": False,
        "emailservice:5000": False,
        "paymentservice:50051": False,
        "redis-cart:6379": False
    }
}

    starttime=time.time()
    result = correctness_check(expected_results)
    print(f"\nFinal result: {result}")
    endtime=time.time()
    print(f"Time taken: {endtime-starttime}")
    exit(0 if result else 1)