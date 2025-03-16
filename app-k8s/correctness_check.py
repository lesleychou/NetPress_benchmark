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

def create_debug_container(pod_name_prefix, timeout=3):
    """Create a debug container in the specified pod.
    
    Args:
        pod_name: Name of the target pod
        timeout: Timeout for command execution (default: 3 seconds)
        
    Returns:
        Name of the created debug container or None if failed
    """
    # Determine target container based on pod name patterns
    if 'loadgenerator' in pod_name_prefix:
        target = "main"  # Verify actual container name for loadgenerator
    elif 'redis-cart' in pod_name_prefix:
        target = "redis"  # Container name from pod spec
    else:
        target = "server"  # Default assumption for other services
    pod_name = find_pod_by_prefix(pod_name_prefix)
    # Construct debug command with dynamic target container
    debug_command = [
        "kubectl", "debug", "-it", pod_name,
        "--image=busybox",
        f"--target={target}",  # Dynamically set target container
        "--quiet",  # Suppress verbose output
        "--attach=false",  # Run in detached mode
        "--", "sleep", "infinity"  # Keep container alive
    ]
    print(f"Creating debug container: {' '.join(debug_command)}")
    try:
        # Execute debug container creation
        result=subprocess.run(
            debug_command, 
            capture_output=True, 
            text=True, 
            timeout=timeout, 
            check=True
        )
        print(f"{pod_name}{result.stdout}")

    except subprocess.TimeoutExpired:
        print(f"Timeout creating debug container in pod {pod_name}")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to create debug container in pod {pod_name}: {e.stderr}")
        return None

    except Exception as e:
        print(f"Error creating debug container in pod {pod_name}: {e}")
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
            "kubectl", "exec", pod_name,  
            "-c", debug_container_name,
            "--", "nc", "-zv", "-w", str(timeout), host, str(port)
        ]

        print(f"Checking connectivity: {' '.join(nc_command)}")
        result = subprocess.run(nc_command, capture_output=True, text=True, timeout=timeout+1)
        output = (result.stdout + result.stderr).strip()

        print(f"{pod_name}output: ", output)
        if "open" in output:
            return True
        else:
            return False
    except subprocess.TimeoutExpired:
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing nc command: {e}")
        return False


def correctness_check(expected_results, debug_container_mapping):
    """
    Check the connectivity of all pods specified in expected_results to their target services.
    For each pod, use the provided debug container to execute the `nc` command to check all targets.
    """

    all_match = True
    mismatch_messages = []  # Used to record all mismatch information

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pod, pod_prefix, targets, debug_container_mapping): pod_prefix for pod_prefix, targets in expected_results.items()}
        for future in as_completed(futures):
            pod_all_match, pod_mismatch_summary = future.result()
            if not pod_all_match:
                all_match = False
                mismatch_messages.append(pod_mismatch_summary)
    
    # Combine all mismatch information into a single string
    mismatch_summary = "\n".join(mismatch_messages) if mismatch_messages else "No mismatches found."

    return all_match, mismatch_summary

def process_pod(pod_prefix, targets, debug_container_mapping):
    pod_name = find_pod_by_prefix(pod_prefix)
    if not pod_name:
        print(f"Pod {pod_prefix} not found")
        return False, f"Pod {pod_prefix} not found"

    debug_container_name = debug_container_mapping.get(pod_prefix)
    if not debug_container_name:
        print(f"Debug container for pod {pod_name} not found in mapping")
        return False, f"Debug container for pod {pod_name} not found in mapping"

    pod_all_match = True
    pod_mismatch_messages = []

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(check_connectivity_with_debug, pod_name, debug_container_name, *target.split(":")): target for target in targets}
        for future in as_completed(futures):
            target = futures[future]
            expected = targets[target]
            try:
                actual = future.result()
                if actual != expected:
                    mismatch_message = f"Mismatch: {pod_prefix} → {target} (Expected: {expected}, Actual: {actual})"
                    pod_mismatch_messages.append(mismatch_message)  # Record mismatch information
                    pod_all_match = False
            except Exception as e:
                print(f"Error checking connectivity for {pod_prefix} → {target}: {e}")
                pod_mismatch_messages.append(f"Error checking connectivity for {pod_prefix} → {target}: {e}")
                pod_all_match = False

    return pod_all_match, "\n".join(pod_mismatch_messages)

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
    all_match, mismatch_summary = correctness_check(expected_results, debug_container_mapping)
    print(f"\nFinal result: All tests passed: {all_match}")
    print(f"Mismatch details: {mismatch_summary}")
    endtime=time.time()
    print(f"Time taken: {endtime-starttime}")
    exit(0 if all_match else 1)
    # cleanup_debug_containers()