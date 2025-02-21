import subprocess
import json
import time

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

def create_debug_container(pod_name, timeout=2):
    """
    Create a debug container in the pod in detached mode.
    Here, the busybox image is used, and the container executes `sleep infinity` to keep it alive.
    """
    try:
        if 'loadgenerator' in pod_name:
            debug_command = [
                "kubectl", "debug", "-it", pod_name,
                "--image=busybox",
                "--target=main",
                "--quiet",
                "--attach=false",
                "--", "sleep", "infinity"
            ]
        else:
            debug_command = [
                "kubectl", "debug", "-it", pod_name,
                "--image=busybox",
                "--target=server",
                "--quiet",
                "--attach=false",
                "--", "sleep", "infinity"
            ]
        subprocess.run(debug_command, capture_output=True, text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        print("Timeout while creating debug container")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error creating debug container: {e}")
        return None

    # Wait for the debug container to start
    debug_container_name = wait_for_debug_container(pod_name)
    if not debug_container_name:
        print(f"Failed to detect debug container in pod {pod_name}")
    return debug_container_name

def check_connectivity_with_debug(pod_name, debug_container_name, host, port, timeout=2):
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
        print(f"Executing command: {' '.join(nc_command)}")
        result = subprocess.run(nc_command, capture_output=True, text=True, timeout=timeout+5)
        output = (result.stdout + result.stderr).strip()
        if "open" in output:
            print(f"Success: {pod_name} (debug container {debug_container_name}) can reach {host}:{port}")
            return True
        else:
            print(f"Failure: {pod_name} (debug container {debug_container_name}) cannot reach {host}:{port}. Output: {output}")
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

    for pod_prefix, targets in expected_results.items():
        pod_name = find_pod_by_prefix(pod_prefix)
        if not pod_name:
            print(f"Pod {pod_prefix} not found")
            all_match = False
            continue

        print(f"\nCreating debug container for pod {pod_name} ...")
        debug_container_name = create_debug_container(pod_name)
        if not debug_container_name:
            print(f"Failed to create debug container for pod {pod_name}")
            all_match = False
            continue

        for target, expected in targets.items():
            try:
                host, port = target.split(":")
                port = int(port)
            except ValueError:
                print(f"Invalid target {target}")
                all_match = False
                continue

            actual = check_connectivity_with_debug(pod_name, debug_container_name, host, port)
            if actual != expected:
                mismatch_message = f"Mismatch: {pod_prefix} → {target} (Expected: {expected}, Actual: {actual})"
                print(mismatch_message)
                mismatch_messages.append(mismatch_message)  # Record mismatch information
                all_match = False

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
            "shippingservice:50051": True
        },
        "checkoutservice": {
            "cartservice:7070": True,
            "currencyservice:7000": True,
            "emailservice:5000": True,
            "paymentservice:50051": True,
            "productcatalogservice:3550": True,
            "shippingservice:50051": True
        },
        "cartservice": {
            "redis-cart:6379": True
        },
        "loadgenerator": {
            "frontend:80": True  # HTTP service requires additional checks
        }
    }

    result = correctness_check(expected_results)
    print(f"\nFinal result: {result}")
    exit(0 if result else 1)