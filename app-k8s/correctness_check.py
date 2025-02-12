import subprocess
import shlex

def find_pod_by_prefix(prefix):
    """Find the pod name that matches the given prefix."""
    try:
        result = subprocess.run(
            shlex.split("kubectl get pods --no-headers -o custom-columns=:metadata.name"),
            capture_output=True, text=True, check=True
        )
        pods = result.stdout.strip().split("\n")
        for pod in pods:
            if pod.startswith(prefix):
                return pod
    except subprocess.CalledProcessError as e:
        print(f"Error listing pods: {e}")
    return None

def curl_from_pod(pod_name, url, timeout=5):
    """Execute curl to check if pod_name can access url.
       - Timeout (curl: (28) Connection timed out) is considered a failure, returns False.
       - Any other response (including Empty reply from server) is considered success, returns True.
    """
    try:
        print(f"Checking connectivity from {pod_name} to {url}...")
        command = f"kubectl exec -it {pod_name} -- curl --max-time {timeout} {url}"
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=timeout + 1)

        # Check if stderr and stdout contain timeout information
        if "Connection timed out" in result.stderr or "Operation timed out" in result.stderr or "exit code 28" in result.stderr:
            print(f"Timeout: {pod_name} cannot reach {url} (Timeout after {timeout}s).")
            return False  # Timeout returns False
        elif result.stdout or "Empty reply from server" in result.stderr:
            print(f"Success: {pod_name} can talk to {url}.\nOutput:\n{result.stdout}{result.stderr}")
            return True  # Any response (including Empty reply) is considered success
        else:
            print(f"Failure: No response from {url} in {timeout} seconds.")
            return False  # No response is considered failure

    except subprocess.TimeoutExpired:
        print(f"Timeout: {pod_name} cannot reach {url} within {timeout} seconds.")
        return False  # Timeout directly returns False

def correctness_check(expected_results):
    """ Iterate through all pods for connectivity tests and compare with expected_results """
    all_match = True  # Record whether all results match

    for pod_prefix, urls in expected_results.items():
        pod_name = find_pod_by_prefix(pod_prefix)
        if not pod_name:
            print(f"Error: No pod found with prefix {pod_prefix}")
            return False  # Directly return False

        for url, expected_value in urls.items():
            actual_value = curl_from_pod(pod_name, url)
            if actual_value != expected_value:
                print(f"Mismatch: Expected {expected_value} but got {actual_value} for {pod_name} → {url}")
                all_match = False  # Found a mismatch

    return all_match  # Return True if all checks pass, otherwise return False

if __name__ == "__main__":
    expected_results = {
        "payment": {"http://database-service:5432": True},  # Empty reply is considered True
        "analytics": {
            "http://database-service:80": False,  # Timeout (exit code 28) is considered False
            "http://payment-service:80": False
        },
        "gateway": {"http://payment-service:80": True}
    }
    
    result = correctness_check(expected_results)
    print(f"\nFinal correctness check result: {result}")
    exit(0 if result else 1)  # Process returns 0 for success, 1 for failure
