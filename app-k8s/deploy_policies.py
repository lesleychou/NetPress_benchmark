import os
import subprocess

# Define the folder for storing policies

def deploy_policies(policy_names=None, root_dir=None):
    """Deploy policies to the Kubernetes cluster."""
    for name in policy_names:
        filename = os.path.join(root_dir, "policies", f"{name}.yaml")
        try:
            result = subprocess.run(["kubectl", "apply", "-f", filename], check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to deploy {filename}:\n{e.stderr}")

if __name__ == "__main__":
    deploy_policies()