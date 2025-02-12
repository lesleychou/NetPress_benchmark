import os
import subprocess

# Define the folder for storing policies
POLICIES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")

def deploy_policies():
    policy_names = ["allow-gateway-to-prod", "allow-payment-to-db", "block-dev-from-prod"]

    for name in policy_names:
        filename = os.path.join(POLICIES_FOLDER, f"{name}.yaml")
        try:
            result = subprocess.run(["kubectl", "apply", "-f", filename], check=True, text=True, capture_output=True)
            print(f"Deployed {filename}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to deploy {filename}:\n{e.stderr}")

if __name__ == "__main__":
    deploy_policies()