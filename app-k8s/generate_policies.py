import os
import yaml

# Define the folder for storing policies
POLICIES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")

def generate_cilium_policy(policy_name):
    # Define policy templates
    policies = {
        "allow-gateway-to-prod": {
            "metadata_name": "allow-gateway-to-prod",
            "endpoint_labels": {"app": "payment", "env": "prod"},
            "ingress": {
                "from_labels": {"app": "gateway"},
                "port": 80,
                "protocol": "TCP"
            }
        },
        "allow-payment-to-db": {
            "metadata_name": "allow-payment-to-db",
            "endpoint_labels": {"app": "database", "env": "prod"},
            "ingress": {
                "from_labels": {"app": "payment"},
                "port": 5432,
                "protocol": "TCP"
            }
        },
        "block-dev-from-prod": {
            "metadata_name": "block-dev-from-prod",
            "endpoint_labels": {"env": "prod"},
            "ingress": {
                "from_labels": {"env": "shared"},
                "port": "any",
                "protocol": "any"
            }
        }
    }

    policy = policies[policy_name]

    # Fit the policy into the template
    template = {
        "apiVersion": "cilium.io/v2",
        "kind": "CiliumNetworkPolicy",
        "metadata": {
            "name": policy["metadata_name"]
        },
        "spec": {
            "endpointSelector": {
                "matchLabels": policy["endpoint_labels"]
            },
            "ingress": [
                {
                    "fromEndpoints": [
                        {
                            "matchLabels": policy["ingress"]["from_labels"]
                        }
                    ],
                    "toPorts": [
                        {
                            "ports": [
                                {
                                    "port": str(policy["ingress"]["port"]),
                                    "protocol": policy["ingress"]["protocol"]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }

    return template

def generate_yaml_files():
    if not os.path.exists(POLICIES_FOLDER):
        os.makedirs(POLICIES_FOLDER)
        print(f"Created folder: {POLICIES_FOLDER}")

    policy_names = ["allow-gateway-to-prod", "allow-payment-to-db", "block-dev-from-prod"]

    for name in policy_names:
        policy = generate_cilium_policy(name)
        filename = os.path.join(POLICIES_FOLDER, f"{name}.yaml")
        with open(filename, "w") as f:
            yaml.dump(policy, f)
        print(f"Generated: {filename}")

if __name__ == "__main__":
    generate_yaml_files()