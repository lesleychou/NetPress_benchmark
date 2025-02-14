import os
import random
import yaml

# Define the folder for storing policies
POLICIES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")

# Possible values for introducing errors
APPS = ["payment", "database", "gateway", "analytics"]
ENVS = ["dev", "prod", "shared"]

def inject_errors(policy, error_count=2):
    """
    Injects errors into a valid CiliumNetworkPolicy while maintaining deployability.
    """
    if "spec" not in policy:
        return policy  # Ensure spec exists

    for _ in range(error_count):
        error_type = random.choice([
            "app_value", "env_value", "protocol_value", "port_value", 
            "extra_egress"
        ])

        # Modify `endpointSelector.matchLabels`
        if error_type == "app_value" and "endpointSelector" in policy["spec"]:
            match_labels = policy["spec"]["endpointSelector"].get("matchLabels", {})
            if "app" in match_labels:
                wrong_app = random.choice([app for app in APPS if app != match_labels["app"]])
                match_labels["app"] = wrong_app

        elif error_type == "env_value" and "endpointSelector" in policy["spec"]:
            match_labels = policy["spec"]["endpointSelector"].get("matchLabels", {})
            if "env" in match_labels:
                wrong_env = random.choice([env for env in ENVS if env != match_labels["env"]])
                match_labels["env"] = wrong_env

        # Modify `ingress` rules correctly
        elif error_type == "protocol_value" and "ingress" in policy["spec"]:
            for rule in policy["spec"]["ingress"]:
                if "toPorts" in rule:
                    for toPort in rule["toPorts"]:
                        for port in toPort.get("ports", []):
                            if "protocol" in port:
                                port["protocol"] = "UDP"  # Introduce an invalid protocol

        elif error_type == "port_value" and "ingress" in policy["spec"]:
            for rule in policy["spec"]["ingress"]:
                if "toPorts" in rule:
                    for toPort in rule["toPorts"]:
                        for port in toPort.get("ports", []):
                            if "port" in port:
                                port["port"] = random.choice(["-1", "99999"])  # Invalid port numbers

        # Add an extra invalid egress rule
        elif error_type == "extra_egress":
            policy["spec"]["egress"] = [{
                "toCIDR": ["192.168.1.100/32"],
                "toPorts": [{"ports": [{"port": "5432", "protocol": "TCP"}]}]
            }]

    return policy

def inject_errors_into_policies():
    policy_names = ["allow-gateway-to-prod", "allow-payment-to-db", "block-dev-from-prod"]

    for name in policy_names:
        filename = os.path.join(POLICIES_FOLDER, f"{name}.yaml")
        with open(filename, "r") as f:
            policy = yaml.safe_load(f)

        policy_with_errors = inject_errors(policy)

        with open(filename, "w") as f:
            yaml.dump(policy_with_errors, f, default_flow_style=False)
        print(f"Injected errors into: {filename}")

if __name__ == "__main__":
    inject_errors_into_policies()
