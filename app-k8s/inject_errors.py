import os
import random
import yaml
import itertools
import json
from typing import List, Dict

# Define the folder for storing policies
POLICIES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")

# Possible values for introducing errors
APPS = ["payment", "database", "gateway", "analytics"]
ENVS = ["dev", "prod", "shared"]

POSSIBLE_APPS = [
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontend", "paymentservice", "productcatalogservice",
    "recommendationservice", "redis-cart", "shippingservice"
]

def inject_errors(policy, error_count=2, complexity_level=1, error_types=None):
    """
    Injects errors into a valid Kubernetes NetworkPolicy while maintaining the correct order.
    """
    if "spec" not in policy:
        return policy  # Return if there is no spec field

    if error_types is None:
        error_types = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]
        selected_error_types = random.sample(error_types, min(error_count, len(error_types)))
    else:
        # Ensure error_types is a list
        selected_error_types = error_types if isinstance(error_types, list) else [error_types]
    print(f"Selected error types: {selected_error_types}")
    for error_type in selected_error_types:
        print(f"Injecting error: {error_type}")
        if error_type == "remove_ingress" and "ingress" in policy["spec"]:
            if policy["spec"]["ingress"]:
                # Directly remove the first ingress rule
                policy["spec"]["ingress"].pop(0)
        if error_type == "add_ingress":
            # Get all podSelector app values in existing ingress rules
            existing_apps = {
                f.get("podSelector", {}).get("matchLabels", {}).get("app")
                for rule in policy["spec"].get("ingress", [])
                for f in rule.get("from", [])
            }
            # Select an app from POSSIBLE_APPS that is not in existing rules, or use "newapp" if none
            new_app = random.choice([app for app in POSSIBLE_APPS if app not in existing_apps] or ["newapp"])
            # Ensure ingress field exists
            if "ingress" not in policy["spec"]:
                policy["spec"]["ingress"] = []
            
            # Directly append a new ingress rule
            new_rule = {
                "from": [
                    {"podSelector": {"matchLabels": {"app": new_app}}}
                ]
            }
            policy["spec"]["ingress"].append(new_rule)

        if error_type == "change_port" and "ingress" in policy["spec"]:
            for rule in policy["spec"]["ingress"]:
                for port_block in rule.get("ports", []):
                    if "port" in port_block:
                        port_block["port"] = random.choice([5432, 1])

        if error_type == "change_protocol" and "ingress" in policy["spec"]:
            for rule in policy["spec"]["ingress"]:
                for port_block in rule.get("ports", []):
                    if "protocol" in port_block:
                        port_block["protocol"] = "UDP"

        if error_type == "add_egress":
            # Get all podSelector app values in existing egress rules
            existing_apps = {
                f.get("podSelector", {}).get("matchLabels", {}).get("app")
                for rule in policy["spec"].get("egress", [])
                for f in rule.get("to", [])
            }
            # Select an app from POSSIBLE_APPS that is not in existing rules, or use "newapp" if none
            new_app = random.choice([app for app in POSSIBLE_APPS if app not in existing_apps] or ["newapp"])
            
            # Ensure egress field exists and clean empty rules
            policy["spec"].setdefault("egress", [])
            policy["spec"]["egress"] = [rule for rule in policy["spec"]["egress"] if rule]  # Remove empty rules
            
            # Construct a new rule (ensure non-empty)
            new_rule = {
                "to": [
                    {"podSelector": {"matchLabels": {"app": new_app}}}
                ]
            }
            policy["spec"]["egress"].append(new_rule)

    # Strictly adjust the order within spec
    ordered_spec = {
        "podSelector": policy["spec"].get("podSelector", {}),
        "policyTypes": policy["spec"].get("policyTypes", []),
        "ingress": policy["spec"].get("ingress", []),
        "egress": policy["spec"].get("egress", []),
    }

    # Final structure
    ordered_policy = {
        "apiVersion": policy.get("apiVersion", "networking.k8s.io/v1"),
        "kind": policy.get("kind", "NetworkPolicy"),
        "metadata": policy.get("metadata", {}),
        "spec": {k: v for k, v in ordered_spec.items() if v}  # Ensure order
    }

    return ordered_policy

def inject_errors_into_policies(policy_names, root_dir, complexity_level, error_type=None):
    error_count = 1
    # Randomly select two different policy names
    selected_policies = random.sample(policy_names, 1)

    for name in selected_policies:
        filename = os.path.join(root_dir, 'policies', f"{name}.yaml")
        with open(filename, "r") as f:
            policy = yaml.safe_load(f)

        policy_with_errors = inject_errors(policy, error_count, complexity_level, error_type)

        with open(filename, "w") as f:
            yaml.dump(policy_with_errors, f, default_flow_style=False)
        print(f"Injected errors into: {filename}")

def generate_config(root_dir, policy_names, num_queries):
    basic_errors = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]
    error_combinations = list(itertools.combinations(basic_errors, 2))
    basic_errors = ["remove_ingress", "change_port", "add_egress"]#
    error_config = []

    # Generate basic errors
    for error in basic_errors:
        for _ in range(num_queries):
            detail = {"type": error}
            if error == "add_ingress":
                detail["app"] = random.choice(POSSIBLE_APPS)
                # Exclude "loadgenerator" and "frontend" from the policies to inject
                policies = random.sample([p for p in policy_names if p not in ["network-policy-loadgenerator", "network-policy-frontend"]], 1)
                # Get all podSelector app values in existing ingress rules
                with open(os.path.join(root_dir, 'policies', f"{policies[0]}.yaml")) as f:
                    policy = yaml.safe_load(f)
                existing_apps = {
                    f.get("podSelector", {}).get("matchLabels", {}).get("app")
                    for rule in policy["spec"].get("ingress", [])
                    for f in rule.get("from", [])
                }
                # Ensure the selected app is not the same as the policy itself and not already in the ingress rules
                while detail["app"] == policies[0].replace("network-policy-", "") or detail["app"] in existing_apps:
                    detail["app"] = random.choice([app for app in POSSIBLE_APPS if app != policies[0].replace("network-policy-", "") and app not in existing_apps])
            elif error == "change_port":
                detail["new_port"] = random.choice([5432, 1])
                policies = random.sample([p for p in policy_names if p not in ["network-policy-loadgenerator", "network-policy-frontend"]], 1)
            elif error == "change_protocol":
                detail["new_protocol"] = "UDP"
                policies = random.sample([p for p in policy_names if p not in ["network-policy-loadgenerator", "network-policy-frontend"]], 1)
            elif error == "add_egress":
                detail["app"] = random.choice(POSSIBLE_APPS)
                policies = random.sample(["network-policy-checkoutservice", "network-policy-frontend"], 1)
            elif error == "remove_ingress":
                policies = random.sample([p for p in policy_names if p not in ["network-policy-loadgenerator", "network-policy-frontend"]], 1)
            
            error_config.append({
                "policies_to_inject": policies,
                "inject_error_num": [1],
                "error_detail": [detail]
            })

    # Generate combination errors
    for combo in error_combinations:
        for _ in range(num_queries):
            details = []
            for error in combo:
                detail = {"type": error}
                if error == "add_ingress":
                    detail["app"] = random.choice(POSSIBLE_APPS)
                    # Exclude "loadgenerator" and "frontend" from the policies to inject
                    policies = random.sample([p for p in policy_names if p not in ["network-policy-loadgenerator", "network-policy-frontend"]], 1)
                    # Get all podSelector app values in existing ingress rules
                    with open(os.path.join(root_dir, 'policies', f"{policies[0]}.yaml")) as f:
                        policy = yaml.safe_load(f)
                    existing_apps = {
                        f.get("podSelector", {}).get("matchLabels", {}).get("app")
                        for rule in policy["spec"].get("ingress", [])
                        for f in rule.get("from", [])
                    }
                    # Ensure the selected app is not the same as the policy itself and not already in the ingress rules
                    while detail["app"] == policies[0].replace("network-policy-", "") or detail["app"] in existing_apps:
                        detail["app"] = random.choice([app for app in POSSIBLE_APPS if app != policies[0].replace("network-policy-", "") and app not in existing_apps])
                elif error == "change_port":
                    detail["new_port"] = random.choice([5432, 1])
                    policies = random.sample([p for p in policy_names if p not in ["network-policy-loadgenerator", "network-policy-frontend"]], 1)
                elif error == "change_protocol":
                    detail["new_protocol"] = "UDP"
                    policies = random.sample([p for p in policy_names if p not in ["network-policy-loadgenerator", "network-policy-frontend"]], 1)
                elif error == "add_egress":
                    detail["app"] = random.choice(POSSIBLE_APPS)
                    policies = random.sample(["network-policy-checkoutservice", "network-policy-frontend"], 1)
                details.append(detail)

            policies = random.sample(
                ["network-policy-checkoutservice", "network-policy-frontend"] if "add_egress" in combo else policy_names,
                1
            )

            error_config.append({
                "policies_to_inject": policies,
                "inject_error_num": [len(combo)],
                "error_detail": details
            })

    # Save to file (ensure correct structure)
    output_path = os.path.join(root_dir, "error_config.json")
    with open(output_path, "w") as f:
        json.dump({"details": error_config}, f, indent=2)
    
    return error_config

def inject_config_errors_into_policies(
    policy_names: List[str],
    root_dir: str,
    inject_error_num: List[int],  # Strictly validate this parameter
    policies_to_inject: List[str],
    error_detail: List[Dict]
):
    """
    Strict validation for precise error injection
    
    Parameter structure as required:
    policy_names, root_dir, inject_error_num, policies_to_inject, error_detail
    """
    # Strict parameter validation (three new validation layers)
    if not isinstance(inject_error_num, list):
        raise TypeError("inject_error_num must be a list")
    
    if len(inject_error_num) != 1:
        raise ValueError("inject_error_num must be a single-element list, e.g., [2]")
    
    if inject_error_num[0] != len(error_detail):
        raise ValueError(
            f"Error count mismatch! Config declares {inject_error_num[0]} errors, "
            f"but actually provided {len(error_detail)} error details"
        )

    # Validate policy name validity
    invalid_policies = [name for name in policies_to_inject if name not in policy_names]
    if invalid_policies:
        raise ValueError(f"Invalid policy names: {invalid_policies}")

    # Iterate over each target policy to inject errors
    for policy_name in policies_to_inject:
        policy_path = os.path.join(root_dir, 'policies', f"{policy_name}.yaml")
        
        # Read policy file
        try:
            with open(policy_path, "r") as f:
                original_policy = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"[ERROR] Policy file not found: {policy_path}")
            continue

        # Perform injection and carry error count validation
        modified_policy = _inject_errors_with_detail(
            original_policy,
            error_detail,
            expected_errors=inject_error_num[0]  # Pass validation value
        )

        # Write back to file
        with open(policy_path, "w") as f:
            yaml.dump(modified_policy, f, default_flow_style=False)
        
        print(f"Successfully injected {len(error_detail)} errors into {policy_name}")
        return modified_policy

def _inject_errors_with_detail(
    policy: Dict,
    error_details: List[Dict],
    expected_errors: int  # New parameter for secondary validation
) -> Dict:
    """Enhanced core logic for error injection"""
    # Pre-validation
    if len(error_details) != expected_errors:
        raise RuntimeError(
            f"Fatal error: Expected to inject {expected_errors} errors, "
            f"but received {len(error_details)} error configurations"
        )

    modified_policy = policy.copy()
    
    for detail in error_details:
        error_type = detail["type"]
        
        # Use match-case for readability (Python 3.10+)
        match error_type:
            case "remove_ingress":
                if modified_policy["spec"].get("ingress"):
                    if modified_policy["spec"]["ingress"]:
                        modified_policy["spec"]["ingress"].pop(0)
            
            case "add_ingress":
                _validate_required_fields(detail, ["app"])
                # Remove empty ingress rules
                modified_policy["spec"]["ingress"] = [
                    rule for rule in modified_policy["spec"].get("ingress", []) if rule
                ]
                new_rule = {
                    "from": [{
                        "podSelector": {"matchLabels": {"app": detail["app"]}}
                    }]
                }
                modified_policy["spec"].setdefault("ingress", []).append(new_rule)
            
            case "change_port":
                _validate_required_fields(detail, ["new_port"])
                if modified_policy["spec"].get("ingress"):
                    for rule in modified_policy["spec"]["ingress"]:
                        for port in rule.get("ports", []):
                            port["port"] = detail["new_port"]
            
            case "change_protocol":
                _validate_required_fields(detail, ["new_protocol"])
                if modified_policy["spec"].get("ingress"):
                    for rule in modified_policy["spec"].get("ingress", []):
                        for port in rule.get("ports", []):
                            port["protocol"] = detail["new_protocol"]
            
            case "add_egress":
                _validate_required_fields(detail, ["app"])
                # Remove empty egress rules
                modified_policy["spec"]["egress"] = [
                    rule for rule in modified_policy["spec"].get("egress", []) if rule
                ]
                new_rule = {
                    "to": [{
                        "podSelector": {"matchLabels": {"app": detail["app"]}}
                    }]
                }
                modified_policy["spec"].setdefault("egress", []).append(new_rule)
            
            case _:
                raise ValueError(f"Unknown error type: {error_type}")

    # Maintain field order
    return {
        "apiVersion": modified_policy.get("apiVersion", "networking.k8s.io/v1"),
        "kind": "NetworkPolicy",
        "metadata": modified_policy.get("metadata", {}),
        "spec": {
            "podSelector": modified_policy["spec"].get("podSelector", {}),
            "policyTypes": modified_policy["spec"].get("policyTypes", []),
            "ingress": modified_policy["spec"].get("ingress", []),
            "egress": modified_policy["spec"].get("egress", [])
        }
    }

def _validate_required_fields(detail: Dict, required_fields: List[str]):
    """Validate required fields are present"""
    missing = [field for field in required_fields if field not in detail]
    if missing:
        raise ValueError(
            f"Error type {detail['type']} is missing required fields: {missing}"
        )

# Example usage
if __name__ == "__main__":
    generate_config("app-k8s", [ "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice", "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend", "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice", "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice" ], 5)
