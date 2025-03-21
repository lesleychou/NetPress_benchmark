import os
import random
import yaml
import itertools
import json
from typing import List, Dict

# Define the folder for storing policies
POLICIES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")

detail_range = {
  "add_ingress_rule": {
    "adservice": ["adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "recommendationservice": ["adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "productcatalogservice": ["adservice","recommendationservice","productcatalogservice","cartservice","paymentservice","emailservice","shippingservice","currencyservice"],
    "redis-cart": ["adservice","recommendationservice","productcatalogservice","cartservice","paymentservice","emailservice","shippingservice","currencyservice"],
    "checkoutservice": ["adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "shippingservice": ["adservice","recommendationservice","productcatalogservice","cartservice","paymentservice","emailservice","shippingservice","currencyservice"],
    "currencyservice": ["adservice","recommendationservice","productcatalogservice","cartservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "paymentservice": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","paymentservice","emailservice","shippingservice","currencyservice"],
    "emailservice": ["loadgenerator","frontend","adservice","recommendationservice","productcatalogservice","cartservice","paymentservice","emailservice","shippingservice","currencyservice"]
  },

  "add_grress_rule": {
    "adservice": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "recommendationservice": ["frontend","adservice","recommendationservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "productcatalogservice": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "redis-cart": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "checkoutservice": ["frontend","adservice","recommendationservice"],
    "shippingservice": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "currencyservice": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "paymentservice": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"],
    "emailservice": ["frontend","adservice","recommendationservice","productcatalogservice","cartservice","checkoutservice", "paymentservice","emailservice","shippingservice","currencyservice"]
  },


  "remove_ingress_rule": {
    "adservice": ["frontend"],
    "recommendationservice": ["frontend"],
    "productcatalogservice": ["frontend","recommendationservice","checkoutservice"],
    "cartservice": ["frontend","checkoutservice"],
    "checkoutservice": ["frontend"],
    "shippingservice": ["frontend","checkoutservice"],
    "currencyservice": ["frontend","checkoutservice"],
    "paymentservice": ["checkoutservice"],
    "emailservice": ["checkoutservice"]
  },
}




def generate_config(root_dir, policy_names, num_queries):
    # Define error types and combinations
    basic_errors = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]
    error_combinations = list(itertools.combinations(basic_errors, 2))
    error_config = []

    # Predefined detail ranges
    detail_range = {
        "add_ingress_rule": {
            "adservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "recommendationservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "productcatalogservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "redis": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "checkoutservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "shippingservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "currencyservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "paymentservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "emailservice": ["loadgenerator", "frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "cartservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "loadgenerator": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "frontend": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"]
        },
        "add_grress_rule": {
            "adservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "recommendationservice": ["fronstend", "adservice", "recommendationservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "productcatalogservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "redis": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "checkoutservice": ["frontend", "adservice", "recommendationservice"],
            "shippingservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "currencyservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "paymentservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "emailservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "cartservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "loadgenerator": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "frontend": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"]
        },
        "remove_ingress_rule": {
            "adservice": ["frontend"],
            "recommendationservice": ["frontend"],
            "productcatalogservice": ["frontend", "recommendationservice", "checkoutservice"],
            "cartservice": ["frontend", "checkoutservice"],
            "checkoutservice": ["frontend"],
            "shippingservice": ["frontend", "checkoutservice"],
            "currencyservice": ["frontend", "checkoutservice"],
            "paymentservice": ["checkoutservice"],
            "emailservice": ["checkoutservice"]
        },
    }

    # ------------------------------
    # Process single error: remove_ingress (target: min(num_queries, 14))
    target_remove = num_queries if num_queries < 14 else 14
    remove_ingress_entries = []
    for pol, apps in detail_range["remove_ingress_rule"].items():
        for app in apps:
            remove_ingress_entries.append((pol, app))
    remove_ingress_entries = remove_ingress_entries[:target_remove]
    for pol, app in remove_ingress_entries:
        detail = {"type": "remove_ingress", "app": app}
        policies = [f"network-policy-{pol}"]
        error_config.append({
            "policies_to_inject": policies,
            "inject_error_num": [1],
            "error_detail": [detail]
        })

    # ------------------------------
    # Process single error: change_protocol (target: min(num_queries, 18))

    for policy in policy_names:
        for key_value in ["UDP", "SCTP"]:
            detail = {"type": "change_protocol", "new_protocol": key_value}
            policies = [policy]
            error_config.append({
                "policies_to_inject": policies,
                "inject_error_num": [1],
                "error_detail": [detail]
            })


    # ------------------------------
    # Process other single errors (change_port, add_egress)
    for error in basic_errors:
        if error == "remove_ingress" or error == "change_protocol":
            continue  # already processed
        for _ in range(num_queries):
            detail = {"type": error}
            policy = random.choice(policy_names)
            policy_name = policy.replace("network-policy-", "")
            if error == "add_ingress":
                if policy_name in detail_range["add_ingress_rule"]:
                    detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
            elif error == "change_port":
                detail["new_port"] = random.randint(1, 65535)
                if policy_name in detail_range["add_ingress_rule"]:
                    detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
            elif error == "add_egress":
                if policy_name in detail_range["add_ingress_rule"]:
                    detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
            error_config.append({
                "policies_to_inject": [policy],
                "inject_error_num": [1],
                "error_detail": [detail]
            })

    # ------------------------------
    # Process combination errors: generate num_queries records for each combination
    for combo in error_combinations:
        for _ in range(num_queries):
            details = []
            policy = random.choice(policy_names)
            policy_name = policy.replace("network-policy-", "")
            for error in combo:
                detail = {"type": error}
                if error == "add_ingress":
                    if policy_name in detail_range["add_ingress_rule"]:
                        detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
                elif error == "change_port":
                    detail["new_port"] = random.randint(1, 65535)
                    if policy_name in detail_range["add_ingress_rule"]:
                        detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
                elif error == "change_protocol":
                    detail["new_protocol"] = random.choice(["UDP", "SCTP"])
                elif error == "add_egress":
                    if policy_name in detail_range["add_ingress_rule"]:
                        detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
                elif error == "remove_ingress":
                    if policy_name in detail_range["remove_ingress_rule"]:
                        detail["app"] = random.choice(detail_range["remove_ingress_rule"][policy_name])
                details.append(detail)
            error_config.append({
                "policies_to_inject": [policy],
                "inject_error_num": [len(combo)],
                "error_detail": details
            })

    # ------------------------------
    # Save result to file
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
        print(modified_policy)
        return modified_policy

def _inject_errors_with_detail(
    policy: Dict,
    error_details: List[Dict],
    expected_errors: int
) -> Dict:
    """Enhanced core logic for error injection with correct ingress/egress structure"""
    
    # Pre-validation
    if len(error_details) != expected_errors:
        raise RuntimeError(
            f"Fatal error: Expected to inject {expected_errors} errors, "
            f"but received {len(error_details)} error configurations"
        )

    modified_policy = policy.copy()
    
    for detail in error_details:
        error_type = detail["type"]
        
        match error_type:
            case "remove_ingress":
                if modified_policy["spec"].get("ingress"):
                    if modified_policy["spec"]["ingress"]:
                        modified_policy["spec"]["ingress"].pop(0)

            case "add_ingress":
                _validate_required_fields(detail, ["app"])
                if not isinstance(detail["app"], list) or not detail["app"]:
                    raise ValueError(f"Invalid app list in add_ingress: {detail['app']}")

                new_rules = [
                    {
                        "from": [{"podSelector": {"matchLabels": {"app": app}}}]
                    }
                    for app in detail["app"]
                ]

                modified_policy["spec"].setdefault("ingress", []).extend(new_rules)
            
            case "change_port":
                _validate_required_fields(detail, ["new_port"])
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"]["ingress"]:
                        for port in rule.get("ports", []):
                            port["port"] = detail["new_port"]
            
            case "change_protocol":
                _validate_required_fields(detail, ["new_protocol"])
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"].get("ingress", []):
                        for port in rule.get("ports", []):
                            port["protocol"] = detail["new_protocol"]
            
            case "add_egress":
                _validate_required_fields(detail, ["app"])
                if not isinstance(detail["app"], list) or not detail["app"]:
                    raise ValueError(f"Invalid app list in add_egress: {detail['app']}")

                # Remove empty egress rules
                modified_policy["spec"]["egress"] = [
                    rule for rule in modified_policy["spec"].get("egress", []) if rule
                ]

                new_rules = [
                    {
                        "to": [{"podSelector": {"matchLabels": {"app": app}}}]
                    }
                    for app in detail["app"]
                ]

                modified_policy["spec"].setdefault("egress", []).extend(new_rules)
            
            case _:
                raise ValueError(f"Unknown error type: {error_type}")

    # Maintain field order and ensure correct format
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
    generate_config("/home/ubuntu/jiajun_benchmark/app-k8s", [ "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice", "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend", "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice", "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice" ], 30)
