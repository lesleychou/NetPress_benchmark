import os
import random
import yaml

# Define the folder for storing policies
POLICIES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")

# Possible values for introducing errors
APPS = ["payment", "database", "gateway", "analytics"]
ENVS = ["dev", "prod", "shared"]

import random

import random

def inject_errors(policy, error_count=2, complexity_level=1):
    """
    Injects errors into a valid Kubernetes NetworkPolicy while maintaining the correct order.
    """
    if "spec" not in policy:
        return policy  # 没有 spec 字段时直接返回

    POSSIBLE_APPS = ["frontend", "backend", "redis", "paymentservice", "shippingservice", "randomapp"]

    error_types = ["remove_ingress", "add_ingress", "change_port", "change_protocol", "add_egress"]
    selected_error_types = random.sample(error_types, min(error_count, len(error_types)))

    for error_type in selected_error_types:
        if error_type == "remove_ingress" and "ingress" in policy["spec"]:
            for rule in policy["spec"]["ingress"]:
                if "from" in rule:
                    rule["from"] = [f for f in rule["from"] if f.get("podSelector", {}).get("matchLabels", {}).get("app") != "frontend"]

        elif error_type == "add_ingress":
            existing_apps = {f.get("podSelector", {}).get("matchLabels", {}).get("app") for rule in policy["spec"].get("ingress", []) for f in rule.get("from", [])}
            new_app = random.choice([app for app in POSSIBLE_APPS if app not in existing_apps] or ["newapp"])
            if "ingress" not in policy["spec"]:
                policy["spec"]["ingress"] = []
            if not policy["spec"]["ingress"]:
                policy["spec"]["ingress"].append({"from": [{"podSelector": {"matchLabels": {"app": new_app}}} ]})
            else:
                policy["spec"]["ingress"][0].setdefault("from", []).append({"podSelector": {"matchLabels": {"app": new_app}}})

        elif error_type == "change_port" and "ingress" in policy["spec"]:
            for rule in policy["spec"]["ingress"]:
                for port_block in rule.get("ports", []):
                    if "port" in port_block:
                        port_block["port"] = random.choice(["99999", "-1"])

        elif error_type == "change_protocol" and "ingress" in policy["spec"]:
            for rule in policy["spec"]["ingress"]:
                for port_block in rule.get("ports", []):
                    if "protocol" in port_block:
                        port_block["protocol"] = "UDP"

        elif error_type == "add_egress":
            policy["spec"].setdefault("egress", []).append({})

    # **严格调整 spec 内部顺序**
    ordered_spec = {
        "podSelector": policy["spec"].get("podSelector", {}),
        "policyTypes": policy["spec"].get("policyTypes", []),
        "ingress": policy["spec"].get("ingress", []),
        "egress": policy["spec"].get("egress", []),
    }

    # **最终结构**
    ordered_policy = {
        "apiVersion": policy.get("apiVersion", "networking.k8s.io/v1"),
        "kind": policy.get("kind", "NetworkPolicy"),
        "metadata": policy.get("metadata", {}),
        "spec": {k: v for k, v in ordered_spec.items() if v}  # **确保顺序**
    }

    return ordered_policy




def inject_errors_into_policies(root_dir, complexity_level):
    policy_names = [ "network-policy-adservice", "network-policy-cartservice", "network-policy-checkoutservice", "network-policy-currencyservice", "network-policy-emailservice", "network-policy-frontend", "network-policy-loadgenerator", "network-policy-paymentservice", "network-policy-productcatalogservice", "network-policy-recommendationservice", "network-policy-redis", "network-policy-shippingservice" ]
    error_count = 2
    # 随机选择两个不同的策略名称
    # selected_policies = random.sample(policy_names, 2)
    selected_policies = [ "network-policy-adservice" ]
    for name in selected_policies:
        filename = os.path.join(root_dir, 'kustomize/components/network-policies', f"{name}.yaml")
        with open(filename, "r") as f:
            policy = yaml.safe_load(f)

        policy_with_errors = inject_errors(policy, error_count, complexity_level)
        print(policy_with_errors)
        with open(filename, "w") as f:
            yaml.dump(policy_with_errors, f, default_flow_style=False)
        print(f"Injected errors into: {filename}")

if __name__ == "__main__":
    inject_errors_into_policies("/home/ubuntu/microservices-demo", 1)
