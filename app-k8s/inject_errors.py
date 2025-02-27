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

def inject_errors(policy, error_count=2, complexity_level=1,error_types=None):
    """
    Injects errors into a valid Kubernetes NetworkPolicy while maintaining the correct order.
    """
    if "spec" not in policy:
        return policy  # 没有 spec 字段时直接返回

    POSSIBLE_APPS = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend",  "paymentservice", "productcatalogservice", "recommendationservice", "redis-cart", "shippingservice"]
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
                # 直接删除 ingress 规则中的第一条
                policy["spec"]["ingress"].pop(0)
        if error_type == "add_ingress":
            # 获取现有 ingress 规则中所有 podSelector 的 app 值
            existing_apps = {
                f.get("podSelector", {}).get("matchLabels", {}).get("app")
                for rule in policy["spec"].get("ingress", [])
                for f in rule.get("from", [])
            }
            # 从 POSSIBLE_APPS 中选出一个不在现有规则中的 app，若没有则使用 "newapp"
            new_app = random.choice([app for app in POSSIBLE_APPS if app not in existing_apps] or ["newapp"])
            # 确保 ingress 字段存在
            if "ingress" not in policy["spec"]:
                policy["spec"]["ingress"] = []
            
            # 直接追加一条新的 ingress 规则
            new_rule = {
                "from": [
                    {"podSelector": {"matchLabels": {"app": new_app}}}
                ]
            }
            policy["spec"]["ingress"].append(new_rule)
            print("111")

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




def inject_errors_into_policies(policy_names, root_dir, complexity_level, error_type=None):
    error_count = 1
    # 随机选择两个不同的策略名称
    selected_policies = random.sample(policy_names, 1)

    for name in selected_policies:
        filename = os.path.join(root_dir, 'policies', f"{name}.yaml")
        with open(filename, "r") as f:
            policy = yaml.safe_load(f)

        policy_with_errors = inject_errors(policy, error_count, complexity_level, error_type)

        with open(filename, "w") as f:
            yaml.dump(policy_with_errors, f, default_flow_style=False)
        print(f"Injected errors into: {filename}")

if __name__ == "__main__":
    inject_errors_into_policies("/home/ubuntu/microservices-demo", 1)    
