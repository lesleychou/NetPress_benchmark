import os
import random
import yaml
import itertools
import json
from typing import List, Dict
import copy

"""
For "remove_ingress" error:
Remove all ingress rules
Remove specific ingress rules (targeted by pod, namespace, etc.)
Replace ingress rules with overly restrictive ones
Method 4: Make ingress more restrictive (remove specific selectors)
Method 5: Add invalid ingress fields

For "add_ingress" error:
Add ingress from unexpected pods
Add ingress with wrong ports
Add ingress with specific IPs instead of pod selectors
Method 4: Add malformed CIDR blocks
Method 5: Add incorrect label selectors

For "change_port" error:
Change to a completely different port
Add port range instead of a specific port
Use string naming for ports instead of numbers
Method 4: Use port ranges instead of specific ports
Method 5: Use malformed port names

For "change_protocol" error:
Change from TCP to UDP, SCTP
Use invalid protocol names
Add multiple conflicting protocol specifications
Method 4: Remove protocol field completely
Method 5: Use case-sensitive/invalid formatting for protocols

5. For "add_egress" error:
Add egress to unexpected pods
Add egress with CIDR ranges
Add overly permissive egress rules
Method 4: Add DNS selectors
Method 5: Add malformed selectors
"""


def generate_config(root_dir, policy_names, num_queries):
    # Define error types and combinations
    basic_errors = [
        "remove_ingress", 
        "add_ingress", 
        "change_port", 
        "change_protocol", 
        "add_egress",
        # New error types
        "remove_egress",
        "modify_selectors",
        "remove_policy_types",
        "add_invalid_fields",
        "conflicting_rules"
    ]
    error_combinations = list(itertools.combinations(basic_errors, 2))
    error_config = []

    # Predefined detail ranges
    detail_range = {
        "add_ingress_rule": {
            "adservice": ["recommendationservice", "productcatalogservice", "cartservice", "checkoutservice",  "emailservice", "shippingservice"],
            "recommendationservice": ["adservice", "productcatalogservice", "cartservice", "checkoutservice", "emailservice", "shippingservice"],
            "productcatalogservice": ["adservice", "recommendationservice", "cartservice", "emailservice", "shippingservice"],
            "redis": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "checkoutservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "shippingservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice"],
            "currencyservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "paymentservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "emailservice": ["loadgenerator", "frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "shippingservice"],
            "cartservice": ["adservice", "recommendationservice", "productcatalogservice","emailservice", "shippingservice"],
            "loadgenerator": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "emailservice", "shippingservice"],
            "frontend": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "emailservice", "shippingservice"]
        },
        "add_egress_rule": {
            "adservice": ["recommendationservice", "productcatalogservice", "cartservice", "checkoutservice",  "emailservice", "shippingservice","redis-cart"],
            "productcatalogservice": ["adservice", "recommendationservice", "cartservice", "emailservice", "shippingservice","redis-cart"],
            "shippingservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice","redis-cart"],
            "paymentservice": ["frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"],
            "currencyservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice", "redis-cart"],
            "emailservice": ["loadgenerator", "frontend", "adservice", "recommendationservice", "productcatalogservice", "cartservice", "shippingservice", "redis-cart"],
            "recommendationservice": ["adservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice","redis-cart"],
            "checkoutservice": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "paymentservice", "emailservice", "shippingservice", "currencyservice", "redis-cart"],
            "cartservice": ["adservice", "recommendationservice", "productcatalogservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice"],
            "frontend": ["adservice", "recommendationservice", "productcatalogservice", "cartservice", "checkoutservice", "paymentservice", "emailservice", "shippingservice", "currencyservice", "resid-cart"],
            "redis":["adservice", "recommendationservice", "productcatalogservice", "cartservice", "emailservice", "shippingservice"]
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
    count = 0
    for policy in policy_names:
        if not (policy == "network-policy-frontend" or policy == "netwwork-policy-loadgenerator" or policy == "network-policy-loadgenerator"):
            for key_value in ["UDP", "SCTP"]:
                detail = {"type": "change_protocol", "new_protocol": key_value}
                policies = [policy]
                count = count + 1
                if count <= num_queries:
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
                allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                policy_name = random.choice(allowed_policies)
                policy = f"network-policy-{policy_name}"  
                if policy_name in detail_range["add_ingress_rule"]:
                    detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
            elif error == "change_port":
                allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                policy_name = random.choice(allowed_policies) 
                policy = f"network-policy-{policy_name}" 
                detail["new_port"] = random.randint(1, 65535)
            elif error == "add_egress":
                allowed_policies = ["recommendationservice", "checkoutservice", "cartservice", "frontend"]
                policy_name = random.choice(allowed_policies)  
                policy = f"network-policy-{policy_name}"  
                detail["app"] = random.sample(detail_range["add_egress_rule"][policy_name], 2)
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
            policies = []
            
            for error in combo:    
                detail = {"type": error}
                if error == "add_ingress":
                    policy_name = random.choice(["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice", "frontend"])
                    policy = f"network-policy-{policy_name}"
                    policies.append(policy)
                    if policy_name in detail_range.get("add_ingress_rule", {}):
                        detail["app"] = random.sample(detail_range["add_ingress_rule"][policy_name], 2)
                elif error == "change_port":
                    allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                    policy_name = random.choice(allowed_policies) 
                    policy = f"network-policy-{policy_name}" 
                    policies.append(policy)
                    detail["new_port"] = random.randint(1, 65535)
                elif error == "change_protocol":
                    allowed_policies =["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "redis", "shippingservice" ]
                    policy_name = random.choice(allowed_policies) 
                    policy = f"network-policy-{policy_name}" 
                    policies.append(policy)
                    detail["new_protocol"] = random.choice(["UDP", "SCTP"])
                elif error == "add_egress":
                    allowed_policies = ["recommendationservice", "checkoutservice", "cartservice", "frontend"]
                    policy_name = random.choice(allowed_policies)  
                    policy = f"network-policy-{policy_name}"  
                    policies.append(policy)
                    if policy_name in detail_range.get("add_egress_rule", {}):
                        detail["app"] = random.sample(detail_range["add_egress_rule"][policy_name], 2)
                elif error == "remove_ingress":
                    allowed_policies = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice"]
                    policy_name = random.choice(allowed_policies)
                    policy = f"network-policy-{policy_name}"
                    if policy_name in detail_range.get("remove_ingress_rule", {}):
                        detail["app"] = random.choice(detail_range["remove_ingress_rule"][policy_name])
                    policies.append(policy)
                details.append(detail)
            

            
            error_config.append({
                "policies_to_inject": policies,
                "inject_error_num": [len(combo)],
                "error_detail": details
            })

    # ------------------------------
    # For the new error types, add similar blocks as existing ones
    for error in ["remove_egress", "modify_selectors", "remove_policy_types", "add_invalid_fields", "conflicting_rules"]:
        for _ in range(num_queries):
            detail = {"type": error, "method": random.randint(1, 3)}
            
            # Choose appropriate policies based on error type
            if error == "remove_egress":
                # Only target policies with egress rules
                allowed_policies = ["frontend", "currencyservice", "checkoutservice"]
                policy_name = random.choice(allowed_policies)
                policy = f"network-policy-{policy_name}"
            elif error == "modify_selectors":
                # Any policy can have selectors modified
                policy = random.choice(policy_names)
            else:
                policy = random.choice(policy_names)
                
            error_config.append({
                "policies_to_inject": [policy],
                "inject_error_num": [1],
                "error_detail": [detail]
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
    print(f"policies_to_inject:", policies_to_inject)
    # Validate policy name validity
    invalid_policies = [name for name in policies_to_inject if name not in policy_names]
    if invalid_policies:
        raise ValueError(f"Invalid policy names: {invalid_policies}")

    # Iterate over each target policy to inject errors
    for i, policy_name in enumerate(policies_to_inject):
        policy_path = os.path.join(root_dir, 'policies', f"{policy_name}.yaml")
        
        # Read policy file
        try:
            with open(policy_path, "r") as f:
                original_policy = yaml.safe_load(f)
            print(f"[INFO] Original policy loaded: {policy_path}")
        except FileNotFoundError:
            print(f"[ERROR] Policy file not found: {policy_path}")
            continue

        # Perform injection and carry error count validation
        modified_policy = _inject_errors_with_detail(
            original_policy,
            error_detail,
            i + 1  # number_of_errors represents the current iteration
        )
        print(f"[INFO] Modified policy: {modified_policy},{policy_name}")
        # Write back to file
        with open(policy_path, "w") as f:
            yaml.dump(modified_policy, f, default_flow_style=False)
        
        print(f"Successfully injected {len(error_detail)} errors into {policy_name}")
        print(modified_policy)
        print(i)

    return modified_policy

def _inject_errors_with_detail(
    policy: Dict,
    error_details: List[Dict],
    number_of_error: int
) -> Dict:
    """Enhanced core logic for error injection with diverse error patterns"""

    modified_policy = policy.copy()
    
    detail = error_details[number_of_error - 1]
    error_type = detail["type"]
    
    match error_type:
        case "remove_ingress":
            method = detail.get("method", random.randint(1, 5))
            
            # Method 1: Remove first ingress rule
            if method == 1:
                if modified_policy["spec"].get("ingress") and modified_policy["spec"]["ingress"]:
                    modified_policy["spec"]["ingress"].pop(0)
            
            # Method 2: Remove all ingress rules
            elif method == 2:
                modified_policy["spec"]["ingress"] = []
            
            # Method 3: Replace with empty ingress rule (more restrictive than removing)
            elif method == 3:
                modified_policy["spec"]["ingress"] = [{}]
                
            # Method 4: Keep structure but remove podSelector contents
            elif method == 4:
                if modified_policy["spec"].get("ingress"):
                    for rule in modified_policy["spec"]["ingress"]:
                        if "from" in rule:
                            for selector in rule["from"]:
                                if "podSelector" in selector:
                                    selector["podSelector"] = {}
                                    
            # Method 5: Remove ports specification
            elif method == 5:
                if modified_policy["spec"].get("ingress"):
                    for rule in modified_policy["spec"]["ingress"]:
                        if "ports" in rule:
                            del rule["ports"]

        case "add_ingress":
            _validate_required_fields(detail, ["app"])
            if not isinstance(detail["app"], list) or not detail["app"]:
                raise ValueError(f"Invalid app list in add_ingress: {detail['app']}")
                
            method = detail.get("method", random.randint(1, 5))
            
            # Method 1: Standard app selector based ingress
            if method == 1:
                new_rules = [
                    {
                        "from": [{"podSelector": {"matchLabels": {"app": app}}}]
                    }
                    for app in detail["app"]
                ]
                modified_policy["spec"].setdefault("ingress", []).extend(new_rules)
            
            # Method 2: Add overly permissive rule
            elif method == 2:
                modified_policy["spec"].setdefault("ingress", []).append({
                    "from": []  # Empty from means allow from anywhere
                })
            
            # Method 3: Add namespace selector instead of pod selector
            elif method == 3:
                namespaces = ["default", "kube-system", "monitoring"]
                new_rules = [
                    {
                        "from": [{"namespaceSelector": {"matchLabels": {"name": ns}}}]
                    }
                    for ns in random.sample(namespaces, min(2, len(namespaces)))
                ]
                modified_policy["spec"].setdefault("ingress", []).extend(new_rules)
                
            # Method 4: Add malformed CIDR blocks
            elif method == 4:
                bad_cidrs = ["192.168.1.1/33", "300.168.1.0/24", "10.0.0.0"]
                modified_policy["spec"].setdefault("ingress", []).append({
                    "from": [{"ipBlock": {"cidr": random.choice(bad_cidrs)}}]
                })
                
            # Method 5: Add incorrect label selectors
            elif method == 5:
                # Using non-existent labels or incorrect format
                bad_selectors = [
                    {"app.kubernetes.io/name": "frontend"},  # Doesn't match app label
                    {"service": detail["app"][0]},           # Wrong label key
                    {"app": f"{detail['app'][0]}-service"}   # Wrong label value format
                ]
                modified_policy["spec"].setdefault("ingress", []).append({
                    "from": [{"podSelector": {"matchLabels": random.choice(bad_selectors)}}]
                })
        
        case "change_port":
            _validate_required_fields(detail, ["new_port"])
            method = detail.get("method", random.randint(1, 5))
            
            # Method 1: Change to specific port number
            if method == 1:
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"]["ingress"]:
                        for port in rule.get("ports", []):
                            port["port"] = detail["new_port"]
            
            # Method 2: Change to named port instead of number
            elif method == 2:
                port_names = ["http", "https", "metrics", "admin", "debug"]
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"]["ingress"]:
                        for port in rule.get("ports", []):
                            port["port"] = random.choice(port_names)
            
            # Method 3: Remove port specification entirely
            elif method == 3:
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"]["ingress"]:
                        if "ports" in rule:
                            del rule["ports"]
                            
            # Method 4: Use port ranges (which may not be supported)
            elif method == 4:
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"]["ingress"]:
                        if "ports" in rule:
                            for port in rule["ports"]:
                                port["port"] = f"{detail['new_port']}-{detail['new_port']+1000}"
                                
            # Method 5: Use malformed port name
            elif method == 5:
                bad_ports = ["Port 80", "http:80", "$ervice", "app_port"]
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"]["ingress"]:
                        if "ports" in rule:
                            for port in rule["ports"]:
                                port["port"] = random.choice(bad_ports)
        
        case "change_protocol":
            _validate_required_fields(detail, ["new_protocol"])
            method = detail.get("method", random.randint(1, 5))
            
            # Method 1: Standard protocol change
            if method == 1:
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"].get("ingress", []):
                        for port in rule.get("ports", []):
                            port["protocol"] = detail["new_protocol"]
            
            # Method 2: Use lowercase protocol (should work but sometimes causes issues)
            elif method == 2:
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"].get("ingress", []):
                        for port in rule.get("ports", []):
                            port["protocol"] = detail["new_protocol"].lower()
            
            # Method 3: Use invalid protocol
            elif method == 3:
                invalid_protocols = ["ICMP", "GRE", "ESP", "ALL"]
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"].get("ingress", []):
                        for port in rule.get("ports", []):
                            port["protocol"] = random.choice(invalid_protocols)
                            
            # Method 4: Remove protocol field completely
            elif method == 4:
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"].get("ingress", []):
                        for port in rule.get("ports", []):
                            if "protocol" in port:
                                del port["protocol"]
                                
            # Method 5: Use mixed-case protocol (can cause issues)
            elif method == 5:
                mixed_case = ["Tcp", "tCp", "TCp", "tcP"]
                if "ingress" in modified_policy["spec"]:
                    for rule in modified_policy["spec"].get("ingress", []):
                        for port in rule.get("ports", []):
                            port["protocol"] = random.choice(mixed_case)
        
        case "add_egress":
            _validate_required_fields(detail, ["app"])
            if not isinstance(detail["app"], list) or not detail["app"]:
                raise ValueError(f"Invalid app list in add_egress: {detail['app']}")
                
            method = detail.get("method", random.randint(1, 5))
            
            # Remove empty egress rules
            modified_policy["spec"]["egress"] = [
                rule for rule in modified_policy["spec"].get("egress", []) if rule
            ]
                
            # Method 1: Standard pod selector based egress
            if method == 1:
                new_rules = [
                    {
                        "to": [{"podSelector": {"matchLabels": {"app": app}}}]
                    }
                    for app in detail["app"]
                ]
                modified_policy["spec"].setdefault("egress", []).extend(new_rules)
            
            # Method 2: Add CIDR range based egress
            elif method == 2:
                cidrs = [
                    "10.0.0.0/8", 
                    "172.16.0.0/12", 
                    "192.168.0.0/16",
                    "0.0.0.0/0"  # Extremely permissive
                ]
                cidr = random.choice(cidrs)
                modified_policy["spec"].setdefault("egress", []).append({
                    "to": [{"ipBlock": {"cidr": cidr}}]
                })
            
            # Method 3: Add port-specific egress
            elif method == 3:
                ports = [80, 443, 8080, 9090]
                new_rules = [
                    {
                        "to": [{"podSelector": {"matchLabels": {"app": app}}}],
                        "ports": [{"port": random.choice(ports), "protocol": "TCP"}]
                    }
                    for app in detail["app"]
                ]
                modified_policy["spec"].setdefault("egress", []).extend(new_rules)
                
            # Method 4: Add DNS-based rules (not standard in NetworkPolicy)
            elif method == 4:
                domains = ["api.example.com", "database.svc.cluster.local", "*.google.com"]
                modified_policy["spec"].setdefault("egress", []).append({
                    "to": [{"dnsSelector": {"domains": [random.choice(domains)]}}]
                })
                
            # Method 5: Add malformed selector
            elif method == 5:
                bad_selectors = [
                    {"matchNames": {"app": detail["app"][0]}},  # Wrong field name
                    {"matchLabels": {"app": [detail["app"][0]]}},  # Value as list instead of string
                    {"matchLabels": {detail["app"][0]: "app"}}  # Reversed key-value
                ]
                modified_policy["spec"].setdefault("egress", []).append({
                    "to": [{"podSelector": random.choice(bad_selectors)}]
                })
        
        # NEW ERROR TYPES
        case "remove_egress":
            method = detail.get("method", random.randint(1, 3))
            
            # Method 1: Remove all egress rules (blocks all outbound)
            if method == 1:
                modified_policy["spec"]["egress"] = []
                
            # Method 2: Replace with restrictive empty list
            elif method == 2:
                modified_policy["spec"]["egress"] = [{"to": []}]
                
            # Method 3: Allow egress only to specific unreachable destinations
            elif method == 3:
                modified_policy["spec"]["egress"] = [{
                    "to": [{"ipBlock": {"cidr": "192.0.2.0/24"}}]  # RFC 5737 TEST-NET-1 (unusable)
                }]
                
        case "modify_selectors":
            method = detail.get("method", random.randint(1, 3))
            
            # Method 1: Change podSelector to non-existent app
            if method == 1:
                non_existent = ["nonexistent-app", "temp-service", "debug-pod"]
                modified_policy["spec"]["podSelector"] = {
                    "matchLabels": {"app": random.choice(non_existent)}
                }
                
            # Method 2: Use incorrect selector syntax
            elif method == 2:
                bad_selectors = [
                    {"app": modified_policy["spec"]["podSelector"]["matchLabels"].get("app", "unknown")},  # Missing matchLabels
                    {"matchingLabels": {"app": modified_policy["spec"]["podSelector"]["matchLabels"].get("app", "unknown")}},  # Wrong key
                    {"matchLabels": {"name": modified_policy["spec"]["podSelector"]["matchLabels"].get("app", "unknown")}}  # Wrong label key
                ]
                modified_policy["spec"]["podSelector"] = random.choice(bad_selectors)
                
            # Method 3: Use expressions instead of matchLabels
            elif method == 3:
                app_name = modified_policy["spec"]["podSelector"]["matchLabels"].get("app", "unknown")
                modified_policy["spec"]["podSelector"] = {
                    "matchExpressions": [
                        {"key": "app", "operator": "In", "values": [f"{app_name}-typo"]}
                    ]
                }
                
        case "remove_policy_types":
            method = detail.get("method", random.randint(1, 3))
            
            # Method 1: Remove Ingress from policyTypes but keep ingress rules
            if method == 1:
                if "Ingress" in modified_policy["spec"].get("policyTypes", []):
                    modified_policy["spec"]["policyTypes"] = [
                        pt for pt in modified_policy["spec"]["policyTypes"] if pt != "Ingress"
                    ]
                    
            # Method 2: Remove Egress from policyTypes but keep egress rules
            elif method == 2:
                if "Egress" in modified_policy["spec"].get("policyTypes", []):
                    modified_policy["spec"]["policyTypes"] = [
                        pt for pt in modified_policy["spec"]["policyTypes"] if pt != "Egress"
                    ]
                    
            # Method 3: Remove policyTypes entirely
            elif method == 3:
                if "policyTypes" in modified_policy["spec"]:
                    del modified_policy["spec"]["policyTypes"]
                    
        case "add_invalid_fields":
            method = detail.get("method", random.randint(1, 3))
            
            # Method 1: Add non-standard fields
            if method == 1:
                custom_fields = [
                    {"timeout": 30},
                    {"priority": "high"},
                    {"allowPrivileged": False}
                ]
                modified_policy["spec"].update(random.choice(custom_fields))
                
            # Method 2: Add duplicate ingress/egress with different rules
            elif method == 2:
                if "ingress" in modified_policy["spec"]:
                    modified_policy["spec"]["ingressRules"] = [
                        {"from": []}  # Empty allowing all
                    ]
                
            # Method 3: Use incorrect field hierarchy
            elif method == 3:
                if "ingress" in modified_policy["spec"]:
                    # Move ingress up to the top level
                    modified_policy["ingress"] = modified_policy["spec"]["ingress"]
                    del modified_policy["spec"]["ingress"]
                    
        case "conflicting_rules":
            method = detail.get("method", random.randint(1, 3))
            
            # Method 1: Add allow-all and deny-specific rules
            if method == 1:
                allow_all = {"from": []}  # Empty means allow from anywhere
                deny_specific = {
                    "from": [{"podSelector": {"matchLabels": {"app": "frontend"}}}],
                    "ports": []  # Empty ports means no ports allowed
                }
                modified_policy["spec"]["ingress"] = [allow_all, deny_specific]
                
            # Method 2: Add port conflicts
            elif method == 2:
                if "ingress" in modified_policy["spec"] and modified_policy["spec"]["ingress"]:
                    original_rule = modified_policy["spec"]["ingress"][0]
                    # Clone the rule but with conflicting ports
                    conflicting_rule = copy.deepcopy(original_rule)
                    for port in conflicting_rule.get("ports", []):
                        if isinstance(port.get("port"), int):
                            port["port"] = port["port"] + 1
                    modified_policy["spec"]["ingress"].append(conflicting_rule)
                    
            # Method 3: Add ipBlock with except that contains the CIDR
            elif method == 3:
                cidr = "10.0.0.0/8"
                # Exception that nullifies the rule
                exception = "10.0.0.0/8"
                modified_policy["spec"].setdefault("ingress", []).append({
                    "from": [{
                        "ipBlock": {
                            "cidr": cidr,
                            "except": [exception]
                        }
                    }]
                })
                
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
