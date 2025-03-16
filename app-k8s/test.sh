#!/bin/bash

pods=(
    "adservice-66c577968c-bbcqx"
    "cartservice-5db54c864c-8wxbp"
    "checkoutservice-5c544d4688-gtmll"
    "currencyservice-795896b8c9-4k5tb"
    "emailservice-65d55b678-vmb45"
    "frontend-7f784cd5d4-6wf2w"
    "loadgenerator-b768b7967-b84pm"
    "paymentservice-7dd8c78d58-p4dnc"
    "productcatalogservice-59bbbc9bd-jvfhg"
    "recommendationservice-7c65f8fc6b-fmlj7"
    "redis-cart-6648b57c9b-n8s9r"
    "shippingservice-64f7bb586b-bd79c"
)

function ping_pod() {
    local source_pod=$1
    local target_pod=$2

    if [[ "$source_pod" == "$target_pod" ]]; then
        return
    fi

    target_ip=$(kubectl get pod "$target_pod" -o jsonpath='{.status.podIP}')
    
    echo "Pinging $target_pod ($target_ip) from $source_pod..."
    kubectl exec "$source_pod" -- ping -c 1 "$target_ip" > /dev/null
    
    if [ $? -ne 0 ]; then
        echo "ERROR: ********** Ping from $source_pod to $target_pod ($target_ip) failed **********"
    else
        echo "$(date): $source_pod -> $target_pod ($target_ip) - OK"
    fi
}



for source_pod in "${pods[@]}"; do
    for target_pod in "${pods[@]}"; do
        ping_pod "$source_pod" "$target_pod"
    done
done

