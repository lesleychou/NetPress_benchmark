#!/bin/bash
cd ..
cd app-route

# Define common parameters
NUM_QUERIES=2
COMPLEXITY_LEVEL="level1"
ROOT_DIR="/home/ubuntu/jiajun_benchmark/app-route"
MAX_ITERATION=5
FULL_TEST=1

# Function to clean up existing controller processes
cleanup_controllers() {
    echo "Cleaning up existing controller processes..."
    sudo killall controller 2>/dev/null
    sudo mn -c >/dev/null 2>&1
    sleep 2  # Give some time for processes to fully terminate
}

# Function to run experiment for a model
run_experiment() {
    local model=$1
    echo "Running experiment for $model..."
    
    # Clean up before each experiment
    cleanup_controllers
    
    sudo -E $(which python) main.py \
        --llm_agent_type "$model" \
        --num_queries $NUM_QUERIES \
        --complexity_level $COMPLEXITY_LEVEL \
        --root_dir "$ROOT_DIR" \
        --max_iteration $MAX_ITERATION \
        --full_test $FULL_TEST
}

# Run experiments for each model
models=(
    "meta-llama/Meta-Llama-3.1-70B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    # "Microsoft/Phi4"
    # "google/gemma-7b"
    "Qwen/QwQ-32B-Preview"
    # "GPT-Agent"
    # "Google/Gemini"
)

# models=(
#     "GPT-Agent"
#     "Google/Gemini"
# )

# Run experiments for each model
for model in "${models[@]}"; do
    echo "==============================================="
    echo "Starting experiment with model: $model"
    run_experiment "$model"
    echo "Finished experiment with model: $model"
    echo "==============================================="
    # Add a small delay between experiments
    sleep 5
done

echo "All experiments completed!" 