#!/bin/bash 
cd ..  # Navigate to the root directory
cd app-route  # Enter the application directory

# Define common parameters
NUM_QUERIES=1
ROOT_DIR="/home/ubuntu/NetPress_benchmark/app-route"
MAX_ITERATION=10
FULL_TEST=1
STATICGEN=1
PROMPT_TYPE="base"
# Define the model and parallel mode parameters
MODEL="Qwen/Qwen2.5-72B-Instruct"  # Replace with the model you want to use
PARALLEL=1  # Default to parallel execution. Set to 0 for single process.

# Create a log file with a timestamp to avoid overwriting
LOG_FILE="experiment_$(date +'%Y-%m-%d_%H-%M-%S').log"

# Function to clean up existing controller processes
cleanup_controllers() {
    echo "Cleaning up existing controller processes..." | tee -a "$LOG_FILE"
    sudo killall controller 2>/dev/null
    sudo mn -c >/dev/null 2>&1
    sleep 2  # Give some time for processes to fully terminate
}

# Function to run the benchmark
run_benchmark() {
    echo "Running experiment with model: $MODEL and parallel mode: $PARALLEL..." | tee -a "$LOG_FILE"
    
    cleanup_controllers
    
    nohup sudo -E $(which python) main.py \
        --llm_agent_type "$MODEL" \
        --num_queries $NUM_QUERIES \
        --root_dir "$ROOT_DIR" \
        --max_iteration $MAX_ITERATION \
        --static_benchmark_generation $STATICGEN \
        --prompt_type "$PROMPT_TYPE" \
        --parallel "$PARALLEL" >> "$LOG_FILE" 2>&1 &
}


# Run the benchmark based on the specified parallel mode
run_benchmark


