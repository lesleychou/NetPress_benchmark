#!/bin/bash
cd ../app-k8s

# Parameter settings
LLM_AGENT_TYPE="Qwen/Qwen2.5-72B-Instruct"  # Type of LLM agent
NUM_QUERIES=1                              # Number of queries to generate
ROOT_DIR="/home/ubuntu/NetPress_benchmark/app-k8s"  # Root directory for output
MICROSERVICE_DIR="/home/ubuntu/microservices-demo"  # Directory for microservice demo
MAX_ITERATION=10                           # Maximum number of iterations for a query
CONFIG_GEN=1                               # Whether to generate a new configuration (1 = yes, 0 = no)
PROMPT_TYPE="base"                         # Type of prompt to use
AGENT_TEST=1                               # Whether to test multiple agents (1 = yes, 0 = no)

# Replace special characters in LLM_AGENT_TYPE to make it a valid file name
SAFE_LLM_AGENT_TYPE=$(echo "$LLM_AGENT_TYPE" | tr '/' '_')

# Log file path
LOG_FILE="${ROOT_DIR}/${SAFE_LLM_AGENT_TYPE}_${PROMPT_TYPE}.log"

# Create the log file if it does not exist
touch "$LOG_FILE"

# Print the log file location
echo "Log saved to $LOG_FILE"

# Run the Python script and log the output
nohup python3 run_workflow.py \
    --llm_agent_type "$LLM_AGENT_TYPE" \
    --num_queries "$NUM_QUERIES" \
    --root_dir "$ROOT_DIR" \
    --microservice_dir "$MICROSERVICE_DIR" \
    --max_iteration "$MAX_ITERATION" \
    --config_gen "$CONFIG_GEN" \
    --prompt_type "$PROMPT_TYPE" \
    --agent_test "$AGENT_TEST" > "$LOG_FILE" 2>&1 &