#!/bin/bash
cd ..
cd app-malt

# Define common parameters
NUM_QUERIES=1
OUTPUT_DIR="logs/llm_agents"
BENCHMARK_PATH="data/benchmark_malt.jsonl"

# Function to run experiment for a model
run_experiment() {
    local model=$1
    local complexity=$2
    local output_file=$3
    
    # Create agent-specific output directory
    local agent_output_dir="logs/${model}"
    
    echo "Running experiment for $model..."
    
    python main.py \
        --llm_agent_type "$model" \
        --num_queries $NUM_QUERIES \
        --complexity_level $complexity \
        --output_dir "$agent_output_dir" \
        --output_file "$output_file" \
        --dynamic_benchmark_path "$BENCHMARK_PATH"
}

# Define models and their configurations
declare -A model_configs=(
    ["AzureGPT4Agent"]="level1 level2:gpt4o.jsonl"
    ["GoogleGeminiAgent"]="level1 level2:gemini.jsonl"
    ["Qwen/Qwen2.5-72B-Instruct"]="level1 level2:qwen.jsonl"
)

# Define the desired order of execution
model_order=("AzureGPT4Agent" "GoogleGeminiAgent")

# Run experiments in specified order
for model in "${model_order[@]}"; do
    echo "==============================================="
    echo "Starting experiment with model: $model"
    
    IFS=':' read -r complexity output_file <<< "${model_configs[$model]}"
    run_experiment "$model" "$complexity" "$output_file"
    
    echo "Finished experiment with model: $model"
    echo "==============================================="
    # Add a small delay between experiments
    sleep 5
done

echo "All experiments completed!" 