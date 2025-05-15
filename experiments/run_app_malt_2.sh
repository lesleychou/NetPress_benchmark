#!/bin/bash
cd ..
cd app-malt

# Define common parameters
NUM_QUERIES=500
BENCHMARK_PATH="data/malt_benchmark_500.jsonl"
PROMPT_TYPE="cot"  # Define prompt_type
# PROMPT_TYPE="few_shot_basic"  # Define prompt_type
# PROMPT_TYPE="zero_shot_cot"  # Define prompt_type


# Function to run experiment for a model
run_experiment() {
    local llm_model_type=$1
    local prompt_type=$2
    local complexity=$3
    local output_file=$4
    
    # Create agent-specific output directory
    local agent_output_dir="logs/${llm_model_type}_${prompt_type}"
    
    echo "Running experiment for $llm_model_type..."
    
    python main.py \
        --llm_model_type "$llm_model_type" \
        --prompt_type "$PROMPT_TYPE" \
        --num_queries $NUM_QUERIES \
        --complexity_level $complexity \
        --output_dir "$agent_output_dir" \
        --output_file "$output_file" \
        --dynamic_benchmark_path "$BENCHMARK_PATH" \
        --start_index 500
        }

# Define models and their configurations
declare -A model_configs=(
    ["AzureGPT4Agent"]="level1 level2 level3:gpt4o_cot_500.jsonl"
)

# Define the desired order of execution
model_order=("AzureGPT4Agent")

# Run experiments in specified order
for model in "${model_order[@]}"; do
    echo "==============================================="
    echo "Starting experiment with model: $model, prompt type: $PROMPT_TYPE"
    
    IFS=':' read -r complexity output_file <<< "${model_configs[$model]}"
    run_experiment "$model" "$PROMPT_TYPE" "$complexity" "$output_file"
    
    echo "Finished experiment with model: $model, prompt type: $PROMPT_TYPE"
    echo "==============================================="
    # Add a small delay between experiments
    sleep 5
done

echo "All experiments completed!" 