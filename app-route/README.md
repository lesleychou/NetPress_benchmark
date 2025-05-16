# README


## Python Prerequisites

To set up the Python environment, we use `conda` to create a virtual environment. You can install the required dependencies by running the following commands:

```bash
conda env create -f environment_mininet.yml
conda activate mininet
```

### Install Mininet Emulator Environment
To install the Mininet emulator, run the following command (we tested on Ubuntu 22.04):

```
chmod +x install_mininet.sh
./install_mininet.sh
```

If you see `Enjoy Mininet`, you install mininet environment successfully!
### Set the environment variable:
You need to set environment variable so that you can access open source LLM. You can do it by using the following command:
```
export HUGGINGFACE_TOKEN="your_huggingface_token"
```
## Running Benchmark Tests
### 1. Navigate to experiments
To run the benchmark tests, you can use our `run_app_route.sh` in `experiments`:
```
cd experiments
```
### 2. Modify Parameters in `run_app_k8s.sh`

Before running the benchmarking script, you need to modify the parameters in `run_app_route.sh` to suit your setup.  Below is a list of configurable parameters and their explanations.

### `MODEL`:
- **Description**: Specifies the type of LLM agent to be used in the benchmark. The format typically includes the name and version of the agent, such as `Qwen/Qwen2.5-72B-Instruct`. This determines which LLM model will be evaluated during the benchmarking process.
- **How to modify**: Replace this with the desired LLM model type.
- **Example**: `Qwen/Qwen2.5-72B-Instruct`

### `NUM_QUERIES`:
- **Description**: Defines the number of queries to generate during the benchmarking process. This determines how many individual queries for each error type will be tested.
- **Example**: `10` (Test with one query)

### `ROOT_DIR`:
- **Description**: The root directory where output logs and results will be stored. This path should point to the location on your machine where the benchmark results will be saved. Ensure that the specified directory exists and is accessible.
- **Example**: `/home/ubuntu/nemo_benchmark/app-route`

### `MAX_ITERATION`:
- **Description**: The maximum number of iterations to run for each query. This helps control the number of times the agent will execute the query in each benchmark run. 
- **Example**: `10` (Run each query up to 10 iterations)

### `STATIC_GEN`:
- **Description**: This parameter controls whether a new configuration should be generated for each benchmark. Set it to `1` to generate a new configuration, or `0` to skip this step and use the existing configuration.
- **Example**: `1` (Generate new configuration)

### `PROMPT_TYPE`:
- **Description**: Specifies the type of prompt to use when interacting with the LLM. The prompt type affects the nature of the queries sent to the LLM. You can choose between basic and more advanced prompts, depending on your test requirements.
- **Example**: `base` (Use the basic prompt type)

### 3. Run the Benchmark
After modifying the parameters, you can execute the benchmarking process by running the script with the following command:
```bash
bash run_app_route.sh
```
# Testing Your Own Model

To test your own model, follow these steps:

1. **Modify Model Name and Initialization Method:**
   - In the `llm_model.py` file, locate the `LLMModel` class. 
   - In the `_create_model` and `_initialize_YourModel` methods, update the model name and the initialization method to match your own model.

2. **Modify Model Loading and Prediction:**
   - In the `llm_model.py` file, locate the `YourModel` class.
   - In the `_load_model` method, update the correct way to load your own model.
   - In the `predict` method, modify the code to generate results using your LLM based on the provided prompt.

Once these modifications are complete, your model will be ready to be integrated into the benchmark environment. You can then proceed with testing by following the instructions in the `Running Benchmark Tests` section.


### Note: Azure GPT usage
Obtain GPT resources and endpoints
If you use Azure GPT on a Azure VM, need to use the following
```python
from azure.identity import AzureCliCredential
# Get the Azure Credential
credential = AzureCliCredential()
```
Otherwise, use the following
```python
from azure.identity import DefaultAzureCredential
# Get the Azure Credential
credential = DefaultAzureCredential()
```
And please update the below with your own endpoint information
```python
#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("please_update").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "please_update"

```