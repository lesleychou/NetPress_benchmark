# README

## Install Environment
Initialize a new virtual env (conda or pip) by your self. I test it on Python 3.10

### Install Python Environment

```
conda create -n mininet python=3.10
conda activate mininet
pip install -r requirements.txt
```

### Install Mininet Emulator Environment
To install the Mininet emulator, run the following command (we tested on Ubuntu 22.04):

```
git clone git://github.com/mininet/mininet
cd mininet
cd util
sudo ./install.sh -a
```

If you see `Enjoy Mininet`, you install mininet environment successfully!

### Install Fast Ping

To accelerate network simulation speed, we use `fping` for faster simulation.
```
sudo apt-get install fping
```
## Instructions for Benchmark Testing
In our benchmark, there are four models to test:

- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- `Microsoft/Phi4`
- `google/gemma-7b`
- `Qwen/QwQ-32B-Preview`

To test these models in the benchmark, you need to obtain **access** to them on Hugging Face. Then, update the `llm_model.py` code with the following command to include your Hugging Face token:

```
login(token="your_token")
```
## Running Benchmark Tests
To run the benchmark tests, you can use the default configuration by executing the following command:
```
sudo -E $(which python) main.py
```
If you want to customize the test configuration, you can specify parameters using the following command:
```
sudo -E $(which python) main.py \
  --llm_agent_type "Qwen/Qwen2.5-72B-Instruct" \
  --num_queries 10 \
  --complexity_level level2 \
  --root_dir "/your/path/to/nemo_benchmark/app-route" \
  --max_iteration 15
  --full_test 1
```
### Explanation of Parameters:
- **`--llm_agent_type`**: Specifies the model to test (e.g., `Qwen/Qwen2.5-72B-Instruct`).
- **`--num_queries`**: Defines the number of queries for the test.
- **`--complexity_level`**: Sets the complexity level of the benchmark (e.g., `level2`).
- **`--root_dir`**: Indicates the path to the benchmark directory (e.g., `/your/path/to/nemo_benchmark/app-route`).
- **`--max_iteration`**: Specifies the maximum number of iterations(e.g., `15`).
- **`--full_test`**: Defines whether to run the test by error_type to test LLM's capability on specific error type (e.g., '1' represents testing on every error type).

# To run the benchmark with GPT-4o
You have to run `sudo az login` to login to Azure first.

```
sudo -E $(which python) main.py   \
--llm_agent_type "GPT-Agent"   \
--num_queries 10   \
--complexity_level level2   \
--root_dir "/your/path/to/nemo_benchmark/app-route"   \
--max_iteration 15
```

# To run the benchmark with Google Gemini
```
sudo -E $(which python) main.py   \
--llm_agent_type "Google/Gemini"   \
--num_queries 10   \
--complexity_level level2   \
--root_dir "/your/path/to/nemo_benchmark/app-route"   \
--max_iteration 15
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

In React agent.
```
pip install -U duckduckgo-search
pip install langchain_experimental
```