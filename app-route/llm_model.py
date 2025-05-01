import time
import json
import os
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from dotenv import load_dotenv
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent, FewShot_Semantic_PromptAgent
from datetime import datetime
from huggingface_hub import login
# Login huggingface
login(token="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")
from vllm import LLM, SamplingParams
# Load environ variables from .env, will not override existing environ variables
load_dotenv()

# For Google Gemini
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY")

# if "GOOGLE_API_KEY" not in os.environ:
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
        
# For Azure OpenAI GPT4
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.chat_models import AzureChatOpenAI
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ztn-oai-sweden.openai.azure.com/"

class LLMModel:
    """
    A simplified class for handling language models.

    Parameters:
    -----------
    model : str
        The name of the model to be used.
    max_new_tokens : int, optional
        The maximum number of new tokens to be generated (default is 20).
    temperature : float, optional
        The temperature for text generation (default is 0).
    device : str, optional
        The device to be used for inference (default is "cuda").
    api_key : str or None, optional
        The API key for API-based models (default is None).
    """

    @staticmethod
    def model_list():
        return [
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "Microsoft/Phi4",
            "google/gemma-7b",
            "Qwen/QwQ-32B-Preview",
            "GPT-Agent"
        ]

    def __init__(self, model: str, max_new_tokens: int = 256, temperature: float = 0.1, device: str = "cuda", api_key: str = None,vllm: bool = True, prompt_type: str = "cot"):
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.api_key = api_key
        self.vllm = vllm
        self.prompt_type=prompt_type
        self.model = self._create_model()

    @staticmethod
    def extract_value(text, keyword):
        """Extract a specific value from the text based on a keyword."""
        import re
        pattern = rf'"{keyword}"\s*:\s*"([^"]+)"'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None
    
    @staticmethod
    def extract_number_before_percentage(text):
        """Extract the number that appears before the '%' symbol in the text."""
        import re
        pattern = r'(\d+)(?=\s*%)'  # Looks for digits before the '%'
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
        return None
    
    def _create_model(self):
        """Creates and returns the appropriate model based on the model name."""
        if self.model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct":
            return self._initialize_meta_llama()
        elif self.model_name == "Qwen/Qwen2.5-72B-Instruct":
            return self._initialize_qwen()
        elif self.model_name == "Microsoft/Phi4":
            return self._initialize_Phi4()
        elif self.model_name == "google/gemma-7b":
            return self._initialize_gemma()
        elif self.model_name == "Qwen/QwQ-32B-Preview":
            return self._initialize_qwq()
        elif self.model_name == "GPT-Agent":
            return self._initialize_gpt_agent()
        elif self.model_name == "Google/Gemini":
            return self._initialize_gemini()
        elif self.model_name == "all-hands/openhands-lm-32b-v0.1":
            return self._initialize_openhands()
        elif self.model_name == "YourModel":
            return self._initialize_YourModel()
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported!")

    def _initialize_meta_llama(self):
        """Initialize the Meta-Llama model."""
        return LlamaModel(
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
        )
    def _initialize_openhands(self):
        """Initialize the OpenHands model."""
        return OpenHandsLLMModel(model_path=self.model_name, prompt_type=self.prompt_type)
    def _initialize_qwen(self):
        """Initialize the Qwen model."""
        if self.vllm:
            return Qwen_vllm_Model(
                model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                device=self.device
            )
        else:
            return QwenModel(
                model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ",
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                device=self.device,
                prompt_type=self.prompt_type
            )
    
    def _initialize_Phi4(self):
        """Initialize the Phi-4 model."""
        return Phi4Model(
            model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
        )
    
    def _initialize_gemma(self):
        """Initialize the Gemma model."""
        return GemmaModel(
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
        )
    
    def _initialize_qwq(self):
        """Initialize the   QwQ model."""
        return QwQModel(
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
        )
    
    def _initialize_gpt_agent(self):
        """Initialize the GPT Agent model."""
        return GPTAgentModel(prompt_type=self.prompt_type)

    def _initialize_gemini(self):
        """Initialize the Google Gemini model."""
        return GeminiAgentModel()

    def _initialize_YourModel(self):
        """Initialize the your model."""
        return YourModel

    def __call__(self, input_text: str, **kwargs):
        """Perform inference with the loaded model."""
        # Replace with actual inference logic
        return f"Generating response for: '{input_text}' using {self.model_name}"
    
class OpenHandsLLMModel:
    def __init__(self, model_path="/home/ubuntu/openhands-lm-32b-v0.1", max_new_tokens=512, temperature=0.0, device="cuda", prompt_type="base"):
        self.model_path = "/home/ubuntu/openhands-lm-32b-v0.1"
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.prompt_type = prompt_type
        self._initialize_prompt_agent()
        self._load_model()

    def _initialize_prompt_agent(self):
        if self.prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif self.prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            print("few_shot_semantic")
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}")

    def _load_model(self):
        self.llm = LLM(
            model=self.model_path,  
            device=self.device,
            max_model_len=35904
        )
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens
        )

    def _generate_prompt(self, file_content, log_content):
        connectivitity_status = file_content + log_content
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivitity_status)
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "few_shot_basic":
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        else:  # base
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the previous commands and the current pingall output:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        return prompt

    def predict(self, log_content, file_path, json_path):
        with open(file_path, 'r') as f:
            file_content = f.read()

        connectivitity_status = file_content + log_content
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivitity_status)
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "few_shot_basic":
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        else:  # base
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the previous commands and the current pingall output:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)

        start_time = time.time()
        result = self.llm.generate([prompt], sampling_params=self.sampling_params)
        content = result[0].outputs[0].text
        print('LLM output:', content)
        end_time = time.time()

        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(json_path, "r") as f:
            data = json.load(f)
        data.append({"packet_loss": loss_rate, "elapsed_time": end_time - start_time})
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        return machine, commands

# a =OpenHandsLLMModel(model_path="/home/ubuntu/openhands-lm-32b-v0.1", max_new_tokens=512, temperature=0.0, device="cuda")
# machine, commmands = a.predict("tenvist", "/home/ubuntu/jiajun_benchmark/app-route/result/GPT-Agent/agenttest/20250410-182938/cot_GPT/disable_routing/result_1.txt", "/home/ubuntu/jiajun_benchmark/app-route/result/GPT-Agent/agenttest/20250410-182938/cot_GPT/disable_routing/result_1.json")
# print("machine:", machine)
# print("commands:", commmands)

class QwenModel:
    """
    A specialized class for handling Qwen/Qwen2.5-72B-Instruct models.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device, prompt_type="base"):
        self.model_name = "Qwen/Qwen2.5-72B-Instruct"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._load_model()
        self.prompt_type = prompt_type
        if prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif prompt_type == "few_shot_semantic":
            print("few_shot_semantic")
            self.prompt_agent = FewShot_Semantic_PromptAgent()

    def _load_model(self):
        """Load the Qwen model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        # Create BitsAndBytesConfig for 4-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device
        )

        # Load the Qwen model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device,
            quantization_config=quantization_config  # Use the quantization config
        )

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""

        with open(file_path, 'r') as f:
            file_content = f.read()

        connectivitity_status = file_content + log_content
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivitity_status)
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the previous commands and the current pingall output:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        # print("prompt:", prompt)
        start_time = time.time()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            **kwargs
        )
        content = str(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands

class Qwen_vllm_Model:
    """
    A specialized class for handling Qwen/Qwen2.5-72B-Instruct models with GPTQ 4-bit quantization.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device="cuda", prompt_type="base"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._load_model()
        self.prompt_type = prompt_type
        if prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif prompt_type == "few_shot_semantic":
            print("few_shot_semantic")
            self.prompt_agent = FewShot_Semantic_PromptAgent()

    def _load_model(self):
        """Load the Qwen model using vllm with GPTQ 4-bit quantization."""

        self.llm = LLM(
            model=self.model_name,
            device=self.device,
            quantization="gptq"  # Enable GPTQ 4-bit loading
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )

    def predict(self, log_content, file_path, json_path, **kwargs):
        """根据日志内容和文件内容生成响应"""
        with open(file_path, 'r') as f:
            file_content = f.read()

        # 根据 prompt_type 和文件内容、日志内容生成 prompt
        connectivitity_status = file_content + log_content
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivitity_status)
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "few_shot_basic":
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivitity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)
        else:  # base
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the previous commands and the current pingall output:\n{input}"
            )
            prompt = prompt.format(input=connectivitity_status)

        # 开始生成响应
        start_time = time.time()

        # Generate response using vllm
        result = self.llm.generate([prompt], sampling_params=self.sampling_params)
        content = result[0].outputs[0].text
        print('LLM output:', content)

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands


class GPTAgentModel:
    """
    A specialized class for handling GPT Agent models.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    api_key : str
        The API key for GPT Agent.
    """

    def __init__(self, prompt_type="base"):
        self.prompt_type = prompt_type
        self._load_model()

    def _load_model(self):
        """Initialize the GPT Agent client."""

        self.client = AzureChatOpenAI(
            openai_api_version="2024-12-01-preview",
            deployment_name='ztn-sweden-gpt-4o',
            model_name='ztn-sweden-gpt-4o',
            temperature=0.0,
            max_tokens=4000,
        )
        if self.prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif self.prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            print("few_shot_semantic")
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        print("======GPT-4o successfully loaded=======")

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""
        with open(file_path, 'r') as f:
            file_content = f.read()
        connectivitity_status = file_content + log_content

        # content = response.choices[0].message.content
        max_length = 127000  
        if len(connectivitity_status) > max_length:
            connectivitity_status = connectivitity_status[:max_length]

        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivitity_status)
            input_data = {"input": connectivitity_status}
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            input_data = {"input": connectivitity_status}
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivitity_status}
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivitity_status}
        start_time = time.time()
        chain = LLMChain(llm=self.client, prompt=prompt)
        content = chain.run(input_data)
        print("LLM output:", content)
        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands

class GeminiAgentModel:
    """
    A specialized class for handling GPT Agent models.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    api_key : str
        The API key for GPT Agent.
    """

    def __init__(self):
        
        self._load_model()

    def _load_model(self):
        """Initialize the GPT Agent client."""

        self.client = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        print("======Google Gemini 1.5-pro successfully loaded=======")

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""
        with open(file_path, 'r') as f:
            file_content = f.read()

        prompt = LLMModel._generate_prompt(file_content, log_content)

        start_time = time.time()

        content = self.client.invoke(prompt).content
        print("LLM output:", content)

        # content = response.choices[0].message.content

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        time.sleep(60)
        
        return machine, commands

    
class YourModel:
    """
    A specialized class for handling YourModel.

    Parameters:
    -----------
    model_name : str
        The name of the model.
    max_new_tokens : int
        The maximum number of new tokens to be generated.
    temperature : float
        The temperature for text generation.
    device : str
        The device for inference.
    """

    def __init__(self, model_name, max_new_tokens, temperature, device):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load the your model and tokenizer."""


    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""

        with open(file_path, 'r') as f:
            file_content = f.read()

        prompt = LLMModel._generate_prompt(file_content, log_content)

        start_time = time.time()

        """Use your model to generate your reposnse here, results should be a str."""
        content = "..."

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Read LLM output
        machine = LLMModel.extract_value(content, "machine")
        commands = LLMModel.extract_value(content, "command")
        loss_rate = LLMModel.extract_number_before_percentage(log_content)

        with open(file_path, "a") as f:
            f.write("Log Content:\n")
            f.write(log_content + "\n\n")
            f.write(f"Machine: {machine}\n")
            f.write(f"Commands: {commands}\n")
            f.write("=" * 50 + "\n")

        with open(json_path, "r") as json_file:
            data = json.load(json_file)

        data.append({"packet_loss": loss_rate, "elapsed_time": elapsed_time})

        with open(json_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
        
        return machine, commands
