import time
import json
import os
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from dotenv import load_dotenv
# Load environ variables from .env, will not override existing environ variables
load_dotenv()
# For Google Gemini
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_GEMINI_API_KEY")

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
        
# For Azure OpenAI GPT4
from langchain.chat_models import AzureChatOpenAI
from azure.identity import AzureCliCredential

credential = AzureCliCredential()
#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ztn-oai-fc.openai.azure.com/"

from huggingface_hub import login

# Login huggingface
login(token="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")

from vllm import LLM, SamplingParams

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

    def __init__(self, model: str, max_new_tokens: int = 256, temperature: float = 0.1, device: str = "cuda", api_key: str = None):
        self.model_name = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.api_key = api_key
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

    @staticmethod
    def _generate_prompt(file_content, log_content):
        """
        Generates a JSON format prompt for fixing a Mininet network issue.

        Args:
            file_content (str): The content of the file containing previous actions and feedback.
            log_content (str): The latest feedback from the Mininet.

        Returns:
            str: A JSON-formatted string with 'machine' and 'command' keys.
        """
        prompt = (
            """There is a mininet network, but there are some kinds of problems in the router r0, 
            so it cannot function well and PingAll() fails at some nodes. You need to fix it.
            I highly recommend you to use some commands to know the information of the router and 
            the network to know the cause of the problem. But if you think the information is enough 
            and you know the reason causing the problem, you have to give commands to fix it.
            You need to give the output in JSON format, which contains the machine and its command.
            Then I will give you the latest PingAll() feedback from the network, and also your 
            previous actions to the network and the actions' feedback to let you know more information.
            """
            + "Here are the previous actions and their feedbacks:\n"
            + file_content
            + "This is the latest feedback from the mininet:\n"
            + log_content
            + "Please only give me the JSON format output, with key 'machine' and 'command' and their value. You can only give one command at a time and don't include 'sudo', and you are not allowed to use vtysh command."
        )
        return prompt
    
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

    def _initialize_qwen(self):
        """Initialize the Qwen model."""
        return QwenModel(
            model_name="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            device=self.device
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
        return GPTAgentModel()

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


class LlamaModel:
    """
    A specialized class for handling Meta-Llama models.

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
        """Load the model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        # Create BitsAndBytesConfig for 4-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            device_map=self.device
        )

        # Load the model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            quantization_config=quantization_config  # Use the quantization config
        )

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""

        with open(file_path, 'r') as f:
            file_content = f.read()

        prompt = LLMModel._generate_prompt(file_content, log_content)

        start_time = time.time()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        prompt_input_ids = model_inputs["input_ids"]
        start_index = prompt_input_ids.shape[-1]
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            **kwargs
        )
        generation_output = generated_ids[0][start_index:]
        content = self.tokenizer.decode(generation_output, skip_special_tokens=True)
        print("LLM output:", content)
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


class QwenModel:
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

    def __init__(self, model_name, max_new_tokens, temperature, device="cuda"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._load_model()

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
        """Generate a response based on the log content and file content."""
        with open(file_path, 'r') as f:
            file_content = f.read()

        # Generate prompt
        prompt = LLMModel._generate_prompt(file_content, log_content)

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

class Phi4Model:
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

    def __init__(self, model_name, max_new_tokens, temperature, device):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load the Phi-4 model and tokenizer."""
        # from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        # import torch

        # model_path = "/home/ubuntu/Phi4"

        # # Create BitsAndBytesConfig for 4-bit quantization
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = self.model.to(self.device)
        import transformers
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="microsoft/phi-4",
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )

    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""

        with open(file_path, 'r') as f:
            file_content = f.read()

        # Generate prompt
        # prompt = LLMModel._generate_prompt(file_content, log_content)
        prompt = "hello"
        start_time = time.time()

        # model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        # generated_ids = self.model.generate(
        #     **model_inputs,
        #     max_new_tokens=512,
        #     do_sample=True,
        #     temperature=self.temperature,
        #     **kwargs
        # )
        content = str(self.pipeline(prompt, max_new_tokens=512)[0]["generated_text"])
        print(content)
        # content = str(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
        print(content)
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

class GemmaModel:
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
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)


    def predict(self, log_content, file_path, json_path, **kwargs):
        """Generate a response based on the log content and file content."""

        with open(file_path, 'r') as f:
            file_content = f.read()

        prompt = LLMModel._generate_prompt(file_content, log_content)

        start_time = time.time()

        """Use your model to generate your reposnse here, results should be a str."""
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            **kwargs
        )
        end_time = time.time()
        content = self.tokenizer.decode(outputs[0])
        print(content)

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

class QwQModel:
    """
    A specialized class for handling Qwen/QwQ-32B-Preview  models.

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
        """Load the QwQ model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        # Create BitsAndBytesConfig for 4-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device
        )

        # Load the QwQ model with 4-bit quantization
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

        # Generate prompt
        prompt = LLMModel._generate_prompt(file_content, log_content)

        start_time = time.time()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        prompt_input_ids = model_inputs["input_ids"]
        start_index = prompt_input_ids.shape[-1]
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            **kwargs
        )
        generation_output = generated_ids[0][start_index:]
        content = self.tokenizer.decode(generation_output, skip_special_tokens=True)
        print(content)
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

    def __init__(self):
        self._load_model()

    def _load_model(self):
        """Initialize the GPT Agent client."""

        self.client = AzureChatOpenAI(
            openai_api_type="azure_ad",
            openai_api_version="2024-08-01-preview",
            deployment_name='gpt-4o',
            model_name='gpt-4o',
            temperature=0.0,
            max_tokens=4000,
            )
        print("======GPT-4o successfully loaded=======")

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
