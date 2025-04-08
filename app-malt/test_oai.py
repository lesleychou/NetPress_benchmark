import os
import warnings
from langchain._api import LangChainDeprecationWarning
from azure.identity import DefaultAzureCredential, AzureCliCredential

# Get the Azure Credential
credential = DefaultAzureCredential()


# Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ztn-oai-fc.openai.azure.com/"

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
# Load environ variables from .env, will not override existing environ variables
#load_dotenv()

OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# For GPT-4 in Azure
llm = AzureChatOpenAI(
    openai_api_type=OPENAI_API_TYPE,
    openai_api_base=OPENAI_API_BASE,
    openai_api_version="2023-12-01-preview",
    deployment_name='gpt-4-32k',
    model_name='gpt-4-32k',
    # openai_api_key=OPENAI_API_KEY,
    temperature=0.0,
    max_tokens=4000,
)

# # For GPT-4o in Azure
# llm = AzureChatOpenAI(
#     openai_api_type=OPENAI_API_TYPE,
#     openai_api_base=OPENAI_API_BASE,
#     openai_api_version="2023-05-15",
#     deployment_name='gpt-4o',
#     model_name='gpt-4o',
#     openai_api_key=OPENAI_API_KEY,
#     temperature=0.0,
#     max_tokens=4000,
# )

template = """Tell me a joke related to the given topic. The topic is {topic}
"""

topic = """Soccer"""

prompt = PromptTemplate(template=template, input_variables=["topic"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
# print("llm_chain: ", llm_chain)

response_1 = llm_chain.run(topic)

print(response_1)
