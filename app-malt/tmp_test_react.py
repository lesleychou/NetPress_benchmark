from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent, FewShot_Semantic_PromptAgent
import os
from dotenv import load_dotenv

print("Starting script execution...")

# Load environ variables from .env, will not override existing environ variables
load_dotenv()
print("Environment variables loaded")

# For Azure OpenAI GPT4
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.chat_models import AzureChatOpenAI
print("Initializing Azure credentials...")
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
print("Azure credentials initialized")

#Set the API type to `azure_ad`
os.environ["OPENAI_API_TYPE"] = "azure_ad"
# Set the API_KEY to the token from the Azure credential
os.environ["OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
# Set the ENDPOINT
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://ztn-oai-sweden.openai.azure.com/"
print("Azure environment variables set")

llm = AzureChatOpenAI(
            openai_api_version="2024-08-01-preview",
            deployment_name='ztn-sweden-gpt-4o',
            model_name='ztn-sweden-gpt-4o',
            temperature=0.0,
            max_tokens=4000,
        )
print("LLM initialized")

# ReAct agent
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool

# Content of the prompt template
template = '''
Answer the following question as best you can. 
Do not use a tool if not required. 
Question: {question}
'''

# Create the prompt template
prompt_template = PromptTemplate.from_template(template)
prompt = hub.pull('hwchase17/react')
print("Prompt template created and hub prompt pulled")

# Set up the Python REPL tool
print("Setting up Python REPL tool...")
python_repl = PythonAstREPLTool()
python_repl_tool = Tool(
    name = 'Python REPL',
    func = python_repl.run,
    description = '''
    A Python shell. Use this to execute python commands. 
    Input should be a valid python command. 
    When using this tool, sometimes output is abbreviated - make sure 
    it does not look abbreviated before using it in your answer.
    '''
)
print("Python REPL tool set up")

# Set up the DuckDuckGo Search tool
print("Setting up DuckDuckGo Search tool...")
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name = 'DuckDuckGo Search',
    func = search.run,
    description = '''
    A wrapper around DuckDuckGo Search. 
    Useful for when you need to answer questions about current events. 
    Input should be a search query.
    '''
)
print("DuckDuckGo Search tool set up")

# Create an array that contains all the tools used by the agent
tools = [python_repl_tool, duckduckgo_tool]
print("Tools array created")

# Create a ReAct agent
print("Creating ReAct agent...")
agent = create_react_agent(llm, tools, prompt)
print("ReAct agent created")

print("Setting up agent executor...")
agent_executor = AgentExecutor(
    agent=agent, 
    tools = tools,
    verbose = True, # explain all reasoning steps
    handle_parsing_errors=True, # continue on error 
    max_iterations = 3 # try up to 10 times to find the best answer
)
print("Agent executor set up")

# Ask your question (replace this with your question)
# question = "What is '(4876 * 1032 / 85) ^ 3'?"

question = "What is the Microsoft (MSFT) share price?"
print(f"Asking question: {question}")
output = agent_executor.invoke({'input': prompt_template.format(question=question)})
print("Agent execution completed")

# Print the output from the agent
print("\nAgent Response:")
print(output['output'])