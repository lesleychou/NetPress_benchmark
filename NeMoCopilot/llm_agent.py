import os
import warnings
from langchain._api import LangChainDeprecationWarning
from azure.identity import DefaultAzureCredential

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
import json
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


template = """You are an expert Question Creator. You will receive an instance of network traffic analysis task, including a context, a question and its answer.
You are tasked with creating an alternative question to explore a different aspect of the original problem.
Please do not change the context but just edit the question and the answer.
Please first generate the question. Then think step-by-step in one line to give an brief analysis of the question, Finally, directly present a short answer omitting the intermediate steps, in a single line.

Context: {context}
Original Question: {question}
Original Answer: {answer}
Alternative Question:
"""


prompt = PromptTemplate(template=template, input_variables=["context", "question", "answer"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

context = """Generate the Python code needed to process the network graph to answer the user query. 
The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph) and returns a single object return_object. 
The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query. 
If the output type is 'text' then the 'data' key should be convert to a string. 
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table. 
If the output type is 'graph' then the 'data' key should be a networkx graph.

All of your output should only contain the defined function, and display in a Python code block."""

# Load the first question and answer from data/ta_query_samples.jsonl file
with open('data/ta_query_samples.jsonl', 'r') as file:
    first_line = file.readline()
    data = json.loads(first_line)
    question = data['question']
    answer = data['answer']

print(question)
print(answer)

response_1 = llm_chain.run(context=context, question=question, answer=answer)

print(response_1)
