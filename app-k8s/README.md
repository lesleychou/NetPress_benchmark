# Setting up KIND Simulator and Deploying Google's Microservices Locally
This guide will walk you through setting up KIND (Kubernetes in Docker) on your local machine(we are using ubuntu 22.04) and deploying Google's microservices using the [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) demo app.

## Prerequisites
Before you begin you should check if you have installed Docker and Go, and whether they are from the latest version.

### Install Docker
Follow the steps below to install Docker on your Ubuntu machine.

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt-get update
sudo docker run hello-world
```
Once Docker is installed, you can verify the installation by running the following command, which will download and run a test Docker image called *hello-world*. If you see it, it means that you installed it successfully.

### Installing Go and KIND

Follow these steps to install Go and KIND on your machine.
```bash
sudo apt install golang-go
```
Check the Go version to ensure it's installed correctly
```bash
go version
```
Now, install KIND using the following command:
```bash
go install sigs.k8s.io/kind@v0.26.0
```
By default, the $GOPATH environment variable might not be set. You can set it manually by running the following commands:
Then, source your .bashrc file to apply the changes:
```bash
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
```
```bash
source ~/.bashrc
```
## Setting Up a KIND Cluster

After installing KIND and kubectl, follow these steps to create a KIND cluster and install kubectl if it's not already installed.

### Step 1: Create a KIND Cluster

Create a new KIND cluster using the following command:

```bash
kind create cluster
```
If kubectl is not already installed, you can install it using Snap:
```bash
sudo snap install kubectl --classic
```
You can use the command below to verify your installation of Kind cluster.
```bash
kubectl get nodes
```
## Setting Up the Application with Skaffold

Follow these steps to set up and deploy the microservices demo application using Skaffold.

### Step 1: Clone the Microservices Demo Repository

First, clone the official microservices demo repository from GitHub:

```bash
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git
```
### Step 2: Change to the microservices-demo directory:
Change to the microservices-demo directory:
```bash
cd microservices-demo
```
### Step 3: Install Skaffold
If you haven't installed Skaffold yet, follow these instructions to install the latest version.
For Linux (Ubuntu):
```bash
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/v2.15.0/skaffold-linux-amd64
chmod +x skaffold
sudo mv skaffold /usr/local/bin
```
### Step 4: Modify the Dockerfile
In the `Dockerfile` of the `loadgenerator` service, the `FROM` statement specifies the platform for the base image. By default, Skaffold and Docker attempt to build images for the correct platform and architecture based on your system. However, in some cases, like using the `BUILDPLATFORM` variable, it can cause issues due to improper platform resolution.
So you should go to the `loadgenerator` directory within the `microservices-demo` project:

```bash
cd microservices-demo/src/loadgenerator
```
In the Dockerfile, locate the line that starts with:
```
FROM --platform=$BUILDPLATFORM python:3.12.8-alpine@sha256:54bec49592c8455de8d5983d984efff76b6417a6af9b5dcc8d0237bf6ad3bd20 AS base
```
Replace $BUILDPLATFORM with linux/amd64. The updated line should look like this:
```
FROM --platform=linux/amd64 python:3.12.8-alpine@sha256:54bec49592c8455de8d5983d984efff76b6417a6af9b5dcc8d0237bf6ad3bd20 AS base
```
### Step 4: Build microservice locally
Now that you’ve set up everything, you can build and deploy the microservices locally using Skaffold. This step will trigger the build process, where Skaffold will compile the Docker images for each microservice defined in the project and deploy them to the Kubernetes cluster. This step may take around 20 minutes.
```bash
cd /path/to/microservices-demo
skaffold run
```

# App-K8s Benchmarking with LLM Agent

This project provides a benchmarking framework that allows you to test a specific LLM agent within a microservices environment using Kubernetes. After setup, this framework runs Google's Microservices Demo in a Kind simulator within a single VM.

## Python Prerequisites

To set up the Python environment, we use `conda` to create a virtual environment. You can install the required dependencies by running the following commands:

```bash
conda env create -f environment_mininet.yml
conda activate mininet
```
## Set the environment variable
You need to set environment variable so that you can access open source LLM. You can do it by using the following command:
```
export HUGGINGFACE_TOKEN="your_huggingface_token"
```
## Start benchmarking 

The easiest way to run the benchmark is by using the provided shell script `run_app_k8s.sh`. Below are the steps to execute the benchmarking process.
### 1.Navigate to experiments Directory
Navigate to the `experiments` directory where the script is located:
```
cd experiments
```
### 2.Modify Parameters in `run_app_k8s.sh`
Before running the benchmarking script, you need to modify the parameters in run_app_k8s.sh to suit your setup. Specifically, update the paths for the microservices demo and app-k8s root directory. Below is a list of configurable parameters and their explanations.

### `LLM_AGENT_TYPE`:
- **Description**: Specifies the type of LLM agent to be used in the benchmark. The format typically includes the name and version of the agent, such as `Qwen/Qwen2.5-72B-Instruct`. This determines which LLM model will be evaluated during the benchmarking process.
- **Example**: `Qwen/Qwen2.5-72B-Instruct`

### `NUM_QUERIES`:
- **Description**: Defines the number of queries to generate during the benchmarking process. This determines how many individual queries for each error type will be tested.
- **Example**: `10` (Test with one query)

### `ROOT_DIR`:
- **Description**: The root directory where output logs and results will be stored. This path should point to the location on your machine where the benchmark results will be saved. Ensure that the specified directory exists and is accessible.
- **Example**: `/home/ubuntu/NetPress_benchmark/app-k8s`

### `MICROSERVICE_DIR`:
- **Description**: The path to the microservices demo directory. This directory contains the demo application that will be used in conjunction with the benchmarking framework. Ensure you specify the correct path to the `microservices-demo` on your system.
- **Example**: `/home/ubuntu/microservices-demo` (Replace this with your own path)

### `MAX_ITERATION`:
- **Description**: The maximum number of iterations to run for each query. This helps control the number of times the agent will execute the query in each benchmark run. 
- **Example**: `10` (Run each query up to 10 iterations)

### `CONFIG_GEN`:
- **Description**: This parameter controls whether a new configuration should be generated for each benchmark. Set it to `1` to generate a new configuration, or `0` to skip this step and use the existing configuration.
- **Example**: `1` (Generate new configuration)

### `PROMPT_TYPE`:
- **Description**: Specifies the type of prompt to use when interacting with the LLM. The prompt type affects the nature of the queries sent to the LLM. You can choose between basic and more advanced prompts, depending on your test requirements.
- **Example**: `base` (Use the basic prompt type)

### `AGENT_TEST`:
- **Description**: Determines whether to test multiple LLM agents and prompt types. Set this to `1` if you want to test multiple agents and prompt variations. If you only wish to test a single LLM with one prompt type, set this to `0`.
- **Example**: `0` (Test a single LLM agent with one prompt type)

### 3. Run the Benchmark
After modifying the parameters, you can execute the benchmarking process by running the script with the following command:
```bash
bash run_app_k8s.sh
```

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