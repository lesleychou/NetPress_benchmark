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