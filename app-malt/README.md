# README

## Install env
Initialize a new virtual env (conda or pip) by your self. I test it on Python 3.10

```
cd app-malt
pip install -r requirements.txt
```

## Run simulator
```
cd app-malt
python run_simulator.py
```
The prompt you can use for using other LLMs is below. You may adjust the format of prompt based on exact LLM usage (like how promptbench use prompts).

```text
Generate the Python code needed to process the network graph to answer the user query. 
The graph consists: 
Each node has a ‘type’ attribute and other attributes depending on its type. The ‘type’ attribute is a list, and each element is in the format of ‘EK_{{TYPE}}’. For example, EK_PACKET_SWITCH indicates this node is a packet switch node. Because it is a list, each node can have multiple types include EK_SUPERBLOCK, EK_CHASSIS, EK_RACK, EK_AGG_BLOCK, EK_JUPITER, EK_PORT, EK_SPINEBLOCK, EK_PACKET_SWITCH, EK_CONTROL_POINT, EK_CONTROL_DOMAIN.
Each directed edge also has a ‘type’ attribute, where the value RK_CONTAINS indicates the source node contains the destination node, and the value RK_CONTROLS indicates the source node controls the destination node. 

The Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data (networkx graph) and returns a single object return_object. 
The return_object will be a JSON object with two keys, 'type' and 'data'. The 'type' key should indicate the output format depending on the user query. 
If the output type is 'text' then the 'data' key should be convert to a string. 
If the output type is 'list' then the 'data' key should contain a list of items.
If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table. 
If the output type is 'graph' then the 'data' key should be a networkx graph.

All of your output should only contain the defined function, and display in a Python code block.

"""Begin! Strictly generate Python code with the following format:

Answer:
```python
${{Code that will answer the user question or request}}

Question: {input}
"""

```
