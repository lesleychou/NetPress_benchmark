import json
import networkx as nx
from networkx.readwrite import json_graph
from prototxt_parser.prototxt import parse


def getGraphData():
    input_string = open("../data/malt-example-final.textproto.txt").read()
    parsed_dict = parse(input_string)

    # Load MALT data
    G = nx.DiGraph()

    # Insert all the entities as nodes
    for entity in parsed_dict['entity']:
        # Check if the node exists
        if entity['id']['name'] not in G.nodes:
            G.add_node(entity['id']['name'], type=[entity['id']['kind']], name=entity['id']['name'])
        else:
            G.nodes[entity['id']['name']]['type'].append(entity['id']['kind'])
        # Add all the attributes
        for key, value in entity.items():
            if key == 'id':
                continue
            for k, v in value.items():
                G.nodes[entity['id']['name']][k] = v

    # Insert all the relations as edges
    for relation in parsed_dict['relationship']:
        G.add_edge(relation['a']['name'], relation['z']['name'], type=relation['kind'])

    rawData = json_graph.node_link_data(G)

    return rawData, G


def get_node_value_ranges(graph, saved_path):
    """For malt_real_graph, save node value range of each node type"""
    node_value_ranges = {}
    for node in graph.nodes:
        node_type = graph.nodes[node]['type'][0]
        node_value = graph.nodes[node]['name']
        if node_type not in node_value_ranges:
            node_value_ranges[node_type] = []
        node_value_ranges[node_type].append(node_value)
    # save the node_value_ranges to a json file
    with open(saved_path, 'w') as f:
        json.dump(node_value_ranges, f)

    return node_value_ranges

def solid_step_add_node_to_graph(graph_data, new_node, parent_node_name=None):
    """
    Adds a new node to the graph. Optionally adds an edge to a parent node with a specified relationship type.

    :param graph_data: The existing graph (a NetworkX graph or similar).
    :param new_node: A dictionary containing the new node's attributes (e.g., name, type).
    :param parent_node_name: Name of the parent node (optional). If provided, a relationship edge will be added.
    :return: updated graph data.
    """
    # Create a new unique node ID
    new_node_id = len(graph_data.nodes) + 1

    # Add the new node to the graph
    graph_data.add_node(new_node_id, name=new_node['name'], type=new_node['type'])

    # If a parent node is specified, add an edge between parent and the new node
    if parent_node_name:
        parent_node_id = None
        for node in graph_data.nodes:
            if graph_data.nodes[node].get('name') == parent_node_name:
                parent_node_id = node
                break
    graph_data.add_edge(parent_node_id, new_node_id, type='RK_CONTAINS')

    # For testing
    # parent_node_name = 'ju1.a1.m1'
    # new_node = {'name': 'new_port', 'type': 'EK_PORT'}
    # malt_graph = solid_step_add_node_to_graph(malt_real_graph, new_node, parent_node_name)
    return graph_data

def solid_step_remove_node_from_graph(graph_data, node_name):
    """
    Removes a node from the graph. Also removes any edges connected to the node.

    :param graph_data: The existing graph (a NetworkX graph or similar).
    :param node_name: The name of the node to be removed.
    :return: updated graph data.
    """
    # Find the node ID by name
    node_id = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == node_name:
            node_id = node
            break

    if node_id is None:
        print(f"Node with name '{node_name}' not found.")
        return graph_data

    # Remove the node and its edges from the graph
    graph_data.remove_node(node_id)

    return graph_data



# create a function for calculating the counting queries
def solid_step_counting_query(graph_data, node1, node2):
    """
    Count the number of node2 contained within node1 in the graph.
    """
    # Find the target node1
    target_node1 = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == node1['name']:
            target_node1 = node
            break

    if target_node1 is None:
        print(f"Node1 {target_node1} not found", )
        return {node1, 'not found'}

    # Find all node2 directly contained within node1
    node2_count = 0
    for edge in graph_data.out_edges(target_node1, data=True):
        if edge[2]['type'] == 'RK_CONTAINS':
            destination_node = edge[1]
            if node2['type'] in graph_data.nodes[destination_node]['type']:
                node2_count += 1
    
    # Find node2 contained within node1 recursively, there is heirarchy in the graph
    for edge in graph_data.out_edges(target_node1, data=True):
        if edge[2]['type'] == 'RK_CONTAINS':
            destination_node = edge[1]
            node2_count += solid_step_counting_query(graph_data, graph_data.nodes[destination_node], node2)

    # For testing
    # node1 = {'type': 'EK_AGG_BLOCK', 'name': 'ju1.a1.m1'}
    # node2 = {'type': 'EK_PORT', 'name': None}
    # count = solid_step_counting_query(malt_real_graph, node1, node2)
    # print(count)

    return node2_count

def solid_step_list_child_nodes(graph_data, parent_node):
    """
    list all nodes that are directly contained within the parent node
    """
    child_nodes = []
    parent_node_id = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == parent_node['name']:
            parent_node_id = node
            break

    if parent_node_id is None:
        print(f"Parent node with name '{parent_node['name']}' not found.")
        return child_nodes

    for edge in graph_data.out_edges(parent_node_id, data=True):
        if edge[2]['type'] == 'RK_CONTAINS':
            child_nodes.append(graph_data.nodes[edge[1]])

    return child_nodes
    
def solid_step_update_node_value(graph_data, child_node_name, new_value):
    """
    Update the value of a child node in the graph.
    """
    # Find the node ID by name
    child_node_id = None
    for node in graph_data.nodes:
        if graph_data.nodes[node].get('name') == child_node_name:
            child_node_id = node
            break

    if child_node_id is None:
        print(f"Node with name '{child_node_name}' not found.")
        return graph_data, child_node_name, new_value

    # Update the node's value
    graph_data.nodes[child_node_id]['name'] = new_value

    return graph_data