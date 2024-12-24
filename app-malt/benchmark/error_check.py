import sys
import json
import traceback
import re
import numpy as np
import pandas as pd
import networkx as nx


class SafetyChecker():
    def __init__(self, ret_graph=None, ret_list=None):
        if ret_graph:
            self.graph = ret_graph
        else:
            self.graph = None
        if ret_list:
            self.output_list = ret_list
        else:
            self.output_list = None

    def evaluate_all(self):
        print("Evaluating all checks")
        # TODO: need to check if it atually runs all the checks; verify_port_exist is not returning correct results
        if self.graph:
            graph_checks = [self.verify_node_format_and_type,
                            self.verify_edge_format_and_type,
                            self.verify_node_hierarchy,
                            self.verify_port_exist,
                            self.verify_no_isolated_nodes,
                            self.verify_bandwidth, 
                            self.verify_port_exist]
            for check in graph_checks:
                try:
                    check()
                except Exception as e:
                    print("Check failed:", e)
                    print(traceback.format_exc())
                    return False, e
            return True, ""

    def verify_node_format_and_type(self):
        """
        Graph check: verify node type and format
        """
        valid_types = ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN']

        for node in self.graph.nodes():
            # Check if the node has a 'type' attribute
            if self.graph.nodes[node].get('type'):
                node_types = self.graph.nodes[node]['type']
                for node_type in node_types:
                    if node_type not in valid_types:
                        return False, "verify_node_types failed"
            else:
                return False, "verify_node_types failed"

        return True, ""

    def verify_edge_format_and_type(self):
        """
        Graph check: verify_edge_format_and_type
        """
        valid_edge_types = ["RK_CONTAINS", "RK_CONTROLS"]

        for edge in self.graph.edges(data=True):
            # Check if the edge has a 'type' attribute
            if 'type' not in edge[2]:
                return False, "verify_edge_format_and_type failed"  # Edge does not have a 'type' attribute
            # Check if the edge's type is in the valid_edge_types list
            if not any(edge_type in edge[2]['type'] for edge_type in valid_edge_types):
                return False, "verify_edge_format_and_type failed"
            else:
                return True, ""

    def verify_node_hierarchy(self):
        """
        Graph check: verify_node_hierarchy
        TODO: Add more hierarchy checks, if adding a PORT, then it has be connected to a PACKET_SWITCH
        """
        hierarchy = {
            "EK_JUPITER": ["EK_SPINEBLOCK", "EK_SUPERBLOCK"],
            "EK_SPINEBLOCK": ["EK_PACKET_SWITCH"],
            "EK_SUPERBLOCK": ["EK_AGG_BLOCK"],
            "EK_AGG_BLOCK": ["EK_PACKET_SWITCH"],
            "EK_CHASSIS": ["EK_CONTROL_POINT", "EK_PACKET_SWITCH"],
            "EK_CONTROL_POINT": ["EK_PACKET_SWITCH"],
            "EK_RACK": ["EK_CHASSIS"],
            "EK_PACKET_SWITCH": ["EK_PORT"],
            "EK_CONTROL_DOMAIN": ["EK_CONTROL_POINT"]
        }

        for edge in self.graph.edges(data=True):
            if 'RK_CONTAINS' in edge[2]['type']:
                source_node_types = self.graph.nodes[edge[0]]['type']
                target_node_types = self.graph.nodes[edge[1]]['type']
                for source_type in source_node_types:
                    if source_type in hierarchy and any(target_type in hierarchy[source_type] for target_type in target_node_types):
                        return True, ""
        
        return False, "verify_node_hierarchy failed"
    
    def verify_port_exist(self):
        import pdb; pdb.set_trace()
        # if there is a packet switch, there should be at least one EK_PORT connected to it
        for node in self.graph.nodes():
            if 'EK_PACKET_SWITCH' in self.graph.nodes[node]['type']:
                has_port = False
            for neighbor in self.graph.neighbors(node):
                if 'EK_PORT' in self.graph.nodes[neighbor]['type']:
                    has_port = True
                break
            if not has_port:
                return False, "verify_port_exist failed: Packet switch should have at least one port."
        return True, ""
    
    def verify_no_isolated_nodes(self):
        """
        Graph check: verify_no_isolated_nodes
        """
        # An isolated node is a node with degree 0, i.e., no edges.
        isolated_nodes = list(nx.isolates(self.graph))

        if len(isolated_nodes) == 0:
            return True, ""  # There are no isolated nodes in the graph.
        else:
            return False, "verify_no_isolated_nodes failed"


    def verify_bandwidth(self):
        """
        Verify if the "Bandwidth" column in a given table is never 0.

        Args:
            data (list): A list of lists representing a table.

        Returns:
            bool: True if the "Bandwidth" column is never 0 or doesn't exist, False otherwise.
        """
        # with the given graph, for all nodes with type EK_PORT, check if it has a 'physical_capacity_bps' attribute
        for node in self.graph.nodes():
            if 'EK_PORT' in self.graph.nodes[node]['type']:
                if 'physical_capacity_bps' not in self.graph.nodes[node]:
                    return False, "verify_bandwidth failed: Port node should have a 'physical_capacity_bps' attribute."
                if self.graph.nodes[node]['physical_capacity_bps'] == 0:
                    return False, "verify_bandwidth failed: Bandwidth should be more than 0."
                else: 
                    return True, ""
    

    def verify_port_exist(self):
        # TODO: find a case that this would fail
        """
        Verify with the given graph, for all nodes with type EK_PACKET_SWITCH, check if it has at least one port with type EK_PORT.
        """
        # with the given graph, for all nodes with type EK_PACKET_SWITCH, check if it has at least one port with type EK_PORT
        for node in self.graph.nodes():
            node_types = self.graph.nodes[node]['type']
            if 'EK_PACKET_SWITCH' in node_types:
                has_port = False
                for neighbor in self.graph.successors(node):
                    if 'EK_PORT' in self.graph.nodes[neighbor]['type']:
                        has_port = True
                if not has_port:
                    return False, "verify_port_exist failed: Packet switch should have at least one port."
                # print(node, has_port)
        return has_port, ""

    #TODO: add each port can only be connected with one packet switch
