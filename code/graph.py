from typing import List, Tuple
import numpy as np
import math


class Graph:
    """
    The Graph class representing the graph in which the ants will search for a path.
    """
    def __init__(self,
                 nr_of_nodes: int,
                 edges: List[Tuple[int, int, int]],
                 start_node: int,
                 goal_node: int):
        """
        :param nr_of_nodes: The number of nodes in the graph.
        :param edges: The edges in the graph, with an edge being represented as a Tuple (from, to, weight).
        :param start_node: The node that represents the start of this graph, i.e. the node that all ants should start from.
        :param goal_node: The node that represent the goal of this graph, i.e. the node that all ants are trying to reach.
        """

        # initialize fields with the constructor arguments
        self.nr_of_nodes = nr_of_nodes
        self.start_node = start_node
        self.goal_node = goal_node

        # the graph structure is stored as an adjacency matrix showing which node is connected to which node
        # Remarks: - this is not a good representation for sparse matrices,
        #            but optimal memory storage for graphs is not the main purpose of this project
        #          - unconnected nodes have an edge between them of infinite weight (math.inf)
        #          - we currently assume undirected graphs, so the matrix is symmetrical and therefore only filled
        #            halfway (the bottom half)
        self.adjacency_matrix = np.full((nr_of_nodes, nr_of_nodes), math.inf)
        for from_node, to_node, weight in edges:
            if from_node > to_node:
                self.adjacency_matrix[from_node, to_node] = weight
            else:
                self.adjacency_matrix[to_node, from_node] = weight

    def has_edge(self, from_node: int, to_node: int):
        """
        Convenience method for checking whether there's an edge between two given nodes.

        :param from_node: The node from which the edge starts (if an edge exists).
        :param to_node: The node towards which the edge goes (if an edge exists).
        :return: True if an edge exists between the given nodes.
        """
        if from_node > to_node:
            return self.get_edge_weight(from_node, to_node) != math.inf
        else:
            return self.get_edge_weight(to_node, from_node) != math.inf

    def get_edge_weight(self, from_node: int, to_node: int):
        """
        Convenience method for getting the weight of the edge between two nodes.

        :param from_node: The node from which the edge starts.
        :param to_node: The node towards which the edge goes.
        :return: The weight of the edge between the two given nodes.
        """
        if from_node > to_node:
            return self.adjacency_matrix[from_node, to_node]
        else:
            return self.adjacency_matrix[to_node, from_node]

    def get_neighbors(self, node: int):
        """
        Method for getting the neighbors of the given node, i.e. all nodes that are reachable from the given node.

        :param node: The node for which neighbors are returned.
        :return: A list of nodes that are neighbors of the given node.
        """

        return [to_node for to_node in range(self.nr_of_nodes) if self.has_edge(node, to_node)]

    def get_path_weight(self, path: List[int]):
        """
        Method for computing the weight of the given path in this graph, with the weight of a path being the
        sum of all edge weights on this path.

        :param path: The list of nodes representing the path to be followed,
                     i.e. the nodes that are sequentially reached on this path.
        :return: The full weight of the path.
        """

        # sum the weights of all edges on the path
        path_weight = 0
        for i in range(len(path)-1):
            path_weight += self.get_edge_weight(path[i], path[i + 1])

        return path_weight

    def is_path_complete(self, path: List[int]):
        """
        Convenience method for checking whether a path is complete,
        i.e. whether a path goes from the start node to the goal node.

        :param path: The list of nodes representing the path to be followed,
                     i.e. the nodes that are sequentially reached on this path.
        :return: True if the given path is complete, otherwise False.
        """
        return (path[0] == self.start_node) and (path[-1] == self.goal_node)

    def edge_in_path(self, from_node: int, to_node: int, path: List[int]):
        """
        Convenience method for checking whether an edge is in a specified path.

        :param from_node: The node from which the edge starts.
        :param to_node: The node towards which the edge goes.
        :param path: The list of nodes representing a path.
        :return: True if the given path is complete, otherwise False;
        """

        for i in range(len(path)-1):
            if (from_node == path[i] and to_node == path[i+1]) or (to_node == path[i] and from_node == path[i+1]):
                return True

        return False
