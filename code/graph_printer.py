# Imports for "plotting" a graph
import matplotlib.pyplot as plt
import networkx as nx
import math


def graph_printer(graph):
    """
    Convenience function for printing a graph.
    This is mostly intended for quick visualizations in order to verify a graph has been correctly constructed.

    :param graph: The graph that should be printed
    """

    # create nx version of graph
    printed_graph = nx.Graph()
    
    # more or less square picture => square root
    nr_of_nodes = graph.nr_of_nodes
    sqrt_nr_of_edges = int(math.sqrt(nr_of_nodes))

    # fill with edges
    for i in range(nr_of_nodes):
        for j in range(i, nr_of_nodes):
            if graph.has_edge(i, j):
                printed_graph.add_edge(i, j, weight=graph.get_edge_weight(i, j))

    edges = [(u, v) for (u, v, d) in printed_graph.edges(data=True)]
    
    # create positions dictionary
    pos = {}
    for i in range(nr_of_nodes):
        # x-s move to right, y-s loop
        pos[i] = [i/sqrt_nr_of_edges, -i%sqrt_nr_of_edges]

    # draw vertices
    nx.draw_networkx_nodes(printed_graph, pos, node_size=700)
    
    # draw weights
    labels = nx.get_edge_attributes(printed_graph, 'weight')
    nx.draw_networkx_edge_labels(printed_graph,pos, edge_labels=labels)

    # draw edges
    nx.draw_networkx_edges(printed_graph, pos, edgelist=edges, width=1, edge_color="b")

    # draw labels
    nx.draw_networkx_labels(printed_graph, pos, font_size=20, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.show()
