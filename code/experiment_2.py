# Imports for printing tables
from tabulate import tabulate

# General imports
from multitype_aco import *
from graph import *
from general_helpers import *


def create_test_graph_2(weight_v3_v5 = 1):
    """
    Helper function that generates and returns test graph 2 (from experiment 2), which does not have any fully disjoint
    paths. There are 2 bridges: v3-v5 and v4-v6.

    :param weight_v3_v5: The weight of bridge 1, which is the edge between vertices v3 and v5.
                         This weight is 1 in the first experiment on this graph and 3 in the second experiment.
    """

    return Graph(nr_of_nodes=10,  # v0 to v9
                 edges=[(0, 1, 1),  # v0 - v1 with weight 1
                        (0, 2, 1),  # v0 - v2 with weight 1
                        (1, 2, 1),  # v1 - v2 with weight 1
                        (1, 3, 1),  # v1 - v3 with weight 1
                        (1, 4, 1),  # v1 - v4 with weight 1
                        (2, 3, 1),  # v2 - v4 with weight 1
                        (2, 4, 1),  # v2 - v3 with weight 1
                        (3, 4, 1),  # v3 - v4 with weight 1
                        (3, 5, weight_v3_v5),  # v3 - v5 with weight 1 or 3 (bridge 1)
                        (4, 6, 1),  # v4 - v6 with weight 1 (bridge 2)
                        (5, 6, 1),  # v5 - v6 with weight 1
                        (5, 7, 1),  # v5 - v7 with weight 1
                        (5, 8, 1),  # v5 - v8 with weight 1
                        (6, 7, 1),  # v6 - v7 with weight 1
                        (6, 8, 1),  # v6 - v8 with weight 1
                        (7, 8, 1),  # v7 - v8 with weight 1
                        (7, 9, 1),  # v7 - v9 with weight 1
                        (8, 9, 1)],  # v8 - v9 with weight 1
                 start_node=0, goal_node=9)  # a path from v0 to v9 should be found


def experiment_2_helper(rho, beta, cl, tau_0, t_max, gamma, q0, nr_ants_per_type, nr_ant_types_range, num_runs,
                        minimization_objective, equal_weight):
    """
    Helper function to be used inside the main experiment 2 functions.

    :param equal_weight: If True, we run experiment 2 on the graph with equal weights for both bridges.
                         If False, we run experiment 2 on the graph where one bridge has triple the weight of the other
                         one.
    """

    graph = create_test_graph_2(weight_v3_v5=1) if equal_weight else create_test_graph_2(weight_v3_v5=3)

    # there are 2 bridges in the graph of which we should check the usage
    bridge_1 = [3, 5]
    bridge_2 = [4, 6]

    # declare the results array for equal weights
    results = np.zeros((len(nr_ant_types_range), 2))

    for i, nr_ant_types in enumerate(nr_ant_types_range):
        print("ant_types: " + str(nr_ant_types))

        # keep track of how often each bridge is used
        bridge_1_count = 0.0
        bridge_2_count = 0.0
        for run in range(num_runs):
            print("run: " + str(run+1) + " of " + str(num_runs))

            m_aco = MultiTypeACO(graph, nr_ant_types, nr_ants_per_type, rho, beta, gamma, q0, cl, t_max, tau_0)
            best_paths, _, _, _, _ = m_aco.run(minimization_objective=minimization_objective, print_results=False)

            for ant_type in range(nr_ant_types):
                if graph.edge_in_path(bridge_1[0], bridge_1[1], best_paths[ant_type]):
                    bridge_1_count += 1
                elif graph.edge_in_path(bridge_2[0], bridge_2[1], best_paths[ant_type]):
                    bridge_2_count += 1

        # compute the percentage from the count
        bridge_1_percentage = 100*(bridge_1_count/nr_ant_types)/num_runs
        bridge_2_percentage = 100*(bridge_2_count/nr_ant_types)/num_runs

        # store the percentages
        results[i][0] = bridge_1_percentage
        results[i][1] = bridge_2_percentage

    return results


def run_experiment_2(rho=0.1,  # default by Dorigo
                     beta=2,  # default by Dorigo
                     cl=5,  # default by Dorigo
                     tau_0=0.05,  # default by Dorigo
                     t_max=1000,
                     gamma=2,
                     q0=1,
                     nr_ants_per_type=12,
                     nr_ant_types_range=np.arange(2, 7),
                     num_runs=100,
                     minimization_objective=0,
                     print_table=True):

    """
    Helper function that runs experiment 2 from the paper.

    :param minimization_objective: 0 to minimize for the shared edges cost, 1 to minimize for the shared edges average.
    """

    # run the experiment for bridges with equal weight
    results_equal_w = experiment_2_helper(rho=rho,
                                          beta=beta,
                                          cl=cl,
                                          tau_0=tau_0,
                                          t_max=t_max,
                                          gamma=gamma,
                                          q0=q0,
                                          nr_ants_per_type=nr_ants_per_type,
                                          nr_ant_types_range=nr_ant_types_range,
                                          num_runs=num_runs,
                                          minimization_objective=minimization_objective,
                                          equal_weight=True)

    # run the experiment for bridges with unequal weight
    results_unequal_w = experiment_2_helper(rho=rho,
                                            beta=beta,
                                            cl=cl,
                                            tau_0=tau_0,
                                            t_max=t_max,
                                            gamma=gamma,
                                            q0=q0,
                                            nr_ants_per_type=nr_ants_per_type,
                                            nr_ant_types_range=nr_ant_types_range,
                                            num_runs=num_runs,
                                            minimization_objective=minimization_objective,
                                            equal_weight=False)

    # print the table if requested:
    if print_table:
        # generate top headers
        top_headers = ["types", "bridge1", "bridge2"]

        # add left side headers to the data
        equal_w_data = results_equal_w.copy().tolist()
        for i, types in enumerate(nr_ant_types_range):
            equal_w_data[i].insert(0, types)

        # add left side headers to the data
        unequal_w_data = results_unequal_w.copy().tolist()
        for i, types in enumerate(nr_ant_types_range):
            unequal_w_data[i].insert(0, types)

        # print the table with the results
        print("Results Equal Weights: ")
        print(tabulate(equal_w_data, headers=top_headers))
        print("")

        # print the table with the results
        print("Results Unequal Weights: ")
        print(tabulate(unequal_w_data, headers=top_headers))

    return results_equal_w, results_unequal_w


def run_experiment_2_1(print_table=True):
    """
    Helper function that runs the first part of experiment 2 from the paper.
    This approach minimizes the sum from equation 7.
    """

    return run_experiment_2(minimization_objective=0, print_table=print_table)


def run_experiment_2_2(print_table=True):
    """
    Helper function that runs the second part of experiment 2 from the paper.
    This approach minimizes the sum from equation 8.
    """

    return run_experiment_2(minimization_objective=1, print_table=print_table)


if __name__ == '__main__':

    print("APPROACH 1: ")
    results_equal_w_1, results_unequal_w_1 = run_experiment_2_1()
    #store_var("results_experiment_2_1_equal", results_equal_w_1)
    #store_var("results_experiment_2_1_unequal", results_unequal_w_1)
    #results_equal_w_1 = load_var("results_experiment_2_1_equal")
    #results_unequal_w_1 = load_var("results_experiment_2_1_unequal")

    print("-----------------------------------------------------")

    print("APPROACH 2: ")
    results_equal_w_2, results_unequal_w_2 = run_experiment_2_2()
    #store_var("results_experiment_2_2_equal", results_equal_w_2)
    #store_var("results_experiment_2_2_unequal", results_unequal_w_2)
    #results_equal_w_2 = load_var("results_experiment_2_2_equal")
    #results_unequal_w_2 = load_var("results_experiment_2_2_unequal")
