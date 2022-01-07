# Imports for printing tables
from tabulate import tabulate

# General imports
from general_helpers import *
from multitype_aco import *
from graph import *


def create_test_graph_3():
    """
    Helper function that generates and returns test graph 3 (from experiment 3), which allows for 4 completely disjoint
    paths.
    """

    return Graph(nr_of_nodes=16,  # v0 to v15
                 edges=[(0, 1, 1),  # v0 - v1 with weight 1
                        (0, 5, 5),  # v0 - v5 with weight 5
                        (0, 8, 5),  # v0 - v8 with weight 5
                        (0, 11, 1),  # v0 - v11 with weight 1
                        (1, 2, 1),  # v1 - v2 with weight 1
                        (2, 3, 1),  # v2 - v3 with weight 1
                        (2, 6, 1),  # v2 - v6 with weight 1
                        (3, 4, 2),  # v3 - v4 with weight 2
                        (3, 5, 3),  # v3 - v5 with weight 3
                        (4, 15, 2),  # v4 - v15 with weight 2
                        (5, 6, 1),  # v5 - v6 with weight 1
                        (5, 9, 6),  # v5 - v9 with weight 6
                        (6, 7, 1),  # v6 - v7 with weight 1
                        (6, 8, 6),  # v6 - v8 with weight 6
                        (7, 15, 1),  # v7 - v15 with weight 1
                        (8, 9, 1),  # v8 - v9 with weight 1
                        (8, 13, 3),  # v8 - v13 with weight 3
                        (9, 10, 1),  # v9 - v10 with weight 1
                        (9, 12, 1),  # v9 - v12 with weight 1
                        (10, 15, 1),  # v10 - v15 with weight 1
                        (11, 12, 1),  # v11 - v12 with weight 1
                        (12, 13, 1),  # v12 - v13 with weight 1
                        (13, 14, 2),  # v13 - v14 with weight 2
                        (14, 15, 2)],  # v14 - v15 with weight 2
                 start_node=0, goal_node=15)   # a path from v0 to v15 should be found


def get_optimal_solution_weight(nr_ant_types):
    """
    Helper function that returns the total combined weight of all paths in the optimal solution in test graph 3 for the
    specified number of ant types.
    """

    # For 2 ant types it should be 0-1-2-6-7-15 or 0-11-12-9-10-15, which both have a length of 5 (so a total of 10)
    if nr_ant_types == 2:
        return 10

    # For 3 ant types, the lowest total length can be found as follows:
    # - one ant type follows the shortest possible path (of length 5, e.g. 0-1-2-6-7-15)
    # - second ant type follows the second-shortest possible path (of length 7, e.g. [0,11,12,13,14,15])
    # - third ant type follows the third-shortest possible path (of length 8, e.g. [0,8,9,10,15])
    # this gives a total length of 5 + 7 + 8 = 20
    if nr_ant_types == 3:
        return 20

    # For 4 ant types it should be [0,1,2,3,4,15], [0,5,6,7,15], [0,8,9,10,15] or [0,11,12,13,14,15],
    # which gives a total length of 7 + 8 + 8 + 7 = 30
    if nr_ant_types == 4:
        return 30


def experiment_3_helper(rho, beta, cl, tau_0, t_max, gamma, q0_range, nr_ants_per_type,
                        nr_ant_types_range, num_runs,
                        is_varying_q0=False,
                        t_last_change_before_restart: int = math.inf):
    """
    Helper function that is used to run experiment 3 from the paper and our own extension experiment 4.
    It has 3 modes of running:
    * With t_last_change_before_restart = math.inf
      - is_varying_q0=False: q0_range is considered to be a range of q0-values to have separate ACO runs over.
      - is_varying_q0=True: q0_range is considered to be a varying q0-value that should be varied during each ACO run.
    * With t_last_change_before_restart != math.inf (expected to have is_varying_q0=False)
      - we run our ACO implementation with the specified restart parameter t_last_change_before_restart
    """

    graph = create_test_graph_3()

    # little "dirty" trick to be able to use the same code for a varying q0 and regular q0
    q0_range_to_iterate = q0_range if not is_varying_q0 else [q0_range]

    # three pieces of information are kept: disjoint percentage, optimal percentage, coefficient of variation
    results = np.zeros((len(nr_ant_types_range), len(q0_range_to_iterate), 3), dtype=float)

    # for each combination of q0 and nr_ant_types
    for q0_index, q0 in enumerate(q0_range_to_iterate):
        for nr_ant_types_idx, nr_ant_types in enumerate(nr_ant_types_range):
            print("q0: " + str(q0) + " - types: " + str(nr_ant_types))

            # get the total length of the paths in the optimal solution (for this specific nr_ant_types)
            optimal_solution_length = get_optimal_solution_weight(nr_ant_types)

            disjoint_count = 0.0
            optimal_count = 0.0
            squared_solution_diffs = []  # needed to compute the coefficient of variation later on

            # repeat the specified number of runs
            for run in range(num_runs):
                print("run: " + str(run+1) + " of " + str(num_runs))

                m_aco = MultiTypeACO(graph, nr_ant_types, nr_ants_per_type, rho, beta, gamma, q0, cl, t_max, tau_0,
                                     t_last_change_before_restart=t_last_change_before_restart)
                best_paths, best_path_lengths, best_path_costs, _, _ = m_aco.run(print_results=False)

                # if all the computed shared edges costs are 0, this means all the paths are disjoint
                if sum(best_path_costs) == 0:
                    disjoint_count += 1

                    # get the total length of all the paths in the solution
                    solution_length = sum(best_path_lengths)

                    # if the total length of all the disjoint paths is equal to the total length of the
                    # paths in the optimal solution, then this is an optimal solution as well
                    if solution_length == optimal_solution_length:
                        optimal_count += 1

                    # we will compute VarCoef as variation in total length for all paths
                    # of the solution and the optimal solution
                    squared_solution_diffs.append((solution_length - optimal_solution_length)**2)

            # store the disjoint percentage
            results[nr_ant_types_idx, q0_index, 0] = 100.0 * disjoint_count / num_runs

            # store the percentage of disjoint solutions that were also optimal
            # (so the percentage should be computed based on the number of disjoint solutions that were found,
            #  not on the total number of runs)
            if disjoint_count == 0:
                results[nr_ant_types_idx, q0_index, 1] = None
            else:
                results[nr_ant_types_idx, q0_index, 1] = (100.0 * optimal_count / disjoint_count)

            # Coefficient of Variation is defined as deviation / optimal_solution_length
            if disjoint_count == 0:
                results[nr_ant_types_idx, q0_index, 2] = None  # without disjoint solutions, there are no optimal ones
            else:
                deviation_of_optimal = math.sqrt(np.mean(squared_solution_diffs))
                results[nr_ant_types_idx, q0_index, 2] = deviation_of_optimal / optimal_solution_length

    return results


def run_experiment_3(rho=0.1,  # default by Dorigo
                     beta=2,  # default by Dorigo
                     cl=5,  # default by Dorigo
                     tau_0=0.05,  # default by Dorigo
                     t_max=1000,
                     gamma=2,
                     q0_range=np.arange(0, 1, 0.1),  # range from 0 to 0.9
                     nr_ants_per_type=12,
                     nr_ant_types_range=np.arange(2, 5),
                     num_runs=100,
                     print_table=True):

    """
    Helper function that runs experiment 3 from the paper.
    """

    results_1 = experiment_3_helper(rho=rho,
                                    beta=beta,
                                    cl=cl,
                                    tau_0=tau_0,
                                    t_max=t_max,
                                    gamma=gamma,
                                    q0_range=q0_range,
                                    nr_ants_per_type=nr_ants_per_type,
                                    nr_ant_types_range=nr_ant_types_range,
                                    num_runs=num_runs,
                                    is_varying_q0=False)

    results_2 = experiment_3_helper(rho=rho,
                                    beta=beta,
                                    cl=cl,
                                    tau_0=tau_0,
                                    t_max=t_max,
                                    gamma=gamma,
                                    q0_range=q0_range,
                                    nr_ants_per_type=nr_ants_per_type,
                                    nr_ant_types_range=nr_ant_types_range,
                                    num_runs=num_runs,
                                    is_varying_q0=True)

    # print the table if requested:
    if print_table:
        for ant_types_idx, nr_ant_types in enumerate(nr_ant_types_range):
            print("TYPES: " + str(nr_ant_types))
            print("---------------")

            top_headers = ["q0", "Disj", "Opt", "VarCoef"]

            # add left side headers to the data
            data = results_1[ant_types_idx].copy().tolist()
            # add varying result
            data.append(results_2[ant_types_idx][0].copy().tolist())

            for q0_idx, q0 in enumerate(q0_range):
                data[q0_idx].insert(0, round(q0, 1))
            data[len(q0_range)].insert(0, "var")

            # print the table with the results
            print(tabulate(data, headers=top_headers))
            print("\n")

    return results_1, results_2


def run_experiment_3_1(rho=0.1,  # default by Dorigo
                       beta=2,  # default by Dorigo
                       cl=5,  # default by Dorigo
                       tau_0=0.05,  # default by Dorigo
                       t_max=1000,
                       gamma=2,
                       q0_range=np.arange(0, 1, 0.1),  # q0 varies from 0 to 0.9,
                       nr_ants_per_type=12,
                       nr_ant_types_range=np.arange(2, 5),
                       num_runs=100,
                       print_table=True):
    """
    Helper function that runs experiment 3 from the paper, without the varying q0-part.
    """

    results_1 = experiment_3_helper(rho=rho,
                                    beta=beta,
                                    cl=cl,
                                    tau_0=tau_0,
                                    t_max=t_max,
                                    gamma=gamma,
                                    q0_range=q0_range,
                                    nr_ants_per_type=nr_ants_per_type,
                                    nr_ant_types_range=nr_ant_types_range,
                                    num_runs=num_runs,
                                    is_varying_q0=False)

    # print the table if requested:
    if print_table:
        for ant_types_idx, nr_ant_types in enumerate(nr_ant_types_range):
            print("TYPES: " + str(nr_ant_types))
            print("---------------")

            top_headers = ["q0", "Disj", "Opt", "VarCoef"]

            # add left side headers to the data
            data = results_1[ant_types_idx].copy().tolist()

            for q0_idx, q0 in enumerate(q0_range):
                data[q0_idx].insert(0, q0)

            # print the table with the results
            print(tabulate(data, headers=top_headers))
            print("\n")

    return results_1


def run_experiment_3_2(rho=0.1,  # default by Dorigo
                       beta=2,  # default by Dorigo
                       cl=5,  # default by Dorigo
                       tau_0=0.05,  # default by Dorigo
                       t_max=1000,
                       gamma=2,
                       q0_range=np.arange(0, 1, 0.1),  # q0 varies from 0 to 0.9,
                       nr_ants_per_type=12,
                       nr_ant_types_range=np.arange(2, 5),
                       num_runs=100,
                       print_table=True):
    """
    Helper function that runs the part of experiment 3 with the varying q0-value.
    """

    results_2 = experiment_3_helper(rho=rho,
                                    beta=beta,
                                    cl=cl,
                                    tau_0=tau_0,
                                    t_max=t_max,
                                    gamma=gamma,
                                    q0_range=q0_range,
                                    nr_ants_per_type=nr_ants_per_type,
                                    nr_ant_types_range=nr_ant_types_range,
                                    num_runs=num_runs,
                                    is_varying_q0=True)

    # print the table if requested:
    if print_table:
        for ant_types_idx, nr_ant_types in enumerate(nr_ant_types_range):
            print("TYPES: " + str(nr_ant_types))
            print("---------------")

            top_headers = ["q0", "Disj", "Opt", "VarCoef"]

            # add left side headers to the data
            data = results_2[ant_types_idx][0].copy().tolist()
            data.insert(0, "var")

            # print the table with the results
            print(tabulate([data], headers=top_headers))
            print("\n")

    return results_2


if __name__ == '__main__':

    print("EXPERIMENT 3")
    results_experiment_3_1, results_experiment_3_2 = run_experiment_3()
    #store_var("results_experiment_3_1", results_experiment_3_1)
    #store_var("results_experiment_3_2", results_experiment_3_2)
    #results_experiment_3_1 = load_var("results_experiment_3_1")
    #results_experiment_3_2 = load_var("results_experiment_3_2")

    #print("EXPERIMENT 3 (FIXED Q0): ")
    #run_experiment_3_1()

    #print("-----------------------------------------------------")

    #print("EXPERIMENT 3 (VARYING Q0): ")
    #run_experiment_3_2()
