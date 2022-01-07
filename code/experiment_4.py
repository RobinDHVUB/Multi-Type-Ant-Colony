# experiment 4 has the same basic setup as experiment 3, so we reuse what is possible
from experiment_3 import *


def run_experiment_4_1(rho=0.1,  # default by Dorigo
                       beta=2,  # default by Dorigo
                       cl=5,  # default by Dorigo
                       tau_0=0.05,  # default by Dorigo
                       t_max=1000,
                       gamma=2,
                       q0_range=np.arange(0, 1, 0.1),  # range from 0 to 0.9
                       nr_ants_per_type=12,
                       nr_ant_types_range=np.arange(4, 5),  # only investigate 4 types (i.e. the hardest case)
                       num_runs=100,
                       print_table=True):
    """
    Helper function that runs an experiment in which we keep track of the first iterations in which we find a
    disjoint/optimal solution. This helps us to get insight into the convergence speed of the algorithm.
    """

    # same graph is used as in experiment 3
    graph = create_test_graph_3()

    # we keep track of 6 pieces of information:
    # - min, first quartile, median, third quartile, max for finding a disjoint solution
    # - min, first quantile, median, third quartile, max for finding an optimal solution
    results = np.full((len(nr_ant_types_range), len(q0_range), 10), math.inf)

    # repeat for every q0 and ant types combination
    for q0_index, q0 in enumerate(q0_range):
        for nr_ant_types_idx, nr_ant_types in enumerate(nr_ant_types_range):
            print("q0: " + str(q0) + " - types: " + str(nr_ant_types))

            # get the total length of the paths in the optimal solution (for this specific nr_ant_types)
            optimal_solution_length = get_optimal_solution_weight(nr_ant_types)

            # keep track of the first iteration a disjoint or optimal solution is found
            disjoint_founds = []
            optimal_founds = []

            # repeat the specified number of runs
            for run in range(num_runs):
                print("run: " + str(run+1) + " of " + str(num_runs))
                m_aco = MultiTypeACO(graph, nr_ant_types, nr_ants_per_type, rho, beta, gamma, q0, cl, t_max, tau_0)
                _, _, _, _, (disjoint_found, optimal_found) = m_aco.run(print_results=False,
                                                                        opt_sol_weight=optimal_solution_length)

                if disjoint_found != math.inf:
                    disjoint_founds.append(disjoint_found)
                else:
                    disjoint_founds.append(np.NAN)

                if optimal_found != math.inf:
                    optimal_founds.append(optimal_found)
                else:
                    optimal_founds.append(np.NAN)

            # discard NAN results (these are not taken into account)
            disjoint_founds = [value for value in disjoint_founds if not np.isnan(value)]
            optimal_founds = [value for value in optimal_founds if not np.isnan(value)]

            # results for the disjoint solutions
            results[nr_ant_types_idx, q0_index, 0] = np.min(disjoint_founds) if disjoint_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 1] = np.quantile(disjoint_founds, 0.25) if disjoint_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 2] = np.mean(disjoint_founds) if disjoint_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 3] = np.quantile(disjoint_founds, 0.75) if disjoint_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 4] = np.max(disjoint_founds) if disjoint_founds else np.NAN

            # results for the optimal solutions
            results[nr_ant_types_idx, q0_index, 5] = np.min(optimal_founds) if optimal_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 6] = np.quantile(optimal_founds, 0.25) if optimal_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 7] = np.median(optimal_founds) if optimal_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 8] = np.quantile(optimal_founds, 0.75) if optimal_founds else np.NAN
            results[nr_ant_types_idx, q0_index, 9] = np.max(optimal_founds) if optimal_founds else np.NAN

    # print the table if requested:
    if print_table:
        for ant_types_idx, nr_ant_types in enumerate(nr_ant_types_range):
            print("TYPES: " + str(nr_ant_types))
            print("---------------")

            top_headers = ["q0", 'd-min', 'd-quart1', 'd-median', 'd-quant3', 'd-max',
                           'o-min', 'o-quart1', 'o-median', 'o-quart3', 'o-max']

            # add left side headers to the data
            data = results[ant_types_idx].copy().tolist()

            for q0_idx, q0 in enumerate(q0_range):
                data[q0_idx].insert(0, round(q0, 1))

            # print the table with the results
            print(tabulate(data, headers=top_headers))
            print("\n")

    return results


def run_experiment_4_2(rho=0.1,  # default by Dorigo
                       beta=2,  # default by Dorigo
                       cl=5,  # default by Dorigo
                       tau_0=0.05,  # default by Dorigo
                       t_max=1000,
                       gamma=2,
                       q0_range=np.arange(0, 1, 0.1),  # q0 varies from 0 to 0.9,
                       nr_ants_per_type=12,
                       nr_ant_types_range=np.arange(2, 5),
                       num_runs=100,
                       t_last_change_before_restart=50,  # the experiment is run by default with a threshold of 50
                       print_table=True):
    """
    Helper function that repeats experiment 3, but now with our extension of restarts.
    """

    results = experiment_3_helper(rho=rho,
                                  beta=beta,
                                  cl=cl,
                                  tau_0=tau_0,
                                  t_max=t_max,
                                  gamma=gamma,
                                  q0_range=q0_range,
                                  nr_ants_per_type=nr_ants_per_type,
                                  nr_ant_types_range=nr_ant_types_range,
                                  num_runs=num_runs,
                                  is_varying_q0=False,
                                  t_last_change_before_restart=t_last_change_before_restart)

    # print the table if requested:
    if print_table:
        for ant_types_idx, nr_ant_types in enumerate(nr_ant_types_range):
            print("TYPES: " + str(nr_ant_types))
            print("---------------")

            top_headers = ["q0", "Disj", "Opt", "VarCoef"]

            # add left side headers to the data
            data = results[ant_types_idx].copy().tolist()

            for q0_idx, q0 in enumerate(q0_range):
                data[q0_idx].insert(0, q0)

            # print the table with the results
            print(tabulate(data, headers=top_headers))
            print("\n")

    return results


if __name__ == '__main__':

    print("EXPERIMENT 4.1:")
    results_experiment_4_1 = run_experiment_4_1()
    #store_var("results_experiment_4_1", results_experiment_4_1)
    #results_experiment_4_1 = load_var("results_experiment_4_1")

    print("EXPERIMENT 4.2:")
    results_experiment_4_2 = run_experiment_4_2()
    #store_var("results_experiment_4_2", results_experiment_4_2)
    #results_experiment_4_2 = load_var("results_experiment_4_2")

