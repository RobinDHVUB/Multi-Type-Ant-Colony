# Imports for printing tables
from tabulate import tabulate

# Plotting imports
import matplotlib.pylab as plt

# General imports
from multitype_aco import *
from graph import *
from general_helpers import *


def create_test_graph_1():
    """
    Helper function that generates and returns test graph 1 (from experiment 1), which should allow finding
    two fully disjoint paths.
    """

    return Graph(nr_of_nodes=4,  # v0, v1, v2, v3
                 edges=[(0, 1, 1),  # v0 - v1 with weight 1
                        (0, 2, 3),  # v0 - v2 with weight 3
                        (1, 2, 1),  # v1 - v2 with weight 1
                        (1, 3, 3),  # v1 - v3 with weight 3
                        (2, 3, 1)],  # v2 - v3 with weight 1
                 start_node=0, goal_node=3)  # a path from v0 to v3 should be found


def equal_paths(path1, path2):
    """
    Helper function that checks whether two paths are identical.
    """
    return path1 == path2


def run_experiment_1_1(rho=0.1,  # default by Dorigo
                       beta=2,  # default by Dorigo
                       cl=5,  # default by Dorigo
                       tau_0=0.05,  # default by Dorigo
                       t_max=20,
                       gamma_range=np.arange(0, 6),
                       q0_range=np.arange(0, 1, 0.1),
                       num_runs=100,
                       print_table=False):
    """
    Helper function that runs experiment 1 from the paper.
    Returns table of percentages of finding an optimal solution for each (gamma,q0) pair.
    """

    # nr_ant_types and nr_ants_per_type are fixed values (for the sake of code clarity)
    nr_ant_types = 2
    nr_ants_per_type = 5

    # simple graph which has the path combination of '0-1-3' and '0-2-3' as optimal solution
    graph = create_test_graph_1()
    optimal_subpath_1 = [0, 1, 3]
    optimal_subpath_2 = [0, 2, 3]

    # we will count the number of runs in which we get to the optimal solution
    optimal_solution_count = np.zeros((len(q0_range), len(gamma_range)), dtype=float)

    # for every q0-gamma combination...
    for q0_index, q0 in enumerate(q0_range):
        for gamma_index, gamma in enumerate(gamma_range):
            # ... repeat the specified number of runs...
            for run in range(num_runs):
                m_aco = MultiTypeACO(graph, nr_ant_types, nr_ants_per_type, rho, beta, gamma, q0, cl, t_max, tau_0)
                best_paths, _, _, _, _ = m_aco.run(print_results=False)
                
                # It doesn't matter whether ant type 1 follows 0-1-3, while ant type 2 follows 0-2-3
                # or whether it's the other way around. Both are optimal solutions.
                if (equal_paths(best_paths[0], optimal_subpath_1)) and (equal_paths(best_paths[1], optimal_subpath_2)):
                    optimal_solution_count[q0_index][gamma_index] += 1
                elif (equal_paths(best_paths[0], optimal_subpath_2)) and (equal_paths(best_paths[1], optimal_subpath_1)):
                    optimal_solution_count[q0_index][gamma_index] += 1

    convergence_to_optimal_solution = (optimal_solution_count / num_runs) * 100

    # print the table if requested:
    if print_table:
        # generate top headers
        top_headers = ["gamma / q0"]
        top_headers.extend(gamma_range.tolist())

        # add left side headers to the data
        data = convergence_to_optimal_solution.tolist().copy()
        for i, q0 in enumerate(q0_range):
            data[i].insert(0, q0)

        # print the table with the results
        print(tabulate(data, headers=top_headers))

    return convergence_to_optimal_solution


def plot_pheromone_intensity(t_max, avg_intens, save_figs):
    """
    Helper function that visualizes the pheromone intensities for experiment 1.
    This function will create a plot for v1-v2, v0-v1-v3 and v0-v2-v3.
    """

    # Fonts that are used in the plots
    font_small = 18
    font_medium = 19
    font_large = 20

    # PLOT 1
    # compute the average pheromone intensity on path v0-v1-v3 for both ant types
    intensity_013_type_0 = (avg_intens[:, 1, 0, 0] + avg_intens[:, 3, 1, 0]) / 2
    intensity_013_type_1 = (avg_intens[:, 1, 0, 1] + avg_intens[:, 3, 1, 1]) / 2

    # plot the pheromone intensity on v0-v1-v3
    x_values = np.arange(0, t_max + 1)
    fig, axes = plt.subplots(figsize=(11, 9))

    y_ticks = np.arange(0, 0.14, 0.01)
    y_tick_labels = [f"{i:,.2f}" for i in y_ticks]
    y_tick_labels[5] = '$\\tau_0$ = ' + y_tick_labels[5]
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(y_tick_labels)

    plt.plot(x_values,
             intensity_013_type_0,
             label="Ant type 1",
             color="blue",
             linewidth=3,
             zorder=5)
    plt.plot(x_values,
             intensity_013_type_1,
             label="Ant type 2",
             color="orange",
             linewidth=3,
             linestyle='dashed',
             zorder=5)

    plt.xticks(fontsize=font_small)
    plt.yticks(fontsize=font_small)

    plt.ylim(bottom=0, top=0.14)
    plt.xlim(left=0, right=25)

    # plot a horizontal line for indicating tau_0
    plt.axhline(y=0.05, color='black', linestyle='-', zorder=1, linewidth=1)

    plt.legend(loc = "lower right", fontsize = font_small)
    plt.xlabel('Iteration\n', fontsize=font_medium)
    plt.ylabel('Pheromones', fontsize=font_medium)
    plt.title('Pheromone Intensity on v0-v1-v3',
              fontsize=font_large)

    if save_figs:
        plt.savefig("graphs/experiment_1_graph_1.png", bbox_inches='tight', dpi=300)

    plt.show()

    # PLOT 2
    # compute the average pheromone intensity on path v0-v2-v3 for both ant types
    intensity_023_type_0 = (avg_intens[:, 2, 0, 0] + avg_intens[:, 3, 2, 0]) / 2
    intensity_023_type_1 = (avg_intens[:, 2, 0, 1] + avg_intens[:, 3, 2, 1]) / 2

    # plot the pheromone intensity on v0-v2-v3
    x_values = np.arange(0, t_max + 1)
    fig, axes = plt.subplots(figsize=(11, 9))

    y_ticks = np.arange(0, 0.14, 0.01)
    y_tick_labels = [f"{i:,.2f}" for i in y_ticks]
    y_tick_labels[5] = '$\\tau_0$ = ' + y_tick_labels[5]
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(y_tick_labels)

    plt.plot(x_values,
             intensity_023_type_0,
             label="Ant type 1",
             color="blue",
             linewidth=3,
             zorder=5)
    plt.plot(x_values,
             intensity_023_type_1,
             label="Ant type 2",
             color="orange",
             linewidth=3,
             linestyle='dashed',
             zorder=5)

    plt.xticks(fontsize=font_small)
    plt.yticks(fontsize=font_small)

    plt.ylim(bottom=0, top=0.14)
    plt.xlim(left=0, right=25)

    # plot a horizontal line for indicating tau_0
    plt.axhline(y=0.05, color='black', linestyle='-', zorder=1, linewidth=1)

    plt.legend(loc="lower right", fontsize=font_small)
    plt.xlabel('Iteration\n', fontsize=font_medium)
    plt.ylabel('Pheromones', fontsize=font_medium)
    plt.title('Pheromone Intensity on v0-v2-v3',
              fontsize=font_large)

    if save_figs:
        plt.savefig("graphs/experiment_1_graph_2.png", bbox_inches='tight', dpi=300)

    plt.show()

    # PLOT 3
    # compute the average pheromone intensity on path v1-v2 for both ant types
    intensity_12_type_0 = avg_intens[:, 2, 1, 0]
    intensity_12_type_1 = avg_intens[:, 2, 1, 1]

    # plot the pheromone intensity on v1-v2
    x_values = np.arange(0, t_max + 1)
    fig, axes = plt.subplots(figsize=(11, 9))

    y_ticks = np.arange(0, 0.14, 0.01)
    y_tick_labels = [f"{i:,.2f}" for i in y_ticks]
    y_tick_labels[5] = '$\\tau_0$ = ' + y_tick_labels[5]
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(y_tick_labels)

    plt.plot(x_values,
             intensity_12_type_0,
             label="Ant type 1",
             color="blue",
             linewidth=3,
             zorder=5)
    plt.plot(x_values,
             intensity_12_type_1,
             label="Ant type 2",
             color="orange",
             linestyle='dashed',
             linewidth=3,
             zorder=5)

    plt.xticks(fontsize=font_small)
    plt.yticks(fontsize=font_small)

    plt.ylim(bottom=0, top=0.14)
    plt.xlim(left=0, right=25)

    # plot a horizontal line for indicating tau_0
    plt.axhline(y=0.05, color='black', linestyle='-', zorder=1, linewidth=1)

    plt.legend(loc = "lower right", fontsize = font_small)
    plt.xlabel('Iteration', fontsize=font_medium)
    plt.ylabel('Pheromones', fontsize=font_medium)
    plt.title('Pheromone Intensity on v1-v2',
              fontsize=font_large)

    if save_figs:
        plt.savefig("graphs/experiment_1_graph_3.png", bbox_inches='tight', dpi=300)

    plt.show()


def run_experiment_1_2(rho=0.1,  # default by Dorigo
                       beta=2,  # default by Dorigo
                       cl=5,  # default by Dorigo
                       tau_0=0.05,  # default by Dorigo
                       t_max=25,
                       gamma=1,
                       q0=0.1,
                       num_runs=1,  # the plot should only be done on one run (averaging would distort the results)
                       save_figs=False):
    """
    Helper function that runs experiment 1 from the paper for one specific value of gamma and q0
    and visualizes the pheromone intensities in a graph.
    """

    # nr_ant_types and nr_ants_per_type are fixed values (for the sake of code clarity)
    nr_ant_types = 2
    nr_ants_per_type = 5

    # simple graph which has the path combination of '0-1-3' and '0-2-3' as optimal solution
    graph = create_test_graph_1()

    # we will keep track of the pheromone levels on a path
    pheromone_intensity_sum = np.zeros(((t_max + 1), graph.nr_of_nodes, graph.nr_of_nodes, nr_ant_types))
    for run in range(num_runs):
        m_aco = MultiTypeACO(graph, nr_ant_types, nr_ants_per_type, rho, beta, gamma, q0, cl, t_max, tau_0)
        _, _, _, pheromones, _ = m_aco.run(track_iter_pheromones=True, print_results=False)
        pheromone_intensity_sum += pheromones

    # average the pheromone intensities over all runs
    avg_intens = pheromone_intensity_sum / num_runs

    # plot the pheromone intensity
    plot_pheromone_intensity(t_max, avg_intens, save_figs)

    return avg_intens


if __name__ == '__main__':

    print("EXPERIMENT 1.1: ")
    results_experiment_1_1 = run_experiment_1_1(print_table=True)
    #store_var("results_experiment_1_1", results_experiment_1_1)
    #results_experiment_1_1 = load_var("results_experiment_1_1")

    print("-----------------------------------------------------")

    print("EXPERIMENT 1.2: ")
    results_experiment_1_2 = run_experiment_1_2()
    #store_var("results_experiment_1_2", results_experiment_1_2)
    #results_experiment_1_2 = load_var("results_experiment_1_2")
    #plot_pheromone_intensity(25, results_experiment_1_2, True)
