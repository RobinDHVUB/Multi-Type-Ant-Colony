import numpy as np
from graph import Graph
import math
import numbers


class MultiTypeACO:
    """
    The Multi-type Ant Colony Optimization class, which represents some kind of "super agent" that knows and steers
     the behavior of all ants for a given graph.
    """

    def __init__(self, graph: Graph, nr_ant_types: int, nr_ants_per_type: int, rho: float, beta: float, gamma: int,
                 q0_range, cl: int, t_max: int, tau_0: float, t_last_change_before_restart: int = math.inf):
        """
        :param graph: The graph on which the Multi-type Ant Colony Optimisation will be run.
        :param nr_ant_types: The number of ant types (i.e. n).
        :param nr_ants_per_type: The number of ants per type (i.e. m).
        :param rho: The pheromone decay.
        :param beta: The sensitivity to heuristic.
        :param gamma: The sensitivity to foreign pheromones
        :param q0_range: The exploration rate. If multiple values are given a varying exploration rate is used.
        :param cl: The candidate list length.
        :param t_max: The maximum number of iterations.
        :param tau_0: The initial amount of pheromone present on all edges.
        :param t_last_change_before_restart: A restart is performed when no better solution has been found in the
                                             last t_last_change_before_restart solutions.
                                             Default = math.inf, which means no restarts will be done by default.
        """

        # to allow the constructor to work with a single value as q0 argument (instead of needing a list with 1 value)
        if isinstance(q0_range, numbers.Number):
            q0_range = [q0_range]

        # initialize fields with the constructor arguments
        self.graph = graph
        self.nr_ant_types = nr_ant_types
        self.nr_ants_per_type = nr_ants_per_type
        self.rho = rho
        self.beta = beta
        self.gamma = gamma
        self.q0_range = q0_range
        self.q0 = q0_range[0]
        self.cl = cl
        self.t_max = t_max
        self.tau_0 = tau_0

        # additional parameter for the restart extension (how many iterations to wait before restarting)
        self.t_last_change_before_restart = t_last_change_before_restart

        # the initial amount of each type of pheromones (i.e. of each ant type) on each edge is the given tau_0
        self.pheromones = np.empty((self.graph.nr_of_nodes, self.graph.nr_of_nodes, self.nr_ant_types))

        # create all the ants that will be used in optimization:
        # the first row has all ants of ant_type = 0, the second row has all ants of ant_type = 1, etc.
        self.ants = np.array([[MultiTypeACO.Ant(ant_type) for _ in range(nr_ants_per_type)] for ant_type in range(nr_ant_types)])

        # keep track of the best combination of solutions found so far in the current search
        # (with one solution per type of ant):
        # - keep track of the best paths per ant type
        self.best_paths = np.empty(self.nr_ant_types, dtype=object)
        # - keep track of the path cost of the best path per ant type
        #   (path cost is based on the length and the number of types that use parts of this path, cfr. formula 7 in
        #    the paper)
        self.best_path_costs = np.full(self.nr_ant_types, math.inf)
        # - keep track of the lengths of the best paths per ant type
        self.best_path_lengths = np.full(self.nr_ant_types, math.inf)

        # same thing as best_paths, etc. but now taking restarts into account,
        # so it shows the best ones over all restarts (not just the best one in the current search)
        self.best_paths_overall = np.empty(self.nr_ant_types, dtype=object)
        self.best_path_costs_overall = np.full(self.nr_ant_types, math.inf)
        self.best_path_lengths_overall = np.full(self.nr_ant_types, math.inf)

        # possible minimization objective functions that can be used
        self.minimization_objectives = [self.compute_shared_edges_cost, self.compute_shared_edges_average]

    def run(self, minimization_objective=0, track_iter_pheromones=False, opt_sol_weight=None, print_results=False):
        """
        Perform the actual Multi-type Ant Colony Optimization for its graph, ants and optimization parameters.

        :param minimization_objective: The objective function to use, with 0 being compute_shared_edges_cost
                                       and 1 being compute_shared_edges_average
        :param track_iter_pheromones: Boolean that states whether to keep the pheromone levels for each iteration
                                      (used in experiment 1)
        :param opt_sol_weight: The combined weight of all best paths in the optimal solution. This parameter is only
                               used for experimentally checking when the first optimal solution occurs in a run.
        :param print_results: If True, the best paths and their lengths will be printed before returning them.
        :return: A Tuple containing the best paths overall, their lengths and their costs as found by the optimization process.
                 Additionally, the pheromones for each iteration and the first iterations where a disjoint or optimal solution
                 is found might be included as well (dependent on the parameters the algorithm is run with).
        """

        # compute how often we have to modify our q-value (for the experiment with a varying q-value)
        nr_q_values = len(self.q0_range)
        vary_every = self.t_max // nr_q_values

        # choose the minimization method to use for this run
        value_to_minimize = self.minimization_objectives[minimization_objective]

        # LOOP 1 of the pseudocode of Vrancx' paper
        # put the initial amount tau_0 of each type of pheromones (i.e. of each ant type) on each edge
        self.pheromones = np.full((self.graph.nr_of_nodes, self.graph.nr_of_nodes, self.nr_ant_types), self.tau_0)

        # LOOP 2 from the pseudocode of Vrancx' paper
        # place each ant on the start node of the graph
        for ant_index in range(self.nr_ants_per_type):
            for ant_type_index in range(self.nr_ant_types):
                self.ants[ant_type_index][ant_index].start_new_path(self.graph.start_node)

        # LOOP 3 from the pseudocode of Vrancx' paper
        self.initialize_best_paths()

        # keep track of the pheromone_levels per iteration to plot the pheromone intensity evolution,
        # used for Fig.4 in the paper (for experiment 1)
        if track_iter_pheromones:
            pheromones_per_iteration = np.empty(((self.t_max + 1), self.graph.nr_of_nodes, self.graph.nr_of_nodes, self.nr_ant_types))
            pheromones_per_iteration[0] = self.pheromones.copy()
        else:
            pheromones_per_iteration = None

        # keep track of the number of iterations without finding a new solution (used for restart extension)
        t_last_change = 0	   

        # keep track of the first iteration a disjoint/optimal solution is found (used for experiment 4.1)
        disjoint_found = math.inf
        optimal_found = math.inf

        # START OF MAIN LOOP from the pseudocode of Vrancx' paper
        # perform t_max iterations to converge to a solution
        for t in range(self.t_max):

            # check whether progress seems to be stalling in which case a restart will be performed
            if t_last_change >= self.t_last_change_before_restart:
                # we consider a better overall solution to be a solution that either has a lower total cost
                # or has a lower total weight if the total cost is the same
                if (sum(self.best_path_costs) < sum(self.best_path_costs_overall)
                        or (sum(self.best_path_costs) == sum(self.best_path_costs_overall)
                            and sum(self.best_path_lengths) < sum(self.best_path_lengths_overall))):
                    # keep track of the best overall solution
                    self.best_path_lengths_overall = self.best_path_lengths
                    self.best_path_costs_overall = self.best_path_costs
                    self.best_paths_overall = self.best_paths

                # reset everything to start "fresh"
                self.initialize_best_paths()
                self.best_path_costs = np.full(self.nr_ant_types, math.inf)
                self.best_path_lengths = np.full(self.nr_ant_types, math.inf)
                self.pheromones = np.full((self.graph.nr_of_nodes, self.graph.nr_of_nodes, self.nr_ant_types),
                                          self.tau_0)

                # we've restarted so reset t_last_change as well
                t_last_change = 0

            # we also keep track of how much each edge is used for easy computation of the shared edge cost later on
            # Remark: currently unused in the final implementation (but kept for experimenting with the alternative
            #         compute_shared_cost_alt implementation)
            edge_usage = np.zeros((self.graph.nr_of_nodes, self.graph.nr_of_nodes, self.nr_ant_types))

            # update value of q0, if we're running with a varying q0
            if t % vary_every == 0:
                q0_index = t // vary_every
                # it's possible that the total number of iterations is not perfectly dividable
                # by the number of q0-values, in which case we simply let the last q0-value have the remaining
                # iteration
                if q0_index < nr_q_values:
                    self.q0 = self.q0_range[q0_index]

            # for every ant type...
            for ant_type_index in range(self.nr_ant_types):
                # ...for every ant...
                for ant_index in range(self.nr_ants_per_type):
                    current_ant = self.ants[ant_type_index][ant_index]

                    # the ant starts again from the source node and searches for a new path
                    current_ant.start_new_path(self.graph.start_node)

                    # as long as the current ant hasn't reached the goal node in the graph, it keeps taking steps
                    while not self.graph.is_path_complete(current_ant.path):
                        # get the current node the ant is on
                        current_node = current_ant.get_current_node()
                        # find the next node for the ant to take
                        next_node = self.choose_next_node(current_ant)
                        # let the ant take the step to the next node
                        current_ant.take_step(next_node)
                        # perform a local pheromone update
                        self.update_pheromones(current_node, next_node, current_ant.ant_type, self.rho * self.tau_0)

                        # keep track of the edge usage
                        # (the edge is used in both directions, since the graph is bidirectional)
                        edge_usage[current_node, next_node, current_ant.ant_type] += 1
                        edge_usage[next_node, current_node, current_ant.ant_type] += 1

            # once every ant has reached the goal state, we can look at the solutions that were found by each of them
            # and keep track of the best solution so far per ant type
            changed = False  # keep track of whether the best found paths have changed this iteration
            for ant_type_index in range(self.nr_ant_types):
                for ant_index in range(self.nr_ants_per_type):
                    current_ant = self.ants[ant_type_index][ant_index]

                    current_path_length = self.graph.get_path_weight(current_ant.path)
                    shared_edges_cost = value_to_minimize(current_ant)
                    # Alternative implementation (currently unused, but kept for experimentation):
                    # shared_edges_cost = self.compute_shared_edges_cost_alt(current_ant, edge_usage)

                    min_path_length = self.best_path_lengths[ant_type_index]
                    min_path_cost = self.best_path_costs[ant_type_index]

                    # - if we found a better solution for this ant type (i.e. a solution with a smaller path cost),
                    #   we will use this solution from now on
                    # - if we found an equal path cost for this ant type, we prefer the shortest path
                    if (min_path_cost > shared_edges_cost
                            or (min_path_cost == shared_edges_cost and current_path_length < min_path_length)):
                        # a new best path was found, so update the best paths
                        self.best_paths[ant_type_index] = current_ant.path
                        self.best_path_costs[ant_type_index] = shared_edges_cost
                        self.best_path_lengths[ant_type_index] = current_path_length
                        changed = True

            # increment t_last_change if no improved paths were found this iteration or otherwise reset it to 0
            t_last_change = 0 if changed else (t_last_change + 1)

            # code for global evaporation (not included in the final implementation but kept for experimentation)
            # we evaporate or decrease the pheromones on all edges that don't belong to the best path a little
            # by performing a global update with delta=0 on them

            # for ant_type_index in range(self.nr_ant_types):
            #     for i in range(self.graph.nr_of_nodes):
            #         for j in range(i, self.graph.nr_of_nodes):
            #             best_path_found = self.best_paths[ant_type_index]
            #             if not self.graph.edge_in_path(i, j, best_path_found):
            #                 self.update_pheromones(from_node=i,
            #                                        to_node=j,
            #                                        ant_type=ant_type_index,
            #                                        delta=0)
            
            # global update rule
            # we reward the best path found for each ant type with an increase in pheromones on their edges
            for ant_type_index in range(self.nr_ant_types):
                best_path_found = self.best_paths[ant_type_index]
                # we assumed the factor rho was forgotten in Vrancx' pseudocode description and included it in our code
                delta = self.rho * 1 / self.graph.get_path_weight(self.best_paths[ant_type_index])

                # go over all edges on the best path that was found and update the pheromones for this ant type:
                for i in range(len(best_path_found)-1):
                    self.update_pheromones(from_node=best_path_found[i],
                                           to_node=best_path_found[i+1],
                                           ant_type=ant_type_index,
                                           delta=delta)

            # keep track of the pheromone levels for this iteration (if requested)
            if pheromones_per_iteration is not None:
                pheromones_per_iteration[t+1] = self.pheromones.copy()

            # code for finding the first iteration with a disjoint/optimal solution
            # if all path costs are zero, we have a disjoint solution
            if sum(self.best_path_costs) == 0:
                if disjoint_found == math.inf:
                    disjoint_found = t+1

                # if a disjoint solution has the optimal length, it is optimal
                if opt_sol_weight is not None and sum(self.best_path_lengths) == opt_sol_weight:
                    if optimal_found == math.inf:
                        optimal_found = t+1
                        # we don't need to keep going after finding the optimal solution,
                        # so a break was included to increase experimental speed
                        break

        # we consider a better overall solution to be a solution that either has a lower total cost or has a lower total
        # weight if the total cost is the same
        # REMARK: If no restarts are performed the best path costs overall will be math.inf at this point, which ensures
        #         the found solution will be copied into the overall variables
        if (sum(self.best_path_costs) < sum(self.best_path_costs_overall)
                or (sum(self.best_path_costs) == sum(self.best_path_costs_overall)
                    and sum(self.best_path_lengths) < sum(self.best_path_lengths_overall))):
            self.best_path_costs_overall = self.best_path_costs
            self.best_paths_overall = self.best_paths
            self.best_path_lengths_overall = self.best_path_lengths

        # Print the results (if requested)
        if print_results:
            print("T+: " + str(self.best_paths_overall))
            print("L+: " + str(self.best_path_lengths_overall))
            print("Shared Edges Sum: " + str(self.best_path_costs_overall))

        return self.best_paths_overall, self.best_path_lengths_overall, self.best_path_costs_overall, pheromones_per_iteration, (disjoint_found, optimal_found)

    def choose_next_node(self, ant):
        """
        Method that selects the next node for the given ant.

        :param ant: The ant for which the next node should be found by ACO.
        :return: The next node the ant should take.
        """

        # find the current node the ant is on
        current_node = ant.get_current_node()

        # find neighboring nodes in the graph
        neighbors = self.graph.get_neighbors(current_node)
        # only unvisited neighboring nodes are potential candidates for the ant's next step
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in ant.path]

        # if no unvisited nodes are found, just choose a random node to take (to avoid ants getting stuck)
        if len(unvisited_neighbors) == 0:
            return np.random.choice(neighbors)

        # if potential candidates are found, then we will first compute a "utility" score for each candidate node with:
        # utility = (own pheromones on edge) x (1/weight of edge)^beta x (1/foreign pheromones on edge)^gamma
        # Remark: utility = the argument of argMax in the formula of Vrancx' pseudocode
        #                   (which is also the numerator for the probability used later on)

        # START OF COMPUTATION OF UTILITIES
        # get the weights for the edges to the unvisited neighbors
        unvisited_weights = [self.graph.get_edge_weight(current_node, neighbor) for neighbor in unvisited_neighbors]
        # get the indices for sorting the neighbors on the weight of the edge towards that neighbor
        sorting_indices = np.argsort(unvisited_weights)
        # restrict the list of candidates (and their weights) to the specified number of top candidates to consider
        nr_of_candidates = min(self.cl, len(unvisited_neighbors))  # there could be less than cl neighbors
        candidates = np.empty(nr_of_candidates, dtype=int)
        utilities = np.zeros(nr_of_candidates)
        for i in range(nr_of_candidates):
            # the first candidate to consider is the nearest neighbor (i.e. the one with the lowest edge weight),
            # the second candidate is the second-nearest neighbor (i.e. the one with the second lowest edge weight),...
            candidate_index = sorting_indices[i]
            current_candidate = unvisited_neighbors[candidate_index]
            current_edge_weight = unvisited_weights[candidate_index]

            # compute the tau for the current candidate (or the total amount of your own type of pheromone on the edge)
            current_tau = self.get_pheromones(current_node, current_candidate, ant.ant_type)

            # compute the eta for the current candidate (the inverse of the weight of the edge)
            current_eta = (1/current_edge_weight)

            # compute the phi for the current candidate (or the total amount of foreign pheromones on the edge)
            current_phi = sum([self.get_pheromones(current_node, current_candidate, ant_type) for ant_type in range(self.nr_ant_types) if ant_type != ant.ant_type])

            # store the actual candidate node and the utility for the candidate node
            utilities[i] = current_tau * (current_eta ** self.beta) * ((1/current_phi) ** self.gamma)
            candidates[i] = current_candidate
    
        # END OF COMPUTATION OF UTILITIES

        # after computing the utilities, the ant should...
        # ... exploit (based on the utility) with a probability of q0
        if np.random.uniform() <= self.q0:
            # the best candidate (i.e. the one with the highest utility) is our next step to take
            # if multiple best candidates exist, we choose a random one among them
            max_util_idx = np.where(utilities == max(utilities))[0]
            return candidates[np.random.choice(max_util_idx)]
        # ... explore with a probability (1-q0)
        else:
            # exploration stills occurs based on the utility:
            # candidates with a higher utility have a higher probability of being chosen
            exploration_probs = utilities / sum(utilities)
            return np.random.choice(candidates, p=exploration_probs)

    def update_pheromones(self, from_node: int, to_node: int, ant_type: int, delta: float):
        """
        Method for updating the amount of pheromones of the given ant type that is on the edge between from_node and to_node.

        :param from_node: The node the edge starts from.
        :param to_node: The node the edge goes to.
        :param ant_type: The type of the ant whose pheromone we will deposit.
        :param delta: Delta offset to be used for updating the amount of pheromones on the given edge.
        """

        # update the pheromone trail
        # since we're using only half of the graph's adjacency matrix, make sure to store the value in the correct half
        if from_node > to_node:
            old_value = self.pheromones[from_node, to_node, ant_type]
            self.pheromones[from_node, to_node, ant_type] = (1 - self.rho) * old_value + delta
        else:
            old_value = self.pheromones[to_node, from_node, ant_type]
            self.pheromones[to_node, from_node, ant_type] = (1 - self.rho) * old_value + delta

    def get_pheromones(self, from_node: int, to_node: int, ant_type: int):
        """
        Convenience method for getting the pheromone amount of the specified ant type that is currently on
        the specified edge.

        :param from_node: The node the edge starts from.
        :param to_node: The node the edge goes to.
        :param ant_type: The type of ant we want to return the pheromone for.
        """

        # since we're using only half of the graph's adjacency matrix, make sure to get the value from the correct half
        if from_node > to_node:
            return self.pheromones[from_node, to_node, ant_type]
        else:
            return self.pheromones[to_node, from_node, ant_type]

    def compute_shared_edges_cost(self, ant):
        """
        Method that computes for a given ant how valuable its path is.
        This method will return the value obtained from computing the sum of formula 7 of the paper.

        :param ant: The ant whose path the value will be computed for.
        :return: the value obtained from computing the sum of formula 7 of the paper.
        """

        # the shared edges cost of an ant's path is computed by summing for all edges on its path the following terms:
        # (the number of ant types using this edge) * (the edge weight)
        shared_edges_cost = 0.0
        # go over all edges on this ant's path and compute the shared edges cost
        for i in range(len(ant.path) - 1):
            from_node = ant.path[i]
            to_node = ant.path[i + 1]

            # get the weight of this edge
            edge_weight = self.graph.get_edge_weight(from_node, to_node)

            # the number of other ant types using this edge can be found by counting how many ant types (that aren't our
            # own type) have this edge in their optimal path
            num_foreign_ant_types = 0
            for ant_type_idx in range(len(self.best_paths)):
                if ant_type_idx != ant.ant_type and self.graph.edge_in_path(from_node, to_node, self.best_paths[ant_type_idx]):
                    num_foreign_ant_types += 1

            # update the shared edges cost with the total cost term for this edge
            shared_edges_cost += num_foreign_ant_types * edge_weight

        return shared_edges_cost

    def compute_shared_edges_cost_alt(self, ant, edge_usage):
        """
        Alternative implementation of compute_shared_edges_cost. This is currently unused, but kept for further
        experimentation.

        :param ant: The ant whose path the value will be computed for.
        :param edge_usage: Edge matrix containing the number of times each edge was
                           used by each ant type in the current iteration.
        :return: the value obtained from computing the sum of formula 7 of the paper.
        """

        # the shared edges cost of an ant's path is computed by summing for all edges on its path the following terms:
        # (the number of ant types using this edge) * (the edge weight)
        shared_edges_cost = 0.0
        # go over all edges on this ant's path and compute the shared edges cost
        for i in range(len(ant.path) - 1):
            from_node = ant.path[i]
            to_node = ant.path[i + 1]

            # get the weight of this edge
            edge_weight = self.graph.get_edge_weight(from_node, to_node)

            # the number of other ant types using this edge can be found by counting how many ant types used this edge
            # in total (i.e. the number of nonzero entries there are for the different ant types for this edge)
            # and subtracting 1 since our own ant type should not be counted as a foreign one
            # (and clearly we used this edge since it's in our path)
            num_foreign_ant_types = np.count_nonzero(edge_usage[from_node, to_node]) - 1

            # update the shared edges cost with the total cost term for this edge
            shared_edges_cost += num_foreign_ant_types * edge_weight

        return shared_edges_cost

    def compute_shared_edges_average(self, ant):
        """
        Method that computes for a given ant how valuable its path is.
        This method will return the value obtained from averaging the number of ant types on shared edges, i.e.,
        the value of formula 8 in the paper. This method is only used in experiment 2.

        :param ant: The ant whose path the value will be computed for.
        :return: the value obtained from averaging the number of ant types on shared edges.
        """

        # the shared edges average of an ant's path is computed by summing for all edges on its path
        # the number of foreign ant types using this edge, divided by the total amount of shared edges on this path

        # go over all edges on the path and count the number of foreign ant types on this path and the total number of
        # shared edges
        shared_edges_count = 0.0
        num_foreign_ant_types = 0.0
        for i in range(len(ant.path) - 1):
            from_node = ant.path[i]
            to_node = ant.path[i + 1]

            # count how many foreign ant types used this edge of the path
            for ant_type_idx in range(len(self.best_paths)):
                if ant_type_idx != ant.ant_type and self.graph.edge_in_path(from_node, to_node, self.best_paths[ant_type_idx]):
                    num_foreign_ant_types += 1

            # if there's any foreign ant type on this edge, this is a shared edge
            if num_foreign_ant_types > 0:
                shared_edges_count += 1

        # compute the shared edges average
        shared_edges_average = (1.0/shared_edges_count) * num_foreign_ant_types if shared_edges_count > 0 else 0.0

        return shared_edges_average

    def initialize_best_paths(self):
        """
        Convenience method that performs one initial run of ACO to prefill the best paths fields.
        """

        # initialize the best paths
        self.best_paths[...] = [[] for _ in range(self.nr_ant_types)]

        # for every ant type we let an ant perform one search
        for ant_type_index in range(self.nr_ant_types):
            current_ant = MultiTypeACO.Ant(ant_type_index)

            # the ant starts from the source node and searches for a new path
            current_ant.start_new_path(self.graph.start_node)

            # as long as the current ant hasn't reached the goal node in the graph, it keeps taking steps
            while not self.graph.is_path_complete(current_ant.path):
                # find the next node for the ant to take
                next_node = self.choose_next_node(current_ant)
                # let the ant take the step to the next node
                current_ant.take_step(next_node)

            # initialize this ant type's best path as the path found by this ant
            self.best_paths[ant_type_index] = current_ant.path

    class Ant:
        """
        Inner class representing one ant agent.
        """

        def __init__(self, ant_type: int):
            """
            :param ant_type: The type of this ant.
            """

            self.ant_type = ant_type
            self.path = []

        def start_new_path(self, start_node):
            """
            Method for letting the ant start a new search path from the given node.

            :param start_node: The node that the ant will start a new path from.
            """

            self.path = [start_node]

        def get_current_node(self):
            """
            Convenience method for getting the node this ant is currently on, i.e. the last node in its path.

            :return: The node this ant is currently on.
            """

            return self.path[-1]

        def take_step(self, to_node):
            """
            Method for letting the ant take a step to the given node. This method will update the ants path
            and current position.
            """

            self.path.append(to_node)
