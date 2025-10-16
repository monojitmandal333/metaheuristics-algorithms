import random
import numpy as np
from typing import List
import math
import datetime as dt

def calculate_total_demand(demand_by_node,nodes):
    tot_demand = 0
    for n in nodes:
        tot_demand = tot_demand + demand_by_node[n]
    return tot_demand

def normalize_vehicle_probability(rand_prob_by_vehicle):
    tot = 0
    for vehicle in rand_prob_by_vehicle:
        tot = tot + rand_prob_by_vehicle[vehicle]
    return {v:rand_prob_by_vehicle[v]/tot for v in rand_prob_by_vehicle}

def rand_max_node_by_vehicle(
    vehicles:list[int|str],
    n_nodes:int,
    flexibility_factor:float
):
    rand_prob_by_vehicle = {
        i:abs(np.random.normal(1/len(vehicles),flexibility_factor)) 
        for i in vehicles
    }
    normalized_prob_by_vehicle = normalize_vehicle_probability(
        rand_prob_by_vehicle
    )
    # print(normalized_prob_by_vehicle)
    max_nodes_by_vehicle = {
        i:math.ceil(normalized_prob_by_vehicle[i]*n_nodes) 
        for i in vehicles
    }
    return max_nodes_by_vehicle

def calculate_actual_time(
    nodes_visited,
    time_matrix,
    time_window_by_node
):
    time = 0
    for i in range(len(nodes_visited) - 1):
        time = max(
            time + time_matrix[nodes_visited[i],nodes_visited[i+1]],
            time_window_by_node[nodes_visited[i+1]]["start_time"]
        )
    return time

def route_to_subroute(
    route:list[int|str],
    vehicles:list[int|str],
    time_matrix,
    demand_by_node:dict[str|int: float],
    capacity_by_vehicle:dict[str|int,list[float]],
    time_window_by_node:dict[str|int,dict[str,float]],
    flexibility_factor:float
):
    max_nodes_by_vehicle = rand_max_node_by_vehicle(
        vehicles = vehicles,
        n_nodes = len(route),
        flexibility_factor = flexibility_factor
    )
    subroute = {i:[] for i in vehicles}
    start_node = 0
    for i in range(len(vehicles)):
        v = vehicles[i]
        for j in range(start_node,len(route)):
            n = route[j]
            from_node = 0 if len(subroute[v]) == 0 else subroute[v][-1]
            to_node = v
            if (
                    ((calculate_total_demand(
                        demand_by_node,subroute[v]) + demand_by_node[n]
                    ) <= capacity_by_vehicle[v]
                    ) and 
                    (len(subroute[v]) + 1 <= max_nodes_by_vehicle[v]) and
                    ((calculate_actual_time(
                        nodes_visited = [0] + subroute[v],
                        time_matrix = time_matrix,
                        time_window_by_node = time_window_by_node
                    ) + time_matrix[from_node, to_node]) 
                     <= time_window_by_node[to_node]["end_time"]
                    )
                ):
                subroute[v] = subroute[v] + [n]
        start_node = start_node + len(subroute[v])
    return subroute

def is_feasible(subroute_by_vehicle,n_nodes):
    tot_nodes = 0
    for n in subroute_by_vehicle:
        tot_nodes = tot_nodes + len(subroute_by_vehicle[n])
    return tot_nodes == n_nodes

def generate_random_route(nodes):
    return random.sample(nodes,len(nodes))

# def fitness_function(subroute_by_vehicle,time_matrix):
#     total_time = 0
#     for vehicle in subroute_by_vehicle:
#         route = [0] + subroute_by_vehicle[vehicle] + [0]
#         for i in range(len(route)-1):
#             total_time = total_time + time_matrix[route[i],route[i+1]]
#     return 1/total_time

# def selection_pair(population, fitness_func,time_matrix):
#     return random.choices(
#         population=population,
#         weights=[fitness_func(gene) for gene in population],
#         k=2
#     )

def generate_feasible_route(
    route:list[int|str],
    vehicles:list[int|str],
    time_matrix,
    demand_by_node:dict[str|int: float],
    capacity_by_vehicle:dict[str|int,list[float]],
    time_window_by_node:dict[str|int,dict[str,float]],
    flexibility_factor:float
):
    route = generate_random_route(route)
    subroute = route_to_subroute(
        route = route,
        vehicles = vehicles,
        time_matrix= time_matrix,
        demand_by_node = demand_by_node,
        capacity_by_vehicle = capacity_by_vehicle,
        time_window_by_node=time_window_by_node,
        flexibility_factor = flexibility_factor,
    )
    start = dt.datetime.now()
    time_diff_sec = 0
    while time_diff_sec < 2*60:
        i = 0
        while i <= 10:
            subroute = route_to_subroute(
                route = route,
                vehicles = vehicles,
                time_matrix= time_matrix,
                demand_by_node = demand_by_node,
                capacity_by_vehicle = capacity_by_vehicle,
                time_window_by_node=time_window_by_node,
                flexibility_factor = flexibility_factor,
            )
            if is_feasible(subroute,len(route)):
                return subroute
            i = i + 1
        route = generate_random_route(route)
        subroute = route_to_subroute(
            route = route,
            vehicles = vehicles,
            time_matrix= time_matrix,
            demand_by_node = demand_by_node,
            capacity_by_vehicle = capacity_by_vehicle,
            time_window_by_node=time_window_by_node,
            flexibility_factor = flexibility_factor,
        )
        if is_feasible(subroute,len(route)):
            return subroute
        else:
            end = dt.datetime.now()
            time_diff_sec = (end-start).total_seconds()
    
    return None

def initialize_population(
    route:list[int|str],
    vehicles:list[int|str],
    time_matrix,
    demand_by_node:dict[str|int: float],
    capacity_by_vehicle:dict[str|int,list[float]],
    time_window_by_node:dict[str|int,dict[str,float]],
    flexibility_factor:float
):
    population = []
    for i in range(100):
        subroute = generate_feasible_route(
            route = route,
            vehicles = vehicles,
            time_matrix= time_matrix,
            demand_by_node = demand_by_node,
            capacity_by_vehicle = capacity_by_vehicle,
            time_window_by_node=time_window_by_node,
            flexibility_factor = flexibility_factor,
        )
        population = population + [subroute]
    return population

def subroute_to_route(subroute):
    route = []
    for i in subroute:
        route = route + subroute[i]
    return route

def crossover(
    subroute_a, 
    subroute_b,
    vehicles:list[int|str],
    time_matrix,
    demand_by_node:dict[str|int: float],
    capacity_by_vehicle:dict[str|int,list[float]],
    time_window_by_node:dict[str|int,dict[str,float]],
    flexibility_factor:float
):
    a = subroute_to_route(subroute_a)
    b = subroute_to_route(subroute_b)
    if len(a) != len(b):
        raise ValueError("Genomes a and b must be of same length")

    length = len(a)
    if length < 2:
        return a, b
    
    start = dt.datetime.now()
    time_passed_sec = 0
    while time_passed_sec < 60:
        p = random.randint(1, length - 1)
        crossover_a = a[0:p] + b[p:]
        crossover_b = b[0:p] + a[p:]
        crossover_a_subroute = route_to_subroute(
            route = crossover_a,
            vehicles = vehicles,
            time_matrix= time_matrix,
            demand_by_node = demand_by_node,
            capacity_by_vehicle = capacity_by_vehicle,
            time_window_by_node=time_window_by_node,
            flexibility_factor = flexibility_factor,
        )
        crossover_b_subroute = route_to_subroute(
            route = crossover_b,
            vehicles = vehicles,
            time_matrix= time_matrix,
            demand_by_node = demand_by_node,
            capacity_by_vehicle = capacity_by_vehicle,
            time_window_by_node=time_window_by_node,
            flexibility_factor = flexibility_factor,
        )
        end = dt.datetime.now()
        time_passed_sec = (end - start).total_seconds()
        if (is_feasible(crossover_a_subroute,len(a)) and 
            is_feasible(crossover_b_subroute,len(b))):
            return crossover_a_subroute, crossover_b_subroute
    print(f"Could not find feasible solution after crossover for : {a,b}")
    return a,b

def mutation(
    subroute,
    vehicles:list[int|str],
    time_matrix,
    demand_by_node:dict[str|int: float],
    capacity_by_vehicle:dict[str|int,list[float]],
    time_window_by_node:dict[str|int,dict[str,float]],
    flexibility_factor:float,
    probability: float = 0.5
):
    genome = subroute_to_route(subroute)
    start = dt.datetime.now()
    time_passed_sec = 0
    while time_passed_sec < 60:
        n = math.floor(len(genome)/2)
        i = random.randrange(start = 0, stop = n)
        if random.random() > probability:
            a = genome[i]
            b = genome[-(i+1)]
            genome[i] = b
            genome[-(i+1)] = a
        subroute = route_to_subroute(
            route = genome,
            vehicles = vehicles,
            time_matrix= time_matrix,
            demand_by_node = demand_by_node,
            capacity_by_vehicle = capacity_by_vehicle,
            time_window_by_node=time_window_by_node,
            flexibility_factor = flexibility_factor,
        )
        end = dt.datetime.now()
        time_passed_sec = (end - start).total_seconds()
        if is_feasible(subroute,len(genome)):
            return subroute
        else:
            pass
    print(f"Could not find feasible solution after mutation for: {genome}")
    return subroute