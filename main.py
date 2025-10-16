import polars as pl
from metaheuristics.cvrptw import *
from metaheuristics.genetic_algorithm import *

def create_data(file_path:str,n_vehicle:int):
    data = pl.read_csv(file_path)
    data = data.filter(pl.col("node_type") != "depot")
    data = data.with_columns(
        pl.col("start_time")
        .str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S')
        .alias("start_time")
    )
    data = data.with_columns(
        pl.col("end_time")
        .str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S')
        .alias("end_time")
    )

    nodes = []
    for row in data.rows(named = True):
        nodes.append(
            Node(
                id = str(row["node_nbr"]),
                longitude = row["X"],
                latitude = row["Y"],
                demand = 1,
                ready_timestamp=row["start_time"],
                due_timestamp = row["end_time"],
                service_time_sec=0
            )
        )
    depot_node = Node(
        id = "0",
        longitude=0,
        latitude = 0,
        demand = 0,
        ready_timestamp=data["start_time"].min(),
        due_timestamp=data["end_time"].max(),
        service_time_sec=0
    )
    vehicles = []
    for i in range(n_vehicle):
        vehicles.append(
            Vehicle(
                id = str(i),
                capacity = 5,
                speed_per_sec=1,
                depot_node=depot_node
            )
        )
    return {
        "nodes":nodes,
        "vehicles": vehicles
    }

def print_solution(solution: GeneticAlgorithmOutput) -> bool:
    time = 1/solution.best_fitness_score
    subroute = solution.best_genome.subroute
    print(f"Time taken for best solution: {1/solution.best_fitness_score} seconds")
    for i in subroute:
        nodes = ["0"]
        for n in subroute[i]:
            nodes.append(n.id)
        nodes.append("0")
        print(f"Route for vehicle {i} is: {nodes}")
    return True


if __name__ == "__main__":
    data = create_data(file_path="data/data.csv",n_vehicle=23)
    vehicles = data["vehicles"]
    nodes = data["nodes"]
    utils = CVRPTWGeneticAlgorithmUtils(
        nodes = nodes,
        vehicles=vehicles,
        population_size=100
    )
    ga = GeneticAlgorithm(
        iteration = 100,
        fitness_limit = 10*10,
        population_init_function = utils.generate_population,
        selection_function = utils.selection_function,
        fitness_function = utils.calculate_fitness,
        crossover_function = utils.crossover,
        mutation_function = utils.mutate,
    )
    ga = ga.optimize()
    optimal_solution = ga.find_best_solution()
    print_solution(optimal_solution)