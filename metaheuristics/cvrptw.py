import random
import numpy as np
import math
import datetime as dt
from pydantic import BaseModel
from typing import List,Tuple,Dict,Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f'logs/{dt.datetime.now().strftime("%Y%m%d%H%M%S")}.log', 
    encoding='utf-8', 
    level=logging.DEBUG
)

Distance = float
FitnessScore = float
VehicleID = str

class Node(BaseModel):
    id: str
    longitude: float
    latitude: float
    demand: float
    ready_timestamp: dt.datetime
    due_timestamp: dt.datetime
    service_time_sec: float

    def calculate_distance(
        self: 'Node', 
        other_node: 'Node'
    ) -> Distance:
        return (
             (self.latitude - other_node.latitude) ** 2 +
             (self.longitude - other_node.longitude) ** 2
        ) ** 0.5

class Vehicle(BaseModel):
    id: VehicleID
    capacity: float
    speed_per_sec: float
    depot_node: Node
    
Route = List[Node]
SubRoute = Dict[VehicleID,List[Node]]

@dataclass
class CVRPTWGenome:
    vehicles: List[Vehicle]
    nodes: Route
    subroute: SubRoute = None


    def __randomized_max_visits_by_vehicle(
        self: 'CVRPTWGenome'
    ) -> Dict[VehicleID,float]:
        n_nodes = len(self.nodes)
        n_vehicles = len(self.vehicles)
        rand_prob = [
            abs(np.random.normal(1/n_vehicles,1/(3*n_vehicles))) 
            for i in self.vehicles
        ]
        normalized_prob = [
            i/sum(rand_prob) 
            for i in rand_prob
        ]
        max_allowable_nodes = {
            self.vehicles[i].id: math.ceil(normalized_prob[i]*n_nodes) 
            for i in range(n_vehicles)
        }
        return max_allowable_nodes
    
    def check_capacity_constraint(
        self: 'CVRPTWGenome',
        vehicle: Vehicle,
        nodes: List[Node]
    ) -> bool:
        demand = 0
        for node in nodes:
            demand = demand + node.demand
        return demand <= vehicle.capacity
    
    def check_time_window_constraint(
        self: 'CVRPTWGenome',
        vehicle: Vehicle,
        nodes: List[Node]
    ) -> bool:
        time = vehicle.depot_node.ready_timestamp + dt.timedelta(
            seconds = vehicle.depot_node.service_time_sec + nodes[0].service_time_sec
        )
        for n in range(len(nodes)-1):
            time = time + dt.timedelta(
                seconds = (
                    nodes[n].calculate_distance(nodes[n+1])
                )/vehicle.speed_per_sec + nodes[n + 1].service_time_sec
            )
        return time <= nodes[-1].due_timestamp
    
    def check_max_visited_limit_constraint(
        self: 'CVRPTWGenome',
        vehicle: Vehicle,
        nodes: List[Node],
        max_visits_by_vehicle:  Dict[Vehicle,float]
    ) -> bool:
        return len(nodes) <= max_visits_by_vehicle[vehicle.id]
    
    def check_all_constraints(
        self: 'CVRPTWGenome',
        vehicle: Vehicle,
        nodes: List[Node],
        max_visits_by_vehicle: Dict[Vehicle, float]
    ) -> bool:
        return (
            self.check_capacity_constraint(
                vehicle = vehicle,
                nodes = nodes
            ) and 
            self.check_time_window_constraint(
                vehicle = vehicle, 
                nodes = nodes
            ) and 
            self.check_max_visited_limit_constraint(
                vehicle = vehicle, 
                nodes = nodes, 
                max_visits_by_vehicle = max_visits_by_vehicle
            )
        )

    def route_to_subroute(
        self:'CVRPTWGenome'
    ) -> SubRoute:
        # Randomize max nodes a vehicle can visit
        max_visits_by_vehicle = self.__randomized_max_visits_by_vehicle()

        # Assign nodes to vehicle untill it violates one of the three constraints
        # 1. total demand for visited node <= Vehicle Capacity
        # 2. number of visits <= randomized max visits
        # 3. Travel time + service time < due datetime
        subroute = {vehicle.id: [] for vehicle in self.vehicles}
        nodes = self.nodes
        for v in range(len(self.vehicles)):
            vehicle = self.vehicles[v]
            for n in range(len(nodes)):
                # from_node = self.base_node if len(subroute[v]) == 0 else subroute[v][-1]
                to_node = nodes[n]
                if self.check_all_constraints(
                    vehicle=self.vehicles[v],
                    nodes = subroute[vehicle.id] + [to_node],
                    max_visits_by_vehicle = max_visits_by_vehicle
                ):
                    subroute[vehicle.id] = subroute[vehicle.id] + [to_node]
                else:
                    break
            nodes = nodes[n:] if n < len(nodes)-1 else []
        return subroute
    
    def check_fesible(
        self: 'CVRPTWGenome',
        subroute: SubRoute
    ) -> bool:
        n_nodes = 0
        for v in self.vehicles:
            n_nodes = n_nodes + len(subroute[v.id])
        return n_nodes == len(self.nodes)
    
    def find_feasible_subroute(self: 'CVRPTWGenome') -> bool:
        start = dt.datetime.now()
        while True:
            subroute = self.route_to_subroute()
            end = dt.datetime.now()
            if (
                self.check_fesible(subroute) and 
                (end-start).total_seconds() < 30
                ):
                self.subroute = subroute
                break
        return True
    
    def __post_init__(self: 'CVRPTWGenome'):
        self.find_feasible_subroute()
        if self.subroute == None:
            logger.warning(f"Feasible subroute could not be found for the route: {self.nodes}")
        else:
            logger.info(f"Feasible subroute found for the route: {self.nodes}")

class CVRPTWGeneticAlgorithmUtils(BaseModel):
    nodes: List[Node]
    vehicles: List[Vehicle]
    population_size: int
    cut_off_sec: int = 30

    def find_vehicle_by_vehicle_id(self,genome,vehicle_id):
        for v in genome.vehicles:
            if v.id == vehicle_id:
                return v

    def generate_population(
        self: 'CVRPTWGeneticAlgorithmUtils'
    ) -> List[CVRPTWGenome]:
        population = []
        i,duration = 0,0
        start = dt.datetime.now()
        while (
            (i < self.population_size) and 
            (duration < self.cut_off_sec)
            ):
            route = random.sample(self.nodes,len(self.nodes))
            cvrptw_genome = CVRPTWGenome(
                vehicles=self.vehicles,
                nodes=route
            )
            if cvrptw_genome.subroute != None:
                population.append(cvrptw_genome)
                i = i + 1
            end = dt.datetime.now()
            duration = (end-start).total_seconds()
        return population
    
    def calculate_fitness(
        self: 'CVRPTWGeneticAlgorithmUtils',
        genome: CVRPTWGenome
    ) -> FitnessScore:
        time = 0
        subroute = genome.subroute
        for v in subroute:
            for n in range(len(subroute[v])):
                vehicle = self.find_vehicle_by_vehicle_id(genome,v)
                if n == 0:
                    from_node = vehicle.depot_node
                    to_node = subroute[v][n]
                elif n == len(subroute[v])-1:
                    from_node = subroute[v][n]
                    to_node = vehicle.depot_node
                else:
                    from_node = subroute[v][n]
                    to_node = subroute[v][n+1]
                time = time + from_node.calculate_distance(to_node)/vehicle.speed_per_sec
        return 1/time
    
    def selection_function(
        self: 'CVRPTWGeneticAlgorithmUtils', 
        population: List[CVRPTWGenome],
        fitness_function:Callable[[CVRPTWGenome],FitnessScore]
    ) -> List[CVRPTWGenome]:
        return random.choices(
            population = population,
            weights = [
                fitness_function(genom) 
                for genom in population
            ],
            k = 2
        )
    
    def crossover(
        self: 'CVRPTWGeneticAlgorithmUtils',
        genome_a: CVRPTWGenome, 
        genome_b: CVRPTWGenome
    ) -> Tuple[CVRPTWGenome,CVRPTWGenome]:
        route_a = genome_a.nodes
        route_b = genome_b.nodes
        if len(route_a) != len(route_b):
            raise ValueError("Genomes a and b must be of same length")
        
        length = len(route_a)
        if length < 2:
            return route_a,route_b
        
        p = random.randint(1, length - 1)

        crossover_a = route_a[0:p] + route_a[p:]
        crossover_b = route_a[0:p] + route_a[p:]

        crossover_a = CVRPTWGenome(
            vehicles=self.vehicles,
            nodes = crossover_a
        )
        crossover_b = CVRPTWGenome(
            vehicles=self.vehicles,
            nodes = crossover_b
        )
        if (
            (crossover_a.subroute != None) and 
            (crossover_b.subroute != None)
            ):
            return crossover_a, crossover_b
        else:
            return genome_a,genome_b
    
    def mutate(
        self: 'CVRPTWGeneticAlgorithmUtils',
        genome: CVRPTWGenome,
        probability: float = 0.5
    ) -> CVRPTWGenome:
        route = genome.nodes
        n = math.floor(len(route)/2)
        i = random.randrange(start = 0, stop = n)
        if random.random() > probability:
            a = route[i]
            b = route[-(i+1)]
            route[i] = b
            route[-(i+1)] = a
        genome_mutated = CVRPTWGenome(
            vehicles=self.vehicles,
            nodes = route
        )
        if genome_mutated.subroute != None:
            return genome_mutated
        else:
            return genome