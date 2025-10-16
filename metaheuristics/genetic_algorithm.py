from pydantic import BaseModel
from typing import List, Dict, Callable,Any,Tuple

Genome = Any
Population = List[Genome]
PopulationInitFunc = Callable[[],Population]
FitnessFunc = Callable[[Genome], int]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]

class GeneticAlgorithmOutput(BaseModel):
    best_genome: Genome
    best_fitness_score: float
    population: Population


class GeneticAlgorithm(BaseModel):
    iteration: int
    fitness_limit: int
    population_init_function: PopulationInitFunc
    selection_function: SelectionFunc
    fitness_function: FitnessFunc
    crossover_function: CrossoverFunc
    mutation_function: MutationFunc
    population:Population = None

    def optimize(self:'GeneticAlgorithm') -> 'GeneticAlgorithm':
        population = self.population_init_function()
        for i in range(self.iteration):
            population = sorted(
                population, 
                key=lambda genome: self.fitness_function(genome), 
                reverse=True
            )
            if self.fitness_function(population[0]) >= self.fitness_limit:
                break
            next_generation = population[0:2]
            for j in range(int(len(population) / 2) - 1):
                # Selection-----------------------------------------
                parents = self.selection_function(
                    population, 
                    self.fitness_function
                )
                # Crossover----------------------------------------
                offspring_a, offspring_b = self.crossover_function(
                    parents[0], 
                    parents[1]
                )
                # Mutation------------------------------------------
                offspring_a = self.mutation_function(offspring_a)
                offspring_b = self.mutation_function(offspring_b)
                # Child generation----------------------------------
                next_generation += [offspring_a, offspring_b]
            population = next_generation
        self.population = population
        return self
    
    def find_best_solution(
        self: 'GeneticAlgorithm'
    ) -> GeneticAlgorithmOutput:
        population = sorted(
            self.population, 
            key=lambda genome: self.fitness_function(genome), 
            reverse=True
        )
        return GeneticAlgorithmOutput(
            best_genome = population[0],
            best_fitness_score = self.fitness_function(population[0]),
            population = population
        )
