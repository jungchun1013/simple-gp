import numpy as np
from typing import List, Callable, Tuple, Union
import random

class GeneticProgramming:
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 20,
                 tournament_size: int = 3,
                 mutation_rate: float = 0.1,
                 max_depth: int = 4):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.max_depth = max_depth
        
        # Define basic operators
        self.operators = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / y if abs(y) > 1e-10 else 1.0
        }
        
    class Node:
        def __init__(self, value: Union[str, float], left=None, right=None):
            self.value = value
            self.left = left
            self.right = right
    
    def print_best_program(self):
        def print_tree(node, depth=0):
            if node is not None:
                if isinstance(node.value, float):
                    print(f"{node.value:.2f}", end='')
                    return
                else:
                    print(str(node.value), end='(')
                print_tree(node.left, depth + 1)
                print(',', end='')
                print_tree(node.right, depth + 1)
                print(')', end='')
        
        if hasattr(self, 'best_program'):
            print_tree(self.best_program)
            print()
        else:
            print("No best program found.")
            
    def _create_random_tree(self, depth: int = 0) -> Node:
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.4):
            return self.Node(random.uniform(-10, 10))
            
        op = random.choice(list(self.operators.keys()))
        return self.Node(op,
                        self._create_random_tree(depth + 1),
                        self._create_random_tree(depth + 1))
    
    def _evaluate_tree(self, node: Node, X: np.ndarray) -> np.ndarray:
        if isinstance(node.value, (int, float)):
            return float(node.value)
            
        left_val = self._evaluate_tree(node.left, X)
        right_val = self._evaluate_tree(node.right, X)
        
        return self.operators[node.value](left_val, right_val)
    
    def _fitness(self, node: Node, X: np.ndarray, y: np.ndarray) -> float:
        try:
            predictions = self._evaluate_tree(node, X)
            mse = np.mean((predictions - y) ** 2)
            return 1 / (1 + mse)
        except:
            return 0.0
            
    def _tournament_selection(self, population: List[Node], 
                            X: np.ndarray, y: np.ndarray) -> Node:
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, 
                  key=lambda node: self._fitness(node, X, y))
    
    def _crossover(self, parent1: Node, parent2: Node) -> Tuple[Node, Node]:
        def copy_tree(node):
            if node is None:
                return None
            return self.Node(node.value, 
                           copy_tree(node.left),
                           copy_tree(node.right))
        
        child1 = copy_tree(parent1)
        child2 = copy_tree(parent2)
        
        # Perform crossover
        if random.random() < 0.5:
            child1.left, child2.left = child2.left, child1.left
        else:
            child1.right, child2.right = child2.right, child1.right
            
        return child1, child2
    
    def _mutate(self, node: Node) -> Node:
        if random.random() < self.mutation_rate:
            return self._create_random_tree()
        
        if node.left:
            node.left = self._mutate(node.left)
        if node.right:
            node.right = self._mutate(node.right)
            
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Initialize population
        population = [self._create_random_tree() for _ in range(self.population_size)]
        
        # Evolution loop
        for gen in range(self.generations):
            new_population = []
            
            # Elitism - keep best individual
            best_individual = max(population,
                                key=lambda node: self._fitness(node, X, y))
            new_population.append(best_individual)
            
            # Create new individuals
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, X, y)
                parent2 = self._tournament_selection(population, X, y)
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # Print progress
            best_fitness = self._fitness(best_individual, X, y)
            if gen % 5 == 0:
                print(f"Generation {gen}: Best Fitness = {best_fitness:.4f}")
        
        self.best_program = max(population,
                               key=lambda node: self._fitness(node, X, y))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._evaluate_tree(self.best_program, X)
    
import numpy as np

# Sample test code
if __name__ == "__main__":
    # Create sample data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    # Initialize and fit the model
    model = GeneticProgramming(population_size=1000, generations=100, mutation_rate=0.1)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    print("Predictions:", predictions)
    print("Best Program:", end=' ')
    model.print_best_program()
