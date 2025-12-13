import math
import numpy as np

class RZ():
    def __init__(self, name: str, mean_cost: int, sigma: int, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.name = name
        self.mean_cost = mean_cost
        self.sigma = sigma
        self.cost = self.draw_cost()
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = {}
    
    def draw_cost(self) -> int:
        if self.sigma == 0:
            self.cost = self.mean_cost
        else:
            self.cost = math.floor(np.random.normal(self.mean_cost, self.sigma))
        self.cost = min(max(1, self.cost), 10)
        return self.cost 
    
    def choose_action():
        pass
    
    def act():
        pass
    
    def learn():
        pass
    
