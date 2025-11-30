import math
import numpy as np

class RZ():
    def __init__(self, name, mean_cost, sigma, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.name = name
        self.mean_cost = mean_cost
        self.sigma = sigma
        self.cost = None
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = {}
    
    def draw_cost(self):
        if self.sigma == 0.0:
            self.cost = self.mean_cost
        else:
            self.cost = math.floor(np.random.normal(self.mean_cost, self.sigma) * 100) / 100.0
        self.cost = max(0.0, self.cost)
        return self.cost
    
    def choose_action():
        pass
    
    def act():
        pass
    
    def learn():
        pass
    
