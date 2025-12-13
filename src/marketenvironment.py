import math
import statistics
import random
import numpy as np

from dataclasses import dataclass


@dataclass
class Bid():
    bidder: str
    bid: int


class RZ():
    def __init__(
        self, name, mean_cost, sigma, state_space, action_space,
        alpha=0.1, gamma=0.95, epsilon=0.1
    ):
        self.name = name
        self.mean_cost = mean_cost
        self.sigma = sigma

        self.state_space = state_space
        self.action_space = action_space

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.cost = self.draw_cost()

        self.q_table = np.zeros((state_space, action_space))

    def draw_cost(self) -> int:
        if self.sigma == 0:
            self.cost = self.mean_cost
        else:
            self.cost = math.floor(
                np.random.normal(self.mean_cost, self.sigma)
            )
        self.cost = min(max(1, self.cost), 10)
        return self.cost

    def choose_action(self, state_id):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_space - 1)
        else:
            action = np.argmax(self.q_table[state_id])
        return action

    def learn(self):
        pass


class MarketEnvironment():
    def __init__(self, num_agents=6, costs=5, sigma=2):
        self.state_space = 9
        self.action_space = 6

        self.jobs = []
        self.bids = []

        self.market_load_prev = 0

        self.agents = self.generate_rz_list(
            num_rz=num_agents, costs=costs, sigma=sigma
        )
        self.price = self.determine_price()

        self.observations = {}

    def reset(self):
        self.jobs = []
        self.bids = []
        self.market_load_prev = 0
        for agent in self.agents:
            agent.cost = agent.draw_cost()
            self.observations[agent.name] = (
                self.get_cost_level(agent.cost) * 3 + self.market_load_prev
            )
        self.price = self.determine_price()

    def step(self):
        # return next_state, rewards, done, info
        pass

    def generate_rz_list(self, num_rz, costs, sigma):
        rz_list = []
        for i in range(num_rz):
            rz = RZ(
                name=f'RZ{i+1}', mean_cost=costs, sigma=sigma,
                state_space=self.state_space, action_space=self.action_space
            )
            rz_list.append(rz)
        return rz_list

    def determine_price(self):
        costs = []
        for rz in self.agents:
            costs.append(rz.cost)
        return math.ceil(statistics.mean(costs))

    def get_cost_level(self, cost):
        if cost <= 3:
            return 0
        elif cost <= 7:
            return 1
        else:
            return 2

    def list_job(self, job):
        self.jobs.append(job)

    def place_bid(self, bidder, bid):
        bid = Bid(bidder, bid)
        self.bids.append(bid)
        self.bids.sort(key=lambda b: b.bid, reverse=True)

    def determine_winner(self):
        if not self.jobs:
            raise ValueError("No jobs listed")
        return self.bids[:len(self.jobs)]

    def pay_seller(self, job):
        if not self.jobs:
            raise ValueError("No jobs listed")
        if not self.bids:
            raise ValueError("No bids placed")
        earning = self.bids.pop(0).bid
        self.jobs.remove(job)
        return earning
