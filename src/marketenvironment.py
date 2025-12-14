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
        alpha=0.1, gamma=0.95, epsilon=0.5
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

    def choose_action(self, state):
        state_id = state[0] * 3 + state[1]
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
        for agent in self.agents.values():
            agent.cost = agent.draw_cost()
            self.observations[agent.name] = (
                self.get_cost_level(agent.cost), self.market_load_prev
            )
        self.price = self.determine_price()
        return self.observations

    def step(self, actions):
        rewards = {}
        for agent_name, action in actions.items():
            print(agent_name, action)
            earnings = self.price - self.agents[agent_name].cost
            if action == 0:
                print('List job on market')
                self.list_job(agent_name)
            elif action == 1:
                print('self processing without bidding')
                rewards[agent_name] = earnings
            elif action == 2:
                print('bid 0.25 of earnings from self processing')
                self.place_bid(agent_name, earnings * 0.25)
            elif action == 3:
                print('bid 0.5 of earnings from self processing')
                self.place_bid(agent_name, earnings * 0.5)
            elif action == 5:
                print('bid 0.75 of earnings from self processing')
                self.place_bid(agent_name, earnings * 0.75)
            elif action == 6:
                print('bid all from self processing')
                self.place_bid(agent_name, earnings * 1.0)
        self.market_load_prev = len(self.jobs)
        # determine winners
        # clear jobs and bids
        # calculate rewards
        # generate new costs, price
        # get next_state (cost_level, prev_marketload)
        # return next_state, rewards, done and info

        return self.observations, {}, False, {}

    def generate_rz_list(self, num_rz, costs, sigma):
        rz_list = {}
        for i in range(num_rz):
            name = f'RZ{i+1}'
            rz = RZ(
                name=name, mean_cost=costs, sigma=sigma,
                state_space=self.state_space, action_space=self.action_space
            )
            rz_list[name] = rz
        return rz_list

    def determine_price(self):
        costs = []
        for rz in self.agents.values():
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
