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
        alpha=0.1, gamma=0.95, epsilon=0.9, learning=True
    ):
        self.name = name
        self.mean_cost = mean_cost
        self.sigma = sigma

        self.state_space = state_space
        self.action_space = action_space

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning = learning

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
        if random.random() < self.epsilon and self.learning:
            action = random.randint(0, self.action_space - 1)
        else:
            action = np.argmax(self.q_table[state_id])
        self.epsilon *= 0.999  # decay epsilon
        return action

    def learn(self, state, action, reward, next_state):
        state_id = state[0] * 3 + state[1]
        next_state_id = next_state[0] * 3 + next_state[1]
        self.q_table[state_id][action] = (
            self.q_table[state_id][action] + self.alpha * (
                reward + self.gamma *
                np.max(self.q_table[next_state_id]) -
                self.q_table[state_id][action]
            )
        )


class MarketEnvironment():
    def __init__(self, num_agents=6, costs=5, sigma=2):
        self.num_agents = num_agents

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
        self.price = self.determine_price()
        for agent in self.agents.values():
            self.observations[agent.name] = (
                self.get_profitability_level(agent.cost),
                self.get_market_level()
            )
        return self.observations.copy()

    def step(self, actions):
        action_names = ["list_job", "self_processing", "bid_0.25",
                        "bid_0.5", "bid_0.75", "bid_1.0"]
        state_names = [
            "(loss;empty_market)",
            "(loss;some_jobs)",
            "(loss;full_market)",
            "(break_even;empty_market)",
            "(break_even;some_jobs)",
            "(break_even;full_market)",
            "(profit;empty_market)",
            "(profit;some_jobs)",
            "(profit;full_market)"
        ]
        current = {}
        rewards = {}
        # print(f"======Price: {self.price}========")
        for agent_name, action in actions.items():
            earnings = self.price - self.agents[agent_name].cost
            rewards[agent_name] = 0
            current[agent_name] = (
                f"Agent {agent_name} with state "
                f"{self.observations[agent_name]} "
                f"{state_names[self.observations[agent_name][0] * 3 + self.observations[agent_name][1]]} "
                f"chooses {action} ({action_names[action]}) "
                f"with cost {self.agents[agent_name].cost} "
                f"and possible earnings {earnings} "
            )
            # list job
            if action == 0:
                self.list_job(agent_name)
            # self processing
            if action == 1:
                rewards[agent_name] += earnings
            # place bids
            if earnings > 0:
                if action == 2:
                    self.place_bid(agent_name, earnings * 0.25)
                if action == 3:
                    self.place_bid(agent_name, earnings * 0.5)
                if action == 4:
                    self.place_bid(agent_name, earnings * 0.75)
                if action == 5:
                    self.place_bid(agent_name, earnings * 1.0)
            elif earnings <= 0 and action >= 2:
                rewards[agent_name] -= 10  # earnings * 2  # loss incurred

        self.market_load_prev = len(self.jobs)
        # determine winners
        winners = self.determine_winner()
        # print(f"Jobs listed: {self.jobs}")
        # print(f"Placed bids: {self.bids}")
        # print(f"Winners: {[bid.bidder for bid in winners]}")
        for bid in winners:
            # calculate rewards of bidder
            rewards[bid.bidder] += (
                (self.price - self.agents[bid.bidder].cost) * 2 - bid.bid
            )
            # calculate rewards of job provider
            winner = random.choice(self.jobs)
            rewards[winner] += bid.bid
            self.jobs.remove(winner)
            self.bids.remove(bid)
        else:
            # if there are bidders without succes, there reward will be self processing
            # print(f"Left over jobs: {self.jobs}")
            for job in self.jobs:
                rewards[job] += self.price - self.agents[job].cost
            # print(f"Left over bids: {self.bids}")
            for bid in self.bids:
                rewards[bid.bidder] = self.price - self.agents[bid.bidder].cost

        # clear jobs and bids
        self.jobs = []
        self.bids = []
        # generate new costs, price
        for agent in self.agents.values():
            current[agent.name] += f"-> actual reward: {rewards[agent.name]}"
            agent.cost = agent.draw_cost()
        # print("\n".join(current.values()))
        self.price = self.determine_price()
        for agent in self.agents.values():
            self.observations[agent.name] = (
                self.get_profitability_level(agent.cost),
                self.get_market_level()
            )
        current['social_welfare'] = sum(rewards.values())
        return self.observations.copy(), rewards, False, current

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

    def get_profitability_level(self, cost):
        earnings = self.price - cost
        if earnings < 0:
            return 0
        if earnings == 0:
            return 1
        return 2

    def get_market_level(self):
        if self.market_load_prev == 0:
            return 0
        if self.market_load_prev >= self.num_agents // 2:
            return 2
        return 1

    def list_job(self, job):
        self.jobs.append(job)

    def place_bid(self, bidder, bid):
        bid = Bid(bidder, bid)
        self.bids.append(bid)
        self.bids.sort(key=lambda b: b.bid, reverse=True)

    def determine_winner(self):
        if not self.jobs:
            return []
        return self.bids[:len(self.jobs)]
