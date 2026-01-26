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
        alpha=0.4, gamma=0.1, epsilon=0.9, learning=True
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

        self.min_epsilon = 0.05

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
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.999)
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
    def __init__(self, num_agents, costs, sigma):
        self.num_agents = num_agents
        costs = self._expand_param(costs, num_agents)
        sigma = self._expand_param(sigma, num_agents)

        self.state_space = 9
        self.action_space = 6

        self.jobs = []
        self.bids = []

        self.agents = self.generate_rz_list(
            num_rz=num_agents, costs=costs, sigma=sigma
        )
        self.price = self.determine_price()

        self.pressure_history = []
        self.price_window = 10

        self.observations = {}

    def _expand_param(self, param, num_agents):
        if np.isscalar(param):
            return np.full(num_agents, param)

        param = np.asarray(param)
        if len(param) != num_agents:
            raise ValueError("Wrong length")
        return param

    def reset(self):
        self.jobs = []
        self.bids = []
        self.pressure_history = []

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
            "(loss;low_competition)",
            "(loss;medium_competition)",
            "(loss;high_competition)",
            "(break_even;low_competition)",
            "(break_even;medium_competition)",
            "(break_even;high_competition)",
            "(profit;low_competition)",
            "(profit;medium_competition)",
            "(profit;high_competition)"
        ]
        current = {}
        rewards = {}
        current["price"] = self.price
        for agent_name, action in actions.items():
            earnings = self.price - self.agents[agent_name].cost
            rewards[agent_name] = 0
            obs = self.observations[agent_name]
            state_idx = obs[0] * 3 + obs[1]

            current[agent_name] = {
                "obs": obs,
                "state": state_names[state_idx],
                "action": action,
                "action_name": action_names[action],
                "cost": self.agents[agent_name].cost,
                "possible_earnings": earnings
            }

            # list job
            if action == 0:
                if earnings <= 0:
                    # small reward for listing unprofitable job
                    rewards[agent_name] += 5
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
                # penalty for bidding with no possible earnings
                rewards[agent_name] -= 10

        # determine winners
        winners = self.determine_winner()

        # calculate market statistics
        round_winning_bids = [bid.bid for bid in winners]
        round_avg_bid = (
            statistics.mean(round_winning_bids)
            if round_winning_bids else 0.0
        )
        trade_count = len(winners)

        # Protect against division by zero
        max_trades = max(1, self.num_agents)

        round_pressure = round_avg_bid * (trade_count / max_trades)
        self.pressure_history.append(round_pressure)

        if len(self.pressure_history) > self.price_window:
            self.pressure_history.pop(0)

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
            current[bid.bidder]["won_bid"] = bid.bid
            current[winner]["received_bid"] = bid.bid
        else:
            # if there are bidders without succes,
            # there reward will be self processing
            for job in self.jobs:
                rewards[job] += self.price - self.agents[job].cost
                current[job]["received_bid"] = 0

            for bid in self.bids:
                rewards[bid.bidder] = self.price - self.agents[bid.bidder].cost
                current[bid.bidder]["won_bid"] = 0

        # clear jobs and bids
        self.jobs = []
        self.bids = []
        # generate new costs, price
        for agent in self.agents.values():
            current[agent.name]["actual_reward"] = rewards[agent.name]
            agent.cost = agent.draw_cost()

        self.price = self.determine_price()
        for agent in self.agents.values():
            self.observations[agent.name] = (
                self.get_profitability_level(agent.cost),
                self.get_market_level()
            )
        current["social_welfare"] = sum(rewards.values())
        current["market_situation"] = self.pressure_history.copy()
        return self.observations.copy(), rewards, False, current

    def generate_rz_list(self, num_rz, costs, sigma):
        rz_list = {}
        for i in range(num_rz):
            name = f"RZ{i+1}"
            rz = RZ(
                name=name, mean_cost=costs[i], sigma=sigma[i],
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
        if len(self.pressure_history) < 3:
            return 1  # medium competition by default

        avg_pressure = statistics.mean(self.pressure_history)

        q33 = np.percentile(self.pressure_history, 33)
        q66 = np.percentile(self.pressure_history, 66)

        if avg_pressure < q33:
            return 0  # low competition
        if avg_pressure < q66:
            return 1  # medium competition
        else:
            return 2  # high competition

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
