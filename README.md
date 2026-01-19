# FederatedComputeCluster
This repository contains a simulation of a federated compute cluster, where multiple data centers act as autonomous agents. Each agent receives computational jobs with varying costs and must decide whether to process them locally or offer them on a decentralized market. Using tabular Q-learning, agents learn bidding and pricing strategies over repeated interactions. The goal is to study whether stable, socially beneficial strategies emerge and how profits and workloads distribute among agents in a self-organizing market.

# State Space
Tabular Q-learning requires a compact, discrete representation of the environment.
In this single-phase market model, an agent must base its decision solely on information available before the actions of the current round are executed. This includes:
- its own current cost level,
- the current price, and therefore
- the current margin, and
- aggregated market activity from the previous round.

To keep the state space small and learning stable, both components (Profitability level and Previous market load) are discretized into fixed categorical levels.

### 1. Profitability level

The agent’s production cost is sampled from a clipped normal distribution (e.g., values 1–10). For Q-learning, these continuous costs are mapped into three discrete buckets:

- 0: loss (costs higher than price)
- 1: break even (costs equal price)
- 2: win (costs lower than price)

This discretization preserves the essential economic signal—how cost-competitive an agent currently is—while keeping the state space compact.

### 2. Previous market load

Since all agents act simultaneously in a single phase, the current market does not yet exist at decision time. Therefore, the agent relies on market activity from the previous round as an indicator of supply and competition.

The previous number of listings is discretized as:
- 0: empty market (no listings last round)
- 1: a few jobs listed (less jobs than agents)
- 2: full market (more or enogh jobs as agents)

This captures whether the market was recently inactive, lightly active, or saturated.

### Combined state representation

The full state is the tuple:
```
state = (profitability_level, market_load_prev)
```
This yields 3 × 3 = 9 possible states, ideal for tabular Q-learning.

Example interpretations:
- (0, 2) → the agent is making losses, and last round the market was highly active
- (2, 0) → the agent makes profits, and last round nobody listed a job
- (1, 1) → the agent breaks even, moderately active market in previous round

# Action space
Tabular Q-learning requires a fixed, discrete set of actions.
The overall action design is:

- 0: offer the job on the market
- 1: process the job locally
- 2: place a bid with factor F={0.25, 0.5, 0.75, 1} of the revenue from self-processing

Since bidding with a continuous factor F is not compatible with a discrete Q-table, the bid action is expanded into fixed, predefined bidding levels. This yields the following action set:

- 0: list - list the job on the market
- 1: self_process - handle the job locally
- 2: bid(F=0.25)
- 3: bid(F=0.50)
- 4: bid(F=0.75)
- 5: bid(F=1.00)

The resulting discrete action space is:

A = {0, 1, 2, 3, 4, 5}

These discrete bid levels ensure that agents can learn stable bidding strategies within the tabular Q-learning framework.

# Policy Controll

1. Policy pro Zustand anschauen
    - Für jeden der 9 Zustände: welche Aktion wird am häufigsten gewählt?
    - Erwartbar:
        - loss → eher list
        - profit → eher bid
        - market_level=0 → weniger bid (weil historisch wenig Angebot), eher self/list abhängig von profit
        - market_level=2 → bids aggressiver oder eher “nicht bieten”, je nachdem wie dein Matching die Gewinne macht

2. Reward-Verteilung nach Zustand/Aktion
    - mittlerer Reward für (state, action) sollte “logisch” aussehen:
        - loss + self_process sollte eher negativ/klein sein
        - loss + list sollte besser sein, wenn genügend Bids existieren