from simulation import run_episode
from marketenvironment import MarketEnvironment

if __name__ == '__main__':
    sim_env = MarketEnvironment(num_agents=6, costs=5, sigma=2)
    steps = 87600  # simulate for one year (hourly steps)
    run_episode(sim_env=sim_env, max_steps=1)
