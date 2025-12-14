from simulation import run_episode
from marketenvironment import MarketEnvironment

if __name__ == '__main__':
    num_of_episodes = 1
    sim_env = MarketEnvironment()
    run_episode(sim_env=sim_env, max_steps=10)  # Placeholder call