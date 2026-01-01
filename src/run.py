from simulation import run_episode
from marketenvironment import MarketEnvironment

if __name__ == '__main__':
    sim_env = MarketEnvironment()
    run_episode(sim_env=sim_env, max_steps=87600)
