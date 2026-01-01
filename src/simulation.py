from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


def plot_q_tables(agent):
    states = [
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

    actions = ["list_job", "self_processing", "bid_0.25",
               "bid_0.5", "bid_0.75", "bid_1.0"]

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(agent.q_table, annot=True, cmap="YlGnBu", cbar=True)
    ax.set_xticklabels(actions, rotation=45)
    ax.set_yticklabels(states, rotation=0)
    plt.show()


def run_episode(sim_env, max_steps):
    states = sim_env.reset()
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        actions = {}
        for agent in sim_env.agents.values():
            actions[agent.name] = agent.choose_action(states[agent.name])

        next_state, rewards, done, info = sim_env.step(actions)

        for agent in sim_env.agents.values():
            agent.learn(
                states[agent.name], actions[agent.name],
                rewards[agent.name], next_state[agent.name]
            )

        states = next_state

        step_count += 1

    print("Final Q-tables:")
    for agent in sim_env.agents.values():
        print(f"Agent {agent.name}:")
        print(agent.epsilon)
        plot_q_tables(agent)

    print(f"Episode finished after {step_count} steps.")
