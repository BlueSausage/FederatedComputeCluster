from pprint import pprint


def run_episode(sim_env, max_steps):
    states = sim_env.reset()
    done = False
    step_count = 0

    print("Initial Q-tables:")
    for agent in sim_env.agents.values():
        print(f"Agent {agent.name}:")
        print(agent.epsilon)
        pprint(agent.q_table)

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

    print("==================================")

    print("Final Q-tables:")
    for agent in sim_env.agents.values():
        print(f"Agent {agent.name}:")
        print(agent.epsilon)
        pprint(agent.q_table)

    print(f"Episode finished after {step_count} steps.")
