def run_episode(sim_env, max_steps):
    states = sim_env.reset()
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        actions = {}
        for agent in sim_env.agents.values():
            actions[agent.name] = agent.choose_action(states[agent.name])

        next_state, rewards, done, info = sim_env.step(actions)

        states = next_state
        step_count += 1
