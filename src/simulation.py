from marketenvironment import MarketEnvironment
import statistics


def run_episode(sim_env: MarketEnvironment, max_steps: int):
    states = sim_env.reset()
    done = False
    step_count = 0

    round_info = {}

    snapshot_intervall = 100
    q_snapshots = {name: [] for name in sim_env.agents.keys()}
    snap_steps = []

    while not done and step_count < max_steps:
        actions = {}
        for agent in sim_env.agents.values():
            actions[agent.name] = agent.choose_action(states[agent.name])

        next_state, rewards, done, info = sim_env.step(actions)

        round_info[step_count] = rewards
        round_info[step_count]["mean_reward"] = statistics.mean(
            rewards.values()
        )
        round_info[step_count]["social_welfare"] = sum(rewards.values())
        round_info[step_count]["price"] = info["price"]
        round_info[step_count]["bids"] = info["bids"]

        for agent in sim_env.agents.values():
            agent.learn(
                states[agent.name], actions[agent.name],
                rewards[agent.name], next_state[agent.name]
            )
            round_info[step_count][agent.name] = info[agent.name]

        states = next_state
        step_count += 1

        if step_count % snapshot_intervall == 0:
            snap_steps.append(step_count)
            for agent in sim_env.agents.values():
                q_snapshots[agent.name].append(agent.q_table.copy())

    return sim_env, round_info, q_snapshots, snap_steps
