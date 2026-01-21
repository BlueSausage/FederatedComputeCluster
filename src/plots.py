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

    ax = sns.heatmap(
        agent.q_table,
        annot=True,
        cmap="RdYlGn",
        center=0,
        cbar=True
    )

    ax.set_xticklabels(actions, rotation=45)
    ax.set_yticklabels(states, rotation=0)
    plt.show()


def plot_rewards(rewards):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    steps = sorted(rewards.keys())

    inner_dict = rewards[steps[0]]
    rz_keys = [key for key in inner_dict.keys() if key.startswith("RZ")]

    palette = sns.color_palette("tab10", n_colors=len(rz_keys))

    for rz, color in zip(rz_keys, palette):
        y = [rewards[s][rz] for s in steps]
        plt.plot(steps, y, label=rz, color=color, linewidth=2)

    mean = [rewards[step]["mean_reward"] for step in steps]
    plt.plot(
        steps, mean,
        label="mean_reward",
        linestyle="--",
        color="black",
        linewidth=3
    )

    social_welfare = [rewards[step]["social_welfare"] for step in steps]
    plt.plot(
        steps, social_welfare,
        label="social_welfare",
        linestyle="--",
        color="gray",
        linewidth=1.5
    )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Agent Rewards Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_reward_for(agent_name, rewards, window_size=100):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    steps = sorted(rewards.keys())

    y = [rewards[s][agent_name] for s in steps]

    # calculate moving average
    y_ma = []
    for i in range(len(y)):
        if i < window_size:
            y_ma.append(sum(y[:i+1]) / (i+1))
        else:
            y_ma.append(sum(y[i-window_size+1:i+1]) / window_size)

    plt.plot(steps, y, label=agent_name, color="blue", linewidth=2)
    plt.plot(steps, y_ma, label=f"{agent_name} (MA)", color="black", linestyle="--", linewidth=1)

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Reward Over Time for {agent_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()
