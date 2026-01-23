import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.patches import Rectangle


def plot_q_tables(agent):
    states = [
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

    best_actions = np.argmax(agent.q_table, axis=1)
    for row, col in enumerate(best_actions):
        if agent.q_table[row][col] > 0:
            rect = Rectangle(
                (col, row), 1, 1,
                fill=False,
                edgecolor="red",
                linewidth=2.5
            )
            ax.add_patch(rect)

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
    plt.plot(
        steps, y_ma,
        label=f"{agent_name} (MA)",
        color="black",
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Reward Over Time for {agent_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()
