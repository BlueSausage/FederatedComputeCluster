import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics

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


def plot_q_convergance(q_snapshots, snap_steps):
    sns.set_style("whitegrid")

    deltas = {}

    for name, snaps in q_snapshots.items():
        d = []
        for i in range(1, len(snaps)):
            diff = snaps[i] - snaps[i-1]
            d.append(np.linalg.norm(diff))  # 
        deltas[name] = np.array(d, dtype=float)

    x = snap_steps[1:]

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("tab10", n_colors=len(deltas))

    for (name, d), color in zip(deltas.items(), palette):
        plt.plot(x, d, label=name, color=color, linewidth=2)

    plt.xlabel("Steps")
    plt.ylabel("L2 change of Q-Table")
    plt.title("Q-table changes over time")
    plt.legend(title="Agents", loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_reward_for(agent_name, round_info, window_size=100):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    steps = sorted(round_info.keys())

    if agent_name.startswith("RZ"):
        reward = [round_info[s][agent_name]["actual_reward"] for s in steps]
    else:
        reward = [round_info[s][agent_name] for s in steps]

    # calculate moving average
    reward_ma = []
    for i in range(len(reward)):
        if i < window_size:
            reward_ma.append(sum(reward[:i+1]) / (i+1))
        else:
            reward_ma.append(sum(reward[i-window_size+1:i+1]) / window_size)

    plt.plot(
        steps, reward,
        label=agent_name,
        color="blue",
        linewidth=2
    )

    plt.plot(
        steps, reward_ma,
        label=f"{agent_name} (MA)",
        color="black",
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"Reward Over Time for {agent_name}")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_cumulative_rewards(round_info):
    sns.set_style("whitegrid")

    steps = sorted(round_info.keys())
    rz_keys = [k for k in round_info[steps[0]].keys() if k.startswith("RZ")]
    palette = sns.color_palette("tab10", n_colors=len(rz_keys))

    final_sum = {}

    for rz, _ in zip(rz_keys, palette):
        y = [float(round_info[s][rz]["actual_reward"]) for s in steps]
        y_cum = np.cumsum(y)
        final_sum[rz] = y_cum[-1].item()

    plt.figure(figsize=(12, 6))
    agents = list(final_sum.keys())
    values = list(final_sum.values())

    plt.bar(agents, values, color=palette)

    plt.ylabel("Final Cumulative Reward")
    plt.title("Final Cumulative Rewards per Agent")
    plt.tight_layout()
    plt.show()


def plot_price(round_info, window_size=100):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    steps = sorted(round_info.keys())

    price = [v["price"] for v in round_info.values()]
    price_ma = []
    for i in range(len(price)):
        if i < window_size:
            mean_price = statistics.mean(price[:i+1])
        else:
            mean_price = statistics.mean(price[i-window_size+1:i+1])
        price_ma.append(mean_price)

    plt.plot(
        steps, price,
        label="Price",
        color="green",
        linewidth=2
    )

    plt.plot(
        steps, price_ma,
        label="Price (MA)",
        color="black",
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.title("Price Over Time")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
