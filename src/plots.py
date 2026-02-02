import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statistics

from matplotlib.patches import Rectangle


def plot_q_tables(agent_name, q_table):
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
        q_table,
        annot=True,
        cmap="RdYlGn",
        center=0,
        cbar=True
    )

    best_actions = np.argmax(q_table, axis=1)
    for row, col in enumerate(best_actions):
        if q_table[row][col] > 0:
            rect = Rectangle(
                (col, row), 1, 1,
                fill=False,
                edgecolor="red",
                linewidth=2.5
            )
            ax.add_patch(rect)

    ax.set_xticklabels(actions, rotation=45)
    ax.set_yticklabels(states, rotation=0)
    plt.title(f"Q-Tabelle für {agent_name}", loc=)
    plt.show()


def plot_q_convergance(q_snapshots, snap_steps):
    sns.set_style("whitegrid")

    deltas = {}

    for name, snaps in q_snapshots.items():
        d = []
        for i in range(1, len(snaps)):
            diff = np.array(snaps[i]) - np.array(snaps[i-1])
            d.append(np.linalg.norm(diff))
        deltas[name] = np.array(d, dtype=float)

    x = snap_steps[1:]

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("tab10", n_colors=len(deltas))

    for (name, d), color in zip(deltas.items(), palette):
        plt.plot(x, d, label=name, color=color, linewidth=2)

    plt.xlabel("Schritte")
    plt.ylabel("L2 Änderung der Q-Tabelle")
    plt.title("Änderungen der Q-Tabelle über die Zeit")
    plt.legend(title="Agenten", loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_reward_for(agent_name, round_info, window_size=100, only_ma=False):
    print(agent_name)
    sns.set_style("whitegrid")

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

    plt.figure(figsize=(12, 6))
    if not only_ma:
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

    plt.xlabel("Schritte")
    plt.ylabel("Reward")
    plt.title(f"{agent_name} über Zeit")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_average_rewards(round_info):
    window_size = 100
    steps = sorted(round_info.keys())
    rz_keys = [k for k in round_info[steps[0]].keys() if k.startswith("RZ")]
    sns.set_style("whitegrid")

    reward = {}
    reward_ma = {}

    for rz in rz_keys:
        reward[rz] = [round_info[s][rz]["actual_reward"] for s in steps]

    # calculate moving average
    for rz in rz_keys:
        ma = []
        for i in range(len(reward[rz])):
            if i < window_size:
                ma.append(sum(reward[rz][:i+1]) / (i+1))
            else:
                ma.append(sum(reward[rz][i-window_size+1:i+1]) / window_size)
        reward_ma[rz] = ma

    plt.figure(figsize=(12, 6))

    for rz in rz_keys:
        plt.plot(
            steps,
            reward_ma[rz],
            label=f"{rz}"
        )

    plt.xlabel("Schritte")
    plt.ylabel("Durchschnitts-Reward")
    plt.title("Durchschnitts-Reward pro Agent über Zeit")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_cumulative_rewards(round_info):
    sns.set_style("whitegrid")

    steps = sorted(round_info.keys())
    rz_keys = [k for k in round_info[steps[0]].keys() if k.startswith("RZ")]
    palette = sns.color_palette("tab10", n_colors=len(rz_keys))

    final_sum = {}
    mean_values = {}

    for rz, _ in zip(rz_keys, palette):
        y = [float(round_info[s][rz]["actual_reward"]) for s in steps]
        mean_values[rz] = statistics.mean(y)
        final_sum[rz] = sum(y)

    plt.figure(figsize=(12, 6))
    agents = list(final_sum.keys())
    values = list(final_sum.values())

    bars = plt.bar(agents, values, color=palette)

    plt.bar_label(
        bars,
        labels=[f"{v:.2f}" for v in values],
        padding=3
    )

    plt.bar_label(
        bars,
        labels=[f"Mean\n{mean_values[rz]:.2f}" for rz in agents],
        label_type="center",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    plt.title("Gewinn pro Agent")
    plt.ylabel("Kumulativer Reward")
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


def plot_processed_jobs(round_info):
    sns.set_style("whitegrid")

    steps = round_info.keys()
    rz_keys = [k for k in round_info[0].keys() if k.startswith("RZ")]
    own_jobs = {rz: 0 for rz in rz_keys}
    foreign_jobs = {rz: 0 for rz in rz_keys}

    for step in steps:
        for rz in rz_keys:
            info = round_info[step][rz]

            jobs = info.get("jobs", 0)

            if jobs == 0:
                continue
            elif jobs == 1:
                own_jobs[rz] += 1
            else:
                own_jobs[rz] += 1
                foreign_jobs[rz] += 1

    self_processed = [own_jobs[rz] for rz in rz_keys]
    extra_jobs = [foreign_jobs[rz] for rz in rz_keys]

    plt.figure(figsize=(12, 6))

    own_job_bars = plt.bar(
        rz_keys, self_processed, label="Eigene Aufträge"
    )

    plt.bar_label(
        own_job_bars,
        labels=[f"{v}" for v in self_processed],
        label_type="center",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    extra_jobs_bars = plt.bar(
        rz_keys, extra_jobs, bottom=self_processed, label="Extra Aufträge"
    )

    plt.bar_label(
        extra_jobs_bars,
        labels=[f"{v}" for v in extra_jobs],
        label_type="center",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    plt.ylabel("Bearbeitete Aufträge")
    plt.title("Bearbeitete Aufträge pro Agent")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_processed_foreign_jobs(round_info):
    sns.set_style("whitegrid")

    steps = round_info.keys()
    rz_keys = [k for k in round_info[0].keys() if k.startswith("RZ")]

    jobs = {
        rz: {other_rz: 0 for other_rz in rz_keys if other_rz != rz}
        for rz in rz_keys
    }

    for step in steps:
        for rz in rz_keys:
            info = round_info[step][rz]

            foreign_job = info["job_received_from"]

            if foreign_job is None:
                continue

            jobs[rz][foreign_job] += 1

    bottom = [0] * len(rz_keys)
    palette = sns.color_palette("tab10", n_colors=len(rz_keys))

    plt.figure(figsize=(12, 6))

    for src, color in zip(rz_keys, palette):
        values = [jobs[rz].get(src, 0) for rz in rz_keys]

        bar = plt.bar(
            rz_keys,
            values,
            bottom=bottom,
            label=f"{src}",
            color=color
        )

        plt.bar_label(
            bar,
            labels=[f"{v if v != 0 else ''}" for v in values],
            label_type="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

        bottom = [b + v for b, v in zip(bottom, values)]

    plt.xlabel("Agent (job receiver)")
    plt.ylabel("Foreign jobs processed")
    plt.title("Foreign Jobs Processed by Agents")
    plt.legend(
        title="Job Source Agent",
        bbox_to_anchor=(1, 1)
    )
    plt.tight_layout()
    plt.show()


def plot_paid_on_market(round_info):
    sns.set_style("whitegrid")

    steps = round_info.keys()
    rz_keys = [k for k in round_info[0].keys() if k.startswith("RZ")]
    paid_on_market = {rz: {
                        "paid": []
                    } for rz in rz_keys}

    for step in steps:
        info = round_info[step]
        for rz in rz_keys:
            paid = info[rz]["paid"]
            if paid:
                paid_on_market[rz]["paid"].append(paid)

    plt.figure(figsize=(12, 6))
    palette = sns.color_palette("tab10", n_colors=len(rz_keys))

    sum_values = [
        sum(paid_on_market[rz]["paid"]) for rz in rz_keys
    ]

    mean_values = [
        statistics.mean(paid_on_market[rz]["paid"]) for rz in rz_keys 
        if paid_on_market[rz]["paid"]
    ]

    bars = plt.bar(rz_keys, sum_values, color=palette)

    plt.ylabel("Total Paid on Market")
    plt.title("Total Paid on Market per Agent")

    plt.bar_label(
        bars,
        labels=[f"{v:.2f}" for v in sum_values],
        padding=3
    )

    plt.bar_label(
        bars,
        labels=[f"Mean\n{v:.2f}" for v in mean_values],
        label_type="center",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


def plot_processed_vs_listed(round_info):
    sns.set_style("whitegrid")
    steps = sorted(round_info.keys())
    rz_keys = [key for key in round_info[0].keys() if key.startswith("RZ")]

    temp = {
        rz: {
            "processed": 0,
            "listed": 0
        } for rz in rz_keys
    }

    for step in steps:
        for rz in rz_keys:
            info = round_info[step][rz]
            jobs = info["jobs"]
            if jobs > 0:
                temp[rz]["processed"] += 1
            else:
                temp[rz]["listed"] += 1

    self_processed = [temp[rz]["processed"] for rz in rz_keys]
    listed = [temp[rz]["listed"] for rz in rz_keys]

    plt.figure(figsize=(12, 6))

    self_processed_bar = plt.bar(
        rz_keys, self_processed, label="Selbst bearbeitet"
    )

    plt.bar_label(
        self_processed_bar,
        labels=[f"{v}" for v in self_processed],
        label_type="center",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    listed_bar = plt.bar(
        rz_keys, listed, bottom=self_processed, label="Gelistet"
    )

    plt.bar_label(
        listed_bar,
        labels=[f"{v}" for v in listed],
        label_type="center",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    plt.ylabel("Anzahl Aufträge")
    plt.title("Bearbeitete vs. gelistete Aufträge")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
