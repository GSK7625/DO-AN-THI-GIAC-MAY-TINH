import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(filename):
    steps, cumulative_rewards, queue_lengths = [], [], []
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs', filename)
    if not os.path.exists(filepath):
        print(f"Warning: File {filename} not found.")
        return None, None, None
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                steps.append(int(row[0]))
                cumulative_rewards.append(float(row[1]))
                queue_lengths.append(float(row[2]))
    return steps, cumulative_rewards, queue_lengths

def make_continuous_reward(steps, rewards):
    """
    Convert per-episode cumulative rewards into one continuous series.
    Each episode resets the reward internally; we detect the jump upward
    and carry the previous episode's final value as an offset.
    """
    if not rewards:
        return steps, rewards
    continuous = []
    offset = 0.0
    prev = rewards[0]
    for r in rewards:
        # Episode reset: reward jumps significantly upward (toward 0)
        if r > prev + 100:
            offset += prev
        continuous.append(offset + r)
        prev = r
    return steps, continuous

def main():
    print("=== Traffic Light Control Comparison on OSM Map ===")

    ft_steps,  ft_rewards,  ft_queues  = load_metrics('osm_ft_metrics.csv')
    mp_steps,  mp_rewards,  mp_queues  = load_metrics('osm_mp_metrics.csv')
    ac_steps,  ac_rewards,  ac_queues  = load_metrics('osm_actuated_metrics.csv')

    if ft_steps is None or mp_steps is None or ac_steps is None:
        print("Error: Run all 3 scripts (FT, MP, Actuated) first.")
        return

    # Build continuous cumulative reward (no episode resets)
    ft_steps,  ft_cont  = make_continuous_reward(ft_steps,  ft_rewards)
    mp_steps,  mp_cont  = make_continuous_reward(mp_steps,  mp_rewards)
    ac_steps,  ac_cont  = make_continuous_reward(ac_steps,  ac_rewards)

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(base, exist_ok=True)

    # --- Plot 1: Continuous Cumulative Reward ---
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ft_steps,  ft_cont,  label="Fixed Timing (FT)",     color="blue",   linewidth=1.8)
    ax.plot(mp_steps,  mp_cont,  label="Max-Pressure (MP)",     color="orange", linewidth=1.8)
    ax.plot(ac_steps,  ac_cont,  label="Actuated Control (AC)", color="purple",  linewidth=1.8)
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Cumulative Reward (continuous)")
    ax.set_title("OSM Map: Cumulative Reward Comparison (Higher is Better)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    reward_path = os.path.join(base, 'osm_comparison_cumulative_reward.png')
    fig.savefig(reward_path, dpi=120)
    print(f"Saved reward plot -> {reward_path}")
    plt.close(fig)

    # --- Plot 2: Queue Length with moving average ---
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    window = 5
    datasets = [
        (ft_steps,  ft_queues,  "Fixed Timing (FT)",     "blue",   "darkblue"),
        (mp_steps,  mp_queues,  "Max-Pressure (MP)",     "orange", "darkorange"),
        (ac_steps,  ac_queues,  "Actuated Control (AC)", "purple", "purple"),
    ]
    for steps, queues, label, raw_c, ma_c in datasets:
        ax2.plot(steps, queues, color=raw_c, alpha=0.2, linewidth=1)
        if len(queues) > window:
            ma = np.convolve(queues, np.ones(window)/window, mode='valid')
            ax2.plot(steps[window-1:], ma, label=f"{label} (MA-{window})",
                     color=ma_c, linewidth=2.2)
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Avg Queue Length (vehicles)")
    ax2.set_title("OSM Map: Queue Length Comparison (Lower is Better)")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    fig2.tight_layout()
    queue_path = os.path.join(base, 'osm_comparison_queue_length.png')
    fig2.savefig(queue_path, dpi=120)
    print(f"Saved queue plot  -> {queue_path}")
    plt.close(fig2)

    # --- Summary ---
    print("\n=== SUMMARY STATISTICS ===")
    print(f"{'Method':20} | {'Avg Queue':12} | {'Final Cum. Reward':20}")
    print("-" * 58)
    for name, queues, cont in [
        ("Fixed Timing",     ft_queues,  ft_cont),
        ("Max-Pressure",     mp_queues,  mp_cont),
        ("Actuated Control", ac_queues,  ac_cont),
    ]:
        print(f"{name:20} | {np.mean(queues):12.2f} | {cont[-1]:20.1f}")
    print("-" * 58)

if __name__ == '__main__':
    main()
