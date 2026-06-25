import os
import sys
import random
import numpy as np

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
from core.simulator  import run_simulation
from core.reporting  import (
    save_scenario_comparison_csv,
    generate_grouped_charts,
    STRATEGIES,
)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, 'configs', 'osm_cut_rl.sumocfg')
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

GUI_MODE    = False
SUMO_BINARY = 'sumo-gui' if GUI_MODE else 'sumo'

# 3 kịch bản lưu lượng
SCENARIOS = [
    (0.5, 'Low'),
    (1.0, 'Medium'),
    (1.5, 'High'),
]

NUM_SEEDS    = 5
random.seed()
RANDOM_SEEDS = random.sample(range(1, 100_000), NUM_SEEDS)


def build_sumo_cmd(scale: float, seed: int) -> list:
    return [
        SUMO_BINARY,
        '-c', SUMOCFG_PATH,
        '--step-length', '0.10',
        '--delay', '0',
        '--lateral-resolution', '0',
        '--seed', str(seed),
        '--scale', str(scale),
    ]


def average_metrics(runs: list) -> dict:
    keys = ['avg_queue', 'avg_wait', 'throughput', 'total_delay', 'avg_delay']
    averaged = {}
    for key in keys:
        vals = [r[key] for r in runs]
        averaged[key]          = float(np.mean(vals))
        averaged[f'std_{key}'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return averaged



def main():
    print("=== STARTING SCENARIO COMPARISON SIMULATIONS ===")
    print(f"  Seeds ngẫu nhiên ({NUM_SEEDS} lần chạy): {RANDOM_SEEDS}\n")

    # all_runs[(scale, strat)] = [metrics_run1, metrics_run2, ...]
    all_runs: dict = {(scale, strat): [] for scale, _ in SCENARIOS for strat in STRATEGIES}

    for seed_idx, seed in enumerate(RANDOM_SEEDS, start=1):
        print(f"--- Seed {seed_idx}/{NUM_SEEDS}: seed={seed} ---")
        for scale, name in SCENARIOS:
            print(f"  Scenario: {name} Traffic Flow (Scale: {scale:.1f})")
            sumo_cmd = build_sumo_cmd(scale, seed)
            for strat in STRATEGIES:
                metrics = run_simulation(sumo_cmd, strat)
                all_runs[(scale, strat)].append(metrics)
                m = metrics
                print(
                    f"    {strat} → AvgQueue: {m['avg_queue']:.2f}, "
                    f"AvgWait: {m['avg_wait']:.2f}s, "
                    f"Throughput: {m['throughput']} vehs, "
                    f"TotalDelay: {m['total_delay']:.1f}s"
                )
        print()

    # Tổng hợp kết quả trung bình
    results = {key: average_metrics(runs) for key, runs in all_runs.items()}

    print("=== KẾT QUẢ TRUNG BÌNH QUA CÁC SEED ===")
    for scale, name in SCENARIOS:
        print(f"  [{name}]")
        for strat in STRATEGIES:
            m = results[(scale, strat)]
            print(
                f"    {strat} → AvgQueue: {m['avg_queue']:.2f} ± {m['std_avg_queue']:.2f}, "
                f"AvgWait: {m['avg_wait']:.2f} ± {m['std_avg_wait']:.2f}s, "
                f"Throughput: {m['throughput']:.0f} ± {m['std_throughput']:.0f} vehs, "
                f"TotalDelay: {m['total_delay']:.1f} ± {m['std_total_delay']:.1f}s"
            )

    # CSV
    csv_path = os.path.join(OUTPUTS_DIR, 'scenario_comparison_results.csv')
    save_scenario_comparison_csv(results, csv_path)

    # Charts
    print("\nGenerating comparison charts...")
    generate_grouped_charts(results, OUTPUTS_DIR, prefix='scenario_comparison')



    print("\n=== SIMULATIONS COMPLETED SUCCESSFULLY ===")


if __name__ == '__main__':
    main()
