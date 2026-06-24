"""
SUMO Benchmark: So sanh 3 phuong phap dieu khien den giao thong.
Chay tat ca 3 thuat toan lien tiep (FT, AC, MP) cho cac muc luong,
roi in ket qua + ve bieu do so sanh.

Tieu chi danh gia:
  1. Avg Queue Length        - Chieu dai hang doi trung binh
  2. Avg Waiting Time       - Thoi gian cho trung binh (s)
  3. Avg Delay/Vehicle      - Thoi gian cham trung binh (s)
  4. Total Throughput       - Tong so xe da di qua
  5. Max Queue              - Hang doi lon nhat
  6. Stopped Vehicles       - So xe dung yen trung binh
  7. Total Delay            - Tong thoi gian cham (s)
  8. Travel Time            - Thoi gian di trung binh (s)
"""

import os
import sys
import codecs
import time
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
if sys.stderr and 'utf-8' not in str(sys.stderr.encoding):
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

import numpy as np

if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
import traci

# ─── Paths ───────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_path = os.path.join(script_dir, '..', 'intersection', 'seattle', 'osm_cut_rl.sumocfg')

# ─── Simulation constants ────────────────────────────────────────────────────
tls_id = "cluster_53190763_5896114911"
GREEN_PHASES = [0, 2, 4, 6]
MIN_GREEN_STEPS = 50
MAX_GREEN_STEPS = 500
STEP_LENGTH = 0.10

detector_ids = [
    "det_428067759#0_0",   "det_428067759#0_1",   "det_428067759#0_2",
    "det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2",
    "det_428067750#0_0",   "det_428067750#0_1",   "det_428067750#0_2",
    "det_-577951513_0",    "det_-577951513_1",    "det_-577951513_2", "det_-577951513_3",
]

phase_detectors = {
    0: ["det_428067759#0_0",   "det_428067759#0_1",   "det_428067759#0_2"],
    1: ["det_428067750#0_0",   "det_428067750#0_1",   "det_428067750#0_2"],
    2: ["det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2"],
    3: ["det_-577951513_0",    "det_-577951513_1",    "det_-577951513_2", "det_-577951513_3"],
}

# ─── Scenarios ────────────────────────────────────────────────────────────────
TRAFFIC_CONFIGS = {
    'Low':    0.5,
    'Medium': 1.0,
    'High':   1.5,
}
ALGO_NAMES = {
    'FT': 'Fixed-Time',
    'AC': 'Actuated',
    'MP': 'Max-Pressure',
}


def build_sumo_cmd(scale, delay_ms=0, seed=42, headless=False, sumo_gui=True):
    sumo_exe = 'sumo-gui' if (sumo_gui and not headless) else 'sumo'
    cmd = [
        sumo_exe, '-c', sumocfg_path,
        '--step-length', str(STEP_LENGTH),
        '--seed', str(seed),
        '--scale', str(scale),
        '--start', '--quit-on-end',
    ]
    if headless:
        cmd += ['--no-step-log', '--delay', '0']
    else:
        cmd += ['--delay', str(delay_ms)]
    return cmd


def run_one_simulation(algo, scale, delay_ms=0, seed=42, headless=False, max_steps=10000):
    """Run a single simulation and return metrics dict."""
    sumo_cmd = build_sumo_cmd(scale, delay_ms, seed, headless)
    try:
        traci.start(sumo_cmd)
    except Exception as e:
        return {'error': str(e)}

    # ── State variables ────────────────────────────────────────────────────────
    current_green_idx = 0
    if algo in ('MP', 'AC'):
        traci.trafficlight.setPhase(tls_id, GREEN_PHASES[0])

    yellow_timer = 0
    green_timer  = 0

    # ── Metrics accumulators ───────────────────────────────────────────────────
    step_count      = 0
    arrived_count   = 0
    step_queues     = []
    step_stopped    = []   # vehicles with speed < 0.1 m/s
    all_waiting     = []
    all_delay       = []
    all_travel_time = []
    vehicle_log     = {}   # veh_id -> {entry_time, wait_samples, max_wait}

    t_start = time.time()

    while (traci.simulation.getMinExpectedNumber() > 0
           and traci.simulation.getTime() < max_steps * STEP_LENGTH):
        sim_time = traci.simulation.getTime()

        # ── Control logic ────────────────────────────────────────────────────
        if algo == 'FT':
            pass  # Fixed-time runs automatically

        elif algo == 'MP':
            if yellow_timer > 0:
                yellow_timer -= 1
                if yellow_timer == 0:
                    traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
                    green_timer = 0
            else:
                green_timer += 1
                action = 0
                if green_timer >= MIN_GREEN_STEPS:
                    pressures = [
                        sum(traci.lanearea.getLastStepVehicleNumber(d) for d in phase_detectors[i])
                        for i in range(4)
                    ]
                    target = int(np.argmax(pressures))
                    if target != current_green_idx or green_timer >= MAX_GREEN_STEPS:
                        action = 1
                if action == 1:
                    traci.trafficlight.setPhase(tls_id, (GREEN_PHASES[current_green_idx] + 1) % 8)
                    if target == current_green_idx:
                        target = (current_green_idx + 1) % 4
                    current_green_idx = target
                    yellow_timer = 30

        elif algo == 'AC':
            if yellow_timer > 0:
                yellow_timer -= 1
                if yellow_timer == 0:
                    traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
                    green_timer = 0
            else:
                green_timer += 1
                action = 0
                if green_timer >= MIN_GREEN_STEPS:
                    active = phase_detectors[current_green_idx]
                    if not any(traci.lanearea.getLastStepVehicleNumber(d) > 0 for d in active) \
                       or green_timer >= MAX_GREEN_STEPS:
                        action = 1
                if action == 1:
                    traci.trafficlight.setPhase(tls_id, (GREEN_PHASES[current_green_idx] + 1) % 8)
                    current_green_idx = (current_green_idx + 1) % 4
                    yellow_timer = 30

        # ── Simulation step ───────────────────────────────────────────────────
        try:
            traci.simulationStep()
        except Exception:
            break

        step_count    += 1
        arrived_count += traci.simulation.getArrivedNumber()

        # Queue (detector vehicles)
        q = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detector_ids)
        step_queues.append(q)

        # Per-vehicle metrics
        for veh_id in traci.vehicle.getIDList():
            spd   = traci.vehicle.getSpeed(veh_id)
            wait  = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            delay = traci.vehicle.getTimeLoss(veh_id)

            if veh_id not in vehicle_log:
                vehicle_log[veh_id] = {
                    'entry_time': sim_time,
                    'max_wait': 0.0,
                    'max_delay': 0.0,
                    'stopped_samples': 0,
                    'total_samples': 0,
                }
            v = vehicle_log[veh_id]
            v['max_wait'] = max(v['max_wait'], wait)
            v['max_delay'] = max(v['max_delay'], delay)
            v['total_samples'] += 1
            if spd < 0.1:
                v['stopped_samples'] += 1

        step_stopped.append(
            sum(1 for v in traci.vehicle.getIDList()
                if traci.vehicle.getSpeed(v) < 0.1)
        )

    try:
        traci.close()
    except Exception:
        pass

    elapsed = time.time() - t_start

    # ── Compile metrics ────────────────────────────────────────────────────────
    if not vehicle_log:
        return {
            'error': 'No vehicles in simulation',
            'algo': algo, 'scale': scale,
        }

    all_waiting2 = [v['max_wait']  for v in vehicle_log.values()]
    all_delay2   = [v['max_delay'] for v in vehicle_log.values()]
    all_stopped  = [v['stopped_samples'] / max(v['total_samples'], 1)
                    for v in vehicle_log.values()]

    return {
        'algo':           algo,
        'scale':          scale,
        'steps':          step_count,
        'sim_duration':   step_count * STEP_LENGTH,
        'arrived':        arrived_count,
        'avg_queue':      float(np.mean(step_queues)) if step_queues else 0.0,
        'max_queue':      float(np.max(step_queues))  if step_queues else 0.0,
        'avg_wait':       float(np.mean(all_waiting2)),
        'max_wait':       float(np.max(all_waiting2)),
        'avg_delay':      float(np.mean(all_delay2)),
        'total_delay':    float(np.sum(all_delay2)),
        'avg_stopped_pct': float(np.mean(all_stopped)) * 100,
        'elapsed':        elapsed,
    }


def run_benchmark():
    """Run full benchmark across all traffic levels and algorithms."""
    results = {}   # {traffic_level: {algo: metrics}}

    for tname, scale in TRAFFIC_CONFIGS.items():
        results[tname] = {}
        print(f"\n{'='*60}")
        print(f"  BENCHMARK: {tname} traffic (scale={scale})")
        print(f"{'='*60}")

        for algo in ['FT', 'AC', 'MP']:
            label = ALGO_NAMES[algo]
            print(f"\n  --> Running {label} ...", end='', flush=True)
            m = run_one_simulation(algo, scale, delay_ms=0, seed=42, headless=True)
            results[tname][algo] = m

            if 'error' in m:
                print(f" ERROR: {m['error']}")
            else:
                print(f" Done ({m['elapsed']:.1f}s)")

    return results


def print_summary_table(results):
    """Print formatted comparison table."""
    print("\n")
    print("=" * 90)
    print(" " * 25 + "BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    headers = [
        "Traffic", "Algorithm",
        "Steps", "Arrived",
        "Avg Queue", "Max Queue",
        "Avg Wait (s)", "Max Wait (s)",
        "Avg Delay (s)", "Total Delay (s)",
        "Stopped %",
    ]
    col_w = [8, 14, 6, 7, 10, 10, 11, 11, 11, 14, 9]

    # Print header
    line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(line)
    print("-" * 90)

    for tname in TRAFFIC_CONFIGS:
        for algo in ['FT', 'AC', 'MP']:
            m = results[tname][algo]
            if 'error' in m:
                row = [tname, ALGO_NAMES[algo], 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR', 'ERR']
            else:
                row = [
                    tname,
                    ALGO_NAMES[algo],
                    str(m['steps']),
                    str(m['arrived']),
                    f"{m['avg_queue']:.2f}",
                    f"{m['max_queue']:.1f}",
                    f"{m['avg_wait']:.2f}",
                    f"{m['max_wait']:.2f}",
                    f"{m['avg_delay']:.2f}",
                    f"{m['total_delay']:.1f}",
                    f"{m['avg_stopped_pct']:.1f}",
                ]
            print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))
        print("-" * 90)

    print()
    print("Metrics explanation:")
    print("  Avg Queue      : Mean number of vehicles in all detectors per step")
    print("  Max Queue      : Peak queue length observed")
    print("  Avg Wait (s)   : Mean waiting time per vehicle (accumulated)")
    print("  Max Wait (s)  : Maximum waiting time of any vehicle")
    print("  Avg Delay (s)  : Mean time loss per vehicle vs ideal travel")
    print("  Total Delay(s) : Sum of all vehicle delays")
    print("  Stopped %      : % of samples each vehicle was nearly stopped (<0.1 m/s)")


def print_comparison_by_metric(results):
    """Print side-by-side comparison of each metric."""
    metrics = [
        ('avg_queue',       'Avg Queue Length',       'vehicles'),
        ('max_queue',       'Max Queue Length',        'vehicles'),
        ('avg_wait',        'Avg Waiting Time',        'seconds'),
        ('max_wait',        'Max Waiting Time',        'seconds'),
        ('avg_delay',       'Avg Delay per Vehicle',   'seconds'),
        ('total_delay',     'Total Delay',             'seconds'),
        ('arrived',         'Throughput (Arrived)',     'vehicles'),
        ('avg_stopped_pct', 'Stopped Vehicle Ratio',   '%'),
    ]

    print("\n" + "=" * 80)
    print(" " * 25 + "METRIC COMPARISON")
    print("=" * 80)

    for key, label, unit in metrics:
        print(f"\n  {label} ({unit}):")
        print(f"  {'Traffic':<10}  {'Fixed-Time':>12}  {'Actuated':>12}  {'Max-Pressure':>12}  {'Best':<12}")
        print(f"  {'':10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")

        for tname in TRAFFIC_CONFIGS:
            vals = {}
            for algo in ['FT', 'AC', 'MP']:
                m = results[tname][algo]
                vals[algo] = m.get(key, 0) if 'error' not in m else float('inf')

            if key in ('arrived',):
                best = max(vals, key=vals.get)
            else:
                best = min(vals, key=vals.get)

            def fmt(v):
                return f"{v:.2f}" if isinstance(v, float) else str(v)

            print(f"  {tname:<10}  {fmt(vals['FT']):>12}  {fmt(vals['AC']):>12}  {fmt(vals['MP']):>12}  {ALGO_NAMES[best]:<12}")


def export_json(results, path):
    """Save results to JSON for later plotting."""
    # Convert non-serialisable floats
    clean = {}
    for t, algos in results.items():
        clean[t] = {}
        for a, m in algos.items():
            clean[t][a] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in m.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {path}")


def plot_charts(results):
    """Generate matplotlib comparison charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("\nmatplotlib not installed. Install with: pip install matplotlib")
        return

    traffic_levels = list(TRAFFIC_CONFIGS.keys())
    algos = ['FT', 'AC', 'MP']
    colors = {'FT': '#2196F3', 'AC': '#4CAF50', 'MP': '#FF9800'}
    labels = {'FT': 'Fixed-Time', 'AC': 'Actuated', 'MP': 'Max-Pressure'}

    # Metrics to plot
    bar_metrics = [
        ('avg_queue',     'Avg Queue Length (vehicles)'),
        ('avg_wait',      'Avg Waiting Time (s)'),
        ('avg_delay',     'Avg Delay per Vehicle (s)'),
        ('avg_stopped_pct', 'Stopped Vehicle Ratio (%)'),
        ('arrived',       'Throughput (vehicles)'),
    ]

    n_metrics = len(bar_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Traffic Signal Control: Algorithm Comparison\nSUMO Benchmark', fontsize=14, fontweight='bold')

    for idx, (metric_key, metric_label) in enumerate(bar_metrics):
        ax = axes[idx // 3, idx % 3]
        x = np.arange(len(traffic_levels))
        width = 0.25
        for i, algo in enumerate(algos):
            vals = []
            for tname in traffic_levels:
                m = results[tname][algo]
                v = m.get(metric_key, 0) if 'error' not in m else 0
                vals.append(v)
            bars = ax.bar(x + i * width, vals, width, label=labels[algo], color=colors[algo], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('Traffic Level')
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_xticks(x + width)
        ax.set_xticklabels(traffic_levels)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # Remove unused subplot
    axes[1, 2].axis('off')

    # Summary table in last subplot
    ax_tbl = axes[1, 2]
    ax_tbl.axis('off')
    table_data = []
    headers = ['Metric', 'FT', 'AC', 'MP', 'Best']
    for tname in traffic_levels:
        row = [tname]
        for algo in algos:
            m = results[tname][algo]
            if 'error' not in m:
                row.append(f"{m['avg_delay']:.2f}s")
            else:
                row.append('ERR')
        # best by avg_delay
        vals = {a: results[tname][a].get('avg_delay', 999) for a in algos
                if 'error' not in results[tname][a]}
        best = min(vals, key=vals.get) if vals else '?'
        row.append(labels[best])
        table_data.append(row)

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax_tbl.set_title('Avg Delay Summary', fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(script_dir, 'benchmark_charts.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Chart saved to: {out_path}")


def main():
    print("=" * 60)
    print("  SUMO TRAFFIC SIGNAL BENCHMARK")
    print("  Comparing: Fixed-Time | Actuated | Max-Pressure")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Scenarios: {list(TRAFFIC_CONFIGS.keys())}")
    print(f"  Algorithms: {list(ALGO_NAMES.values())}")

    results = run_benchmark()
    print_summary_table(results)
    print_comparison_by_metric(results)

    # Save data
    json_path = os.path.join(script_dir, 'benchmark_results.json')
    export_json(results, json_path)

    # Charts
    plot_charts(results)

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
