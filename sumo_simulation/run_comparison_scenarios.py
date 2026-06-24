"""
run_comparison_scenarios.py
===========================
So sánh FT / AC / MP trên ba kịch bản lưu lượng (thấp / trung bình / cao).

Chạy:
    python run_comparison_scenarios.py
"""

import os
import sys
import random
import numpy as np

# ------------------------------------------------------------------
# SUMO_HOME setup
# ------------------------------------------------------------------
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# ------------------------------------------------------------------
# Core modules
# ------------------------------------------------------------------
from core.simulator  import run_simulation
from core.reporting  import (
    calculate_los,
    save_scenario_comparison_csv,
    generate_grouped_charts,
    STRATEGIES,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Cấu hình seed ngẫu nhiên
# ------------------------------------------------------------------
NUM_SEEDS    = 5
random.seed()                                      # Dùng entropy hệ thống
RANDOM_SEEDS = random.sample(range(1, 100_000), NUM_SEEDS)


def build_sumo_cmd(scale: float, seed: int) -> list:
    """Tạo lệnh SUMO với scale và seed cụ thể."""
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
    """Tính trung bình và độ lệch chuẩn của các metrics qua nhiều lần chạy."""
    keys = ['avg_queue', 'avg_wait', 'throughput', 'total_delay', 'avg_delay']
    averaged = {}
    for key in keys:
        vals = [r[key] for r in runs]
        averaged[key]          = float(np.mean(vals))
        averaged[f'std_{key}'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return averaged


def build_markdown_report(results: dict, seeds: list) -> str:
    """Tạo nội dung báo cáo Markdown theo kịch bản (mean ± std)."""

    def table(scale):
        header = (
            "| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) "
            "| Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) | LOS |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        rows = []
        for strat in STRATEGIES:
            m   = results[(scale, strat)]
            los = calculate_los(m['avg_delay'])
            rows.append(
                f"| {strat} "
                f"| {m['avg_queue']:.2f} ± {m['std_avg_queue']:.2f} "
                f"| {m['avg_wait']:.2f} ± {m['std_avg_wait']:.2f} s "
                f"| {m['throughput']:.0f} ± {m['std_throughput']:.0f} "
                f"| {m['total_delay']:.1f} ± {m['std_total_delay']:.1f} s "
                f"| {m['avg_delay']:.2f} ± {m['std_avg_delay']:.2f} s "
                f"| **{los}** |"
            )
        return header + "\n" + "\n".join(rows)

    seeds_str = ', '.join(str(s) for s in seeds)

    return f"""# Báo cáo so sánh kịch bản điều khiển giao thông

Báo cáo đánh giá **Fixed-Time (FT)**, **Actuated Control (AC)** và **Max-Pressure (MP)**
trên ba kịch bản lưu lượng:
1. **Lưu lượng thấp** (Scale = 0.5)
2. **Lưu lượng trung bình** (Scale = 1.0)
3. **Lưu lượng cao** (Scale = 1.5)

> **Phương pháp thống kê**: Mỗi thuật toán / kịch bản được chạy {NUM_SEEDS} lần với seed ngẫu nhiên khác nhau.
> Kết quả hiển thị dưới dạng **giá trị trung bình ± độ lệch chuẩn**.
> Seeds đã dùng: `{seeds_str}`

---

## 1. Kết quả chi tiết theo từng Kịch bản (mean ± std)

### Kịch bản 1: Lưu lượng thấp (Scale = 0.5)
{table(0.5)}

### Kịch bản 2: Lưu lượng trung bình (Scale = 1.0)
{table(1.0)}

### Kịch bản 3: Lưu lượng cao (Scale = 1.5)
{table(1.5)}

---

## 2. Biểu đồ trực quan hóa hiệu năng

### 2.1 Độ dài hàng đợi trung bình
![Average Queue Length](scenario_comparison_avg_queue.png)

### 2.2 Thời gian chờ trung bình
![Average Waiting Time](scenario_comparison_avg_wait.png)

### 2.3 Tổng xe thông qua
![Throughput](scenario_comparison_throughput.png)

### 2.4 Tổng thời gian trễ
![Total Delay](scenario_comparison_total_delay.png)

---

## 3. Đánh giá và Phân tích kỹ thuật

1. **Độ ổn định thống kê**: Mỗi cặp (kịch bản, thuật toán) được đánh giá qua {NUM_SEEDS} lần chạy độc lập
   với seed ngẫu nhiên khác nhau mỗi phiên, đảm bảo kết quả khách quan.

2. **Lưu lượng thấp (Scale 0.5)**: AC và MP không lãng phí pha xanh cho hướng trống,
   giảm đáng kể hàng đợi và thời gian chờ so với FT.

3. **Lưu lượng trung bình (Scale 1.0)**: AC tối ưu theo hiện diện thực tế của xe.
   MP bắt đầu thể hiện ưu thế phân bổ đều áp lực hàng đợi.

4. **Lưu lượng cao (Scale 1.5)**: MP vượt trội nhờ trực tiếp giải tỏa hướng có
   hàng đợi lớn nhất, ngăn tắc nghẽ cục bộ kéo dài.
"""


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

    # Markdown report
    report_path = os.path.join(OUTPUTS_DIR, 'scenario_comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(build_markdown_report(results, RANDOM_SEEDS))
    print(f"  Markdown report saved → {report_path}")

    print("\n=== SIMULATIONS COMPLETED SUCCESSFULLY ===")


if __name__ == '__main__':
    main()
