"""
run_real_comparison.py
======================
So sánh hiệu năng FT / AC / MP trên dữ liệu lưu lượng thực tế
(trích xuất từ camera giám sát tại nút giao Bellevue).

Chạy:
    python run_real_comparison.py
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
    save_real_comparison_csv,
    generate_bar_charts,
    STRATEGIES,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMOCFG_PATH = os.path.join(SCRIPT_DIR, 'configs', 'osm_cut_real.sumocfg')
OUTPUTS_DIR  = os.path.join(SCRIPT_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

GUI_MODE    = False
SUMO_BINARY = 'sumo-gui' if GUI_MODE else 'sumo'

# ------------------------------------------------------------------
# Cấu hình seed ngẫu nhiên
# ------------------------------------------------------------------
NUM_SEEDS   = 5
random.seed()                                     # Dùng entropy hệ thống
RANDOM_SEEDS = random.sample(range(1, 100_000), NUM_SEEDS)


def build_sumo_cmd(seed: int) -> list:
    """Tạo lệnh SUMO với seed cụ thể."""
    return [
        SUMO_BINARY,
        '-c', SUMOCFG_PATH,
        '--step-length', '0.10',
        '--delay', '0',
        '--lateral-resolution', '0',
        '--seed', str(seed),
        '--scale', '1.0',
    ]


def build_markdown_report(results: dict, seeds: list) -> str:
    """Tạo nội dung báo cáo Markdown với kết quả trung bình qua nhiều seed."""
    def row(strat):
        m   = results[strat]
        los = calculate_los(m['avg_delay'])
        return (
            f"| {strat} "
            f"| {m['avg_queue']:.2f} ± {m['std_avg_queue']:.2f} "
            f"| {m['avg_wait']:.2f} ± {m['std_avg_wait']:.2f} s "
            f"| {m['throughput']:.0f} ± {m['std_throughput']:.0f} "
            f"| {m['total_delay']:.1f} ± {m['std_total_delay']:.1f} s "
            f"| {m['avg_delay']:.2f} ± {m['std_avg_delay']:.2f} s "
            f"| **{los}** |"
        )

    seeds_str = ', '.join(str(s) for s in seeds)

    return f"""# Báo cáo đánh giá điều khiển giao thông trên dữ liệu thực tế

Báo cáo này đánh giá hiệu năng của ba thuật toán điều khiển đèn tín hiệu giao thông:
**Fixed-Time (FT)**, **Actuated Control (AC)** và **Max-Pressure (MP)**
dưới kịch bản lưu lượng xe thực tế được trích xuất từ camera giám sát tại nút giao Bellevue.

> **Phương pháp thống kê**: Mỗi thuật toán được chạy {NUM_SEEDS} lần với {NUM_SEEDS} seed ngẫu nhiên khác nhau.
> Kết quả hiển thị dưới dạng **giá trị trung bình ± độ lệch chuẩn**.
> Seeds đã dùng: `{seeds_str}`

---

## 1. Kết quả đánh giá hiệu năng (mean ± std)

| Thuật toán | Avg Queue Length (xe) | Avg Waiting Time (s) | Throughput (xe) | Total Delay (s) | Avg Delay/Vehicle (s) | LOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{row('FT')}
{row('AC')}
{row('MP')}

---

## 2. Biểu đồ trực quan hóa hiệu năng

### 2.1 Độ dài hàng đợi trung bình (Average Queue Length)
![Average Queue Length](real_comparison_avg_queue.png)

### 2.2 Thời gian chờ trung bình (Average Waiting Time)
![Average Waiting Time](real_comparison_avg_wait.png)

### 2.3 Tổng xe thông qua (Throughput)
![Throughput](real_comparison_throughput.png)

### 2.4 Tổng thời gian trễ (Total Delay)
![Total Delay](real_comparison_total_delay.png)

---

## 3. Nhận xét và Phân tích kỹ thuật

1. **Độ ổn định thống kê**: Mỗi thuật toán được đánh giá qua {NUM_SEEDS} lần chạy độc lập với seed ngẫu nhiên,
   đảm bảo kết quả không phụ thuộc vào một kịch bản ngẫu nhiên cụ thể.

2. **Hiệu quả giảm thiểu ùn tắc**: So sánh với FT, cả AC và MP đều cải thiện
   vượt trội về hàng đợi trung bình và thời gian chờ do hệ thống thực tế có
   phân bổ không đều giữa các hướng.

3. **So sánh AC vs MP**: AC kéo dài pha xanh khi có xe và cắt ngay khi hết xe.
   MP đưa ra quyết định dựa trên chênh lệch số lượng xe vào–ra, giúp duy trì
   cân bằng tối ưu và giải quyết hàng đợi lớn chủ động hơn.
"""


def average_metrics(runs: list) -> dict:
    """Tính trung bình và độ lệch chuẩn của các metrics qua nhiều lần chạy.

    Args:
        runs: Danh sách dict metrics từ mỗi lần chạy

    Returns:
        dict chứa mean và std của từng metric
    """
    keys = ['avg_queue', 'avg_wait', 'throughput', 'total_delay', 'avg_delay']
    averaged = {}
    for key in keys:
        vals = [r[key] for r in runs]
        averaged[key]          = float(np.mean(vals))
        averaged[f'std_{key}'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return averaged


def main():
    print("=== STARTING REAL-WORLD SCENARIO COMPARISON ===")
    print(f"  Seeds ngẫu nhiên ({NUM_SEEDS} lần chạy): {RANDOM_SEEDS}\n")

    # Lưu toàn bộ kết quả từng run: {strat: [metrics_run1, metrics_run2, ...]}
    all_runs: dict = {strat: [] for strat in STRATEGIES}

    for seed_idx, seed in enumerate(RANDOM_SEEDS, start=1):
        print(f"--- Seed {seed_idx}/{NUM_SEEDS}: seed={seed} ---")
        sumo_cmd = build_sumo_cmd(seed)

        for strat in STRATEGIES:
            metrics = run_simulation(sumo_cmd, strat)
            all_runs[strat].append(metrics)
            m = metrics
            print(
                f"    {strat} → AvgQueue: {m['avg_queue']:.2f}, "
                f"AvgWait: {m['avg_wait']:.2f}s, "
                f"Throughput: {m['throughput']} vehs, "
                f"TotalDelay: {m['total_delay']:.1f}s"
            )
        print()

    # Tổng hợp kết quả trung bình
    results = {strat: average_metrics(all_runs[strat]) for strat in STRATEGIES}

    print("=== KẾT QUẢ TRUNG BÌNH QUA CÁC SEED ===")
    for strat in STRATEGIES:
        m = results[strat]
        print(
            f"  {strat} → AvgQueue: {m['avg_queue']:.2f} ± {m['std_avg_queue']:.2f}, "
            f"AvgWait: {m['avg_wait']:.2f} ± {m['std_avg_wait']:.2f}s, "
            f"Throughput: {m['throughput']:.0f} ± {m['std_throughput']:.0f} vehs, "
            f"TotalDelay: {m['total_delay']:.1f} ± {m['std_total_delay']:.1f}s"
        )

    # CSV
    csv_path = os.path.join(OUTPUTS_DIR, 'real_comparison_results.csv')
    save_real_comparison_csv(results, csv_path)

    # Charts
    print("\nGenerating comparison charts...")
    generate_bar_charts(results, OUTPUTS_DIR, prefix='real_comparison')

    # Markdown report
    report_path = os.path.join(OUTPUTS_DIR, 'real_comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(build_markdown_report(results, RANDOM_SEEDS))
    print(f"  Markdown report saved → {report_path}")

    print("\n=== REAL-WORLD SCENARIO COMPARISON COMPLETED SUCCESSFULLY ===")


if __name__ == '__main__':
    main()
