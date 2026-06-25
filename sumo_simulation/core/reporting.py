import os
import csv
import matplotlib.pyplot as plt
import numpy as np


STRATEGIES   = ['FT', 'AC', 'MP']
COLORS       = ['#3498db', '#9b59b6', '#e67e22']   # Blue, Purple, Orange

METRIC_LABELS = {
    'avg_queue':   'Average Queue Length (vehicles)',
    'avg_wait':    'Average Waiting Time (seconds)',
    'throughput':  'Total Throughput (vehicles)',
    'total_delay': 'Total Time Loss/Delay (seconds)',
}


def calculate_los(avg_delay: float) -> str:
    """Tính mức dịch vụ LOS theo HCM dựa trên thời gian trễ trung bình.
    
    Args:
        avg_delay: Thời gian trễ trung bình mỗi xe (giây)
    
    Returns:
        LOS: 'A', 'B', 'C', 'D', 'E', hoặc 'F'
    """
    if avg_delay <= 10.0:
        return 'A'
    elif avg_delay <= 20.0:
        return 'B'
    elif avg_delay <= 35.0:
        return 'C'
    elif avg_delay <= 55.0:
        return 'D'
    elif avg_delay <= 80.0:
        return 'E'
    else:
        return 'F'

def save_real_comparison_csv(results: dict, csv_path: str) -> None:
    first = results[STRATEGIES[0]]
    has_std = 'std_avg_queue' in first

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if has_std:
            writer.writerow([
                'Strategy',
                'AvgQueueLength', 'StdQueueLength',
                'AvgWaitingTime', 'StdWaitingTime',
                'Throughput',     'StdThroughput',
                'TotalDelay',     'StdTotalDelay',
                'AvgDelay',       'StdAvgDelay',
                'LOS',
            ])
            for strat in STRATEGIES:
                m = results[strat]
                writer.writerow([
                    strat,
                    round(m['avg_queue'],         2), round(m['std_avg_queue'],         2),
                    round(m['avg_wait'],           2), round(m['std_avg_wait'],           2),
                    round(m['throughput'],         0), round(m['std_throughput'],         0),
                    round(m['total_delay'],        1), round(m['std_total_delay'],        1),
                    round(m['avg_delay'],          2), round(m['std_avg_delay'],          2),
                    calculate_los(m['avg_delay']),
                ])
        else:
            writer.writerow(['Strategy', 'AvgQueueLength', 'AvgWaitingTime',
                             'Throughput', 'TotalDelay', 'AvgDelay', 'LOS'])
            for strat in STRATEGIES:
                m = results[strat]
                writer.writerow([
                    strat,
                    round(m['avg_queue'],   2),
                    round(m['avg_wait'],    2),
                    m['throughput'],
                    round(m['total_delay'], 1),
                    round(m['avg_delay'],   2),
                    calculate_los(m['avg_delay']),
                ])
    print(f"  CSV saved → {csv_path}")


def save_scenario_comparison_csv(results: dict, csv_path: str) -> None:
    """Lưu kết quả so sánh kịch bản (3 scale × 3 strategy) ra CSV.
    
    Args:
        results:  dict {(scale, strategy): metrics_dict}
        csv_path: Đường dẫn file CSV xuất ra
    """
    scale_names = {0.5: 'Low', 1.0: 'Medium', 1.5: 'High'}
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ScenarioScale', 'ScenarioName', 'Strategy',
                         'AvgQueueLength', 'AvgWaitingTime', 'Throughput',
                         'TotalDelay', 'AvgDelay', 'LOS'])
        for (scale, strat), m in results.items():
            writer.writerow([
                scale,
                scale_names.get(scale, str(scale)),
                strat,
                round(m['avg_queue'], 2),
                round(m['avg_wait'], 2),
                m['throughput'],
                round(m['total_delay'], 1),
                round(m['avg_delay'], 2),
                calculate_los(m['avg_delay']),
            ])
    print(f"  CSV saved → {csv_path}")


def generate_bar_charts(results: dict, outputs_dir: str, prefix: str = 'real_comparison') -> None:
    for metric_key, label in METRIC_LABELS.items():
        values = [results[strat][metric_key] for strat in STRATEGIES]
        _save_bar_chart(
            labels=STRATEGIES,
            values=values,
            colors=COLORS,
            title=f"Real-World Traffic: {label}",
            ylabel=label.split(' (')[0],
            filename=f"{prefix}_{metric_key}.png",
            outputs_dir=outputs_dir,
            is_int=(metric_key == 'throughput'),
        )


def generate_grouped_charts(results: dict, outputs_dir: str, prefix: str = 'scenario_comparison') -> None:
    scales = [0.5, 1.0, 1.5]
    scenario_labels = ['Thấp (Scale 0.5)', 'Trung bình (Scale 1.0)', 'Cao (Scale 1.5)']
    x = np.arange(len(scales))
    width = 0.25

    metric_titles_vi = {
        'avg_queue':   'Độ dài hàng đợi trung bình (xe)',
        'avg_wait':    'Thời gian chờ trung bình (s)',
        'throughput':  'Tổng xe thông qua (Throughput)',
        'total_delay': 'Tổng thời gian trễ (s)',
    }

    for metric_key, title in metric_titles_vi.items():
        fig, ax = plt.subplots(figsize=(9, 6))
        for idx, strat in enumerate(STRATEGIES):
            values = [results[(scale, strat)][metric_key] for scale in scales]
            ax.bar(x + (idx - 1) * width, values, width,
                   label=strat, color=COLORS[idx], edgecolor='black', alpha=0.85)

        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Kịch bản lưu lượng', fontsize=12, labelpad=10)
        ax.set_ylabel(title.split(' (')[0], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, fontsize=11)
        ax.legend(frameon=True, facecolor='white', edgecolor='gray')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.tight_layout()

        filename = f"{prefix}_{metric_key}.png"
        plt.savefig(os.path.join(outputs_dir, filename), dpi=150)
        plt.close()
        print(f"  Chart saved → {filename}")


def _save_bar_chart(labels, values, colors, title, ylabel, filename, outputs_dir, is_int=False):
    """Helper nội bộ: tạo và lưu một biểu đồ bar đơn."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        label_text = f'{int(h)}' if is_int else f'{h:.2f}'
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + (h * 0.01 + 0.1),
            label_text,
            ha='center', va='bottom', fontweight='bold',
        )
    plt.title(title, fontsize=12, fontweight='bold', pad=15)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, filename), dpi=150)
    plt.close()
    print(f"  Chart saved → {filename}")
