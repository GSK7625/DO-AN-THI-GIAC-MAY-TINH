"""
watch_simulation.py
===================
Giao diện tương tác để xem mô phỏng trực tiếp trong SUMO-GUI.
Người dùng chọn kịch bản lưu lượng, thuật toán, và tốc độ mô phỏng.

Chạy:
    python watch_simulation.py
"""

import os
import sys

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
from core.simulator import run_simulation_interactive

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_RL     = os.path.join(SCRIPT_DIR, 'configs', 'osm_cut_rl.sumocfg')
CFG_REAL   = os.path.join(SCRIPT_DIR, 'configs', 'osm_cut_real.sumocfg')


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------
def get_menu_choice(title: str, options: dict) -> str:
    """Hiển thị menu và yêu cầu người dùng chọn một tùy chọn hợp lệ."""
    print(f"\n=== {title} ===")
    for k, v in options.items():
        print(f"  {k}. {v}")
    while True:
        choice = input("Nhập lựa chọn của bạn: ").strip()
        if choice in options:
            return choice
        print("Lựa chọn không hợp lệ, vui lòng nhập lại.")


def get_delay_input(default: int = 50) -> int:
    """Yêu cầu người dùng nhập độ trễ mô phỏng (ms/bước)."""
    print(f"\n=== CẤU HÌNH ĐỘ TRỄ MÔ PHỎNG (ms) ===")
    val = input(f"Nhập độ trễ mỗi bước (ms, mặc định {default}): ").strip()
    return int(val) if val.isdigit() else default


def print_summary(metrics: dict, scen_name: str, algo_name: str) -> None:
    """In kết quả tổng kết sau khi mô phỏng kết thúc."""
    print("\n==================================================")
    print("          KẾT QUẢ MÔ PHỎNG (SUMMARY METRICS)      ")
    print("==================================================")
    print(f" - Kịch bản:                  {scen_name}")
    print(f" - Thuật toán:                {algo_name}")
    print(f" - Tổng số bước chạy:         {metrics['step_count']}")
    print(f" - Hàng đợi TB (Avg Queue):   {metrics['avg_queue']:.2f} xe")
    print(f" - Thời gian chờ TB:          {metrics['avg_wait']:.2f} giây")
    print(f" - Tổng xe thông qua:         {metrics['throughput']} xe")
    print(f" - Tổng thời gian trễ:        {metrics['total_delay']:.1f} giây")
    print(f" - Thời gian trễ TB/xe:       {metrics['avg_delay']:.2f} giây")
    print("==================================================")


# ------------------------------------------------------------------
# Định nghĩa kịch bản và thuật toán
# ------------------------------------------------------------------
SCENARIOS = {
    '1': ('Lưu lượng thấp (Low - Scale 0.5)',          CFG_RL,   0.5),
    '2': ('Lưu lượng trung bình (Medium - Scale 1.0)', CFG_RL,   1.0),
    '3': ('Lưu lượng cao (High - Scale 1.5)',           CFG_RL,   1.5),
    '4': ('Lưu lượng thực tế từ camera (Real-world)',  CFG_REAL, 1.0),
}

ALGORITHMS = {
    '1': ('Chu kỳ cố định (Fixed-Time - FT)',       'FT'),
    '2': ('Cảm biến lưu lượng (Actuated Control - AC)', 'AC'),
    '3': ('Tối đa hóa áp lực (Max-Pressure - MP)',   'MP'),
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("==================================================")
    print("      SUMO INTERACTIVE SIMULATION WATCHER         ")
    print("==================================================")

    # 1. Chọn kịch bản
    scen_choice           = get_menu_choice("CHỌN KỊCH BẢN LƯU LƯỢNG",
                                            {k: v[0] for k, v in SCENARIOS.items()})
    scen_name, cfg_path, scale = SCENARIOS[scen_choice]

    # 2. Chọn thuật toán
    algo_choice           = get_menu_choice("CHỌN THUẬT TOÁN ĐIỀU KHIỂN",
                                            {k: v[0] for k, v in ALGORITHMS.items()})
    algo_name, algo_code  = ALGORITHMS[algo_choice]

    # 3. Chọn độ trễ
    delay = get_delay_input(default=50)

    print("\n==================================================")
    print(f" Đang khởi chạy mô phỏng:")
    print(f"  - Kịch bản:  {scen_name}")
    print(f"  - Thuật toán: {algo_name}")
    print(f"  - Độ trễ:     {delay} ms/bước")
    print("==================================================")

    # Xây dựng lệnh SUMO-GUI
    sumo_cmd = [
        'sumo-gui',
        '-c', cfg_path,
        '--step-length', '0.10',
        '--delay', str(delay),
        '--lateral-resolution', '0',
        '--seed', '42',
        '--scale', str(scale),
        '--start',       # Tự động bắt đầu
        '--quit-on-end', # Tự đóng GUI khi xong
    ]

    # Chạy mô phỏng
    metrics = run_simulation_interactive(sumo_cmd, algo_code)

    # In kết quả
    if metrics:
        print_summary(metrics, scen_name, algo_name)
    else:
        print("\nKhông thu thập được dữ liệu mô phỏng.")


if __name__ == '__main__':
    main()
