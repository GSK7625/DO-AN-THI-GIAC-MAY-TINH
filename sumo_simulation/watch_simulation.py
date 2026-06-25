import os
import sys

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

from core.simulator import run_simulation_interactive

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_RL     = os.path.join(SCRIPT_DIR, 'configs', 'osm_cut_rl.sumocfg')
CFG_REAL   = os.path.join(SCRIPT_DIR, 'configs', 'osm_cut_real.sumocfg')

def get_menu_choice(title: str, options: dict) -> str:
    print(f"\n=== {title} ===")
    for k, v in options.items():
        label = v[0] if isinstance(v, tuple) else v
        print(f"  {k}. {label}")
    while True:
        choice = input("Nhap lua chon cua ban: ").strip()
        if choice in options:
            return choice
        print("Lua chon khong hop le.")


def get_delay_input(default: int = 50) -> int:
    print(f"\n=== CAU HINH DO TRE MO PHONG (ms) ===")
    val = input(f"Nhap do tre moi buoc (ms, mac dinh {default}): ").strip()
    return int(val) if val.isdigit() else default


def print_summary(metrics: dict, scen_name: str, algo_name: str) -> None:
    print("\n==================================================")
    print("          KET QUA MO PHONG (SUMMARY METRICS)      ")
    print("==================================================")
    print(f" - Kich ban:                  {scen_name}")
    print(f" - Thuat toan:                {algo_name}")
    print(f" - Tong so buoc chay:         {metrics['step_count']}")
    print(f" - Hang doi TB (Avg Queue):   {metrics['avg_queue']:.2f} xe")
    print(f" - Thoi gian cho TB:          {metrics['avg_wait']:.2f} giay")
    print(f" - Tong xe thong qua:         {metrics['throughput']} xe")
    print(f" - Tong thoi gian tre:        {metrics['total_delay']:.1f} giay")
    print(f" - Thoi gian tre TB/xe:       {metrics['avg_delay']:.2f} giay")
    print("==================================================")


SCENARIOS = {
    '1': ('Luu luong thap (Low - Scale 0.5)',          CFG_RL,   0.5),
    '2': ('Luu luong trung binh (Medium - Scale 1.0)', CFG_RL,   1.0),
    '3': ('Luu luong cao (High - Scale 1.5)',           CFG_RL,   1.5),
    '4': ('Luu luong thuc te tu camera (Real-world)',  CFG_REAL, 1.0),
}

ALGORITHMS = {
    '1': ('Chu ky co dinh (Fixed-Time - FT)',       'FT'),
    '2': ('Cam bien luu luong (Actuated Control - AC)', 'AC'),
    '3': ('Toi da hoa ap luc (Max-Pressure - MP)',   'MP'),
}


def main():
    print("==================================================")
    print("      SUMO INTERACTIVE SIMULATION WATCHER         ")
    print("==================================================")

    # 1. Chon kich ban
    scen_choice           = get_menu_choice("CHON KICH BAN LUU LUONG",
                                            {k: v[0] for k, v in SCENARIOS.items()})
    scen_name, cfg_path, scale = SCENARIOS[scen_choice]

    # 2. Chon thuat toan
    algo_choice           = get_menu_choice("CHON THUAT TOAN DIEU KHIEN",
                                            {k: v[0] for k, v in ALGORITHMS.items()})
    algo_name, algo_code  = ALGORITHMS[algo_choice]

    # 3. Chon do tre
    delay = get_delay_input(default=50)

    print("\n==================================================")
    print(f" Dang khoi chay mo phong:")
    print(f"  - Kich ban:  {scen_name}")
    print(f"  - Thuat toan: {algo_name}")
    print(f"  - Do tre:     {delay} ms/buoc")
    print("==================================================")

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
        print("Khong co du lieu.")

if __name__ == '__main__':
    main()

