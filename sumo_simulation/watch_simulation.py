import os
import sys
import codecs

if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
if sys.stderr and sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

try:
    INTERACTIVE = sys.stdin.isatty()
except Exception:
    INTERACTIVE = False

import numpy as np

if 'SUMO_HOME' not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
import traci

script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_path = os.path.join(script_dir, '..', 'intersection', 'seattle', 'osm_cut_rl.sumocfg')

# --- Simulation parameters ---
tls_id = "cluster_53190763_5896114911"
GREEN_PHASES = [0, 2, 4, 6]
MIN_GREEN_STEPS = 50
MAX_GREEN_STEPS = 500
MAX_SIMULATION_TIME = 1000.0

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

# --- Options ---
TRAFFIC_OPTIONS = {
    '1': ('Low traffic',    0.5),
    '2': ('Medium traffic', 1.0),
    '3': ('High traffic',   1.5),
}
ALGO_OPTIONS = {
    '1': 'Fixed-Time (FT)',
    '2': 'Actuated Control (AC)',
    '3': 'Max-Pressure (MP)',
}
ALGO_MAP = {'1': 'FT', '2': 'AC', '3': 'MP'}


def get_choice(title, options):
    print(f"\n=== {title} ===")
    for k, v in options.items():
        label = v[0] if isinstance(v, tuple) else v
        print(f"  {k}. {label}")
    while True:
        choice = input("Nhap lua chon cua ban: ").strip()
        if choice in options:
            return choice
        print("Lua chon khong hop le.")


def choose_or_default(title, options, default_key):
    if INTERACTIVE:
        return get_choice(title, options)
    print(f"\n=== {title} ===")
    for k, v in options.items():
        label = v[0] if isinstance(v, tuple) else v
        print(f"  {k}. {label}")
    print(f"[Non-interactive: using default = {default_key}]")
    return default_key


def run_simulation(scale, algo, delay):
    sumo_cmd = [
        'sumo-gui', '-c', sumocfg_path,
        '--step-length', '0.10', '--delay', str(delay),
        '--seed', '42', '--scale', str(scale),
        '--start', '--quit-on-end',
    ]
    try:
        traci.start(sumo_cmd)
    except Exception as e:
        print(f"Loi khoi chay SUMO-GUI: {e}")
        return

    current_green_idx = 0
    if algo in ('MP', 'AC'):
        traci.trafficlight.setPhase(tls_id, GREEN_PHASES[0])

    yellow_timer = 0
    green_timer  = 0
    step_queues  = []
    vehicle_data = {}
    step_count   = 0
    arrived_count = 0

    while traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < MAX_SIMULATION_TIME:
        # --- Fixed-Time: no action needed ---
        # --- Max-Pressure ---
        if algo == 'MP':
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

        # --- Actuated Control ---
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

        # --- Simulation step ---
        try:
            traci.simulationStep()
            step_count  += 1
            arrived_count += traci.simulation.getArrivedNumber()
            q_sum = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detector_ids)
            step_queues.append(q_sum)

            for veh in traci.vehicle.getIDList():
                if veh not in vehicle_data:
                    vehicle_data[veh] = {'waiting_time': 0.0, 'time_loss': 0.0}
                vehicle_data[veh]['waiting_time'] = traci.vehicle.getAccumulatedWaitingTime(veh)
                vehicle_data[veh]['time_loss']     = traci.vehicle.getTimeLoss(veh)

            if step_count % 100 == 0:
                print(f"[Buoc {step_count}] Xe={len(traci.vehicle.getIDList())}, Hangdoi={q_sum}")

        except Exception:
            print("SUMO-GUI da dong.")
            break

    try:
        traci.close()
    except Exception:
        pass

    if vehicle_data:
        avg_queue  = np.mean(step_queues) if step_queues else 0.0
        avg_wait    = np.mean([d['waiting_time'] for d in vehicle_data.values()])
        avg_delay   = np.mean([d['time_loss']     for d in vehicle_data.values()])
        total_delay = sum(d['time_loss'] for d in vehicle_data.values())

        print("\n==================================================")
        print("            KET QUA MO PHONG                          ")
        print("==================================================")
        print(f"  Scale  : {scale}")
        print(f"  Algo   : {algo}")
        print(f"  Steps  : {step_count}")
        print(f"  Avg Queue     : {avg_queue:.2f} xe")
        print(f"  Avg Wait      : {avg_wait:.2f} s")
        print(f"  Throughput    : {arrived_count} xe")
        print(f"  Avg Delay/Veh : {avg_delay:.2f} s")
        print("==================================================")
    else:
        print("Khong co du lieu.")


def main():
    print("==================================================")
    print("      SUMO INTERACTIVE SIMULATION WATCHER          ")
    print("==================================================")

    scen_key   = choose_or_default("CHON KICH BAN", TRAFFIC_OPTIONS, '2')
    algo_key   = choose_or_default("CHON THUAT TOAN", ALGO_OPTIONS, '2')
    scen_name, scale = TRAFFIC_OPTIONS[scen_key]
    algo_name  = ALGO_OPTIONS[algo_key]
    algo       = ALGO_MAP[algo_key]

    if INTERACTIVE:
        raw = input("Do tre (ms, mac dinh 50): ").strip()
        delay = int(raw) if raw.isdigit() else 50
    else:
        delay = 50

    print(f"\n  Kich ban : {scen_name}")
    print(f"  Thuat toan: {algo_name}")
    print(f"  Do tre   : {delay} ms")
    print("==================================================")

    run_simulation(scale, algo, delay)


if __name__ == '__main__':
    main()
