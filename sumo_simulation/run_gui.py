"""
run_gui.py - Run SUMO-GUI to visually observe the simulation.
Uses Fixed Time controller. Press Play in SUMO-GUI to start.
"""
import os
import sys
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_path = os.path.join(script_dir, 'configs', 'osm_cut_rl.sumocfg')

sumo_cmd = [
    'sumo-gui',
    '-c', sumocfg_path,
    '--step-length', '0.10',
    '--delay', '100',          # delay 100ms mỗi step → dễ quan sát
    '--lateral-resolution', '0',
    '--start',                  # tự động start khi mở
    '--quit-on-end',            # tự đóng khi xong
]

detector_ids = [
    "det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2",
    "det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2",
    "det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2",
    "det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"
]
tls_id = "cluster_53190763_5896114911"

def get_queues():
    return [traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids]

def main():
    print("=== SUMO-GUI Mode: Fixed Time Controller ===")
    print(">>> SUMO-GUI window will open. Press Play if not auto-started.")
    print(">>> Press Ctrl+C in terminal to stop early.\n")

    traci.start(sumo_cmd)

    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

            if step % 100 == 0:
                queues = get_queues()
                total_q = sum(queues)
                sim_time = traci.simulation.getTime()
                print(f"  t={sim_time:.1f}s | Queue total: {total_q} veh | Step: {step}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except traci.exceptions.FatalTraCIError:
        # Normal: SUMO-GUI closed the window (quit-on-end or user closed it)
        print("\nSUMO-GUI closed.")
    finally:
        try:
            traci.close()
        except Exception:
            pass
        print(f"\nFinished after {step} simulation steps.")

if __name__ == '__main__':
    main()
