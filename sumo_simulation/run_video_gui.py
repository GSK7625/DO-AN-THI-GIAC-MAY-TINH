import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_path = os.path.join(script_dir, 'configs', 'osm_cut_video.sumocfg')

sumo_cmd = [
    'sumo-gui',
    '-c', sumocfg_path,
    '--step-length', '0.10',
    '--delay', '1000',
    '--start',
    '--quit-on-end',
]

def main():
    print("=== SUMO-GUI Mode: Video Reconstructed Simulation ===")
    print(">>> SUMO-GUI window will open. Press Play if not auto-started.")
    print(">>> Press Ctrl+C in terminal to stop early.\n")

    traci.start(sumo_cmd)
    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except traci.exceptions.FatalTraCIError:
        print("\nSUMO-GUI closed.")
    finally:
        try:
            traci.close()
        except Exception:
            pass
        print(f"\nFinished after {step} simulation steps.")

if __name__ == '__main__':
    main()
