import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Establish SUMO_HOME path for TraCI
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# sumo config
script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_path = os.path.join(script_dir, 'osm_cut_rl.sumocfg')

GUI_MODE = False
sumo_binary = 'sumo-gui' if GUI_MODE else 'sumo'

Sumo_config = [
    sumo_binary,
    '-c', sumocfg_path,
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0'
]

# Detectors and TLS info
detector_ids = [
    "det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2",
    "det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2",
    "det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2",
    "det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"
]
tls_id = "cluster_53190763_5896114911"

def get_reward(queues):
    return -float(sum(queues))

def get_queues():
    return [traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids]

def main():
    NUM_EPISODES = 20
    print(f"\n=== Starting Online Fixed Time Simulation on OSM Map ({NUM_EPISODES} Episodes) ===")
    
    episode_rewards = []
    episode_avg_queues = []
    
    step_history = []
    reward_history = []
    queue_history = []
    
    global_step = 0
    
    for ep in range(NUM_EPISODES):
        print(f"\n--- Episode {ep+1}/{NUM_EPISODES} ---")
        traci.start(Sumo_config)
        
        ep_reward = 0.0
        ep_queues = []
        
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            global_step += 1
            
            queues = get_queues()
            reward = get_reward(queues)
            ep_reward += reward
            ep_queues.append(sum(queues))
            
            if global_step % 500 == 0:
                print(f"  Step {global_step}, Queue: {sum(queues)}, Reward: {reward:.1f}")
                step_history.append(global_step)
                reward_history.append(ep_reward)
                # Rolling avg of last 500 steps (not cumulative episode avg)
                queue_history.append(np.mean(ep_queues[-500:]) if ep_queues else 0.0)
                
        traci.close()
        
        avg_q = np.mean(ep_queues) if ep_queues else 0.0
        episode_rewards.append(ep_reward)
        episode_avg_queues.append(avg_q)
        print(f"Episode {ep+1} Finished. Total Steps: {global_step}, Cum. Reward: {ep_reward:.1f}, Avg Queue: {avg_q:.2f}")

    # Save to CSV
    csv_path = os.path.join(script_dir, 'osm_ft_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'cumulative_reward', 'queue_length'])
        writer.writerows(zip(step_history, reward_history, queue_history))
    print(f"\nSaved metrics to {csv_path}")

if __name__ == '__main__':
    main()
