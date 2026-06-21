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
sumocfg_path = os.path.join(script_dir, 'configs', 'osm_cut_rl.sumocfg')

GUI_MODE = False
sumo_binary = 'sumo-gui' if GUI_MODE else 'sumo'

Sumo_config = [
    sumo_binary,
    '-c', sumocfg_path,
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0',
    '--seed', '42'
]

detector_ids = [
    "det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2",
    "det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2",
    "det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2",
    "det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"
]
tls_id = "cluster_53190763_5896114911"
GREEN_PHASES = [0, 2, 4, 6]

# Map green phase index (0 to 3) to detector IDs
phase_detectors = {
    0: ["det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2"],  # East (Phase 0)
    1: ["det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2"],  # North (Phase 2)
    2: ["det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2"],  # West (Phase 4)
    3: ["det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"]  # South (Phase 6)
}

MIN_GREEN_STEPS = 50   # 5.0 seconds minimum green
MAX_GREEN_STEPS = 500  # 50.0 seconds maximum green (safety limit)

def get_reward(queues):
    return -float(sum(queues))

def get_queues():
    return [traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids]

def main():
    NUM_EPISODES = 1
    print(f"\n=== Starting Online Max-Pressure Control Simulation on OSM Map ({NUM_EPISODES} Episodes) ===")
    
    episode_rewards = []
    episode_avg_queues = []
    
    step_history = []
    reward_history = []
    queue_history = []
    
    global_step = 0
    
    for ep in range(NUM_EPISODES):
        print(f"\n--- Episode {ep+1}/{NUM_EPISODES} ---")
        traci.start(Sumo_config)
        
        current_green_idx = 0
        traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
        
        yellow_timer = 0
        green_timer = 0
        
        ep_reward = 0.0
        ep_queues = []
        
        def step_and_record():
            nonlocal ep_reward, global_step
            traci.simulationStep()
            global_step += 1
            queues = get_queues()
            ep_reward += get_reward(queues)
            ep_queues.append(sum(queues))

        def do_log():
            if global_step % 500 == 0:
                if not step_history or step_history[-1] != global_step:
                    queues = get_queues()
                    print(f"  Step {global_step}, Queue: {sum(queues)}, Reward: {get_reward(queues):.1f}")
                    step_history.append(global_step)
                    reward_history.append(ep_reward)
                    queue_history.append(np.mean(ep_queues[-500:]) if ep_queues else 0.0)

        while traci.simulation.getMinExpectedNumber() > 0:
            if yellow_timer > 0:
                step_and_record()
                do_log()
                yellow_timer -= 1
                
                if yellow_timer == 0:
                    # Switch to target green phase
                    traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
                    green_timer = 0
                continue
            
            # Inside Green phase
            green_timer += 1
            
            action = 0  # 0: Keep, 1: Switch
            if green_timer >= MIN_GREEN_STEPS:
                # Calculate pressure (queue lengths) for all 4 approaches
                pressures = [
                    sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[0]),
                    sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[1]),
                    sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[2]),
                    sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[3])
                ]
                
                # Max-pressure target phase index
                target_green_idx = int(np.argmax(pressures))
                
                # Switch if target phase is different, or if we reach safety max green steps
                if target_green_idx != current_green_idx or green_timer >= MAX_GREEN_STEPS:
                    action = 1
                    
            if action == 1:
                # Transition to yellow phase for the CURRENT green phase to clear traffic
                yellow_phase = (GREEN_PHASES[current_green_idx] + 1) % 8
                traci.trafficlight.setPhase(tls_id, yellow_phase)
                
                # If safety max green triggered a switch but pressures say same phase,
                # we force shift to the next logical phase in sequence to prevent deadlock
                if target_green_idx == current_green_idx:
                    target_green_idx = (current_green_idx + 1) % 4
                
                current_green_idx = target_green_idx
                yellow_timer = 30
                
                step_and_record()
                do_log()
                yellow_timer -= 1
                continue
            
            step_and_record()
            do_log()
            
        traci.close()
        
        avg_q = np.mean(ep_queues) if ep_queues else 0.0
        episode_rewards.append(ep_reward)
        episode_avg_queues.append(avg_q)
        print(f"Episode {ep+1} Finished. Total Steps: {global_step}, Cum. Reward: {ep_reward:.1f}, Avg Queue: {avg_q:.2f}")

    # Save to CSV
    outputs_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    csv_path = os.path.join(outputs_dir, 'osm_mp_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'cumulative_reward', 'queue_length'])
        writer.writerows(zip(step_history, reward_history, queue_history))
    print(f"\nSaved metrics to {csv_path}")

    # Plot Max-Pressure progress
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Episode Rewards
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPISODES + 1), episode_rewards, marker='o', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Max-Pressure: Episode Reward')
    plt.grid(True)
    
    # Plot 2: Episode Average Queue Length
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPISODES + 1), episode_avg_queues, marker='o', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Avg Queue Length')
    plt.title('Max-Pressure: Avg Queue Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'osm_mp_progress.png'))
    print("Saved Max-Pressure progress plot to: osm_mp_progress.png")

if __name__ == '__main__':
    main()
