import os
import sys
import random
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

# Hyperparameters and IDs
detector_ids = [
    "det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2",
    "det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2",
    "det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2",
    "det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"
]
tls_id = "cluster_53190763_5896114911"
GREEN_PHASES = [0, 2, 4, 6]

state_size = len(detector_ids) + 1  # 13 + 1
action_size = 2  # 0: Keep, 1: Switch
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
MIN_GREEN_STEPS = 50  # 5.0 seconds minimum green time

# Tabular Q-table
Q_table = {}

def get_max_Q_value_of_state(s):
    if s not in Q_table:
        Q_table[s] = np.zeros(action_size)
    return np.max(Q_table[s])

def get_reward(queues):
    total_queue = sum(queues)
    return -float(total_queue)

def get_state(current_green_idx):
    # Discretize: 1 if queue > 0, else 0
    queues = [1 if traci.lanearea.getLastStepVehicleNumber(det) > 0 else 0 for det in detector_ids]
    return tuple(queues) + (current_green_idx,)

def get_raw_queues():
    return [traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids]

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice([0, 1])
    else:
        if state not in Q_table:
            Q_table[state] = np.zeros(action_size)
        return int(np.argmax(Q_table[state]))

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(action_size)
    
    old_q = Q_table[old_state][action]
    best_future_q = get_max_Q_value_of_state(new_state)
    
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

def main():
    NUM_EPISODES = 20
    print(f"\n=== Starting Online Q-Learning Training on OSM Map ({NUM_EPISODES} Episodes) ===")
    
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
        
        state = get_state(current_green_idx)
        
        yellow_timer = 0
        green_timer = 0
        
        # Variables to store transition state when action == 1
        pending_update = False
        transition_old_state = None
        transition_action = None
        
        ep_reward = 0.0
        ep_queues = []
        
        def step_and_record():
            nonlocal ep_reward, global_step
            traci.simulationStep()
            global_step += 1
            raw_qs = get_raw_queues()
            q_sum = sum(raw_qs)
            ep_reward += get_reward(raw_qs)
            ep_queues.append(q_sum)

        def do_log():
            """Log every 500 steps regardless of which action branch we are in."""
            if global_step % 500 == 0:
                # Avoid duplicate entries for the same step
                if not step_history or step_history[-1] != global_step:
                    raw_qs = get_raw_queues()
                    print(f"  Step {global_step}, Queue: {sum(raw_qs)}, Reward: {get_reward(raw_qs):.1f}")
                    step_history.append(global_step)
                    reward_history.append(ep_reward)
                    # Rolling avg of last 500 steps (not cumulative episode avg)
                    queue_history.append(np.mean(ep_queues[-500:]) if ep_queues else 0.0)
        
        while traci.simulation.getMinExpectedNumber() > 0:
            if yellow_timer > 0:
                step_and_record()
                do_log()
                yellow_timer -= 1
                
                if yellow_timer == 0:
                    traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
                    green_timer = 0
                    
                    new_state = get_state(current_green_idx)
                    if pending_update:
                        reward = get_reward(get_raw_queues())
                        update_Q_table(transition_old_state, transition_action, reward, new_state)
                        pending_update = False
                        
                    state = new_state
                continue
            
            green_timer += 1
            action = 0
            if green_timer >= MIN_GREEN_STEPS:
                action = get_action_from_policy(state)
                
            if action == 1:
                transition_old_state = state
                transition_action = action
                pending_update = True
                
                yellow_phase = (GREEN_PHASES[current_green_idx] + 1) % 8
                traci.trafficlight.setPhase(tls_id, yellow_phase)
                
                current_green_idx = (current_green_idx + 1) % 4
                yellow_timer = 30
                
                step_and_record()
                do_log()
                yellow_timer -= 1
                continue
                
            step_and_record()
            do_log()
            
            new_state = get_state(current_green_idx)
            reward = get_reward(get_raw_queues())
            
            update_Q_table(state, action, reward, new_state)
            state = new_state
                
        traci.close()
        
        avg_q = np.mean(ep_queues) if ep_queues else 0.0
        episode_rewards.append(ep_reward)
        episode_avg_queues.append(avg_q)
        print(f"Episode {ep+1} Finished. Total Steps: {global_step}, Cum. Reward: {ep_reward:.1f}, Avg Queue: {avg_q:.2f}")

    # Save to CSV
    csv_path = os.path.join(script_dir, 'osm_ql_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'cumulative_reward', 'queue_length'])
        writer.writerows(zip(step_history, reward_history, queue_history))
    print(f"\nSaved metrics to {csv_path}")

if __name__ == '__main__':
    main()
