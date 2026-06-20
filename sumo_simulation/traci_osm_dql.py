import os
import sys
import random
import csv
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Establish SUMO_HOME path for TraCI
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import torch
import torch.nn as nn
import torch.optim as optim

# sumo config
script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_path = os.path.join(script_dir, 'osm_cut_rl.sumocfg')

GUI_MODE = False  # Set to True to launch sumo-gui
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

state_size = len(detector_ids) + 1  # 13 queues + current_green_idx
action_size = 2  # 0: Keep, 1: Switch
GAMMA = 0.9
MIN_GREEN_STEPS = 50  # 5.0 seconds minimum green time before allowing a switch

# PyTorch Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer and Training Hyperparameters
replay_buffer = deque(maxlen=20000)
BATCH_SIZE = 64
TAU = 0.005  # Target network soft update parameter

dqn_model = DQN(state_size, action_size)
target_model = DQN(state_size, action_size)
target_model.load_state_dict(dqn_model.state_dict())
target_model.eval()

# Optimized optimizer with lr=0.0005 for training stability
optimizer = optim.Adam(dqn_model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

def to_tensor(state_tuple):
    return torch.tensor(state_tuple, dtype=torch.float32).unsqueeze(0)

def get_state(current_green_idx):
    # Returns discretized state tuple (1.0 if queue > 0, else 0.0) and sum of raw queues
    raw_queues = [traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids]
    disc_queues = [1.0 if q > 0 else 0.0 for q in raw_queues]
    state = tuple(disc_queues) + (float(current_green_idx) / 3.0,)
    return state, sum(raw_queues)

def get_action_from_policy(state, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1])
    else:
        state_tensor = to_tensor(state)
        with torch.no_grad():
            Q_values = dqn_model(state_tensor)
        return int(torch.argmax(Q_values, dim=1).item())

def update_Q_network():
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    # Sample transition batch from buffer
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
    # Q(s, a) from current model
    q_values = dqn_model(states_t).gather(1, actions_t)
    
    # Double DQN action selection and evaluation
    with torch.no_grad():
        best_actions = dqn_model(next_states_t).argmax(dim=1, keepdim=True)
        next_q_values = target_model(next_states_t).gather(1, best_actions)
        target_q_values = rewards_t + (1.0 - dones_t) * GAMMA * next_q_values
        
    loss = criterion(q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient norm clipping to prevent gradient explosions from reward variances
    torch.nn.utils.clip_grad_norm_(dqn_model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Soft update of target model weights
    for target_param, local_param in zip(target_model.parameters(), dqn_model.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

def main():
    NUM_EPISODES = 20
    print(f"\n=== Starting Online DQL Training on OSM Map using PyTorch ({NUM_EPISODES} Episodes) ===")
    
    episode_rewards = []
    episode_avg_queues = []
    
    step_history = []
    reward_history = []
    queue_history = []
    
    global_step = 0
    
    # Epsilon parameters
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY = 0.85  # Decays epsilon slightly slower to allow enough exploration
    
    epsilon = EPSILON_START
    
    for ep in range(NUM_EPISODES):
        print(f"\n--- Episode {ep+1}/{NUM_EPISODES} (Epsilon: {epsilon:.3f}) ---")
        traci.start(Sumo_config)
        
        current_green_idx = 0
        traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
        
        state, _ = get_state(current_green_idx)
        
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
            queues = [traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids]
            q_sum = sum(queues)
            ep_reward += -float(q_sum)
            ep_queues.append(q_sum)

        def do_log():
            """Log every 500 steps regardless of which action branch we are in."""
            if global_step % 500 == 0:
                if not step_history or step_history[-1] != global_step:
                    raw_qs = [traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids]
                    print(f"  Step {global_step}, Queue: {sum(raw_qs)}, Reward: {-float(sum(raw_qs)):.1f}")
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
                    # Switch to target green phase
                    traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
                    green_timer = 0
                    
                    # Yellow is finished, now we are in new green phase, perform update
                    new_state, raw_q_sum = get_state(current_green_idx)
                    if pending_update:
                        # Scaled reward by 10.0 to stabilize gradients
                        reward = -float(raw_q_sum) / 10.0
                        replay_buffer.append((transition_old_state, transition_action, reward, new_state, False))
                        if global_step % 4 == 0:
                            update_Q_network()
                        pending_update = False
                        
                    state = new_state
                continue
            
            # Inside Green phase
            green_timer += 1
            
            # Determine action: force keep if minimum green time has not passed
            action = 0
            if green_timer >= MIN_GREEN_STEPS:
                action = get_action_from_policy(state, epsilon)
                
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
                
            # Action == 0 (keep current phase)
            step_and_record()
            do_log()
            
            new_state, raw_q_sum = get_state(current_green_idx)
            # Scaled reward by 10.0 to stabilize gradients
            reward = -float(raw_q_sum) / 10.0
            
            replay_buffer.append((state, action, reward, new_state, False))
            if global_step % 4 == 0:
                update_Q_network()
            state = new_state
                
        traci.close()
        
        # Decay epsilon at the end of episode
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        avg_q = np.mean(ep_queues) if ep_queues else 0.0
        episode_rewards.append(ep_reward)
        episode_avg_queues.append(avg_q)
        print(f"Episode {ep+1} Finished. Total Steps: {global_step}, Cum. Reward: {ep_reward:.1f}, Avg Queue: {avg_q:.2f}")

    # Save to CSV
    csv_path = os.path.join(script_dir, 'osm_dql_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'cumulative_reward', 'queue_length'])
        writer.writerows(zip(step_history, reward_history, queue_history))
    print(f"\nSaved metrics to {csv_path}")

    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Episode Rewards
    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPISODES + 1), episode_rewards, marker='o', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQL Training: Episode Reward')
    plt.grid(True)
    
    # Plot 2: Episode Average Queue Length
    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPISODES + 1), episode_avg_queues, marker='o', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Avg Queue Length')
    plt.title('DQL Training: Avg Queue Length')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'osm_dql_training_progress.png'))
    print("Saved training progress plot to: osm_dql_training_progress.png")
    
if __name__ == '__main__':
    main()
