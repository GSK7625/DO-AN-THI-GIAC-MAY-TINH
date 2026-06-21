import os
import sys
import numpy as np

# Establish SUMO_HOME path for TraCI
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sumocfg_path = os.path.join(script_dir, 'configs', 'osm_cut_rl.sumocfg')

detector_ids = [
    "det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2",
    "det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2",
    "det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2",
    "det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"
]
tls_id = "cluster_53190763_5896114911"
GREEN_PHASES = [0, 2, 4, 6]

phase_detectors = {
    0: ["det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2"],        # East
    1: ["det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2"],        # North
    2: ["det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2"],  # West
    3: ["det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"]  # South
}

MIN_GREEN_STEPS = 50
MAX_GREEN_STEPS = 500

def get_menu_choice(title, options):
    print(f"\n=== {title} ===")
    for k, v in options.items():
        print(f"  {k}. {v}")
    while True:
        choice = input("Nhập lựa chọn của bạn: ").strip()
        if choice in options:
            return choice
        print("Lựa chọn không hợp lệ, vui lòng nhập lại.")

def main():
    print("==================================================")
    print("      SUMO INTERACTIVE SIMULATION WATCHER         ")
    print("==================================================")
    
    # 1. Choose Traffic Scenario
    scenarios = {
        '1': 'Lưu lượng thấp (Low - Scale 0.5)',
        '2': 'Lưu lượng trung bình (Medium - Scale 1.0)',
        '3': 'Lưu lượng cao (High - Scale 1.5)'
    }
    scen_choice = get_menu_choice("CHỌN KỊCH BẢN LƯU LƯỢNG", scenarios)
    scale_map = {'1': 0.5, '2': 1.0, '3': 1.5}
    scale = scale_map[scen_choice]
    scen_name = scenarios[scen_choice]
    
    # 2. Choose Control Algorithm
    algorithms = {
        '1': 'Chu kỳ cố định (Fixed-Time - FT)',
        '2': 'Cảm biến lưu lượng (Actuated Control - AC)',
        '3': 'Tối đa hóa áp lực (Max-Pressure - MP)'
    }
    algo_choice = get_menu_choice("CHỌN THUẬT TOÁN ĐIỀU KHIỂN", algorithms)
    algo_map = {'1': 'FT', '2': 'AC', '3': 'MP'}
    algo = algo_map[algo_choice]
    algo_name = algorithms[algo_choice]
    
    # 3. Choose Delay
    print("\n=== CẤU HÌNH ĐỘ TRỄ MÔ PHỎNG (ms) ===")
    delay_input = input("Nhập độ trễ mỗi bước (ms, mặc định 50): ").strip()
    if not delay_input.isdigit():
        delay = 50
    else:
        delay = int(delay_input)
        
    print("\n==================================================")
    print(f" Đang khởi chạy mô phỏng:")
    print(f"  - Kịch bản: {scen_name}")
    print(f"  - Thuật toán: {algo_name}")
    print(f"  - Độ trễ: {delay} ms/bước")
    print("==================================================")
    
    # Configure SUMO command
    sumo_cmd = [
        'sumo-gui',
        '-c', sumocfg_path,
        '--step-length', '0.10',
        '--delay', str(delay),
        '--lateral-resolution', '0',
        '--seed', '42',
        '--scale', str(scale),
        '--start',        # Auto start simulation
        '--quit-on-end'   # Auto close GUI when finished
    ]
    
    try:
        traci.start(sumo_cmd)
    except Exception as e:
        print(f"Lỗi khi khởi chạy SUMO-GUI: {e}")
        print("Đảm bảo bạn đã cài đặt SUMO và thêm 'sumo-gui' vào PATH.")
        return
        
    current_green_idx = 0
    if algo in ['MP', 'AC']:
        traci.trafficlight.setPhase(tls_id, GREEN_PHASES[current_green_idx])
        
    yellow_timer = 0
    green_timer = 0
    
    step_queues = []
    vehicle_data = {}
    step_count = 0
    
    # Run simulation loop
    while traci.simulation.getMinExpectedNumber() > 0:
        if algo == 'FT':
            pass
        elif algo == 'MP':
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
                        sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[0]),
                        sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[1]),
                        sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[2]),
                        sum(traci.lanearea.getLastStepVehicleNumber(det) for det in phase_detectors[3])
                    ]
                    target_green_idx = int(np.argmax(pressures))
                    if target_green_idx != current_green_idx or green_timer >= MAX_GREEN_STEPS:
                        action = 1
                if action == 1:
                    yellow_phase = (GREEN_PHASES[current_green_idx] + 1) % 8
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    if target_green_idx == current_green_idx:
                        target_green_idx = (current_green_idx + 1) % 4
                    current_green_idx = target_green_idx
                    yellow_timer = 30
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
                    active_dets = phase_detectors[current_green_idx]
                    has_vehicles = any(traci.lanearea.getLastStepVehicleNumber(det) > 0 for det in active_dets)
                    if not has_vehicles or green_timer >= MAX_GREEN_STEPS:
                        action = 1
                if action == 1:
                    yellow_phase = (GREEN_PHASES[current_green_idx] + 1) % 8
                    traci.trafficlight.setPhase(tls_id, yellow_phase)
                    current_green_idx = (current_green_idx + 1) % 4
                    yellow_timer = 30

        try:
            traci.simulationStep()
            step_count += 1
            
            # Queue data
            q_sum = sum(traci.lanearea.getLastStepVehicleNumber(det) for det in detector_ids)
            step_queues.append(q_sum)
            
            # Vehicle data
            active_vehs = traci.vehicle.getIDList()
            for veh in active_vehs:
                if veh not in vehicle_data:
                    vehicle_data[veh] = {
                        'waiting_time': 0.0,
                        'time_loss': 0.0
                    }
                vehicle_data[veh]['waiting_time'] = traci.vehicle.getAccumulatedWaitingTime(veh)
                vehicle_data[veh]['time_loss'] = traci.vehicle.getTimeLoss(veh)
                
            # Log real-time info in console every 100 steps (10 seconds)
            if step_count % 100 == 0:
                print(f"Bước {step_count}: Số xe đang hoạt động = {len(active_vehs)}, Tổng hàng đợi hiện tại = {q_sum}")
                
        except Exception as e:
            # Handle user closing the GUI manually before the simulation finishes
            print("\nMô phỏng bị ngắt kết nối (hoặc bạn đã đóng cửa sổ SUMO-GUI).")
            break
            
    # Try to close traci safely
    try:
        traci.close()
    except:
        pass
        
    # Print summary metrics
    if vehicle_data:
        avg_queue = np.mean(step_queues) if step_queues else 0.0
        throughput = len(vehicle_data)
        total_delay = sum(d['time_loss'] for d in vehicle_data.values())
        avg_wait = np.mean([d['waiting_time'] for d in vehicle_data.values()])
        avg_delay = np.mean([d['time_loss'] for d in vehicle_data.values()])
        
        print("\n==================================================")
        print("          KẾT QUẢ MÔ PHỎNG (SUMMARY METRICS)      ")
        print("==================================================")
        print(f" - Kịch bản: {scen_name}")
        print(f" - Thuật toán: {algo_name}")
        print(f" - Tổng số bước chạy: {step_count}")
        print(f" - Hàng đợi trung bình (Avg Queue): {avg_queue:.2f} xe")
        print(f" - Thời gian chờ trung bình (Avg Wait): {avg_wait:.2f} giây")
        print(f" - Tổng xe thông qua (Throughput): {throughput} xe")
        print(f" - Tổng thời gian chậm (Total Delay): {total_delay:.1f} giây")
        print(f" - Thời gian chậm TB (Avg Delay/Vehicle): {avg_delay:.2f} giây")
        print("==================================================")
    else:
        print("\nKhông thu thập được dữ liệu mô phỏng.")

if __name__ == '__main__':
    main()
