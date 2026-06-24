"""
core/simulator.py
=================
Vòng lặp mô phỏng SUMO dùng chung.

Hàm run_simulation() khởi động TraCI, chạy vòng lặp điều khiển,
thu thập metrics, đóng TraCI và trả về kết quả.

Không chứa logic điều khiển (xem core/controllers.py)
và không chứa logic báo cáo (xem core/reporting.py).
"""

import numpy as np
import traci

from core.config import (
    TLS_ID,
    GREEN_PHASES,
    DETECTOR_IDS,
    MAX_SIMULATION_TIME,
)
from core.controllers import ControllerState, get_controller


def run_simulation(sumo_cmd: list, control_type: str, verbose: bool = True) -> dict:
    """Chạy một lần mô phỏng và thu thập metrics.

    Args:
        sumo_cmd:     Lệnh khởi động SUMO (list of strings cho traci.start)
        control_type: Loại điều khiển: 'FT', 'AC', hoặc 'MP'
        verbose:      In thông tin tiến trình nếu True

    Returns:
        dict với các khóa:
            'avg_queue'   — Độ dài hàng đợi trung bình (xe)
            'avg_wait'    — Thời gian chờ trung bình (giây)
            'throughput'  — Tổng số xe đã rời khỏi mạng
            'total_delay' — Tổng time loss của tất cả xe (giây)
            'avg_delay'   — Time loss trung bình mỗi xe (giây)
    """
    if verbose:
        print(f"  Running simulation: control_type={control_type}...")

    controller_cls = get_controller(control_type)

    traci.start(sumo_cmd)

    # Khởi tạo trạng thái pha đèn
    state = ControllerState(initial_green_idx=0)
    if control_type in ('MP', 'AC'):
        traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[state.current_green_idx])

    # Bộ thu thập dữ liệu
    step_queues = []
    vehicle_data = {}
    arrived_count = 0

    # Vòng lặp mô phỏng
    while (
        traci.simulation.getMinExpectedNumber() > 0
        and traci.simulation.getTime() < MAX_SIMULATION_TIME
    ):
        # Bước điều khiển đèn
        state = controller_cls.step(state)

        # Tiến mô phỏng một bước
        traci.simulationStep()
        arrived_count += traci.simulation.getArrivedNumber()

        # Thu thập độ dài hàng đợi tổng (tất cả detector)
        q_sum = sum(
            traci.lanearea.getLastStepVehicleNumber(det) for det in DETECTOR_IDS
        )
        step_queues.append(q_sum)

        # Thu thập dữ liệu từng xe
        for veh in traci.vehicle.getIDList():
            if veh not in vehicle_data:
                vehicle_data[veh] = {'waiting_time': 0.0, 'time_loss': 0.0}
            vehicle_data[veh]['waiting_time'] = traci.vehicle.getAccumulatedWaitingTime(veh)
            vehicle_data[veh]['time_loss']    = traci.vehicle.getTimeLoss(veh)

    traci.close()

    # Tính toán metrics tổng hợp
    avg_queue   = float(np.mean(step_queues)) if step_queues else 0.0
    total_delay = sum(d['time_loss'] for d in vehicle_data.values())
    avg_wait    = float(np.mean([d['waiting_time'] for d in vehicle_data.values()])) if vehicle_data else 0.0
    avg_delay   = float(np.mean([d['time_loss']    for d in vehicle_data.values()])) if vehicle_data else 0.0

    return {
        'avg_queue':   avg_queue,
        'avg_wait':    avg_wait,
        'throughput':  arrived_count,
        'total_delay': total_delay,
        'avg_delay':   avg_delay,
    }


def run_simulation_interactive(sumo_cmd: list, control_type: str) -> dict:
    """Phiên bản có xử lý lỗi cho watch_simulation (SUMO-GUI có thể bị đóng tay).

    Giống run_simulation() nhưng:
    - Bắt exception nếu user đóng cửa sổ GUI
    - In log mỗi 100 bước

    Returns:
        dict tương tự run_simulation(), hoặc dict rỗng nếu không có dữ liệu
    """
    controller_cls = get_controller(control_type)

    traci.start(sumo_cmd)

    state = ControllerState(initial_green_idx=0)
    if control_type in ('MP', 'AC'):
        traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[state.current_green_idx])

    step_queues = []
    vehicle_data = {}
    arrived_count = 0
    step_count = 0

    while (
        traci.simulation.getMinExpectedNumber() > 0
        and traci.simulation.getTime() < MAX_SIMULATION_TIME
    ):
        state = controller_cls.step(state)

        try:
            traci.simulationStep()
            step_count += 1
            arrived_count += traci.simulation.getArrivedNumber()

            q_sum = sum(
                traci.lanearea.getLastStepVehicleNumber(det) for det in DETECTOR_IDS
            )
            step_queues.append(q_sum)

            active_vehs = traci.vehicle.getIDList()
            for veh in active_vehs:
                if veh not in vehicle_data:
                    vehicle_data[veh] = {'waiting_time': 0.0, 'time_loss': 0.0}
                vehicle_data[veh]['waiting_time'] = traci.vehicle.getAccumulatedWaitingTime(veh)
                vehicle_data[veh]['time_loss']    = traci.vehicle.getTimeLoss(veh)

            if step_count % 100 == 0:
                print(f"Bước {step_count}: Xe đang chạy = {len(active_vehs)}, Hàng đợi = {q_sum}")

        except Exception:
            print("\nMô phỏng bị ngắt kết nối (hoặc bạn đã đóng cửa sổ SUMO-GUI).")
            break

    try:
        traci.close()
    except Exception:
        pass

    if not vehicle_data:
        return {}

    return {
        'step_count':  step_count,
        'avg_queue':   float(np.mean(step_queues)) if step_queues else 0.0,
        'avg_wait':    float(np.mean([d['waiting_time'] for d in vehicle_data.values()])),
        'throughput':  arrived_count,
        'total_delay': sum(d['time_loss'] for d in vehicle_data.values()),
        'avg_delay':   float(np.mean([d['time_loss'] for d in vehicle_data.values()])),
    }
