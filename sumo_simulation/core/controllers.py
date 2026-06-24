"""
core/controllers.py
===================
Logic điều khiển đèn tín hiệu giao thông.

Mỗi controller được đóng gói trong một class riêng, nhận trạng thái
hiện tại và trả về trạng thái mới. Không phụ thuộc vào biến global.

Classes:
    ControllerState   — Dữ liệu trạng thái pha hiện tại (dataclass)
    FixedTimeController   — Giữ nguyên lịch SUMO mặc định
    MaxPressureController — Chọn pha có áp suất lớn nhất
    ActuatedController    — Kéo dài pha nếu có xe, ngược lại chuyển pha
"""

import numpy as np
import traci

from core.config import (
    GREEN_PHASES,
    PHASE_DETECTORS,
    PHASE_OUTGOING_EDGES,
    MIN_GREEN_STEPS,
    MAX_GREEN_STEPS,
    YELLOW_STEPS,
    TLS_ID,
)


class ControllerState:
    """Trạng thái bộ điều khiển pha đèn.
    
    Attributes:
        current_green_idx: Chỉ số pha xanh hiện tại trong GREEN_PHASES (0–3)
        green_timer:        Số bước đã ở pha xanh hiện tại
        yellow_timer:       Số bước còn lại ở pha vàng (0 = không có vàng)
    """
    def __init__(self, initial_green_idx: int = 0):
        self.current_green_idx: int = initial_green_idx
        self.green_timer: int = 0
        self.yellow_timer: int = 0


class FixedTimeController:
    """Fixed-Time (FT): Không can thiệp, để SUMO tự chạy theo lịch cố định."""

    @staticmethod
    def step(state: ControllerState) -> ControllerState:
        """Không làm gì. SUMO tự quản lý pha."""
        return state


class MaxPressureController:
    """Max-Pressure (MP): Chọn pha có áp suất (incoming - outgoing) lớn nhất.
    
    Công thức: pressure(p) = Σ incoming_queue(p) − Σ outgoing_queue(p)
    Chuyển pha khi pha hiện tại không còn là pha có áp suất cao nhất
    hoặc khi đã vượt quá MAX_GREEN_STEPS.
    """

    @staticmethod
    def step(state: ControllerState) -> ControllerState:
        if state.yellow_timer > 0:
            state.yellow_timer -= 1
            if state.yellow_timer == 0:
                # Pha vàng kết thúc → chuyển sang pha xanh mới
                traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[state.current_green_idx])
                state.green_timer = 0
        else:
            state.green_timer += 1
            if state.green_timer >= MIN_GREEN_STEPS:
                # Tính áp suất cho từng pha
                pressures = []
                for p in range(4):
                    incoming = sum(
                        traci.lanearea.getLastStepVehicleNumber(det)
                        for det in PHASE_DETECTORS[p]
                    )
                    outgoing = sum(
                        traci.edge.getLastStepVehicleNumber(edge)
                        for edge in PHASE_OUTGOING_EDGES[p]
                    )
                    pressures.append(incoming - outgoing)

                target_idx = int(np.argmax(pressures))
                should_switch = (
                    target_idx != state.current_green_idx
                    or state.green_timer >= MAX_GREEN_STEPS
                )

                if should_switch:
                    yellow_phase = (GREEN_PHASES[state.current_green_idx] + 1) % 8
                    traci.trafficlight.setPhase(TLS_ID, yellow_phase)
                    state.current_green_idx = target_idx
                    state.yellow_timer = YELLOW_STEPS

        return state


class ActuatedController:
    """Actuated Control (AC): Kéo dài pha xanh nếu detector phát hiện xe;
    chuyển pha vòng tròn khi không có xe hoặc vượt MAX_GREEN_STEPS.
    """

    @staticmethod
    def step(state: ControllerState) -> ControllerState:
        if state.yellow_timer > 0:
            state.yellow_timer -= 1
            if state.yellow_timer == 0:
                traci.trafficlight.setPhase(TLS_ID, GREEN_PHASES[state.current_green_idx])
                state.green_timer = 0
        else:
            state.green_timer += 1
            if state.green_timer >= MIN_GREEN_STEPS:
                has_vehicles = any(
                    traci.lanearea.getLastStepVehicleNumber(det) > 0
                    for det in PHASE_DETECTORS[state.current_green_idx]
                )
                if not has_vehicles or state.green_timer >= MAX_GREEN_STEPS:
                    yellow_phase = (GREEN_PHASES[state.current_green_idx] + 1) % 8
                    traci.trafficlight.setPhase(TLS_ID, yellow_phase)
                    state.current_green_idx = (state.current_green_idx + 1) % 4
                    state.yellow_timer = YELLOW_STEPS

        return state


# -------------------------------------------------------------------
# Factory function để lấy controller theo tên
# -------------------------------------------------------------------
_CONTROLLER_MAP = {
    'FT': FixedTimeController,
    'MP': MaxPressureController,
    'AC': ActuatedController,
}

def get_controller(control_type: str):
    """Trả về class controller tương ứng với tên.
    
    Args:
        control_type: 'FT', 'MP', hoặc 'AC'
    
    Returns:
        Controller class (FixedTimeController | MaxPressureController | ActuatedController)
    
    Raises:
        ValueError: Nếu control_type không hợp lệ
    """
    controller = _CONTROLLER_MAP.get(control_type.upper())
    if controller is None:
        raise ValueError(f"Unknown control_type '{control_type}'. Choose from: {list(_CONTROLLER_MAP.keys())}")
    return controller
