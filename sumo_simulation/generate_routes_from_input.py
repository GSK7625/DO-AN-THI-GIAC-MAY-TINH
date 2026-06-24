"""
SUMO Scenario Generator
=======================
Doc: input/calib_config.json + input/tmc.csv + input/analytics.csv
     -> intersection/seattle/osm_cut_rl.rou.xml  (route xe)
     -> intersection/seattle/osm_cut_rl.add.xml  (detector)
     -> intersection/seattle/osm_cut_rl.sumocfg  (config)

Data sources:
- tmc.csv          : track_id, direction, class, movement
- analytics.csv    : direction, Minute, lane_id, n_vehicles, avg_speed_kmh, avg_delay_s, los
- summary.json     : throughput, avg_control_delay, avg_speed_kmh, los per direction
- calib_config.json: ROI polygons per direction (lane geometry)
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
INPUT_DIR = PROJECT_ROOT / "input"
OUT_DIR   = PROJECT_ROOT / "intersection" / "seattle"

OUT_ROU  = OUT_DIR / "osm_cut_rl.rou.xml"
OUT_ADD  = OUT_DIR / "osm_cut_rl.add.xml"
OUT_CFG  = OUT_DIR / "osm_cut_rl.sumocfg"

# ── Network edges (osm_cut.net.xml) ────────────────────────────────────────
# Incoming (approach) edges
EDGE_N_IN = "428067750#0"    # North approach
EDGE_S_IN = "-577951513"     # South approach
EDGE_E_IN = "428067759#0"    # East approach
EDGE_W_IN = "428067756.116"  # West approach

# Outgoing (exit) edges
EDGE_N_OUT = "-428067750#1"  # exit north
EDGE_S_OUT = "428067754#0"   # exit south
EDGE_E_OUT = "378376408#0"   # exit east
EDGE_W_OUT = "160183267#0"   # exit west

# Route definitions
ROUTE_TABLE = {
    ("N", "S"): [EDGE_N_IN,  EDGE_S_OUT],
    ("N", "E"): [EDGE_N_IN,  EDGE_E_OUT],
    ("N", "W"): [EDGE_N_IN,  EDGE_W_OUT],
    ("S", "N"): [EDGE_S_IN,  EDGE_N_OUT],
    ("S", "E"): [EDGE_S_IN,  EDGE_E_OUT],
    ("S", "W"): [EDGE_S_IN,  EDGE_W_OUT],
    ("E", "N"): [EDGE_E_IN,  EDGE_N_OUT],
    ("E", "S"): [EDGE_E_IN,  EDGE_S_OUT],
    ("E", "W"): [EDGE_E_IN,  EDGE_W_OUT],
    ("W", "N"): [EDGE_W_IN,  EDGE_N_OUT],
    ("W", "S"): [EDGE_W_IN,  EDGE_S_OUT],
    ("W", "E"): [EDGE_W_IN,  EDGE_E_OUT],
}

# Lane assignment per approach direction (from calib_config.json / video analysis)
# Layout when approaching the intersection:
#   E approach (428067759#0, 3 lanes index 0-2):
#     lane 2 (leftmost)  -> left turn  -> N  (E→N)
#     lane 1 (middle)    -> straight   -> S  (E→S)
#     lane 0 (rightmost) -> right turn -> W  (E→W)
#   W approach (428067756.116, 3 lanes index 0-2):
#     lane 2 (leftmost)  -> left turn  -> S  (W→S)
#     lane 1 (middle)    -> straight   -> N  (W→N)
#     lane 0 (rightmost) -> right turn -> E  (W→E)
#   N approach (428067750#0, 3 lanes index 0-2):
#     lane 0 (rightmost) -> left turn  -> E  (N→E)   [tightest turn]
#     lane 1 (middle)    -> straight   -> S  (N→S)
#     lane 2 (leftmost)  -> right turn -> W  (N→W)
#   S approach (-577951513, 4 lanes index 0-3):
#     lane 3 (leftmost)  -> left turn  -> W  (S→W)
#     lane 2 (middle)    -> straight   -> N  (S→N)
#     lane 1 (next)      -> right turn -> E  (S→E)
#     lane 0 (rightmost) -> UT / N straight through S→N

DEST_TO_LANE = {
    "N": {
        "E": "0",   # N→E left turn -> lane 0
        "S": "1",   # N→S straight  -> lane 1
        "W": "2",   # N→W right turn -> lane 2
    },
    "S": {
        "W": "3",   # S→W left turn -> lane 3
        "N": "2",   # S→N straight -> lane 2
        "E": "1",   # S→E right turn -> lane 1
    },
    "E": {
        "N": "2",   # E→N left turn -> lane 2
        "S": "1",   # E→S straight  -> lane 1
        "W": "0",   # E→W right turn -> lane 0
    },
    "W": {
        "S": "2",   # W→S left turn -> lane 2
        "N": "1",   # W→N straight  -> lane 1
        "E": "0",   # W→E right turn -> lane 0
    },
}

# Spawn position (distance from stop line toward origin)
# Edge lengths: E/W arms ~34m, N arm ~42m, S arm ~32m
# Spawn at ~5m from edge start = well before the stop line
SPAWN_POS = {
    EDGE_N_IN: "5",
    EDGE_S_IN: "5",
    EDGE_E_IN: "5",
    EDGE_W_IN: "5",
}

MIN_SPAWN_GAP_S = 0.5
MIN_SPEED_MPS   = 5.0

# ── vType definitions ───────────────────────────────────────────────────────
VTYPES = [
    '    <vType id="standard_car" accel="2.6" decel="4.5" length="4.5" minGap="2.0" maxSpeed="13.41"'
    ' sigma="0.5" speedFactor="1.0"'
    ' jmIgnoreKeepClearTime="5" jmDriveAfterYellowTime="1.0" jmTimegapMinor="1.0" />',
    '    <vType id="truck"        accel="1.0" decel="4.0" length="10.0" minGap="2.5" maxSpeed="10.0"'
    ' sigma="0.3" speedFactor="0.9"'
    ' jmIgnoreKeepClearTime="5" jmDriveAfterYellowTime="1.0" jmTimegapMinor="1.5" />',
    '    <vType id="motorcycle"   accel="2.5" decel="5.0" length="2.2" minGap="1.2" maxSpeed="13.41"'
    ' sigma="0.5" speedFactor="1.1"'
    ' jmIgnoreKeepClearTime="3" jmDriveAfterYellowTime="1.0" jmTimegapMinor="0.8" />',
]


# ── Load data ────────────────────────────────────────────────────────────────

DIR_FULL = {"East": "E", "West": "W", "North": "N", "South": "S"}
DIR_SHORT = {"E": "East", "W": "West", "N": "North", "S": "South"}

def load_tmc() -> dict[str, dict[str, int]]:
    """Return {direction: {movement: count}} from tmc.csv."""
    tmc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    path = INPUT_DIR / "tmc.csv"
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            direction = DIR_FULL.get(row["direction"].strip(), row["direction"].strip())
            movement  = row["movement"].strip()
            tmc[direction][movement] += 1
    return tmc


def load_analytics() -> dict[str, dict[str, dict]]:
    """Return {direction: {lane_id: {n_vehicles, avg_speed_kmh, avg_delay_s}}}."""
    analytics: dict[str, dict[str, dict]] = defaultdict(dict)
    path = INPUT_DIR / "analytics.csv"
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            direction = DIR_FULL.get(row["direction"].strip(), row["direction"].strip())
            lane      = row["lane_id"].strip()
            analytics[direction][lane] = {
                "n_vehicles":    int(row["n_vehicles"]),
                "avg_speed_kmh": float(row["avg_speed_kmh"]),
                "avg_delay_s":   float(row["avg_delay_s"]),
                "los":           row["los"].strip(),
            }
    return analytics


def load_summary() -> dict[str, dict]:
    """Return {direction: {total_throughput, avg_control_delay, avg_speed_kmh, overall_los}}."""
    path = INPUT_DIR / "summary.json"
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for direction, data in raw.items():
        out[direction] = {
            "total_throughput":  data["total_throughput"],
            "avg_control_delay": data["avg_control_delay"],
            "avg_speed_kmh":     data["avg_speed_kmh"],
            "overall_los":       data["overall_los"],
            "throughput_per_hour": data["throughput_per_hour"],
            "measurement_duration": data["measurement_duration"],
        }
    return out


# ── Turn ratio computation ──────────────────────────────────────────────────

def compute_turn_ratios(tmc: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Compute destination probabilities from tmc.csv, ignoring Stationary vehicles."""
    STRAIGHT = {"N": "S", "S": "N", "E": "W", "W": "E"}
    LEFT     = {"N": "E", "S": "W", "E": "N", "W": "S"}
    RIGHT    = {"N": "W", "S": "E", "E": "S", "W": "N"}
    ratios = {}
    for direction, movements in tmc.items():
        moving = {k: v for k, v in movements.items() if k != "Stationary"}
        total = sum(moving.values())
        if total == 0:
            ratios[direction] = {"S": 0.6, "W": 0.25, "E": 0.15}
            continue
        ratios[direction] = {
            STRAIGHT[direction]: moving.get("Straight",   0) / total,
            LEFT[direction]:     moving.get("Left Turn",  0) / total,
            RIGHT[direction]:     moving.get("Right Turn", 0) / total,
        }
    return ratios


# ── Vehicle generation ──────────────────────────────────────────────────────

def generate_vehicles(
    tmc: dict[str, dict[str, int]],
    turn_ratios: dict[str, dict[str, float]],
    rng: random.Random,
) -> list[dict]:
    """
    Generate individual vehicle records from tmc.csv.
    Uses real track_id, direction, class, movement from the input data.
    """
    vehicles: list[dict] = []

    # For each direction, build a list of (track_id, vehicle_class, movement)
    by_dir: dict[str, list[tuple]] = defaultdict(list)
    path = INPUT_DIR / "tmc.csv"
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            direction = DIR_FULL.get(row["direction"].strip(), row["direction"].strip())
            movement  = row["movement"].strip()
            vclass   = row.get("class", "car").strip().lower()
            track_id = int(row["track_id"])
            by_dir[direction].append((track_id, vclass, movement))

    # Assign departure times: spread vehicles uniformly over the measurement window
    # measurement_duration from summary.json gives us the real time span
    summary = load_summary()
    all_dirs = ["E", "W", "N", "S"]

    # Collect all vehicles first to know the total count
    all_vehicles_raw: list[tuple] = []
    for direction in all_dirs:
        for item in by_dir[direction]:
            all_vehicles_raw.append((direction,) + item)

    total_vehicles = len(all_vehicles_raw)

    # Use the longest measurement window
    max_duration = max(
        summary.get(d, {}).get("measurement_duration", 866)
        for d in all_dirs
    )

    # Sort by direction then track_id for consistent ordering
    all_vehicles_raw.sort(key=lambda x: (x[0], x[1]))

    last_depart: dict[str, float] = {d: -MIN_SPAWN_GAP_S for d in all_dirs}

    for direction, track_id, vclass, movement in all_vehicles_raw:
        if movement == "Stationary":
            continue  # skip parked/stopped vehicles

        # Destination from movement label
        dest = _dest_from_movement(direction, movement)
        if dest is None:
            dest = rng.choices(
                population=list(turn_ratios[direction].keys()),
                weights=list(turn_ratios[direction].values()),
                k=1,
            )[0]

        # Assign depart time: space evenly over the measurement window
        idx = len(vehicles)
        depart = round(idx * (max_duration / max(total_vehicles, 1)), 2)
        depart = max(depart, round(last_depart[direction] + MIN_SPAWN_GAP_S, 2))
        last_depart[direction] = depart

        # Vehicle type
        vtype = _normalize_vtype(vclass)

        # Depart speed from analytics (per-direction average as fallback)
        avg_speed = summary.get(direction, {}).get("avg_speed_kmh", 8.0)
        speed_mps = avg_speed / 3.6 * rng.uniform(0.85, 1.25)
        speed_mps = max(speed_mps, MIN_SPEED_MPS)

        edge_in = _edge_in(direction)
        depart_lane = DEST_TO_LANE[direction].get(dest, "1")

        vehicles.append({
            "id":          f"veh_{track_id}",
            "type":        vtype,
            "route":       f"{direction}_{dest}",
            "depart":      f"{depart:.2f}",
            "depart_lane": depart_lane,
            "depart_pos":  SPAWN_POS[edge_in],
            "depart_speed": f"{speed_mps:.2f}",
            "direction":   direction,
            "dest":        dest,
        })

    return vehicles


def _dest_from_movement(direction: str, movement: str) -> str | None:
    m = movement.strip().lower()
    if m == "straight":
        return {"N": "S", "S": "N", "E": "W", "W": "E"}[direction]
    if m == "left turn":
        return {"N": "E", "S": "W", "E": "N", "W": "S"}[direction]
    if m == "right turn":
        return {"N": "W", "S": "E", "E": "S", "W": "N"}[direction]
    return None


def _normalize_vtype(vclass: str) -> str:
    vclass = vclass.strip().lower()
    if vclass == "truck":
        return "truck"
    if vclass in ("motorcycle", "motorbike"):
        return "motorcycle"
    return "standard_car"


def _edge_in(direction: str) -> str:
    return {
        "N": EDGE_N_IN,
        "S": EDGE_S_IN,
        "E": EDGE_E_IN,
        "W": EDGE_W_IN,
    }[direction]


# ── Write outputs ───────────────────────────────────────────────────────────

def write_rou(vehicles: list[dict]) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
    ]
    for vt in VTYPES:
        lines.append(vt)
    lines.append("")

    # Route definitions (sorted for consistency)
    for route_id, edges in sorted(ROUTE_TABLE.items()):
        lines.append(f'    <route id="{route_id[0]}_{route_id[1]}" edges="{" ".join(edges)}" />')
    lines.append("")
    lines.append("    <!-- Vehicles from tmc.csv with turn ratios from video detection -->")

    for v in vehicles:
        lines.append(
            f'    <vehicle id="{v["id"]}" type="{v["type"]}"'
            f' route="{v["route"]}" depart="{v["depart"]}"'
            f' departLane="{v["depart_lane"]}"'
            f' departPos="{v["depart_pos"]}"'
            f' departSpeed="{v["depart_speed"]}" />'
        )
    lines.append("</routes>")
    OUT_ROU.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_add(vehicles: list[dict]) -> None:
    """Write detector (lane area) XML for all approach lanes."""
    # Fixed lane counts per approach from net.xml inspection:
    #   E:  3 lanes (0,1,2)   -> approach edge 428067759#0
    #   W:  3 lanes (0,1,2)   -> approach edge 428067756.116
    #   N:  3 lanes (0,1,2)   -> approach edge 428067750#0
    #   S:  4 lanes (0,1,2,3) -> approach edge -577951513
    APPROACH_LANES = {
        "E": [0, 1, 2],
        "W": [0, 1, 2],
        "N": [0, 1, 2],
        "S": [0, 1, 2, 3],
    }

    def _det_id(direction: str, lane: int) -> str:
        edge = _edge_in(direction)
        return f"det_{edge}_{lane}"

    def _lane_id(direction: str, lane: int) -> str:
        edge = _edge_in(direction)
        return f"{edge}_{lane}"

    det_length = {
        EDGE_E_IN: 34,
        EDGE_W_IN: 42,
        EDGE_N_IN: 42,
        EDGE_S_IN: 32,
    }

    lines = ["<additional>"]
    for direction in ["E", "W", "N", "S"]:
        edge = _edge_in(direction)
        length = det_length[edge]
        for lane in APPROACH_LANES[direction]:
            did = _det_id(direction, lane)
            lid = _lane_id(direction, lane)
            lines.append(
                f'    <laneAreaDetector id="{did}" lane="{lid}"'
                f' pos="0" length="{length}" freq="1" file="nul"/>'
            )
    lines.append("</additional>")
    OUT_ADD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sumocfg() -> None:
    OUT_CFG.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '    xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n'
        '    <input>\n'
        '        <net-file value="osm_cut.net.xml"/>\n'
        '        <route-files value="osm_cut_rl.rou.xml"/>\n'
        '        <additional-files value="osm_cut_rl.add.xml"/>\n'
        '    </input>\n'
        '    <time>\n'
        '        <begin value="0"/>\n'
        '        <end value="1000"/>\n'
        '        <step-length value="0.1"/>\n'
        '    </time>\n'
        '    <processing>\n'
        '        <time-to-teleport value="120"/>\n'
        '        <ignore-junction-blocker value="10"/>\n'
        '    </processing>\n'
        '</configuration>\n',
        encoding="utf-8",
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(42)

    print("Loading input data...")
    tmc        = load_tmc()
    analytics  = load_analytics()
    summary    = load_summary()
    turn_ratios = compute_turn_ratios(tmc)

    print("\nTurn ratios from tmc.csv:")
    for d, r in turn_ratios.items():
        print(f"  {d}: " + ", ".join(f"{k}={v:.1%}" for k, v in r.items()))

    print("\nGenerating vehicles...")
    vehicles = generate_vehicles(tmc, turn_ratios, rng)

    print(f"  {len(vehicles)} vehicles (skipped Stationary)")

    # Per-direction stats
    by_dir: dict[str, int] = defaultdict(int)
    by_type: dict[str, int] = defaultdict(int)
    by_dest: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for v in vehicles:
        by_dir[v["direction"]] += 1
        by_type[v["type"]] += 1
        by_dest[v["direction"]][v["dest"]] += 1

    print(f"\n  by direction: {dict(by_dir)}")
    print(f"  by type:      {dict(by_type)}")

    print("\n  by direction->destination:")
    for d in ["E", "W", "N", "S"]:
        if d in by_dest:
            print(f"    {d}: " + ", ".join(f"{k}={v}" for k, v in sorted(by_dest[d].items())))

    print(f"\nWriting {OUT_ROU} ...")
    write_rou(vehicles)

    print(f"Writing {OUT_ADD} ...")
    write_add(vehicles)

    print(f"Writing {OUT_CFG} ...")
    write_sumocfg()

    print("\nDone. Files generated:")
    print(f"  {OUT_ROU}")
    print(f"  {OUT_ADD}")
    print(f"  {OUT_CFG}")
    print("\nRun:  python src/watch_simulation.py")


if __name__ == "__main__":
    main()
