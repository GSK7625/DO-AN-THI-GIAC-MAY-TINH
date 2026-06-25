DETECTOR_IDS = [
    "det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2",   # East
    "det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2",   # North
    "det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2",  # West
    "det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"  # South
]

TLS_ID = "cluster_53190763_5896114911"

GREEN_PHASES = [0, 2, 4, 6]

YELLOW_STEPS = 30

PHASE_DETECTORS = {
    0: ["det_428067759#0_0", "det_428067759#0_1", "det_428067759#0_2"],         # East
    1: ["det_428067750#0_0", "det_428067750#0_1", "det_428067750#0_2"],         # North
    2: ["det_428067756.116_0", "det_428067756.116_1", "det_428067756.116_2"],   # West
    3: ["det_-577951513_0", "det_-577951513_1", "det_-577951513_2", "det_-577951513_3"]  # South
}

PHASE_OUTGOING_EDGES = {
    0: ["160183267#0", "-428067750#1"],   # East → West & North
    1: ["428067754#0", "160183267#0"],   # North → South & West
    2: ["378376408#0", "428067754#0"],   # West → East & South
    3: ["-428067750#1", "378376408#0"]   # South → North & East
}

MIN_GREEN_STEPS = 50  # 5.0 giây tối thiểu
MAX_GREEN_STEPS = 500  # 50.0 giây tối đa
MAX_SIMULATION_TIME = 1000.0  # giây
