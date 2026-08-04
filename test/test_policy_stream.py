import os
import sys
import bindings.bindings as bindings
import time
import numpy as np
from quarc_stream import Streamer
import numpy as np
from jetson_inference.runtime import TensorRTRunner
import time

# Sample time (100 Hz)
SAMPLE_TIME = 0.01

# Hardcoded model path
MODEL_PATH = r"ppo_trajectory_scaled_down2.onnx"


# Distance and offset parameters
Z_OFFSET = 1.0
MIN_DIST = 1.0
MAX_DIST = 1.0

# History sizes
ACTION_HISTORY_SIZE = 10
STATE_HISTORY_SIZE = 10

# Gate parameters
GATE_WIDTH = 0.1
GATE_HEIGHT = 0.1
GATE_CROSSING_RADIUS = 0.6

# IO
INPUT_DIMS = 186
OUTPUT_DIMS=6
runner = TensorRTRunner("artifacts/ppo_trajectory_scaled_down2_fp16.engine")

# Construct, allocate, and warm before arming.
state_host = np.empty((1, INPUT_DIMS), dtype=np.float32)
state_host.fill(0)
for _ in range(100):
    runner.infer({"observation": state_host})

# Optional. Keep disabled until the normal path is verified on the target model.
#runner.capture_cuda_graph()

def control_tick(latest_state):
    # Prefer writing preprocessing results directly into a reusable pinned tensor
    # for truly asynchronous H2D transfer. This NumPy assignment is illustrative.
    state_host[0, :] = latest_state
    outputs = runner.infer(
        {"observation": state_host},
        return_cpu=True,
        synchronize=True,
#        use_cuda_graph=True,
    )
    action = outputs["output_0"][0]  # persistent view, overwritten next call
    action[np.isnan(action)] = 0.0

    return action

# Trajectory processing
feasibility = bindings.FeasibilityLimits() # default
feasibility.max_tilt_rad = 1.1344640137963142
feasibility.max_normal_acceleration = 15.0
waypoint_generator =  bindings.WaypointGenerator()
CURVE_TYPE = "circle"
if CURVE_TYPE == "circle":
    waypoints = np.array(waypoint_generator.GenerateCircleWaypoints(1.5, 0,0,0.8, 0.8, 0.5, feasibility))
else:
    waypoints = np.array(waypoint_generator.GenerateLissajousWaypoints(1.5, 1.5, 2*np.pi/12, 2*np.pi/6, np.pi/2,0, 0.5, 0.8, 0.8, feasibility))
xyz_scaling = np.array([0.10, 0.10, 0.03])
vxyz_scaling = np.array([0.024,0.024, 0.008])*0.8

preprocessor = bindings.TrajectoryPlanningProcessor(ACTION_HISTORY_SIZE, STATE_HISTORY_SIZE, True, True,
                                                    waypoints, 1.0, 1.0, 0.4, xyz_scaling, vxyz_scaling, bindings.GateAcceptanceMode.FullRectangularOpening )

preprocessor.reset(0)
drone_states =np.zeros((13,1)) # must have
drone_state_dots =np.zeros((13,1)) # keep empty
prev_drone_states =np.zeros((13,1)) # must have
prev_drone_state_dots =np.zeros((13,1)) # keep empty
latest_state = state_host.copy()
print("Starting pos: ", waypoints[0][:3])

if __name__ == "__main__":
    received_initial = False
    with Streamer("tcpip://localhost:18002", OUTPUT_DIMS, 13, Streamer.CLIENT) as client:
        ts = time.perf_counter()
        for i in range(100000):
            ti = time.perf_counter()
            drone_states[:, 0] = client.receive()
            if not received_initial:
                prev_drone_states[:] = drone_states[:]
                received_initial = True 
                continue
            latest_state[0, :] = preprocessor.ProcessObservation(drone_states, drone_state_dots,
                                                                prev_drone_states, prev_drone_state_dots)
            u = control_tick(latest_state)
            client.send(u)
            # sleept = max(1.7e-2 - (time.perf_counter() - ti), 0.0)
            print("U: ",u )
            print("Avg freq: ", i/(time.perf_counter() - ts))
            # time.sleep(sleept)
        tf = time.perf_counter()
    print("Total time: ", tf-ts)
    print("Avg. time per step: ", (tf-ts)/1000)
    print("Avg. frequency: ", 1000/(tf-ts))

