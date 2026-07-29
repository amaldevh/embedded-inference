import os
import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from nmpc_controller import NMPCController
import time 
import numpy as np 

if __name__ == "__main__":
    controller = NMPCController(1, np.array([0,0,-9.81]))
    state = np.zeros((13))
    state[6] = 1
    state_dot = state.copy()
    desired = np.array((1, 1, 1, 0, 0 , 0, 1, 0, 0,0))
    ts = time.perf_counter()
    for i in range(1000):
        controller.calculate_control(state, state_dot, desired)
    tf = time.perf_counter()
    print("Total time: ", tf-ts)
    print("Avg. time per step: ", (tf-ts)/1000)
    print("Avg. frequency: ", 1000/(tf-ts))
