import os
import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from nmpc_controller_safe import NMPCController
import time 
import numpy as np 
from quarc_stream import Streamer


def run_client(address="tcpip://localhost:5001", iterations=10):
    :
        for _ in range(iterations):
            request = np.ones(5)*_
            print("Client sending:", request)
            client.send(request)
            received = client.receive()
            print("Client got:", received)


if __name__ == "__main__":
    controller = NMPCController(1, np.array([0,0,-9.81]))
    state = np.zeros((13))
    state[6] = 1
    state_dot = state.copy()
    desired = np.array((1, 1, 1, 0, 0 , 0, 1, 0, 0,0))
    with Streamer("tcpip://192.168.2.12:18002", 4, 5, Streamer.CLIENT) as client:
        ts = time.perf_counter()
        for i in range(1000):
            u = controller.calculate_control(state, state_dot, desired)
            client.send(u)
        tf = time.perf_counter()
    print("Total time: ", tf-ts)
    print("Avg. time per step: ", (tf-ts)/1000)
    print("Avg. frequency: ", 1000/(tf-ts))
