import os
import sys
sys.path.append(os.path.dirname(__file__)+"/..")
from nmpc_controller import NMPCController
import time 
import numpy as np 
from quarc_stream import Streamer


if __name__ == "__main__":
    controller = NMPCController(1.504, np.array([0,0,-9.81]))
    with Streamer("tcpip://localhost:18002", 4, 26, Streamer.CLIENT) as client:
        ts = time.perf_counter()
        for i in range(10000):
            ti = time.perf_counter()
            state_state_des = client.receive()

            u = controller.calculate_control(state_state_des[:13], None,
                 state_state_des[13:26].reshape(1, -1))
            u[0] = u[0] - 1.504*9.81
            client.send(u)
            print(state_state_des)
            # sleept = max(1.7e-2 - (time.perf_counter() - ti), 0.0)
            print("U: ",u )
            print("Avg freq: ", i/(time.perf_counter() - ts))
            # time.sleep(sleept)
        tf = time.perf_counter()
    print("Total time: ", tf-ts)
    print("Avg. time per step: ", (tf-ts)/1000)
    print("Avg. frequency: ", 1000/(tf-ts))
