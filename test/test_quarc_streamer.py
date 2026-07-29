from quarc_stream import Streamer
from threading import Thread 
import numpy as np

if __name__ == "__main__":
    streamer = Streamer("tcpip://localhost:5001", 5, 6, Streamer.CLIENT)
    for i in range(10):
        streamer.receive(6)
        print("Streamer Got: ", streamer.recv)
        print("Streamer sending: ", np.zeros((5)) + 1)
        streamer.send(np.zeros((5)) + 1)