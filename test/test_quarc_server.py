from quarc_stream import  Streamer
from threading import Thread 
import numpy as np

if __name__ == "__main__":

    server =  Streamer("tcpip://localhost:5001", 6, 5, Streamer.SERVER)
    for i in range(10):
        server.receive(5)
        print("Server Got: ", server.recv)
        print("Server sending: ", np.zeros((6)) + 1)
        server.send(np.zeros((6)) + 1)