from quarc_stream import  Server
from threading import Thread 
import numpy as np

if __name__ == "__main__":

    server =  Server("tcpip://localhost:5001")
    for i in range(10):
        server.receive(5)
        print("Server Got: ", server.recv)
        print("Server sending: ", np.zeros((6)) + 1)
        server.send(np.zeros((6)) + 1)