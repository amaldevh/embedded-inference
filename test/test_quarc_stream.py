from quarc_stream import Streamer, Server
from threading import Thread 

if __name__ == "__main__":
    server = None
    def make_server():
        server =  Server("tcpip://localhost:5001")
    t = Thread(target=make_server)
    t.start()
    streamer = Streamer("tcpip://localhost:5001", 5, 6)
    send_val = np.zeros((5))
    recv_val = np.zeros((6))
    streamer.send(send_val +1)
    print("sent: ", send_val+1)
    server.receive(5)
    print("server recvd: ", server.recv)