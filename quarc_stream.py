from quanser.communications import Stream
import numpy as np


class Streamer:
    SERVER = 0
    CLIENT = 1
    def __init__(self, address, double_send_size, 
        double_receive_size, stream_type):
        send_bytes_size = double_send_size*8
        receive_bytes_size = double_receive_size*8
        self.stream = Stream()
        if stream_type == Streamer.CLIENT:
            self.stream.connect(address, False,
            send_bytes_size, receive_bytes_size)
        else:
            self._stream = self.stream
            self._stream.listen(address, False)
            self.stream = self._stream.accept(send_bytes_size, receive_bytes_size)
        self.recv = np.zeros((double_receive_size))

    def send(self, data):
        self.stream.send_double_array(data, len(data))
        self.stream.flush()

    def receive(self, ):
        result = self.stream.receive_double_array(self.recv, len(self.recv))

    def __del__(self):
        self.stream.shutdown()
        self.stream.close()
