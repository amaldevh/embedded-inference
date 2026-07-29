import numpy as np

from quarc_stream import Streamer


class FakeConnection:
    def __init__(self, received):
        self.received = np.asarray(received, dtype=np.float64)
        self.sent = None
        self.shutdown_called = False
        self.close_called = False

    def receive_double_array(self, destination, count):
        destination[:count] = self.received[:count]
        return 1

    def send_double_array(self, values, count):
        self.sent = np.array(values[:count], copy=True)
        return 1

    def flush(self):
        return 1

    def shutdown(self):
        self.shutdown_called = True

    def close(self):
        self.close_called = True


class FakeListener:
    def __init__(self, connection):
        self.connection = connection
        self.listen_args = None
        self.accept_args = None
        self.close_called = False

    def listen(self, address, non_blocking):
        self.listen_args = (address, non_blocking)

    def accept(self, send_buffer_size, receive_buffer_size):
        self.accept_args = (send_buffer_size, receive_buffer_size)
        return self.connection

    def close(self):
        self.close_called = True


def test_server_receives_sends_and_closes_both_streams():
    connection = FakeConnection(np.ones(5))
    listener = FakeListener(connection)

    with Streamer(
        "tcpip://localhost:5001",
        6,
        5,
        Streamer.SERVER,
        stream_factory=lambda: listener,
    ) as server:
        received = server.receive()
        assert np.array_equal(received, np.ones(5))
        assert server.send(np.full(6, -1.0)) == 6

    assert listener.listen_args == ("tcpip://localhost:5001", False)
    assert listener.accept_args == (6 * 8, 5 * 8)
    assert listener.close_called
    assert connection.shutdown_called
    assert connection.close_called
    assert np.array_equal(connection.sent, np.full(6, -1.0))


def run_server(address="tcpip://localhost:5001", iterations=10):
    with Streamer(address, 6, 5, Streamer.SERVER) as server:
        for _ in range(iterations):
            received = server.receive()
            print("Server got:", received)
            reply = np.full(6, -1.0)
            print("Server sending:", reply)
            server.send(reply)


if __name__ == "__main__":
    run_server()
