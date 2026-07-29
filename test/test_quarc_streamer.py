import numpy as np

from quarc_stream import Streamer


class FakeClientStream:
    def __init__(
        self,
        received=None,
        receive_result=1,
        send_result=1,
        flush_result=0,
    ):
        self.received = np.asarray(
            np.full(6, -1.0) if received is None else received,
            dtype=np.float64,
        )
        self.receive_result = receive_result
        self.send_result = send_result
        self.flush_result = flush_result
        self.connect_args = None
        self.sent = None
        self.events = []
        self.shutdown_calls = 0
        self.close_calls = 0

    def connect(
        self,
        address,
        non_blocking,
        send_buffer_size,
        receive_buffer_size,
    ):
        self.connect_args = (
            address,
            non_blocking,
            send_buffer_size,
            receive_buffer_size,
        )
        return True

    def send_double_array(self, values, count):
        self.events.append("send")
        self.sent = np.array(values[:count], copy=True)
        return self.send_result

    def flush(self):
        self.events.append("flush")
        return self.flush_result

    def receive_double_array(self, destination, count):
        self.events.append("receive")
        if self.receive_result == 1:
            destination[:count] = self.received[:count]
        return self.receive_result

    def shutdown(self):
        self.shutdown_calls += 1

    def close(self):
        self.close_calls += 1


def make_client(fake_stream):
    return Streamer(
        "tcpip://localhost:5001",
        5,
        6,
        Streamer.CLIENT,
        stream_factory=lambda: fake_stream,
    )


def test_client_sends_then_receives_and_closes_once():
    fake_stream = FakeClientStream()

    with make_client(fake_stream) as client:
        assert client.send(np.ones(5, dtype=np.float32)) == 5
        received = client.receive()
        assert np.array_equal(received, np.full(6, -1.0))
        client.close()

    assert fake_stream.connect_args == (
        "tcpip://localhost:5001",
        False,
        5 * 8,
        6 * 8,
    )
    assert fake_stream.events == ["send", "flush", "receive"]
    assert fake_stream.sent.dtype == np.float64
    assert fake_stream.sent.flags.c_contiguous
    assert fake_stream.shutdown_calls == 1
    assert fake_stream.close_calls == 1


def test_receive_reports_peer_disconnect():
    import pytest

    client = make_client(FakeClientStream(receive_result=0))
    try:
        with pytest.raises(EOFError, match="closed by the peer"):
            client.receive()
    finally:
        client.close()


def test_flush_accepts_old_and_new_quanser_success_values():
    for flush_result in (0, 1):
        client = make_client(FakeClientStream(flush_result=flush_result))
        try:
            assert client.send(np.ones(5)) == 5
        finally:
            client.close()


def test_negative_flush_result_is_rejected():
    import pytest

    client = make_client(FakeClientStream(flush_result=-1))
    try:
        with pytest.raises(RuntimeError, match="flush returned"):
            client.send(np.ones(5))
    finally:
        client.close()


def test_send_rejects_wrong_shape_or_size():
    import pytest

    invalid_data = (
        np.ones(4),
        np.ones(6),
        np.ones((1, 5)),
    )
    for data in invalid_data:
        client = make_client(FakeClientStream())
        try:
            with pytest.raises(ValueError):
                client.send(data)
        finally:
            client.close()


def test_invalid_stream_type_is_rejected():
    import pytest

    for stream_type in (-1, 2, None, "client"):
        with pytest.raises(ValueError, match="stream_type"):
            Streamer(
                "tcpip://localhost:5001",
                5,
                6,
                stream_type,
                stream_factory=FakeClientStream,
            )


def test_nonpositive_sizes_are_rejected():
    import pytest

    for size in (0, -1):
        with pytest.raises(ValueError):
            Streamer(
                "tcpip://localhost:5001",
                size,
                6,
                Streamer.CLIENT,
                stream_factory=FakeClientStream,
            )


def run_client(address="tcpip://localhost:5001", iterations=10):
    with Streamer(address, 5, 6, Streamer.CLIENT) as client:
        for _ in range(iterations):
            request = np.ones(5)
            print("Client sending:", request)
            client.send(request)
            received = client.receive()
            print("Client got:", received)


if __name__ == "__main__":
    run_client()
