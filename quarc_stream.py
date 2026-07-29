import operator
from threading import Lock

import numpy as np

try:
    from quanser.communications import Stream
except ImportError:
    Stream = None


class Streamer:
    SERVER = 0
    CLIENT = 1

    def __init__(
        self,
        address,
        double_send_size,
        double_receive_size,
        stream_type,
        stream_factory=None,
    ):
        if not isinstance(address, str) or not address:
            raise ValueError("address must be a non-empty string")
        if stream_type not in (self.SERVER, self.CLIENT):
            raise ValueError("stream_type must be Streamer.SERVER or Streamer.CLIENT")

        self.send_size = self._validate_size(double_send_size, "double_send_size")
        self.receive_size = self._validate_size(
            double_receive_size, "double_receive_size"
        )
        self.recv = np.zeros(self.receive_size, dtype=np.float64)

        factory = stream_factory if stream_factory is not None else Stream
        if factory is None:
            raise ImportError(
                "quanser.communications is required when no stream_factory is supplied"
            )
        if not callable(factory):
            raise TypeError("stream_factory must be callable")

        self.stream = None
        self._listener = None
        self._closed = False
        self._send_lock = Lock()
        self._receive_lock = Lock()

        send_buffer_size = self.send_size * np.dtype(np.float64).itemsize
        receive_buffer_size = self.receive_size * np.dtype(np.float64).itemsize

        try:
            if stream_type == self.CLIENT:
                self.stream = factory()
                connected = self.stream.connect(
                    address, False, send_buffer_size, receive_buffer_size
                )
                if connected is False:
                    raise ConnectionError(
                        "blocking Quanser stream connection did not complete"
                    )
            else:
                self._listener = factory()
                self._listener.listen(address, False)
                self.stream = self._listener.accept(
                    send_buffer_size, receive_buffer_size
                )
                if self.stream is None:
                    raise ConnectionError(
                        "blocking Quanser stream accept returned no connection"
                    )
        except Exception:
            self._close_quietly()
            raise

    @staticmethod
    def _validate_size(value, name):
        if isinstance(value, bool):
            raise TypeError("{} must be an integer".format(name))
        try:
            size = operator.index(value)
        except TypeError:
            raise TypeError("{} must be an integer".format(name))
        if size <= 0:
            raise ValueError("{} must be greater than zero".format(name))
        return size

    def _require_open(self):
        if self._closed or self.stream is None:
            raise RuntimeError("Quanser stream is closed")

    def send(self, data):
        self._require_open()
        values = np.asarray(data, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("send data must be a one-dimensional array")
        if values.size != self.send_size:
            raise ValueError(
                "send data has {} elements; expected {}".format(
                    values.size, self.send_size
                )
            )
        values = np.ascontiguousarray(values)

        with self._send_lock:
            self._require_open()
            result = self.stream.send_double_array(values, self.send_size)
            if result != 1:
                if result == 0:
                    raise ConnectionError("Quanser stream closed while sending")
                raise RuntimeError(
                    "Quanser send_double_array returned {}".format(result)
                )
            result = self.stream.flush()
            if result != 1:
                if result == 0:
                    raise ConnectionError("Quanser stream closed while flushing")
                raise RuntimeError("Quanser stream flush returned {}".format(result))
        return self.send_size

    def receive(self):
        self._require_open()
        with self._receive_lock:
            self._require_open()
            result = self.stream.receive_double_array(
                self.recv, self.receive_size
            )
            if result == 0:
                raise EOFError("Quanser stream was closed by the peer")
            if result != 1:
                raise RuntimeError(
                    "Quanser receive_double_array returned {}".format(result)
                )
        return self.recv

    def close(self):
        if self._closed:
            return

        self._closed = True
        stream = self.stream
        listener = self._listener
        self.stream = None
        self._listener = None
        errors = []

        if stream is not None:
            try:
                stream.shutdown()
            except Exception as exc:
                errors.append(exc)
            try:
                stream.close()
            except Exception as exc:
                errors.append(exc)

        if listener is not None and listener is not stream:
            try:
                listener.close()
            except Exception as exc:
                errors.append(exc)

        if errors:
            raise errors[0]

    def _close_quietly(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False

    def __del__(self):
        self._close_quietly()
