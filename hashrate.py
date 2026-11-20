"""Report completed scans on stderr without waiting for a log consumer."""
from collections import deque
from contextlib import contextmanager
import io
import os
import stat
import sys
import threading
import time


@contextmanager
def _status_writer(stream):
    if stream is None:
        yield None
        return
    if isinstance(stream, io.StringIO):
        yield stream.write
        return
    try:
        original = stream.fileno()
        if stat.S_ISREG(os.fstat(original).st_mode):
            fd = os.dup(original)
        else:
            # On Linux, opening this path gives us independent file status flags.
            # Setting O_NONBLOCK on a dup would also change the application's stderr.
            flags = os.O_WRONLY | os.O_NONBLOCK | getattr(os, "O_NOCTTY", 0)
            fd = os.open(f"/proc/self/fd/{original}", flags)
    except (AttributeError, OSError, ValueError):
        yield None
        return
    try:
        yield lambda text: os.write(fd, text.encode("ascii"))
    finally:
        os.close(fd)


class RateDisplay:
    def __init__(self, *, benchmark=False, stream=None, clock=None):
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.perf_counter if clock is None else clock
        self.benchmark = benchmark
        self.started = None
        self.phase_name = "warm-up" if benchmark else "starting"
        self.last_rate = None
        self.total_macs = 0
        self.scans = 0
        self.condition = threading.Condition()
        self.pending = deque(maxlen=16)
        self.closed = False

    def __enter__(self):
        now = self.clock()
        self.started = now
        self.pending.append(self._line(now))
        self.thread = threading.Thread(target=self._report, name="hashrate-display", daemon=True)
        try:
            self.thread.start()
        except RuntimeError:
            # Mining does not depend on the availability of a reporting thread.
            self.thread = None
        return self

    def __exit__(self, exc_type, exc, traceback):
        with self.condition:
            if not self.benchmark or exc_type is not None:
                self.phase_name = "stopped" if exc_type is not None else "finished"
                self.pending.append(self._line(self.clock()))
            self.closed = True
            self.condition.notify()
        if self.thread is not None:
            self.thread.join(timeout=0.2)

    def after_warmup(self, now):
        with self.condition:
            self.started = now
            self.phase_name = "scanning"
            self.pending.append(self._line(now))
            self.condition.notify()

    def phase(self, name):
        with self.condition:
            self.phase_name = name

    def end_benchmark(self):
        # The final completed line uses the exact same end time as the JSON result.
        with self.condition:
            self.closed = True
            self.condition.notify()

    def completed(self, macs, scan_seconds, now, phase="scan complete"):
        with self.condition:
            self.scans += 1
            self.total_macs += macs
            self.last_rate = macs / scan_seconds / 1e12 if scan_seconds > 0 else None
            self.phase_name = phase
            self.pending.append(self._line(now))
            self.condition.notify()

    def _report(self):
        with _status_writer(self.stream) as write:
            if write is None:
                return
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.pending, timeout=10)
                    if self.pending:
                        line = self.pending.popleft()
                    elif self.closed:
                        return
                    else:
                        line = self._line(self.clock())
                # No I/O occurs under the lock shared with the mining thread.
                try:
                    write(line + "\n")
                except (BlockingIOError, InterruptedError):
                    continue
                except (OSError, ValueError):
                    return

    def _line(self, now):
        elapsed = 0.0 if self.started is None else max(0.0, now - self.started)
        avg = self.total_macs / elapsed / 1e12 if self.scans and elapsed > 0 else None
        last_text = "--" if self.last_rate is None else f"{self.last_rate:.2f}"
        avg_text = "--" if avg is None else f"{avg:.2f}"
        return (f"Cyberminer | {self.phase_name} | last scan {last_text} TH/s | "
                f"session avg {avg_text} TH/s | scans {self.scans} | elapsed {elapsed:.1f}s")
