"""Dependency-free progress bars and estimates for concurrent inference."""

from collections import Counter
import concurrent.futures
import math
import sys
import time


def duration(seconds):
    if seconds is None:
        return "--:--:--"
    seconds = math.ceil(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressBar:
    def __init__(self, total, label, workers, *, stream=None, clock=None):
        self.total, self.label, self.workers = total, label, workers
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock if clock is not None else time.monotonic
        self.started = self.clock()
        self.last_display = float("-inf")
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interval = 1.0 if self.tty else 10.0
        self.completed, self.work_count, self.work_seconds = 0, 0, 0.0
        self.counts = Counter()
        self.closed = False

    def update(self, state, seconds):
        self.completed += 1
        self.counts[state] += 1
        # Cached or skipped records do not measure inference throughput.
        if state not in {"cached", "abstained", "missing_generation", "invalid_generation"}:
            self.work_count += 1
            self.work_seconds += seconds

    def estimate_remaining(self):
        remaining = max(0, self.total - self.completed)
        if not remaining:
            return 0.0
        if not self.work_count or self.work_seconds <= 0:
            return None
        mean_work_seconds = self.work_seconds / self.work_count
        return remaining * mean_work_seconds / self.workers

    def format_line(self):
        elapsed = max(0, self.clock() - self.started)
        fraction = self.completed / self.total if self.total else 1.0
        filled = min(16, int(16 * fraction))
        bar = "#" * filled + "-" * (16 - filled)
        remaining = self.estimate_remaining()
        total = elapsed + remaining if remaining is not None else None
        errors = sum(self.counts[state] for state in ("error", "missing_generation", "invalid_generation"))
        return (f"{self.label} [{bar}] {self.completed:,}/{self.total:,} "
                f"({fraction:5.1%}) elapsed {duration(int(elapsed))} ETA {duration(remaining)} "
                f"est total {duration(total)} cached={self.counts['cached']} errors={errors}")

    def display(self, *, force=False, final=False):
        now = self.clock()
        if self.closed or (not force and now - self.last_display < self.interval):
            return
        line = self.format_line()
        if self.tty:
            self.stream.write("\r\033[2K" + line + ("\n" if final else ""))
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self.last_display = now
        self.closed = final

    def close(self):
        if not self.closed and self.tty:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True


def concurrent_results(work, items, *, total, max_workers, label, stream=None):
    """Yield finished results immediately, refreshing progress during slow calls."""
    progress = ProgressBar(total, label, max_workers, stream=stream)
    progress.display(force=True, final=not total)
    if not total:
        return

    def timed(item):
        started = time.monotonic()
        state = work(item)
        return state, time.monotonic() - started

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = {executor.submit(timed, item) for item in items}
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED)
                results = []
                for future in done:
                    state, seconds = future.result()
                    progress.update(state, seconds)
                    results.append(state)
                progress.display(force=not pending, final=not pending)
                yield from results
    finally:
        progress.close()
