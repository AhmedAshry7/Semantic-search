import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import psutil


@dataclass
class MemoryEvent:
    label: str
    start_rss: int
    end_rss: int
    start_uss: int
    end_uss: int
    duration: float

    def rss_delta(self) -> int:
        return self.end_rss - self.start_rss

    def uss_delta(self) -> int:
        return self.end_uss - self.start_uss


class MemoryReporter:
    """Lightweight helper to capture RSS/USS and tracemalloc diffs per block."""

    def __init__(self, top_lines: int = 15, trace_frames: int = 15):
        self.proc = psutil.Process()
        self.top_lines = top_lines
        self.events: List[MemoryEvent] = []
        self._tracemalloc_started = tracemalloc.is_tracing()
        if not self._tracemalloc_started:
            tracemalloc.start(trace_frames)

    def _fmt_bytes(self, value: int) -> str:
        if value < 1024:
            return f"{value} B"
        kb = value / 1024
        if kb < 1024:
            return f"{kb:.1f} KB"
        mb = kb / 1024
        if mb < 1024:
            return f"{mb:.2f} MB"
        gb = mb / 1024
        return f"{gb:.2f} GB"

    def _mem_info(self) -> tuple[int, int]:
        info = self.proc.memory_full_info()
        return info.rss, getattr(info, "uss", info.rss)

    def log_snapshot_diff(self, label: str, before, after) -> None:
        diff = after.compare_to(before, "lineno")
        print(f"\n[MemoryReporter] Hot allocations for '{label}':")
        for stat in diff[: self.top_lines]:
            size = self._fmt_bytes(stat.size_diff)
            count = stat.count_diff
            print(f"  {size:>10} | {count:>6}x | {stat.traceback.format()[-1].strip()}")

    @contextmanager
    def track(self, label: str, emit_tracemalloc: bool = True):
        start_rss, start_uss = self._mem_info()
        snap_before = tracemalloc.take_snapshot() if emit_tracemalloc else None
        t0 = time.time()
        try:
            yield
        finally:
            duration = time.time() - t0
            end_rss, end_uss = self._mem_info()
            snap_after = tracemalloc.take_snapshot() if emit_tracemalloc else None
            self.events.append(
                MemoryEvent(
                    label=label,
                    start_rss=start_rss,
                    end_rss=end_rss,
                    start_uss=start_uss,
                    end_uss=end_uss,
                    duration=duration,
                )
            )
            delta_rss = end_rss - start_rss
            delta_uss = end_uss - start_uss
            print(
                f"[MemoryReporter] {label}: RSS {self._fmt_bytes(start_rss)} -> {self._fmt_bytes(end_rss)} "
                f"(Δ {self._fmt_bytes(delta_rss)}), USS {self._fmt_bytes(start_uss)} -> {self._fmt_bytes(end_uss)} "
                f"(Δ {self._fmt_bytes(delta_uss)}), took {duration:.3f}s"
            )
            if emit_tracemalloc and snap_before and snap_after:
                self.log_snapshot_diff(label, snap_before, snap_after)

    def summary(self) -> None:
        if not self.events:
            print("[MemoryReporter] No events recorded.")
            return
        worst = sorted(self.events, key=lambda e: e.rss_delta(), reverse=True)[:10]
        print("\n[MemoryReporter] Top blocks by RSS growth:")
        for ev in worst:
            print(
                f"  {ev.label:30} | ΔRSS {self._fmt_bytes(ev.rss_delta())} | "
                f"ΔUSS {self._fmt_bytes(ev.uss_delta())} | {ev.duration:.3f}s"
            )

    def current(self) -> None:
        rss, uss = self._mem_info()
        print(f"[MemoryReporter] Current RSS {self._fmt_bytes(rss)}, USS {self._fmt_bytes(uss)}")

    def stop(self) -> None:
        if not self._tracemalloc_started:
            tracemalloc.stop()


def make_reporter(top_lines: int = 15) -> MemoryReporter:
    return MemoryReporter(top_lines=top_lines)
