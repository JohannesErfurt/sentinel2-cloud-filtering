"""Small helpers shared across the pipeline."""
from __future__ import annotations

import sys


def peak_memory_mb() -> float | None:
    """Peak resident set size of this process in MB, or ``None`` if unavailable.

    Reports what the operating system actually reserved, not what Python
    allocated: numpy buffers and the GDAL block cache both count, and both are
    where this pipeline's memory goes. Falls back through the three ways of
    asking, because none of them works everywhere.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            # Without explicit argtypes/restype, ctypes assumes plain c_int for
            # the HANDLE arguments. That truncates the pointer on 64-bit Python
            # and GetProcessMemoryInfo then fails (returns 0) without raising --
            # it looks like "no memory info available" instead of a bug.
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            return None
        return None

    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS bytes.
        divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
        return peak / divisor
    except Exception:
        return None


def human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return "%.1f %s" % (value, unit)
        value /= 1024.0
    return "%.1f GB" % value


__all__ = ["peak_memory_mb", "human_bytes"]
