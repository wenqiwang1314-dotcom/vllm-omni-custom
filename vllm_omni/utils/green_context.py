import contextvars
import ctypes
import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Request ID set by OmniGPUModelRunner._model_forward; used for per-request tracing.
VLLM_REQ_ID = contextvars.ContextVar("VLLM_REQ_ID", default="-")


def _get_stage_id() -> str:
    """Resolve current stage id from environment for stage-local SM policies."""
    for key in ("VLLM_OMNI_STAGE_ID", "OMNI_STAGE_ID", "VLLM_STAGE_ID", "STAGE_ID"):
        value = os.getenv(key)
        if value is not None:
            return str(value)
    return "?"


def _resolve_percent(stage_id: str) -> int | None:
    """Resolve SM percentage with stage-specific override first.

    Priority:
    1) VLLM_GREEN_SM_PERCENT_STAGE_<stage_id> (e.g. ..._STAGE_0)
    2) VLLM_GREEN_SM_PERCENT (global fallback)
    """
    stage_key = f"VLLM_GREEN_SM_PERCENT_STAGE_{stage_id}"
    raw = os.getenv(stage_key)
    if raw is None:
        raw = os.getenv("VLLM_GREEN_SM_PERCENT")
    if raw is None:
        return None
    return int(raw)


class GreenContext:
    def __init__(self):
        self.handle: int | None = None
        self.stream: torch.cuda.ExternalStream | None = None
        self.actual_sm: int | None = None
        self.total_sm: int | None = None
        self.align: int | None = None
        self.max_valid: int | None = None
        self.percent: int | None = None
        self._prev = None
        self.stage_id = _get_stage_id()

        # Critical behavior:
        # each stage process computes its own percent from env, so we can
        # sweep stage-0/1/2 with different SM budgets without touching YAML.
        resolved_percent = _resolve_percent(self.stage_id)
        self.enabled = resolved_percent is not None
        if not self.enabled:
            return

        self.percent = resolved_percent
        device = int(os.getenv("VLLM_GREEN_DEVICE", "0"))
        lib = os.getenv("VLLM_GREEN_LIB", os.path.expanduser("~/Lucas/GreenContext/libgc_rt.so"))

        self._lib = ctypes.CDLL(lib)
        self._lib.gc_create_percent.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.gc_create_percent.restype = ctypes.c_int
        self._lib.gc_destroy.argtypes = [ctypes.c_uint64]
        self._lib.gc_destroy.restype = ctypes.c_int

        h = ctypes.c_uint64(0)
        s = ctypes.c_uint64(0)
        actual_sm = ctypes.c_int(0)
        total_sm = ctypes.c_int(0)
        align = ctypes.c_int(0)
        max_valid = ctypes.c_int(0)

        torch.cuda.set_device(device)

        rc = self._lib.gc_create_percent(
            device,
            self.percent,
            ctypes.byref(h),
            ctypes.byref(s),
            ctypes.byref(actual_sm),
            ctypes.byref(total_sm),
            ctypes.byref(align),
            ctypes.byref(max_valid),
        )
        if rc != 0:
            raise RuntimeError(f"gc_create_percent failed rc={rc}")

        self.handle = h.value
        self.stream = torch.cuda.ExternalStream(s.value)
        self.actual_sm = actual_sm.value
        self.total_sm = total_sm.value
        self.align = align.value
        self.max_valid = max_valid.value

    def __enter__(self):
        if not self.enabled or self.stream is None:
            return self
        self._prev = torch.cuda.current_stream()
        torch.cuda.set_stream(self.stream)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        if self._prev is not None:
            torch.cuda.set_stream(self._prev)
        return False

    def close(self):
        if self.enabled and self.handle:
            rc = self._lib.gc_destroy(self.handle)
            if rc != 0:
                raise RuntimeError(f"gc_destroy failed rc={rc}")
            self.handle = None
