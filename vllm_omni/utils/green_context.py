import time
import os
import ctypes
import torch
import contextvars


# 当前请求ID（由模型forward处设置）
VLLM_REQ_ID = contextvars.ContextVar("VLLM_REQ_ID", default="-")
# 当前stage id（优先从 env 推断）
def _get_stage_id():
    # 这些名字你可以按你实际环境再补充；抓不到就 '?'
    for k in ("VLLM_OMNI_STAGE_ID", "OMNI_STAGE_ID", "VLLM_STAGE_ID", "STAGE_ID"):
        v = os.getenv(k)
        if v is not None:
            return v
    return "?"

def __enter__(self):
    if not self.enabled:
        return self
    self._prev = torch.cuda.current_stream()
    torch.cuda.set_stream(self.stream)

    # ✅ 第1行：enter日志（最小改动）
    print(f"[GreenCtx][enter] pid={os.getpid()} stage={_get_stage_id()} req={VLLM_REQ_ID.get()} "
          f"sm={self.actual_sm}/{self.total_sm} pct={self.percent} stream=0x{int(self.stream.cuda_stream):x}")

    return self

def __exit__(self, exc_type, exc, tb):
    if not self.enabled:
        return False
    torch.cuda.set_stream(self._prev)

    # ✅ 第2行：exit日志（最小改动）
    print(f"[GreenCtx][exit ] pid={os.getpid()} stage={_get_stage_id()} req={VLLM_REQ_ID.get()} "
          f"exc={'None' if exc_type is None else exc_type.__name__}")

    return False

class GreenContext:
    def __init__(self):
        self.enabled = os.getenv("VLLM_GREEN_SM_PERCENT") is not None
        self.handle = None
        self.stream = None
        self.actual_sm = None

        if not self.enabled:
            return

        percent = int(os.getenv("VLLM_GREEN_SM_PERCENT", "50"))
        device = int(os.getenv("VLLM_GREEN_DEVICE", "0"))
        lib = os.getenv("VLLM_GREEN_LIB", os.path.expanduser("~/Lucas/GreenContext/libgc_rt.so"))

        self._lib = ctypes.CDLL(lib)
        self._lib.gc_create_percent.argtypes = [
            ctypes.c_int, ctypes.c_int,
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
            device, percent,
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
        self.percent = percent
        self.total_sm = total_sm.value
        self.align = align.value
        self.max_valid = max_valid.value

    def __enter__(self):
        if not self.enabled:
            return self
        self._prev = torch.cuda.current_stream()
        torch.cuda.set_stream(self.stream)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        torch.cuda.set_stream(self._prev)
        return False

    def close(self):
        if self.enabled and self.handle:
            rc = self._lib.gc_destroy(self.handle)
            if rc != 0:
                raise RuntimeError(f"gc_destroy failed rc={rc}")
            self.handle = None

