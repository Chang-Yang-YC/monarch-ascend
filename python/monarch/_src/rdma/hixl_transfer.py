# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
HiXL transport backend for Ascend NPU single-sided communication.

Implements :class:`TransportBackend` so it can be discovered and used
by Monarch's transport registry without any hardcoded imports.

All HIXL operations (init, register, connect, transfer) use ctypes to
call into a C-API shim library (``libtest_hixl.so`` or ``libcann_hixl.so``).
"""

import ctypes
import logging
import os
import threading
import time
from typing import Optional

from monarch._src.rdma.transport import TransportBackend

logger = logging.getLogger(__name__)

_LIB_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tests", "hixl", "build", "libtest_hixl.so"),
    "/root/monarch/tests/hixl/build/libtest_hixl.so",
]

_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_ctx = None
_engine_id: Optional[str] = None
_connected_peers: set = set()
_registered_addrs: set = set()


def _find_lib_path() -> Optional[str]:
    env = os.environ.get("MONARCH_HIXL_LIB")
    if env and os.path.isfile(env):
        return env
    for p in _LIB_SEARCH_PATHS:
        rp = os.path.realpath(p)
        if os.path.isfile(rp):
            return rp
    return None


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    path = _find_lib_path()
    if path is None:
        raise FileNotFoundError(
            "Cannot find libtest_hixl.so. Set MONARCH_HIXL_LIB env var."
        )
    _lib = ctypes.CDLL(path)

    _lib.hixl_init_engine.restype = ctypes.c_void_p
    _lib.hixl_init_engine.argtypes = [ctypes.c_int, ctypes.c_char_p]

    _lib.hixl_register_mem.restype = ctypes.c_int
    _lib.hixl_register_mem.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]

    _lib.hixl_connect.restype = ctypes.c_int
    _lib.hixl_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    _lib.hixl_transfer_write.restype = ctypes.c_int
    _lib.hixl_transfer_write.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]

    _lib.hixl_transfer_read.restype = ctypes.c_int
    _lib.hixl_transfer_read.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ]

    logger.info("HIXL: loaded %s", path)
    return _lib


def compute_engine_id() -> str:
    """Return the engine_id for this process (ip:port) without initializing HIXL.

    Uses the same formula as the Rust ``global_engine_id()`` in
    ``manager_actor.rs`` so both sides agree on the identity.
    """
    env = os.environ.get("MONARCH_PYTHON_HIXL_ENGINE_ID")
    if env:
        return env
    ip = os.environ.get("MONARCH_HIXL_IP", "127.0.0.1")
    port = 20000 + (os.getpid() % 40000)
    return f"{ip}:{port}"


def _ensure_init() -> None:
    """Lazily initialize the HIXL engine on first use (call under _lock)."""
    global _ctx, _engine_id
    if _ctx is not None:
        return
    lib = _load_lib()
    dev = int(os.environ.get("MONARCH_NPU_DEVICE", "0"))
    eid = compute_engine_id()
    os.environ["MONARCH_PYTHON_HIXL_ENGINE_ID"] = eid
    logger.warning("HIXL: init engine_id=%s dev=%d pid=%d", eid, dev, os.getpid())
    ctx = lib.hixl_init_engine(dev, eid.encode())
    if not ctx:
        raise RuntimeError(f"hixl_init_engine failed for dev={dev} engine_id={eid}")
    _ctx = ctx
    _engine_id = eid
    logger.warning("HIXL: engine ready: %s ctx=%s", eid, hex(ctx))


def _ensure_connected(remote_engine_id: str) -> None:
    if remote_engine_id in _connected_peers:
        return
    lib = _load_lib()
    remote_bytes = remote_engine_id.encode()
    for attempt in range(30):
        ret = lib.hixl_connect(_ctx, remote_bytes)
        if ret == 0:
            break
        logger.warning(
            "HIXL: connect attempt %d/30 to %s failed: %d",
            attempt + 1, remote_engine_id, ret,
        )
        if attempt < 29:
            time.sleep(1.0)
    else:
        raise RuntimeError(f"HIXL connect to {remote_engine_id} failed after 30 attempts")
    _connected_peers.add(remote_engine_id)
    logger.warning("HIXL: connected to %s", remote_engine_id)


def _ensure_registered(addr: int, size: int) -> None:
    if addr in _registered_addrs:
        return
    lib = _load_lib()
    ret = lib.hixl_register_mem(_ctx, addr, size)
    if ret != 0:
        raise RuntimeError(f"HIXL register_mem(addr={hex(addr)}, size={size}) failed: {ret}")
    _registered_addrs.add(addr)


# ---------------------------------------------------------------------------
# Public transfer functions (kept for backward compat and direct use)
# ---------------------------------------------------------------------------

def init_for_rust() -> str:
    """Initialise the HiXL engine via ctypes and pass the pointer to Rust.

    Call this from an actor process (after ``torch.npu.set_device()``)
    **before** any ``RDMABuffer`` / ``XDMABuffer`` is created so that the
    ``HixlManagerActor`` picks up the ctypes-created engine instead of
    calling ``hixl_init_engine`` through the Rust FFI path (which may fail
    on certain platforms due to ACL context / thread-affinity issues).

    Returns the engine_id string.
    """
    with _lock:
        _ensure_init()
        assert _ctx is not None and _engine_id is not None
        try:
            from monarch._rust_bindings.rdma import _RdmaBuffer
            _RdmaBuffer.set_hixl_engine(_ctx, _engine_id)
            logger.warning(
                "HIXL: passed ctypes engine to Rust: eid=%s ptr=%s",
                _engine_id,
                hex(_ctx),
            )
        except Exception as exc:
            logger.warning("HIXL: set_hixl_engine failed (non-fatal): %s", exc)
        return _engine_id


def get_engine_id() -> str:
    with _lock:
        _ensure_init()
        assert _engine_id is not None
        return _engine_id


def register_mem(addr: int, size: int) -> None:
    with _lock:
        _ensure_init()
        _ensure_registered(addr, size)


def transfer_write(
    remote_engine_id: str,
    local_addr: int,
    local_size: int,
    remote_addr: int,
) -> None:
    with _lock:
        _ensure_init()
        _ensure_registered(local_addr, local_size)
        _ensure_connected(remote_engine_id)

    remote_bytes = remote_engine_id.encode()
    logger.warning(
        "HIXL: WRITE local=%s size=%d -> remote=%s on %s",
        hex(local_addr), local_size, hex(remote_addr), remote_engine_id,
    )

    with _lock:
        ret = _load_lib().hixl_transfer_write(
            _ctx, remote_bytes, local_addr, remote_addr, local_size,
        )

    if ret != 0:
        raise RuntimeError(
            f"HIXL transfer WRITE to {remote_engine_id} failed: ret={ret}"
        )


def transfer_read(
    remote_engine_id: str,
    local_addr: int,
    local_size: int,
    remote_addr: int,
) -> None:
    with _lock:
        _ensure_init()
        _ensure_registered(local_addr, local_size)
        _ensure_connected(remote_engine_id)

    remote_bytes = remote_engine_id.encode()
    logger.warning(
        "HIXL: READ remote=%s on %s -> local=%s size=%d",
        hex(remote_addr), remote_engine_id, hex(local_addr), local_size,
    )

    with _lock:
        ret = _load_lib().hixl_transfer_read(
            _ctx, remote_bytes, local_addr, remote_addr, local_size,
        )

    if ret != 0:
        raise RuntimeError(
            f"HIXL transfer READ from {remote_engine_id} failed: ret={ret}"
        )


# ---------------------------------------------------------------------------
# TransportBackend implementation
# ---------------------------------------------------------------------------

class HixlTransportBackend(TransportBackend):
    """Ascend NPU single-sided transport via HiXL."""

    def name(self) -> str:
        return "hixl"

    def is_available(self) -> bool:
        try:
            return _find_lib_path() is not None
        except Exception:
            return False

    def get_engine_id(self) -> str:
        return get_engine_id()

    def transfer_read(
        self,
        remote_engine_id: str,
        local_addr: int,
        local_size: int,
        remote_addr: int,
    ) -> None:
        transfer_read(remote_engine_id, local_addr, local_size, remote_addr)

    def transfer_write(
        self,
        remote_engine_id: str,
        local_addr: int,
        local_size: int,
        remote_addr: int,
    ) -> None:
        transfer_write(remote_engine_id, local_addr, local_size, remote_addr)

    def register_mem(self, addr: int, size: int) -> None:
        register_mem(addr, size)

    def ensure_peer_connected(self, remote_engine_id: str) -> None:
        with _lock:
            _ensure_init()
            _ensure_connected(remote_engine_id)


def create_backend() -> TransportBackend:
    """Factory function called by the transport registry for auto-discovery."""
    return HixlTransportBackend()
