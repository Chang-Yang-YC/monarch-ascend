# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NPU single-sided communication buffer — the NPU counterpart of ``rdma.py``.

On GPU, ``RDMABuffer`` in ``rdma.py`` uses ibverbs via Rust.
On NPU, ``XDMABuffer`` here uses pluggable transport backends (HiXL, etc.)
via the :mod:`transport` registry — all transfers happen in Python.

This file is **completely independent** from ``rdma.py``, so upstream GPU
code updates can be merged without any conflict.

Quick start::

    import torch, torch_npu
    from monarch._src.rdma.xdma import XDMABuffer

    t = torch.randn(1024, device="npu:0")
    buf = XDMABuffer(t)
    # ... send `buf` to consumer via Monarch actor ...
    buf.read_into(dst_tensor).get()
"""

import logging
from typing import List, Optional, Tuple

import torch
from monarch._rust_bindings.monarch_hyperactor.pytokio import PythonTask, Shared
from monarch._src.actor.proc_mesh import ProcMesh
from monarch._src.rdma.transport import get_transport, try_read_into, try_write_from

try:
    from monarch._rust_bindings.rdma import (
        _LocalMemoryHandle,
        _RdmaBuffer,
        _RdmaManager,
    )
except ImportError as e:
    logging.error("RDMA bindings not available: {}".format(e))
    raise e

from collections import defaultdict
from enum import Enum
from typing import Dict

from typing_extensions import Self

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers reused from rdma.py (duplicated to keep zero-import from rdma.py)
# ---------------------------------------------------------------------------

_rdma_manager: Optional[_RdmaManager] = None


def _ensure_init_rdma_manager():
    """Lazily create the process-global RdmaManager. Returns a Shared[None]."""
    from monarch._src.rdma.rdma import _ensure_init_rdma_manager as _upstream_init
    return _upstream_init()


def _make_local_memory_handle(data) -> _LocalMemoryHandle:
    from monarch._src.rdma.rdma import _make_local_memory_handle as _upstream_handle
    return _upstream_handle(data)


def context():
    from monarch._src.rdma.rdma import context as _upstream_ctx
    return _upstream_ctx()


# ---------------------------------------------------------------------------
# Future — minimal async wrapper (same pattern as rdma.py)
# ---------------------------------------------------------------------------

class Future:
    """Lightweight async wrapper compatible with Monarch's event loop."""

    def __init__(self, coro=None, task=None, shared=None):
        if coro is not None:
            self._task = PythonTask.from_coroutine(coro)
            self._shared: Shared = self._task.spawn()
        elif task is not None:
            self._task = task
            self._shared = self._task.spawn()
        elif shared is not None:
            self._shared = shared
            self._task = None
        else:
            raise ValueError("Must provide one of coro, task, or shared")

    def get(self, timeout: Optional[float] = None):
        return self._shared.block_on()

    def __await__(self):
        return self._shared.__await__()


# ---------------------------------------------------------------------------
# XDMABuffer — the NPU equivalent of RDMABuffer
# ---------------------------------------------------------------------------

class XDMABuffer:
    """Single-sided communication buffer for NPU devices.

    Uses the Monarch Rust buffer machinery for metadata / actor lifecycle,
    but routes **all data transfers** through the Python transport registry.
    This keeps the GPU code path (``rdma.py``) completely untouched.
    """

    def __init__(self, data: torch.Tensor | memoryview) -> None:
        backend = get_transport()
        if backend is not None:
            backend.get_engine_id()

        _ensure_init_rdma_manager().block_on()

        handle = _make_local_memory_handle(data)
        if handle.size == 0:
            raise ValueError("Cannot create XDMABuffer with size 0.")

        ctx = context()
        self._buffer: _RdmaBuffer = _RdmaBuffer.create_rdma_buffer_blocking(
            local=handle,
            client=ctx.actor_instance,
        )

        if backend is not None:
            ext_info = self._buffer.external_backend_info()
            if ext_info is not None:
                backend.register_mem(handle.addr, handle.size)

    def size(self) -> int:
        return self._buffer.size()

    def read_into(
        self,
        dst: torch.Tensor | memoryview,
        *,
        timeout: int = 3,
    ) -> Future:
        """Pull data from the remote buffer into *dst*."""
        handle = _make_local_memory_handle(dst)
        if self.size() > handle.size:
            raise ValueError(
                f"Destination size ({handle.size}) must be >= buffer size ({self.size()})"
            )

        ext_info = self._buffer.external_backend_info()
        if ext_info is None:
            raise RuntimeError(
                "XDMABuffer requires an external transport backend. "
                "Use RDMABuffer for ibverbs (GPU) transfers."
            )

        a, s = handle.addr, handle.size

        async def _do_read() -> Optional[int]:
            try_read_into(ext_info, a, s)
            return None

        return Future(coro=_do_read())

    def write_from(
        self,
        src: torch.Tensor | memoryview,
        *,
        timeout: int = 3,
    ) -> Future:
        """Push data from *src* into the remote buffer.

        HIXL WRITE (PUT) requires a bidirectional connection. We achieve
        this by having the **local** engine explicitly connect to the
        remote engine — because the remote side already initialized its
        HIXL engine (when it created the XDMABuffer), HIXL's
        ``AutoConnect=1`` mode will accept the incoming connection from
        the local side. We just need both sides to call ``Connect``.
        """
        handle = _make_local_memory_handle(src)
        if handle.size > self.size():
            raise ValueError(
                f"Source size ({handle.size}) must be <= buffer size ({self.size()})"
            )

        ext_info = self._buffer.external_backend_info()
        if ext_info is None:
            raise RuntimeError(
                "XDMABuffer requires an external transport backend. "
                "Use RDMABuffer for ibverbs (GPU) transfers."
            )

        a, s = handle.addr, handle.size

        async def _do_write() -> None:
            try_write_from(ext_info, a, s)

        return Future(coro=_do_write())

    def drop(self) -> Future:
        """Release the remote buffer handle."""
        client = context().actor_instance

        async def _do_drop() -> None:
            await _ensure_init_rdma_manager()
            await self._buffer.drop(client=client)

        return Future(coro=_do_drop())

    @property
    def owner(self) -> str:
        return self._buffer.owner_actor_id()


# ---------------------------------------------------------------------------
# Convenience alias — users can `from xdma import RDMABuffer` for drop-in use
# ---------------------------------------------------------------------------
RDMABuffer = XDMABuffer
