# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NPU single-sided communication buffer — the NPU counterpart of ``rdma.py``.

**Plan B architecture**: all connect/register/transfer operations go through
Rust (``hixl-sys`` → ``libtest_hixl.so`` → HiXL C++ API).  Python only deals
with metadata and the ``_RdmaBuffer`` PyO3 wrapper.

This file is **completely independent** from ``rdma.py``, so upstream GPU
code updates can be merged without any conflict.

Quick start::

    import torch, torch_npu
    from monarch._src.rdma.xdma import XDMABuffer

    t = torch.randn(1024, device="npu:0")
    buf = XDMABuffer(t)
    # ... send `buf` to consumer via Monarch actor ...
    await buf.read_into(dst_tensor)
"""

import logging
from typing import Optional

import torch

try:
    from monarch._rust_bindings.rdma import (
        _LocalMemoryHandle,
        _RdmaBuffer,
        _RdmaManager,
    )
except ImportError as e:
    logging.error("RDMA bindings not available: {}".format(e))
    raise e

from monarch._src.actor.future import Future

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers reused from rdma.py (duplicated to keep zero-import from rdma.py)
# ---------------------------------------------------------------------------

_rdma_manager: Optional[_RdmaManager] = None
_hixl_engine_inited: bool = False


def _ensure_hixl_engine_for_rust() -> None:
    """Initialise the HiXL engine via ctypes and pass the pointer to Rust.

    Must be called before ``_ensure_init_rdma_manager()`` so that the
    Rust ``HixlManagerActor`` finds the engine already present and skips
    its own (potentially broken) Rust-FFI initialisation path.
    """
    global _hixl_engine_inited
    if _hixl_engine_inited:
        return
    _hixl_engine_inited = True
    try:
        from monarch._src.rdma.hixl_transfer import init_for_rust
        init_for_rust()
    except Exception as exc:
        logger.warning("xdma: HiXL pre-init failed (non-fatal): %s", exc)


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
# XDMABuffer — the NPU equivalent of RDMABuffer
# ---------------------------------------------------------------------------


class XDMABuffer:
    """Single-sided communication buffer for NPU devices.

    All connect / register / transfer operations are handled by Rust
    (``hixl-sys`` ↔ ``libtest_hixl.so``).  Connection establishment is
    automatic and transparent: the first ``read_into`` / ``write_from``
    triggers a sequential connect handshake via actor messages.
    """

    def __init__(self, data: "torch.Tensor | memoryview") -> None:
        _ensure_hixl_engine_for_rust()
        _ensure_init_rdma_manager().block_on()

        handle = _make_local_memory_handle(data)
        if handle.size == 0:
            raise ValueError("Cannot create XDMABuffer with size 0.")

        ctx = context()
        self._buffer: _RdmaBuffer = _RdmaBuffer.create_rdma_buffer_blocking(
            local=handle,
            client=ctx.actor_instance,
        )

    def size(self) -> int:
        return self._buffer.size()

    def read_into(
        self,
        dst: "torch.Tensor | memoryview",
        *,
        timeout: int = 3,
    ) -> "Future[Optional[int]]":
        """Pull data from the remote buffer into *dst*.

        Connection establishment happens automatically on first use.
        """
        _ensure_hixl_engine_for_rust()
        handle = _make_local_memory_handle(dst)
        if self.size() > handle.size:
            raise ValueError(
                f"Destination size ({handle.size}) must be >= buffer size ({self.size()})"
            )

        ctx = context()
        client = ctx.actor_instance
        buffer = self._buffer

        async def _nonblocking() -> Optional[int]:
            await _ensure_init_rdma_manager()
            res = await buffer.read_into(dst=handle, client=client, timeout=timeout)
            return res

        return Future(coro=_nonblocking())

    def write_from(
        self,
        src: "torch.Tensor | memoryview",
        *,
        timeout: int = 3,
    ) -> "Future[None]":
        """Push data from *src* into the remote buffer."""
        _ensure_hixl_engine_for_rust()
        handle = _make_local_memory_handle(src)
        if handle.size > self.size():
            raise ValueError(
                f"Source size ({handle.size}) must be <= buffer size ({self.size()})"
            )

        ctx = context()
        client = ctx.actor_instance
        buffer = self._buffer

        async def _nonblocking() -> None:
            await _ensure_init_rdma_manager()
            await buffer.write_from(src=handle, client=client, timeout=timeout)

        return Future(coro=_nonblocking())

    def drop(self) -> "Future[None]":
        """Release the remote buffer handle."""
        client = context().actor_instance
        buffer = self._buffer

        async def _nonblocking() -> None:
            await _ensure_init_rdma_manager()
            await buffer.drop(client=client)

        return Future(coro=_nonblocking())

    @property
    def owner(self) -> str:
        return self._buffer.owner_actor_id()


# ---------------------------------------------------------------------------
# Convenience alias — users can `from xdma import RDMABuffer` for drop-in use
# ---------------------------------------------------------------------------
RDMABuffer = XDMABuffer
