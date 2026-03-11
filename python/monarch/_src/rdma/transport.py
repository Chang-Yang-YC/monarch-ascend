# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Transport backend plugin interface for single-sided RDMA communication.

To add a new transport (e.g., HiXL for Ascend NPU, or a future backend):

    1. Create a module that implements ``TransportBackend``
    2. Call ``register_transport("my_backend", MyBackend())``

Monarch core never imports any backend directly. Discovery is:
    - Explicit: user calls ``register_transport()`` in bootstrap
    - Automatic: set ``MONARCH_TRANSPORT_BACKEND=hixl`` env var

The registry is process-global and thread-safe.
"""

import abc
import importlib
import logging
import os
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TransportBackend(abc.ABC):
    """Interface that every single-sided transport must implement."""

    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. ``"hixl"``, ``"rdmaxcel"``."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can operate in the current environment."""
        ...

    @abc.abstractmethod
    def get_engine_id(self) -> str:
        """Return the local engine identifier for this process."""
        ...

    @abc.abstractmethod
    def transfer_read(
        self,
        remote_engine_id: str,
        local_addr: int,
        local_size: int,
        remote_addr: int,
    ) -> None:
        """Pull ``local_size`` bytes from ``remote_addr`` on the remote engine
        into ``local_addr`` on the local device."""
        ...

    @abc.abstractmethod
    def transfer_write(
        self,
        remote_engine_id: str,
        local_addr: int,
        local_size: int,
        remote_addr: int,
    ) -> None:
        """Push ``local_size`` bytes from ``local_addr`` on the local device
        to ``remote_addr`` on the remote engine."""
        ...

    @abc.abstractmethod
    def register_mem(self, addr: int, size: int) -> None:
        """Register a device memory region for RDMA access."""
        ...

    def ensure_peer_connected(self, remote_engine_id: str) -> None:
        """Ensure bidirectional connection with a remote peer.

        Some transports (e.g., HiXL) require both sides to have an
        explicit connection before WRITE (PUT) operations succeed.
        Override if your transport needs it; default is a no-op.
        """
        pass


# ---------------------------------------------------------------------------
# Process-global registry
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_backends: Dict[str, TransportBackend] = {}
_active: Optional[TransportBackend] = None

# Backend module paths, keyed by short name.
# Each module must expose ``create_backend() -> TransportBackend``.
_BUILTIN_BACKENDS: Dict[str, str] = {
    "hixl": "monarch._src.rdma.hixl_transfer",
}


def register_transport(name: str, backend: TransportBackend) -> None:
    """Register a transport backend by name. Thread-safe."""
    with _lock:
        _backends[name] = backend
        logger.info("Registered transport backend: %s", name)


def get_transport(name: Optional[str] = None) -> Optional[TransportBackend]:
    """Look up a transport backend by name, or return the active default.

    Resolution order when *name* is ``None``:
        1. Previously activated backend (cached)
        2. ``MONARCH_TRANSPORT_BACKEND`` env var
        3. Probe registered backends via ``is_available()``
    """
    with _lock:
        return _resolve_locked(name)


def _resolve_locked(name: Optional[str]) -> Optional[TransportBackend]:
    global _active

    if name is not None:
        be = _backends.get(name)
        if be is not None:
            return be
        be = _try_load_builtin(name)
        if be is not None:
            return be
        return None

    if _active is not None:
        return _active

    env_name = os.environ.get("MONARCH_TRANSPORT_BACKEND")
    if env_name:
        be = _backends.get(env_name) or _try_load_builtin(env_name)
        if be is not None and be.is_available():
            _active = be
            return _active

    for be in _backends.values():
        if be.is_available():
            _active = be
            return _active

    for builtin_name in _BUILTIN_BACKENDS:
        if builtin_name not in _backends:
            be = _try_load_builtin(builtin_name)
            if be is not None and be.is_available():
                _active = be
                return _active

    return None


def _try_load_builtin(name: str) -> Optional[TransportBackend]:
    """Attempt to import a built-in backend module and call its factory."""
    module_path = _BUILTIN_BACKENDS.get(name)
    if module_path is None:
        return None
    try:
        mod = importlib.import_module(module_path)
        factory = getattr(mod, "create_backend", None)
        if factory is None:
            return None
        be: TransportBackend = factory()
        _backends[name] = be
        return be
    except Exception:
        logger.debug("Could not load builtin backend %r", name, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Intercept helpers — called from rdma.py to keep upstream diffs minimal
# ---------------------------------------------------------------------------

def try_register_mem(buffer_ext_info: Optional[Tuple[str, int]], addr: int, size: int) -> None:
    """Register memory with the external transport if applicable.

    Called once when an ``RDMABuffer`` is created. If ``buffer_ext_info``
    is ``None`` (i.e., native ibverbs path), this is a no-op.
    """
    if buffer_ext_info is None:
        return
    be = get_transport()
    if be is not None:
        be.register_mem(addr, size)


def try_read_into(
    buffer_ext_info: Optional[Tuple[str, int]],
    local_addr: int,
    local_size: int,
) -> bool:
    """Perform a read via the external transport. Returns True if handled."""
    if buffer_ext_info is None:
        return False
    remote_engine_id, remote_addr = buffer_ext_info
    be = get_transport()
    if be is None:
        raise RuntimeError("No transport backend available for external RDMA transfer")
    be.transfer_read(remote_engine_id, local_addr, local_size, remote_addr)
    return True


def try_write_from(
    buffer_ext_info: Optional[Tuple[str, int]],
    local_addr: int,
    local_size: int,
) -> bool:
    """Perform a write via the external transport. Returns True if handled."""
    if buffer_ext_info is None:
        return False
    remote_engine_id, remote_addr = buffer_ext_info
    be = get_transport()
    if be is None:
        raise RuntimeError("No transport backend available for external RDMA transfer")
    be.transfer_write(remote_engine_id, local_addr, local_size, remote_addr)
    return True
