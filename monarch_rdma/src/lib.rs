/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// RDMA requires frequent unsafe code blocks
#![allow(clippy::undocumented_unsafe_blocks)]

use std::sync::Arc;

use local_memory::RdmaLocalMemory;
use serde::Deserialize;
use serde::Serialize;

#[macro_use]
mod macros;

pub mod backend;
#[cfg(not(feature = "hixl"))]
pub mod config;
#[cfg(not(feature = "hixl"))]
pub mod device_selection;
#[cfg(not(feature = "hixl"))]
pub mod efa;
pub mod local_memory;
mod rdma_components;
mod rdma_manager_actor;

#[cfg(not(feature = "hixl"))]
pub use backend::ibverbs::primitives::*;

/// Whether any RDMA backend is available on this system.
///
/// Returns true if ibverbs hardware is present, or if TCP fallback
/// is enabled via [`config::RDMA_ALLOW_TCP_FALLBACK`].
#[cfg(not(feature = "hixl"))]
pub fn rdma_supported() -> bool {
    ibverbs_supported() || hyperactor_config::global::get(config::RDMA_ALLOW_TCP_FALLBACK)
}
pub use rdma_components::RdmaRemoteBuffer;
pub use rdma_components::SegmentScannerFn;
pub use rdma_components::register_segment_scanner;
pub use rdma_components::*;
pub use rdma_manager_actor::*;
#[cfg(not(feature = "hixl"))]
pub use rdmaxcel_sys;
#[cfg(not(feature = "hixl"))]
pub use test_utils::is_cuda_available;

/// Type of RDMA operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RdmaOpType {
    ReadIntoLocal,
    WriteFromLocal,
}

/// A single RDMA operation to be submitted to a backend.
#[derive(Debug)]
pub struct RdmaOp {
    pub op_type: RdmaOpType,
    pub local: Arc<dyn RdmaLocalMemory>,
    pub remote: RdmaRemoteBuffer,
}

/// Transport level for single-sided communication, ordered slowest to fastest.
///
/// Used to describe or select the underlying interconnect.
/// On GPU: typically `Nic` (RoCE/InfiniBand via rdmaxcel).
/// On NPU: `Nic` for inter-supernode (RDMA/RoCE via HIXL),
///          `Hccs` for intra-supernode (HCCS via HIXL),
///          chosen automatically by the HIXL library based on topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RdmaTransportLevel {
    /// TCP/IP sockets (fallback transport).
    Tcp,
    /// RDMA NIC (RoCE, InfiniBand, EFA) — inter-supernode on NPU.
    Nic,
    /// HCCS interconnect (Ascend NPU intra-supernode). Higher bandwidth and
    /// lower latency than NIC, supports both collective and single-sided ops.
    Hccs,
    /// Direct memory access (NVLink, shared memory).
    Memory,
}

#[cfg(not(feature = "hixl"))]
pub fn print_device_info_if_debug_enabled(context: *mut rdmaxcel_sys::ibv_context) {
    if std::env::var("MONARCH_DEBUG_RDMA").is_ok() {
        unsafe {
            rdmaxcel_sys::rdmaxcel_print_device_info(context);
        }
    }
}

#[cfg(not(feature = "hixl"))]
pub fn print_device_info(context: *mut rdmaxcel_sys::ibv_context) {
    unsafe {
        rdmaxcel_sys::rdmaxcel_print_device_info(context);
    }
}

#[cfg(not(feature = "hixl"))]
mod test_utils;
