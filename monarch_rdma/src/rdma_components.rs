/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Components
//!
//! Core RDMA building blocks for establishing and managing RDMA connections.
//! Supports ibverbs (GPU) and HIXL (NPU) backends via feature flags.

use std::sync::Arc;
use std::time::Duration;

use hyperactor::ActorRef;
use hyperactor::context;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::RdmaLocalMemory;
use crate::RdmaManagerActor;
use crate::RdmaOp;
use crate::RdmaOpType;
use crate::ReleaseBufferClient;
#[cfg(feature = "hixl")]
use crate::EnsurePeerConnectedClient;
use crate::backend::RdmaBackendContext;

#[cfg(not(feature = "hixl"))]
use crate::backend::RdmaBackend;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::IbvBuffer;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerActor;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerMessageClient;

/// Lightweight handle representing a registered RDMA buffer.
#[derive(Debug, Named, Clone, Serialize, Deserialize)]
pub struct RdmaRemoteBuffer {
    pub id: usize,
    pub size: usize,
    pub owner: ActorRef<RdmaManagerActor>,
    pub backends: Vec<RdmaBackendContext>,
}
wirevalue::register_type!(RdmaRemoteBuffer);

impl RdmaRemoteBuffer {
    /// Push data from local memory into this remote buffer (local->remote).
    #[cfg(not(feature = "hixl"))]
    pub async fn write_from_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let mut local_ibv_backend = IbvManagerActor::local_handle(client).await?;
        local_ibv_backend
            .submit(
                client,
                vec![RdmaOp {
                    op_type: RdmaOpType::WriteFromLocal,
                    local,
                    remote: self.clone(),
                }],
                Duration::from_secs(timeout),
            )
            .await?;
        Ok(true)
    }

    /// Push data from local memory into this remote buffer (external backend path).
    /// The actual transfer is handled by the Python transport plugin; this Rust
    /// path is kept as a fallback but returns an error directing callers to
    /// use the Python ``write_from`` path instead.
    #[cfg(feature = "hixl")]
    pub async fn write_from_local(
        &self,
        _client: &(impl context::Actor + Send + Sync),
        _local: Arc<dyn RdmaLocalMemory>,
        _timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        Err(anyhow::anyhow!(
            "External transport transfers should be handled by the Python transport plugin, \
             not the Rust path. Check that rdma.py routes through external_backend_info()."
        ))
    }

    /// Pull data from this remote buffer into local memory (remote->local).
    #[cfg(not(feature = "hixl"))]
    pub async fn read_into_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        let mut local_ibv_backend = IbvManagerActor::local_handle(client).await?;
        local_ibv_backend
            .submit(
                client,
                vec![RdmaOp {
                    op_type: RdmaOpType::ReadIntoLocal,
                    local,
                    remote: self.clone(),
                }],
                Duration::from_secs(timeout),
            )
            .await?;
        Ok(true)
    }

    /// Pull data from this remote buffer into local memory (external backend path).
    #[cfg(feature = "hixl")]
    pub async fn read_into_local(
        &self,
        _client: &(impl context::Actor + Send + Sync),
        _local: Arc<dyn RdmaLocalMemory>,
        _timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        Err(anyhow::anyhow!(
            "External transport transfers should be handled by the Python transport plugin, \
             not the Rust path. Check that rdma.py routes through external_backend_info()."
        ))
    }

    /// Drop the buffer and release remote handles.
    pub async fn drop_buffer(&self, client: &impl context::Actor) -> Result<(), anyhow::Error> {
        tracing::debug!("[buffer] dropping buffer id={}", self.id);
        self.owner.release_buffer(client, self.id).await?;
        Ok(())
    }

    /// Resolve ibverbs-specific buffer info (GPU path only).
    #[cfg(not(feature = "hixl"))]
    pub async fn resolve_ibv(
        &self,
        client: &impl context::Actor,
    ) -> Result<(ActorRef<IbvManagerActor>, IbvBuffer), anyhow::Error> {
        let RdmaBackendContext::Ibverbs(remote_ibv_mgr, remote_ibv_buf) =
            self.backends.iter().map(Ok).next().unwrap_or_else(|| {
                Err(anyhow::anyhow!(
                    "ibverbs backend not found for buffer: {:?}",
                    self
                ))
            })?;

        Ok((
            remote_ibv_mgr.clone(),
            remote_ibv_buf
                .get_or_try_init(async || {
                    remote_ibv_mgr
                        .request_buffer(client, self.id)
                        .await?
                        .ok_or_else(|| anyhow::anyhow!("buffer {} not found", self.id))
                })
                .await
                .cloned()?,
        ))
    }
}

/// Utility to validate CUDA execution context (GPU only).
#[cfg(not(feature = "hixl"))]
pub async fn validate_execution_context() -> Result<(), anyhow::Error> {
    use std::fs;
    match fs::read_to_string("/proc/modules") {
        Ok(contents) => {
            if !contents.contains("nvidia_peermem") {
                return Err(anyhow::anyhow!(
                    "nvidia_peermem module not found in /proc/modules"
                ));
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!(e));
        }
    }
    match fs::read_to_string("/proc/driver/nvidia/params") {
        Ok(contents) => {
            if !contents.contains("PeerMappingOverride=1") {
                return Err(anyhow::anyhow!(
                    "PeerMappingOverride=1 not found in /proc/driver/nvidia/params"
                ));
            }
        }
        Err(e) => {
            return Err(anyhow::anyhow!(e));
        }
    }
    Ok(())
}

/// Get all CUDA segments registered with MRs (GPU only).
#[cfg(not(feature = "hixl"))]
pub fn get_registered_cuda_segments() -> Vec<rdmaxcel_sys::rdma_segment_info_t> {
    unsafe {
        let segment_count = rdmaxcel_sys::rdma_get_active_segment_count();
        if segment_count <= 0 {
            return Vec::new();
        }

        let mut segments = vec![
            std::mem::MaybeUninit::<rdmaxcel_sys::rdma_segment_info_t>::zeroed()
                .assume_init();
            segment_count as usize
        ];
        let actual_count =
            rdmaxcel_sys::rdma_get_all_segment_info(segments.as_mut_ptr(), segment_count);

        if actual_count > 0 {
            segments.truncate(actual_count as usize);
            segments
        } else {
            Vec::new()
        }
    }
}

/// Segment scanner callback type alias.
#[cfg(not(feature = "hixl"))]
pub type SegmentScannerFn = rdmaxcel_sys::RdmaxcelSegmentScannerFn;
#[cfg(feature = "hixl")]
pub type SegmentScannerFn = Option<unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> i32>;

/// Register a segment scanner callback.
#[cfg(not(feature = "hixl"))]
pub fn register_segment_scanner(scanner: SegmentScannerFn) {
    unsafe { rdmaxcel_sys::rdmaxcel_register_segment_scanner(scanner) }
}
#[cfg(feature = "hixl")]
pub fn register_segment_scanner(_scanner: SegmentScannerFn) {
    // No-op for HIXL backend: NPU memory registration is handled by HIXL directly.
}
