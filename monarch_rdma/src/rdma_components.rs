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
#[cfg(feature = "hixl")]
use crate::EnsurePeerConnectedClient;
use crate::ReleaseBufferClient;
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

#[cfg(feature = "hixl")]
async fn ensure_bidir_connected(
    client: &(impl context::Actor + Send + Sync),
    owner: &ActorRef<RdmaManagerActor>,
    remote_engine_id: &str,
    timeout_ms: i32,
) -> Result<(), anyhow::Error> {
    let strict_bidir = std::env::var("MONARCH_HIXL_STRICT_BIDIR_CONNECT")
        .map(|v| v != "0")
        .unwrap_or(false);
    let local_engine_id = crate::backend::hixl::manager_actor::get_engine_id_wait(timeout_ms)?;
    let handshake_trace_id = format!(
        "hs:{}->{}:{}",
        local_engine_id,
        remote_engine_id,
        std::process::id()
    );
    tracing::warn!(
        "HIXL: bidir handshake start trace_id={} local_engine={} remote_engine={} timeout_ms={}",
        handshake_trace_id,
        local_engine_id,
        remote_engine_id,
        timeout_ms,
    );

    tracing::warn!(
        "HIXL: bidir handshake remote ensure_peer_connected start trace_id={} peer_engine_id={} strict_bidir={}",
        handshake_trace_id,
        local_engine_id,
        strict_bidir,
    );
    if let Err(e) = owner
        .ensure_peer_connected(client, local_engine_id.clone())
        .await
    {
        if strict_bidir {
            tracing::error!(
                "HIXL: bidir handshake remote ensure_peer_connected failed trace_id={} peer_engine_id={} err={}",
                handshake_trace_id,
                local_engine_id,
                e,
            );
            return Err(e);
        }

        tracing::warn!(
            "HIXL: bidir handshake remote ensure_peer_connected failed (non-strict, continue) trace_id={} peer_engine_id={} err={}",
            handshake_trace_id,
            local_engine_id,
            e,
        );
    } else {
        tracing::warn!(
            "HIXL: bidir handshake remote ensure_peer_connected done trace_id={} peer_engine_id={}",
            handshake_trace_id,
            local_engine_id,
        );
    }

    tracing::warn!(
        "HIXL: bidir handshake local connect start trace_id={} remote_engine_id={}",
        handshake_trace_id,
        remote_engine_id,
    );
    if let Err(e) = crate::backend::hixl::manager_actor::hixl_connect_peer(remote_engine_id, timeout_ms)
    {
        tracing::error!(
            "HIXL: bidir handshake local connect failed trace_id={} remote_engine_id={} err={}",
            handshake_trace_id,
            remote_engine_id,
            e,
        );
        return Err(e);
    }
    tracing::warn!(
        "HIXL: bidir handshake local connect done trace_id={} remote_engine_id={}",
        handshake_trace_id,
        remote_engine_id,
    );

    tracing::warn!(
        "HIXL: bidir handshake complete trace_id={} local_engine={} remote_engine={}",
        handshake_trace_id,
        local_engine_id,
        remote_engine_id
    );
    Ok(())
}

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

    /// Push data from local memory into this remote buffer (HIXL path).
    #[cfg(feature = "hixl")]
    pub async fn write_from_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        tracing::warn!("write_from_local: self.backends = {:?}", self.backends);
        let remote_hixl = self.backends.iter().find_map(|ctx| {
            if let RdmaBackendContext::Hixl(buf) = ctx { Some(buf.clone()) } else { None }
        }).ok_or_else(|| anyhow::anyhow!("No HIXL backend on remote buffer"))?;
        tracing::warn!("write_from_local: remote_hixl = {:?}", remote_hixl);

        if local.size() > remote_hixl.size {
            return Err(anyhow::anyhow!(
                "HIXL write_from_local size overflow: local_size={} remote_size={} remote_engine={}",
                local.size(),
                remote_hixl.size,
                remote_hixl.engine_id,
            ));
        }

        let timeout_ms = timeout as i32 * 1000;
        ensure_bidir_connected(client, &self.owner, &remote_hixl.engine_id, timeout_ms).await?;

        crate::backend::hixl::manager_actor::hixl_register_transfer_memory(
            local.addr(),
            local.size(),
        )?;
        let transfer_result = crate::backend::hixl::manager_actor::hixl_transfer_connected(
            &remote_hixl.engine_id,
            local.addr(),
            local.size(),
            remote_hixl.addr,
            hixl_sys::HixlTransferOp::HIXL_WRITE,
            timeout_ms,
        );
        transfer_result?;
        Ok(true)
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

    /// Pull data from this remote buffer into local memory (HIXL path).
    #[cfg(feature = "hixl")]
    pub async fn read_into_local(
        &self,
        client: &(impl context::Actor + Send + Sync),
        local: Arc<dyn RdmaLocalMemory>,
        timeout: u64,
    ) -> Result<bool, anyhow::Error> {
        tracing::warn!("read_into_local: self.backends = {:?}", self.backends);
        let remote_hixl = self.backends.iter().find_map(|ctx| {
            if let RdmaBackendContext::Hixl(buf) = ctx { Some(buf.clone()) } else { None }
        }).ok_or_else(|| anyhow::anyhow!("No HIXL backend on remote buffer"))?;
        tracing::warn!("read_into_local: remote_hixl = {:?}", remote_hixl);

        if local.size() > remote_hixl.size {
            return Err(anyhow::anyhow!(
                "HIXL read_into_local size overflow: local_size={} remote_size={} remote_engine={}",
                local.size(),
                remote_hixl.size,
                remote_hixl.engine_id,
            ));
        }

        let timeout_ms = timeout as i32 * 1000;
        ensure_bidir_connected(client, &self.owner, &remote_hixl.engine_id, timeout_ms).await?;

        crate::backend::hixl::manager_actor::hixl_register_transfer_memory(
            local.addr(),
            local.size(),
        )?;
        let transfer_result = crate::backend::hixl::manager_actor::hixl_transfer_connected(
            &remote_hixl.engine_id,
            local.addr(),
            local.size(),
            remote_hixl.addr,
            hixl_sys::HixlTransferOp::HIXL_READ,
            timeout_ms,
        );
        transfer_result?;
        Ok(true)
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
