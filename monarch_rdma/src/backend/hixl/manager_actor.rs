/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # HIXL Manager Actor
//!
//! Per-process actor that owns the HIXL engine instance and manages connections,
//! memory registrations, and single-sided data transfers over HCCS (intra-supernode)
//! and RDMA/RoCE (inter-supernode).
//!
//! ## Design
//!
//! All HIXL operations go through the global `PROCESS_HIXL` static, which holds
//! a single HIXL instance per process. The `HixlManagerActor` initializes this
//! instance during its first use and provides actor-based message handling for
//! buffer registration/release.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use super::HixlBuffer;
use crate::RdmaOp;
use crate::RdmaOpType;
use crate::RdmaTransportLevel;
use crate::backend::RdmaBackend;
use crate::rdma_manager_actor::RdmaManagerActor;

/// Send-safe wrapper around `HixlMemHandle` (`*mut c_void`).
#[derive(Copy, Clone)]
struct SendMemHandle(hixl_sys::HixlMemHandle);
unsafe impl Send for SendMemHandle {}
unsafe impl Sync for SendMemHandle {}

impl std::fmt::Debug for SendMemHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemHandle({:p})", self.0)
    }
}

/// Messages handled by [`HixlManagerActor`].
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum HixlManagerMessage {
    /// Register a buffer with HIXL. The caller provides `addr` and `size`
    /// directly to avoid a callback to the parent actor (which would deadlock).
    RequestBuffer {
        remote_buf_id: usize,
        addr: usize,
        size: usize,
        #[reply]
        reply: OncePortRef<Option<HixlBuffer>>,
    },
    /// Release a buffer registration.
    ReleaseBuffer {
        remote_buf_id: usize,
        #[reply]
        reply: OncePortRef<()>,
    },
}
wirevalue::register_type!(HixlManagerMessage);

/// Per-process global HIXL context for transfer operations.
/// Initialized once by `HixlManagerActor` and used by all transfer operations.
static PROCESS_HIXL: OnceLock<Mutex<ProcessHixl>> = OnceLock::new();

/// Per-process HIXL state including engine instance, connections, and registrations.
struct ProcessHixl {
    engine_id: String,
    hixl: hixl_sys::Hixl,
    connected_peers: HashMap<String, bool>,
    registered_buffers: HashMap<usize, SendMemHandle>,
}

// Safety: hixl_sys::Hixl internally manages thread safety.
// The Mutex provides exclusive access for connect/transfer operations.
unsafe impl Send for ProcessHixl {}

impl ProcessHixl {
    fn ensure_connected(&mut self, remote_engine: &str, timeout_ms: i32) -> Result<()> {
        if self.connected_peers.contains_key(remote_engine) {
            return Ok(());
        }
        tracing::info!("HIXL: connecting to remote engine: {}", remote_engine);
        self.hixl.connect(remote_engine, timeout_ms)
            .map_err(|e| anyhow::anyhow!("HIXL connect to {} failed: {}", remote_engine, e))?;
        self.connected_peers.insert(remote_engine.to_string(), true);
        tracing::info!("HIXL: connected to remote engine: {}", remote_engine);
        Ok(())
    }

    fn register_memory(
        &mut self,
        buf_id: usize,
        addr: usize,
        size: usize,
        mem_type: hixl_sys::HixlMemType,
    ) -> Result<SendMemHandle> {
        let handle = self.hixl.register_mem(addr, size, mem_type)
            .map_err(|e| anyhow::anyhow!("HIXL register_mem failed: {}", e))?;
        let mem_handle = SendMemHandle(handle);
        self.registered_buffers.insert(buf_id, mem_handle);
        Ok(mem_handle)
    }

    fn deregister_memory(&mut self, buf_id: usize) -> Result<()> {
        if let Some(mem_handle) = self.registered_buffers.remove(&buf_id) {
            self.hixl.deregister_mem(mem_handle.0)
                .map_err(|e| anyhow::anyhow!("HIXL deregister_mem failed: {}", e))?;
        }
        Ok(())
    }
}

/// Initialize the process-global HIXL instance with the given engine_id.
/// Called once by `HixlManagerActor` during initialization.
fn init_process_hixl(engine_id: String) -> Result<()> {
    if PROCESS_HIXL.get().is_some() {
        tracing::warn!("HIXL: PROCESS_HIXL already initialized, skipping");
        return Ok(());
    }

    tracing::info!("HIXL: creating instance with engine_id={}", engine_id);
    let hixl = hixl_sys::Hixl::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HIXL instance: {}", e))?;
    
    // Initialize with listening port (port > 0 enables server mode)
    tracing::info!("HIXL: initializing with engine_id={}", engine_id);
    hixl.initialize(&engine_id, &[])
        .map_err(|e| anyhow::anyhow!("HIXL initialize failed: {}", e))?;
    tracing::info!("HIXL: engine initialized successfully, server listening on {}", engine_id);

    let _ = PROCESS_HIXL.set(Mutex::new(ProcessHixl {
        engine_id: engine_id.clone(),
        hixl,
        connected_peers: HashMap::new(),
        registered_buffers: HashMap::new(),
    }));

    tracing::info!("HIXL: global PROCESS_HIXL initialized with engine_id={}", engine_id);
    Ok(())
}

/// Get the engine_id from the process-global HIXL instance.
pub fn get_engine_id() -> Option<String> {
    PROCESS_HIXL.get().and_then(|px| {
        px.lock().ok().map(|guard| guard.engine_id.clone())
    })
}

/// Wait for PROCESS_HIXL to be initialized, with a timeout.
/// This is needed because HixlManagerActor::init runs asynchronously,
/// and we need to ensure PROCESS_HIXL is ready before any transfer.
fn wait_for_process_hixl(timeout_ms: i32) -> Result<()> {
    // Check if already initialized
    if PROCESS_HIXL.get().is_some() {
        return Ok(());
    }

    // Wait for initialization with timeout
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_millis(timeout_ms as u64);
    
    while PROCESS_HIXL.get().is_none() {
        if start.elapsed() > timeout {
            return Err(anyhow::anyhow!(
                "HIXL not initialized after {}ms - ensure HixlManagerActor is spawned first",
                timeout_ms
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    
    Ok(())
}

/// Execute a single-sided transfer using the process-global HIXL context.
/// Called from `RdmaRemoteBuffer::read_into_local` / `write_from_local`.
pub fn hixl_transfer_sync(
    remote_engine_id: &str,
    local_addr: usize,
    local_size: usize,
    remote_addr: usize,
    transfer_op: hixl_sys::HixlTransferOp,
    timeout_ms: i32,
) -> Result<()> {
    // Wait for HIXL to be initialized (may be still initializing in background)
    wait_for_process_hixl(timeout_ms)?;
    
    let process_hixl = PROCESS_HIXL
        .get()
        .ok_or_else(|| anyhow::anyhow!("HIXL not initialized - ensure HixlManagerActor is spawned first"))?;
    let mut guard = process_hixl
        .lock()
        .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;

    guard.ensure_connected(remote_engine_id, timeout_ms)?;

    // Register local memory for the transfer
    let local_mem_handle = guard.hixl.register_mem(
        local_addr,
        local_size,
        hixl_sys::HixlMemType::HIXL_MEM_DEVICE,
    ).map_err(|e| anyhow::anyhow!("HIXL register_mem for transfer failed: {}", e))?;

    let op_desc = hixl_sys::HixlTransferOpDesc {
        local_addr,
        remote_addr,
        len: local_size,
    };

    let result = guard.hixl.transfer_sync(
        remote_engine_id,
        transfer_op,
        &[op_desc],
        timeout_ms,
    );

    // Deregister local memory after transfer
    let _ = guard.hixl.deregister_mem(local_mem_handle);

    result.map_err(|e| anyhow::anyhow!("HIXL transfer to {} failed: {}", remote_engine_id, e))
}

/// HIXL manager actor — initializes the process-global HIXL instance and
/// handles buffer registration/release messages.
#[derive(Debug)]
#[hyperactor::export(
    handlers = [
        HixlManagerMessage,
    ],
)]
pub struct HixlManagerActor {
    owner: OnceLock<ActorHandle<RdmaManagerActor>>,
    engine_id: String,
}

impl HixlManagerActor {
    pub fn new(engine_id: String) -> Self {
        Self {
            owner: OnceLock::new(),
            engine_id,
        }
    }
}

#[async_trait]
impl Actor for HixlManagerActor {
    async fn init(&mut self, _this: &Instance<Self>) -> Result<(), anyhow::Error> {
        // Initialize the process-global HIXL instance
        init_process_hixl(self.engine_id.clone())?;
        tracing::info!("HixlManagerActor initialized with engine_id={}", self.engine_id);
        Ok(())
    }
}

impl Drop for HixlManagerActor {
    fn drop(&mut self) {
        // Note: We don't clean up PROCESS_HIXL here because it may still be in use
        // by ongoing transfers. The HIXL instance will be cleaned up when the process exits.
        tracing::info!("HixlManagerActor dropped (engine_id={})", self.engine_id);
    }
}

#[async_trait]
#[hyperactor::handle(HixlManagerMessage)]
impl HixlManagerMessageHandler for HixlManagerActor {
    async fn request_buffer(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
        addr: usize,
        size: usize,
    ) -> Result<Option<HixlBuffer>, anyhow::Error> {
        tracing::debug!("HIXL request_buffer: id={} addr={:#x} size={}", remote_buf_id, addr, size);
        
        let process_hixl = PROCESS_HIXL
            .get()
            .ok_or_else(|| anyhow::anyhow!("HIXL not initialized"))?;
        let mut guard = process_hixl
            .lock()
            .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;

        guard.register_memory(
            remote_buf_id,
            addr,
            size,
            hixl_sys::HixlMemType::HIXL_MEM_DEVICE,
        )?;

        tracing::debug!("HIXL request_buffer: registered id={}", remote_buf_id);

        Ok(Some(HixlBuffer {
            engine_id: self.engine_id.clone(),
            addr,
            size,
        }))
    }

    async fn release_buffer(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<(), anyhow::Error> {
        let process_hixl = PROCESS_HIXL
            .get()
            .ok_or_else(|| anyhow::anyhow!("HIXL not initialized"))?;
        let mut guard = process_hixl
            .lock()
            .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;

        guard.deregister_memory(remote_buf_id)?;
        Ok(())
    }
}

#[async_trait]
impl RdmaBackend for HixlManagerActor {
    type TransportInfo = ();

    async fn submit(
        &mut self,
        _cx: &(impl hyperactor::context::Actor + Send + Sync),
        ops: Vec<RdmaOp>,
        timeout: std::time::Duration,
    ) -> Result<()> {
        let timeout_ms = timeout.as_millis() as i32;

        for op in ops {
            let remote_hixl = op
                .remote
                .backends
                .iter()
                .find_map(|ctx| {
                    if let crate::backend::RdmaBackendContext::Hixl(buf) = ctx {
                        return Some(buf.clone());
                    }
                    None
                })
                .ok_or_else(|| anyhow::anyhow!("No HIXL backend context on remote buffer"))?;

            let transfer_op = match op.op_type {
                RdmaOpType::ReadIntoLocal => hixl_sys::HixlTransferOp::HIXL_READ,
                RdmaOpType::WriteFromLocal => hixl_sys::HixlTransferOp::HIXL_WRITE,
            };

            hixl_transfer_sync(
                &remote_hixl.engine_id,
                op.local.addr(),
                op.local.size(),
                remote_hixl.addr,
                transfer_op,
                timeout_ms,
            )?;
        }

        Ok(())
    }

    fn transport_level(&self) -> RdmaTransportLevel {
        RdmaTransportLevel::Nic
    }

    fn transport_info(&self) -> Option<Self::TransportInfo> {
        None
    }
}