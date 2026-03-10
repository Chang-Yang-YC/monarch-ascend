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
//! Mirrors [`IbvManagerActor`] in structure: the parent [`RdmaManagerActor`] spawns
//! one `HixlManagerActor` per process and delegates transport-specific work to it.

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
/// The `HixlManagerActor` initializes this during its first `request_buffer` call.
/// Transfer operations in `rdma_components.rs` use this to execute transfers
/// without going through actor message passing (avoiding deadlocks).
static PROCESS_HIXL: OnceLock<Mutex<ProcessHixl>> = OnceLock::new();

struct ProcessHixl {
    engine_id: String,
    hixl: hixl_sys::Hixl,
    connected_peers: HashMap<String, bool>,
}

// Safety: hixl_sys::Hixl internally manages thread safety.
// The Mutex provides exclusive access for connect/transfer operations.
unsafe impl Send for ProcessHixl {}

impl ProcessHixl {
    fn ensure_connected(&mut self, remote_engine: &str, timeout_ms: i32) -> Result<()> {
        if self.connected_peers.contains_key(remote_engine) {
            return Ok(());
        }
        self.hixl.connect(remote_engine, timeout_ms)
            .map_err(|e| anyhow::anyhow!("HIXL connect to {} failed: {}", remote_engine, e))?;
        self.connected_peers.insert(remote_engine.to_string(), true);
        tracing::info!("HIXL connected to remote engine: {}", remote_engine);
        Ok(())
    }
}

fn ensure_process_hixl() -> Result<()> {
    PROCESS_HIXL.get_or_init(|| {
        let hixl = hixl_sys::Hixl::new().expect("Failed to create HIXL instance");
        let ip = crate::rdma_manager_actor::local_ip_for_hixl();
        let engine_id = format!("{}:0", ip);
        hixl.initialize(&engine_id, &[]).expect("HIXL initialize failed");
        tracing::warn!("HIXL: lazy-initialized process-global context with id: {}", engine_id);
        Mutex::new(ProcessHixl {
            engine_id,
            hixl,
            connected_peers: HashMap::new(),
        })
    });
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
    ensure_process_hixl()?;
    let process_hixl = PROCESS_HIXL
        .get()
        .ok_or_else(|| anyhow::anyhow!("HIXL not initialized"))?;
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

/// HIXL manager actor — manages a persistent HIXL engine instance, peer
/// connections, and memory registrations for single-sided transfers.
#[derive(Debug)]
#[hyperactor::export(
    handlers = [
        HixlManagerMessage,
    ],
)]
pub struct HixlManagerActor {
    owner: OnceLock<ActorHandle<RdmaManagerActor>>,
    engine_id: String,
    hixl: Option<hixl_sys::Hixl>,
    connected_peers: HashMap<String, bool>,
    registered_buffers: HashMap<usize, SendMemHandle>,
}

impl HixlManagerActor {
    pub fn new(engine_id: String) -> Self {
        Self {
            owner: OnceLock::new(),
            engine_id,
            hixl: None,
            connected_peers: HashMap::new(),
            registered_buffers: HashMap::new(),
        }
    }

    fn ensure_initialized(&mut self) -> Result<&hixl_sys::Hixl> {
        if self.hixl.is_none() {
            tracing::warn!("HIXL: creating instance...");
            let hixl = hixl_sys::Hixl::new()
                .map_err(|e| anyhow::anyhow!("Failed to create HIXL instance: {}", e))?;
            tracing::warn!("HIXL: instance created, initializing with engine_id={}...", self.engine_id);
            hixl.initialize(&self.engine_id, &[])
                .map_err(|e| anyhow::anyhow!("HIXL initialize failed: {}", e))?;
            tracing::warn!("HIXL: engine initialized with id: {}", self.engine_id);

            // Also set up the process-global HIXL for transfer operations.
            // Reuse the same engine_id with port=0 (HCCS mode) for the global context.
            let global_hixl = hixl_sys::Hixl::new()
                .map_err(|e| anyhow::anyhow!("Failed to create global HIXL instance: {}", e))?;
            let xfer_engine_id = format!("{}:0", crate::rdma_manager_actor::local_ip_for_hixl());
            global_hixl.initialize(&xfer_engine_id, &[])
                .map_err(|e| anyhow::anyhow!("Global HIXL initialize failed: {}", e))?;

            let _ = PROCESS_HIXL.set(Mutex::new(ProcessHixl {
                engine_id: self.engine_id.clone(),
                hixl: global_hixl,
                connected_peers: HashMap::new(),
            }));

            self.hixl = Some(hixl);
        }
        Ok(self.hixl.as_ref().unwrap())
    }

    fn register_memory(
        &mut self,
        addr: usize,
        size: usize,
        mem_type: hixl_sys::HixlMemType,
    ) -> Result<SendMemHandle> {
        let hixl = self.ensure_initialized()?;
        let handle = hixl.register_mem(addr, size, mem_type)
            .map_err(|e| anyhow::anyhow!("HIXL register_mem failed: {}", e))?;
        Ok(SendMemHandle(handle))
    }
}

#[async_trait]
impl Actor for HixlManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        let owner = if let Some(owner) = this.parent_handle() {
            owner
        } else {
            anyhow::bail!("RdmaManagerActor not found as parent of HixlManagerActor");
        };
        self.owner
            .set(owner)
            .expect("owner should only be set once during init");
        Ok(())
    }
}

impl Drop for HixlManagerActor {
    fn drop(&mut self) {
        for (_buf_id, mem_handle) in self.registered_buffers.drain() {
            if let Some(ref hixl) = self.hixl {
                let _ = hixl.deregister_mem(mem_handle.0);
            }
        }
        for (peer, _) in self.connected_peers.drain() {
            if let Some(ref hixl) = self.hixl {
                let _ = hixl.disconnect(&peer, 5000);
            }
        }
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
        tracing::warn!("HIXL request_buffer: id={} addr={:#x} size={}", remote_buf_id, addr, size);
        let mem_handle = self.register_memory(
            addr,
            size,
            hixl_sys::HixlMemType::HIXL_MEM_DEVICE,
        )?;
        tracing::warn!("HIXL request_buffer: register_memory done for id={}", remote_buf_id);
        self.registered_buffers.insert(remote_buf_id, mem_handle);

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
        if let Some(mem_handle) = self.registered_buffers.remove(&remote_buf_id) {
            if let Some(ref hixl) = self.hixl {
                hixl.deregister_mem(mem_handle.0)
                    .map_err(|e| anyhow::anyhow!("HIXL deregister_mem failed: {}", e))?;
            }
        }
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
