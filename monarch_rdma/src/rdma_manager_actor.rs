/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # RDMA Manager Actor
//!
//! Per-process actor that owns RDMA buffer registrations and delegates
//! transport-specific work to backend actors.
//!
//! ## Backend Selection
//!
//! - **ibverbs** (default): Uses [`IbvManagerActor`] for RDMA over InfiniBand/RoCE.
//! - **hixl** (feature `hixl`): Uses [`HixlManagerActor`] for HIXL over HCCS/RDMA
//!   on Ascend NPUs.
//!
//! ## Responsibilities
//!
//! - Assigns a unique `remote_buf_id` to each registered local memory handle
//!   and stores the `Arc<dyn RdmaLocalMemory>` for later retrieval.
//! - Produces [`RdmaRemoteBuffer`] tokens that can be sent to remote peers so
//!   they can address this buffer over RDMA.
//! - Delegates MR registration, QP management, and data movement to the
//!   selected backend.
//! - Handles remote [`ReleaseBuffer`] requests to clean up registrations.

use std::collections::HashMap;
#[cfg(feature = "hixl")]
use std::fs;
use std::sync::Arc;

use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorId;
use hyperactor::ActorRef;
use hyperactor::Context;
use hyperactor::HandleClient;
use hyperactor::Handler;
use hyperactor::Instance;
use hyperactor::OncePortHandle;
use hyperactor::OncePortRef;
use hyperactor::RefClient;
use hyperactor::RemoteSpawn;
use hyperactor::context;
use hyperactor::supervision::ActorSupervisionEvent;
use hyperactor_config::Flattrs;
use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

use crate::RdmaLocalMemory;
use crate::backend::RdmaBackendContext;
use crate::rdma_components::RdmaRemoteBuffer;

// ---- ibverbs backend imports ----
#[cfg(not(feature = "hixl"))]
use tokio::sync::OnceCell;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerActor;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::manager_actor::IbvManagerMessageClient;
#[cfg(not(feature = "hixl"))]
use crate::backend::ibverbs::primitives::IbvConfig;

// ---- HIXL backend imports ----
#[cfg(feature = "hixl")]
use crate::backend::hixl::manager_actor::HixlManagerActor;
#[cfg(feature = "hixl")]
use crate::backend::hixl::manager_actor::HixlManagerMessageClient;
/// Helper function to get detailed error messages from RDMAXCEL error codes.
#[cfg(not(feature = "hixl"))]
pub fn get_rdmaxcel_error_message(error_code: i32) -> String {
    unsafe {
        let c_str = rdmaxcel_sys::rdmaxcel_error_string(error_code);
        std::ffi::CStr::from_ptr(c_str)
            .to_string_lossy()
            .into_owned()
    }
}

/// Local-only messages for the [`RdmaManagerActor`].
///
/// These messages carry `Arc<dyn RdmaLocalMemory>` and are therefore
/// not serializable — they can only be sent within the same process.
#[derive(Handler, HandleClient, Debug)]
pub enum RdmaManagerMessage {
    /// Register a local memory handle and return a [`RdmaRemoteBuffer`] that
    /// remote peers can use to address this buffer over RDMA.
    RequestBuffer {
        local: Arc<dyn RdmaLocalMemory>,
        #[reply]
        reply: OncePortHandle<RdmaRemoteBuffer>,
    },
    /// Look up the local memory handle for a given `remote_buf_id`. Returns
    /// `None` if the id does not correspond to a registered buffer.
    RequestLocalMemory {
        remote_buf_id: usize,
        #[reply]
        reply: OncePortHandle<Option<Arc<dyn RdmaLocalMemory>>>,
    },
}

/// Serializable release message for wire transport.
///
/// Used by [`RdmaRemoteBuffer::drop_buffer`] to release a buffer
/// from a remote process.
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct ReleaseBuffer {
    pub id: usize,
}
wirevalue::register_type!(ReleaseBuffer);

/// Serializable cross-process message asking the receiver to establish
/// an HIXL connection to the given peer engine id.
#[cfg(feature = "hixl")]
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct EnsurePeerConnected {
    pub peer_engine_id: String,
    #[reply]
    pub reply: OncePortRef<()>,
}
#[cfg(feature = "hixl")]
wirevalue::register_type!(EnsurePeerConnected);

/// Serializable query for resolving the [`IbvManagerActor`] ref
/// from a remote [`RdmaManagerActor`]. Only used in testing.
#[cfg(not(feature = "hixl"))]
#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub struct GetIbvActorRef {
    #[reply]
    pub reply: OncePortRef<Option<ActorRef<IbvManagerActor>>>,
}
#[cfg(not(feature = "hixl"))]
wirevalue::register_type!(GetIbvActorRef);

#[derive(Debug)]
enum RdmaBackendActor<A: Actor> {
    Uninit,
    Created(A),
    Spawned(ActorHandle<A>),
}

impl<A: Actor> RdmaBackendActor<A> {
    fn spawn(&mut self, rdma_manager: &Instance<RdmaManagerActor>) -> anyhow::Result<()> {
        let created = std::mem::replace(self, RdmaBackendActor::Uninit);
        let actor = if let RdmaBackendActor::Created(actor) = created {
            actor
        } else {
            panic!("rdma backend actor already spawned");
        };
        let handle = rdma_manager.spawn(actor)?;
        *self = RdmaBackendActor::Spawned(handle);
        Ok(())
    }

    fn handle(&self) -> &ActorHandle<A> {
        if let RdmaBackendActor::Spawned(handle) = self {
            handle
        } else {
            panic!("cannot get handle")
        }
    }
}

// ============================================================================
// ibverbs backend (default)
// ============================================================================

#[cfg(not(feature = "hixl"))]
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        GetIbvActorRef,
        ReleaseBuffer,
    ],
)]
pub struct RdmaManagerActor {
    next_remote_buf_id: usize,
    buffers: HashMap<usize, Arc<dyn RdmaLocalMemory>>,
    ibverbs: RdmaBackendActor<IbvManagerActor>,
}

#[cfg(not(feature = "hixl"))]
impl RdmaManagerActor {
    pub fn local_handle(client: &impl context::Actor) -> ActorHandle<Self> {
        let proc_id = client.mailbox().actor_id().0.clone();
        let actor_ref = ActorRef::attest(ActorId(proc_id, "rdma_manager".to_string(), 0));
        actor_ref
            .downcast_handle(client)
            .expect("RdmaManagerActor is not in the local process")
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
impl RemoteSpawn for RdmaManagerActor {
    type Params = Option<IbvConfig>;

    async fn new(params: Self::Params, _environment: Flattrs) -> Result<Self, anyhow::Error> {
        let ibv = RdmaBackendActor::Created(IbvManagerActor::new(params).await?);
        Ok(Self {
            next_remote_buf_id: 0,
            buffers: HashMap::new(),
            ibverbs: ibv,
        })
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
impl Actor for RdmaManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        self.ibverbs.spawn(this)?;
        tracing::debug!("RdmaManagerActor initialized with ibverbs backend");
        Ok(())
    }

    async fn handle_supervision_event(
        &mut self,
        _cx: &Instance<Self>,
        _event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        tracing::error!("rdmaManagerActor supervision event: {:?}", _event);
        tracing::error!("rdmaManagerActor error occurred, stop the worker process, exit code: 1");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(GetIbvActorRef)]
impl GetIbvActorRefHandler for RdmaManagerActor {
    async fn get_ibv_actor_ref(
        &mut self,
        _cx: &Context<Self>,
    ) -> Result<Option<ActorRef<IbvManagerActor>>, anyhow::Error> {
        Ok(Some(self.ibverbs.handle().bind()))
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(ReleaseBuffer)]
impl ReleaseBufferHandler for RdmaManagerActor {
    async fn release_buffer(&mut self, cx: &Context<Self>, id: usize) -> Result<(), anyhow::Error> {
        self.buffers.remove(&id);
        self.ibverbs.handle().release_buffer(cx, id).await
    }
}

#[cfg(not(feature = "hixl"))]
#[async_trait]
#[hyperactor::handle(RdmaManagerMessage)]
impl RdmaManagerMessageHandler for RdmaManagerActor {
    async fn request_buffer(
        &mut self,
        cx: &Context<Self>,
        local: Arc<dyn RdmaLocalMemory>,
    ) -> Result<RdmaRemoteBuffer, anyhow::Error> {
        let remote_buf_id = self.next_remote_buf_id;
        self.next_remote_buf_id += 1;
        let size = local.size();

        self.buffers.insert(remote_buf_id, local);

        Ok(RdmaRemoteBuffer {
            id: remote_buf_id,
            size,
            owner: cx.bind().clone(),
            backends: vec![RdmaBackendContext::Ibverbs(
                self.ibverbs.handle().bind(),
                Arc::new(OnceCell::new()),
            )],
        })
    }

    async fn request_local_memory(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<Option<Arc<dyn RdmaLocalMemory>>, anyhow::Error> {
        Ok(self.buffers.get(&remote_buf_id).cloned())
    }
}

// ============================================================================
// HIXL backend (Ascend NPU)
// ============================================================================

#[cfg(feature = "hixl")]
#[derive(Debug)]
#[hyperactor::export(
    spawn = true,
    handlers = [
        EnsurePeerConnected,
        ReleaseBuffer,
    ],
)]
pub struct RdmaManagerActor {
    next_remote_buf_id: usize,
    buffers: HashMap<usize, Arc<dyn RdmaLocalMemory>>,
    hixl: RdmaBackendActor<HixlManagerActor>,
}

#[cfg(feature = "hixl")]
impl RdmaManagerActor {
    pub fn local_handle(client: &impl context::Actor) -> ActorHandle<Self> {
        let proc_id = client.mailbox().actor_id().0.clone();
        let actor_ref = ActorRef::attest(ActorId(proc_id, "rdma_manager".to_string(), 0));
        actor_ref
            .downcast_handle(client)
            .expect("RdmaManagerActor is not in the local process")
    }
}

/// HIXL configuration for the RdmaManagerActor.

#[cfg(feature = "hixl")]

#[derive(Debug, Named, Clone, Serialize, Deserialize, Default)]

pub struct HixlConfig {

    /// Explicit engine_id (ip:port format). If not set, auto-generated.

    pub engine_id: Option<String>,

    /// Listening port for HIXL server. If not set, auto-assigned from HIXL_BASE_PORT + counter.

    pub port: Option<u16>,

}



/// Get the base port for HIXL from environment variable or default.



#[cfg(feature = "hixl")]



fn hixl_base_port() -> u16 {



    std::env::var("HIXL_BASE_PORT")



        .ok()



        .and_then(|s| s.parse().ok())



        .unwrap_or(16000)



}







/// Allocate a unique port for this RdmaManagerActor instance.



/// Uses process ID to ensure uniqueness across different processes.



#[cfg(feature = "hixl")]



fn allocate_hixl_port(configured_port: Option<u16>) -> u16 {



    if let Some(port) = configured_port {



        return port;



    }



    // Use process ID to generate a unique port offset.



    // This ensures different processes get different ports even if they run concurrently.



    let pid = std::process::id();



    // Use modulo to keep ports in a reasonable range



    let offset = (pid % 1000) as u16;



    let port = hixl_base_port() + offset;



    // Ensure port doesn't exceed max valid port



    port.min(65535)



}


#[cfg(feature = "hixl")]
fn npu_device_id_from_env() -> Option<u32> {
    if let Ok(v) = std::env::var("MONARCH_NPU_DEVICE") {
        if let Ok(id) = v.trim().parse::<u32>() {
            return Some(id);
        }
    }

    if let Ok(v) = std::env::var("ASCEND_RT_VISIBLE_DEVICES") {
        let first = v.split(',').next().unwrap_or("").trim();
        if let Ok(id) = first.parse::<u32>() {
            return Some(id);
        }
    }

    None
}

#[cfg(feature = "hixl")]
fn read_device_ip_from_hccn_conf(device_id: u32) -> Option<String> {
    let mut conf_paths: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("HCCN_CONF_PATH") {
        if !p.trim().is_empty() {
            conf_paths.push(p);
        }
    }
    conf_paths.push("/etc/hccn.conf".to_string());

    let key = format!("address_{}=", device_id);
    for path in conf_paths {
        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(ip) = line.strip_prefix(&key) {
                let ip = ip.trim();
                if !ip.is_empty() {
                    return Some(ip.to_string());
                }
            }
        }
    }

    None
}



#[cfg(feature = "hixl")]

pub(crate) fn local_ip_for_hixl() -> String {
    // 1) Explicit override for engine_id IP.
    if let Ok(ip) = std::env::var("MONARCH_HIXL_IP") {
        let ip = ip.trim();
        if !ip.is_empty() {
            tracing::info!("HIXL: using MONARCH_HIXL_IP={}", ip);
            return ip.to_string();
        }
    }

    // 2) Use host non-loopback IPv4 for HIXL engine_id.
    // NOTE: HIXL engine_id should use HOST IP (not device IP) for RoCE communication.
    // The device IP is used internally by HCCL for HCCS communication, but HIXL's
    // server-server model requires host IP for the listening socket.
    if let Ok(hostname) = hostname::get() {
        if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(
            &format!("{}:0", hostname.to_string_lossy()),
        ) {
            for addr in addrs {
                if addr.is_ipv4() && !addr.ip().is_loopback() {
                    tracing::info!(
                        "HIXL: using host IPv4 for engine_id ip={}",
                        addr.ip()
                    );
                    return addr.ip().to_string();
                }
            }
        }
    }
    // Fallback: use 0.0.0.0 (binds to all interfaces)
    tracing::warn!(
        "HIXL: fallback to 0.0.0.0 for engine_id (unable to resolve host ip)"
    );
    "0.0.0.0".to_string()
}
#[cfg(feature = "hixl")]
wirevalue::register_type!(HixlConfig);

#[cfg(feature = "hixl")]
#[async_trait]
impl RemoteSpawn for RdmaManagerActor {
    type Params = Option<HixlConfig>;

    async fn new(params: Self::Params, _environment: Flattrs) -> Result<Self, anyhow::Error> {
        let config = params.unwrap_or_default();
        
        // Generate engine_id with listening port for server-server model
        let engine_id = config.engine_id.unwrap_or_else(|| {
            let ip = local_ip_for_hixl();
            let port = allocate_hixl_port(config.port);
            tracing::info!("HIXL: allocated port {} for engine_id", port);
            format!("{}:{}", ip, port)
        });
        
        tracing::info!("HIXL: RdmaManagerActor using engine_id={}", engine_id);
        let hixl_actor = HixlManagerActor::new(engine_id);
        let hixl = RdmaBackendActor::Created(hixl_actor);
        Ok(Self {
            next_remote_buf_id: 0,
            buffers: HashMap::new(),
            hixl,
        })
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
impl Actor for RdmaManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        self.hixl.spawn(this)?;
        tracing::debug!("RdmaManagerActor initialized with HIXL backend");
        Ok(())
    }

    async fn handle_supervision_event(
        &mut self,
        _cx: &Instance<Self>,
        _event: &ActorSupervisionEvent,
    ) -> Result<bool, anyhow::Error> {
        tracing::error!("rdmaManagerActor supervision event: {:?}", _event);
        tracing::error!("rdmaManagerActor error occurred, stop the worker process, exit code: 1");
        std::process::exit(1);
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
#[hyperactor::handle(ReleaseBuffer)]
impl ReleaseBufferHandler for RdmaManagerActor {
    async fn release_buffer(&mut self, cx: &Context<Self>, id: usize) -> Result<(), anyhow::Error> {
        self.buffers.remove(&id);
        self.hixl.handle().release_buffer(cx, id).await
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
#[hyperactor::handle(EnsurePeerConnected)]
impl EnsurePeerConnectedHandler for RdmaManagerActor {
    async fn ensure_peer_connected(
        &mut self,
        _cx: &Context<Self>,
        peer_engine_id: String,
    ) -> Result<(), anyhow::Error> {
        let timeout_ms = std::env::var("MONARCH_HIXL_CONNECT_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(20_000);
        let local_engine_id = crate::backend::hixl::manager_actor::get_engine_id()
            .unwrap_or_else(|| "unknown".to_string());
        tracing::warn!(
            "RdmaManager: ensure_peer_connected start peer={} local_engine={} timeout_ms={} acl_device={:?}",
            peer_engine_id,
            local_engine_id,
            timeout_ms,
            hixl_sys::get_acl_device(),
        );
        let connect_result =
            crate::backend::hixl::manager_actor::hixl_connect_peer(&peer_engine_id, timeout_ms);
        match connect_result {
            Ok(()) => {
                tracing::warn!(
                    "RdmaManager: ensure_peer_connected done peer={} local_engine={} acl_device={:?}",
                    peer_engine_id,
                    local_engine_id,
                    hixl_sys::get_acl_device(),
                );
                Ok(())
            }
            Err(e) => {
                tracing::error!(
                    "RdmaManager: ensure_peer_connected failed peer={} local_engine={} timeout_ms={} acl_device={:?} err={}",
                    peer_engine_id,
                    local_engine_id,
                    timeout_ms,
                    hixl_sys::get_acl_device(),
                    e,
                );
                Err(e)
            }
        }
    }
}

#[cfg(feature = "hixl")]
#[async_trait]
#[hyperactor::handle(RdmaManagerMessage)]
impl RdmaManagerMessageHandler for RdmaManagerActor {
    async fn request_buffer(
        &mut self,
        cx: &Context<Self>,
        local: Arc<dyn RdmaLocalMemory>,
    ) -> Result<RdmaRemoteBuffer, anyhow::Error> {
        let remote_buf_id = self.next_remote_buf_id;
        self.next_remote_buf_id += 1;
        let size = local.size();

        let addr = local.addr();
        tracing::warn!("RdmaManager: request_buffer id={} addr={:#x} size={}", remote_buf_id, addr, size);
        self.buffers.insert(remote_buf_id, local);

        tracing::warn!("RdmaManager: sending to HixlManagerActor...");
        let request_timeout_ms = std::env::var("MONARCH_HIXL_REQUEST_BUFFER_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10_000);
        let hixl_buf = tokio::time::timeout(
            std::time::Duration::from_millis(request_timeout_ms),
            self.hixl.handle().request_buffer(cx, remote_buf_id, addr, size),
        )
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "HIXL request_buffer timed out after {}ms (likely HixlManagerActor init failure; check earlier HIXL initialize logs)",
                request_timeout_ms
            )
        })??
            .ok_or_else(|| anyhow::anyhow!("HIXL request_buffer returned None for id {}", remote_buf_id))?;
        tracing::warn!("RdmaManager: got HixlBuffer: {:?}", hixl_buf);

        Ok(RdmaRemoteBuffer {
            id: remote_buf_id,
            size,
            owner: cx.bind().clone(),
            backends: vec![RdmaBackendContext::Hixl(hixl_buf)],
        })
    }

    async fn request_local_memory(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<Option<Arc<dyn RdmaLocalMemory>>, anyhow::Error> {
        Ok(self.buffers.get(&remote_buf_id).cloned())
    }
}
