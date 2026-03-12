/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # HIXL Manager Actor (Rust-managed mode — Plan B)
//!
//! The HIXL engine is owned by a process-global singleton and initialised
//! lazily by [`HixlManagerActor::init`].  All connect, register-mem, and
//! transfer operations go through the C shim exposed by `hixl-sys`.
//!
//! The singleton is shared between:
//! - [`HixlManagerActor`] (metadata & buffer registration via actor messages)
//! - [`RdmaRemoteBuffer::read_into_local`] / [`RdmaRemoteBuffer::write_from_local`]
//!   (data-plane transfers, called from `PyPythonTask` context)
//! - [`RdmaManagerActor::ensure_peer_connected`] (connection requests from
//!   remote peers)
//!
//! A `connect_lock` inside the singleton serialises all `hixl_connect` calls
//! so that the HiXL library never sees two simultaneous Connect() calls on the
//! same engine — a constraint discovered during integration testing.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::OnceLock;

use anyhow::Result;
use async_trait::async_trait;
use hyperactor::Actor;
use hyperactor::ActorHandle;
use hyperactor::ActorRef;
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
use crate::RdmaTransportLevel;
use crate::backend::RdmaBackend;
use crate::rdma_manager_actor::EnsurePeerConnectedClient;
use crate::rdma_manager_actor::RdmaManagerActor;

// ============================================================================
// Process-global HIXL engine state
// ============================================================================

pub struct HixlEngineState {
    pub engine: hixl_sys::HixlEngine,
    pub engine_id: String,
    pub connected_peers: Mutex<HashSet<String>>,
    pub connect_lock: Mutex<()>,
    pub registered_addrs: Mutex<HashSet<usize>>,
}

impl std::fmt::Debug for HixlEngineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HixlEngineState")
            .field("engine_id", &self.engine_id)
            .finish_non_exhaustive()
    }
}

static HIXL_STATE: OnceLock<HixlEngineState> = OnceLock::new();

/// Returns the process-global HiXL state, waiting up to ~10 s for the
/// `HixlManagerActor` child actor to finish initialisation.
pub fn get_hixl_state() -> Result<&'static HixlEngineState> {
    if let Some(s) = HIXL_STATE.get() {
        return Ok(s);
    }
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if let Some(s) = HIXL_STATE.get() {
            return Ok(s);
        }
    }
    Err(anyhow::anyhow!(
        "HiXL engine not initialised after 10 s (HixlManagerActor not started?)"
    ))
}

/// Populate `HIXL_STATE` from a pre-initialised engine pointer (created
/// externally, e.g. from Python ctypes).  No-op if already set.
///
/// SAFETY: `ptr_val` must be a valid `HixlTestCtx*` from `hixl_init_engine`.
pub fn set_hixl_state_from_raw(ptr_val: usize, engine_id: String) -> Result<()> {
    if HIXL_STATE.get().is_some() {
        tracing::info!("[hixl] set_hixl_state_from_raw: already initialised, skipping");
        return Ok(());
    }
    let engine = unsafe {
        hixl_sys::HixlEngine::from_raw(ptr_val as *mut std::ffi::c_void)
    };
    let state = HixlEngineState {
        engine,
        engine_id: engine_id.clone(),
        connected_peers: Mutex::new(HashSet::new()),
        connect_lock: Mutex::new(()),
        registered_addrs: Mutex::new(HashSet::new()),
    };
    let _ = HIXL_STATE.set(state);
    // SAFETY: single-threaded at this point (Python init runs before actor spawn)
    unsafe { std::env::set_var("MONARCH_PYTHON_HIXL_ENGINE_ID", &engine_id) };
    tracing::info!(
        "[hixl] set_hixl_state_from_raw: engine_id={} ptr={:#x}",
        engine_id,
        ptr_val,
    );
    Ok(())
}

/// Connect to `peer_eid` if not already connected.
/// Serialised by `connect_lock` so at most one Connect() is in-flight
/// per process at any time.
pub fn do_connect(peer_eid: &str) -> Result<()> {
    let state = get_hixl_state()?;
    if state.connected_peers.lock().unwrap().contains(peer_eid) {
        return Ok(());
    }
    let _guard = state.connect_lock.lock().unwrap();
    if state.connected_peers.lock().unwrap().contains(peer_eid) {
        return Ok(());
    }
    tracing::info!("[hixl] connecting to peer {}", peer_eid);
    state
        .engine
        .connect(peer_eid)
        .map_err(|ret| anyhow::anyhow!("hixl_connect({}) failed: ret={}", peer_eid, ret))?;
    state
        .connected_peers
        .lock()
        .unwrap()
        .insert(peer_eid.to_string());
    tracing::info!("[hixl] connected to peer {}", peer_eid);
    Ok(())
}

/// Register a device memory region if not already registered.
pub fn register_mem_if_needed(addr: usize, size: usize) -> Result<()> {
    let state = get_hixl_state()?;
    let mut addrs = state.registered_addrs.lock().unwrap();
    if addrs.contains(&addr) {
        return Ok(());
    }
    state
        .engine
        .register_mem(addr, size)
        .map_err(|ret| anyhow::anyhow!("hixl_register_mem(addr={:#x}, size={}) failed: ret={}", addr, size, ret))?;
    addrs.insert(addr);
    Ok(())
}

/// Return the engine_id of the local engine.
pub fn global_engine_id() -> Result<String> {
    get_hixl_state().map(|s| s.engine_id.clone())
}

// ============================================================================
// Ensure connected (async — used from data-plane code)
// ============================================================================

/// Ensure that the local engine is connected to `remote_eid`.
///
/// 1. Sends `EnsurePeerConnected(my_eid)` to the remote `RdmaManagerActor`
///    so the remote side connects to us first.
/// 2. Then does the local `Connect(remote_eid)` sequentially.
pub async fn ensure_connected(
    client: &(impl hyperactor::context::Actor + Send + Sync),
    remote_rdma_mgr: &ActorRef<RdmaManagerActor>,
    remote_eid: &str,
) -> Result<()> {
    let state = get_hixl_state()?;
    if state.connected_peers.lock().unwrap().contains(remote_eid) {
        return Ok(());
    }

    tracing::info!(
        "[hixl] ensure_connected: connecting to {}",
        remote_eid
    );

    // Only the data consumer needs to Connect; skip the reverse direction
    // (EnsurePeerConnected) as it's unnecessary for unidirectional transfer
    // and can cause channel conflicts in the async runtime.
    do_connect(remote_eid)?;

    // Small settling time for HiXL's internal channel setup.
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    Ok(())
}

// ============================================================================
// HixlManagerMessage
// ============================================================================

#[derive(Handler, HandleClient, RefClient, Debug, Serialize, Deserialize, Named)]
pub enum HixlManagerMessage {
    RequestBuffer {
        remote_buf_id: usize,
        addr: usize,
        size: usize,
        #[reply]
        reply: OncePortRef<Option<HixlBuffer>>,
    },
    ReleaseBuffer {
        remote_buf_id: usize,
        #[reply]
        reply: OncePortRef<()>,
    },
    GetEngineId {
        #[reply]
        reply: OncePortRef<String>,
    },
}
wirevalue::register_type!(HixlManagerMessage);

// ============================================================================
// HixlManagerActor
// ============================================================================

#[derive(Debug)]
#[hyperactor::export(
    handlers = [
        HixlManagerMessage,
    ],
)]
pub struct HixlManagerActor {
    engine_id: String,
    device_id: i32,
    owner: OnceLock<ActorHandle<RdmaManagerActor>>,
}

impl HixlManagerActor {
    pub fn new(engine_id: String, device_id: i32) -> Self {
        Self {
            engine_id,
            device_id,
            owner: OnceLock::new(),
        }
    }
}

fn resolve_engine_id(hint: &str) -> String {
    if !hint.is_empty() {
        return hint.to_string();
    }
    if let Ok(eid) = std::env::var("MONARCH_PYTHON_HIXL_ENGINE_ID") {
        return eid;
    }
    let ip = crate::rdma_manager_actor::local_ip_for_hixl();
    let port = 20000 + (std::process::id() % 40000);
    format!("{}:{}", ip, port)
}

fn resolve_device_id(hint: i32) -> i32 {
    if hint >= 0 {
        return hint;
    }
    std::env::var("MONARCH_NPU_DEVICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

#[async_trait]
impl Actor for HixlManagerActor {
    async fn init(&mut self, this: &Instance<Self>) -> Result<(), anyhow::Error> {
        let owner = this
            .parent_handle()
            .ok_or_else(|| anyhow::anyhow!("RdmaManagerActor not found as parent"))?;
        self.owner
            .set(owner)
            .expect("owner should only be set once during init");

        let eid = resolve_engine_id(&self.engine_id);
        let dev = resolve_device_id(self.device_id);
        self.engine_id = eid.clone();
        self.device_id = dev;

        if let Some(existing) = HIXL_STATE.get() {
            tracing::info!(
                "[hixl] engine already initialised by Python ctypes: engine_id={}",
                existing.engine_id,
            );
            self.engine_id = existing.engine_id.clone();
        } else {
            tracing::warn!(
                "[hixl] HIXL_STATE not pre-set by Python; falling back to Rust FFI init \
                 (dev={} engine_id={}). This path may fail on some platforms.",
                dev,
                eid,
            );
            HIXL_STATE.get_or_init(|| {
                let engine = hixl_sys::HixlEngine::new(dev, &eid)
                    .expect("hixl_init_engine failed");
                unsafe { std::env::set_var("MONARCH_PYTHON_HIXL_ENGINE_ID", &eid) };
                HixlEngineState {
                    engine,
                    engine_id: eid,
                    connected_peers: Mutex::new(HashSet::new()),
                    connect_lock: Mutex::new(()),
                    registered_addrs: Mutex::new(HashSet::new()),
                }
            });
        }

        tracing::info!(
            "[hixl] HixlManagerActor ready: engine_id={}",
            self.engine_id
        );
        Ok(())
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
        tracing::debug!(
            "[hixl] request_buffer: id={} addr={:#x} size={}",
            remote_buf_id,
            addr,
            size,
        );

        register_mem_if_needed(addr, size)?;

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
        tracing::debug!("[hixl] release_buffer: id={}", remote_buf_id);
        Ok(())
    }

    async fn get_engine_id(
        &mut self,
        _cx: &Context<Self>,
    ) -> Result<String, anyhow::Error> {
        Ok(self.engine_id.clone())
    }
}

#[async_trait]
impl RdmaBackend for HixlManagerActor {
    type TransportInfo = ();

    async fn submit(
        &mut self,
        _cx: &(impl hyperactor::context::Actor + Send + Sync),
        _ops: Vec<RdmaOp>,
        _timeout: std::time::Duration,
    ) -> Result<()> {
        Err(anyhow::anyhow!(
            "HixlManagerActor::submit() should not be called directly; \
             transfers go through the process-global HixlEngineState"
        ))
    }

    fn transport_level(&self) -> RdmaTransportLevel {
        RdmaTransportLevel::Nic
    }

    fn transport_info(&self) -> Option<Self::TransportInfo> {
        None
    }
}
