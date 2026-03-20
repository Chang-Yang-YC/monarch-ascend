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
use std::collections::HashSet;
use std::sync::mpsc;
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
    // Unified lifecycle: one registration map by addr + refcount for both
    // exported buffers and transfer-time local buffers.
    registered_by_addr: HashMap<usize, (SendMemHandle, usize)>,
    exported_buf_addrs: HashMap<usize, usize>,
    // Transfer-side long-lived registrations keyed by local address.
    // These are intentionally kept registered across transfers to reduce
    // register/deregister churn before TransferSync.
    transfer_registered_addrs: HashSet<usize>,
}

// Safety: hixl_sys::Hixl internally manages thread safety.
// The Mutex provides exclusive access for connect/transfer operations.
unsafe impl Send for ProcessHixl {}

fn parse_env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default)
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| v != "0")
        .unwrap_or(default)
}

fn connect_error_non_retryable(err: &str, patterns: &[String]) -> bool {
    let err_lower = err.to_ascii_lowercase();
    patterns.iter().any(|pattern| {
        let pat = pattern.trim().to_ascii_lowercase();
        !pat.is_empty() && err_lower.contains(&pat)
    })
}

impl ProcessHixl {
    fn ensure_connected(&mut self, remote_engine: &str, timeout_ms: i32) -> Result<()> {
        if self.connected_peers.contains_key(remote_engine) {
            return Ok(());
        }
        let retries = parse_env_u32("MONARCH_HIXL_CONNECT_RETRY", 20).max(1);
        let base_backoff_ms = parse_env_u64("MONARCH_HIXL_CONNECT_RETRY_BACKOFF_MS", 100);
        let max_backoff_ms = parse_env_u64("MONARCH_HIXL_CONNECT_RETRY_MAX_BACKOFF_MS", 500);
        let min_attempt_timeout_ms = parse_env_u64("MONARCH_HIXL_CONNECT_MIN_ATTEMPT_TIMEOUT_MS", 1_000).max(1);
        let total_budget_ms = parse_env_u64(
            "MONARCH_HIXL_CONNECT_TOTAL_BUDGET_MS",
            (timeout_ms as u64).max(1),
        );
        let fast_fail_enabled = parse_env_bool("MONARCH_HIXL_CONNECT_FAST_FAIL", true);
        let fast_fail_min_attempts =
            parse_env_u32("MONARCH_HIXL_CONNECT_FAST_FAIL_MIN_ATTEMPTS", 3).max(1);
        let non_retryable_patterns_raw = std::env::var("MONARCH_HIXL_CONNECT_NON_RETRYABLE_ERRORS")
            .unwrap_or_else(|_| {
                "103900,PARAM_INVALID,invalid parameter,invalid engine".to_string()
            });
        let non_retryable_patterns: Vec<String> = non_retryable_patterns_raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let connect_timeout_cap_ms = (timeout_ms as u64).max(1);
        let connect_start = std::time::Instant::now();

        let mut last_err: Option<String> = None;
        let mut total_attempts = 0u32;
        for attempt in 1..=retries {
            let elapsed_ms = connect_start.elapsed().as_millis() as u64;
            if elapsed_ms >= total_budget_ms {
                break;
            }

            let remaining_budget_ms = total_budget_ms.saturating_sub(elapsed_ms);
            if remaining_budget_ms == 0 {
                break;
            }

            let attempt_timeout_ms = remaining_budget_ms
                .min(connect_timeout_cap_ms)
                .max(min_attempt_timeout_ms.min(remaining_budget_ms));
            total_attempts = attempt;

            tracing::info!(
                "HIXL: connecting to remote engine: {} (attempt {}/{}, local_engine={}, acl_device={:?}, attempt_timeout_ms={}, remaining_budget_ms={})",
                remote_engine,
                attempt,
                retries,
                self.engine_id,
                hixl_sys::get_acl_device(),
                attempt_timeout_ms,
                remaining_budget_ms,
            );

            match self.hixl.connect(remote_engine, attempt_timeout_ms as i32) {
                Ok(()) => {
                    self.connected_peers.insert(remote_engine.to_string(), true);
                    let settle_ms = parse_env_u64("MONARCH_HIXL_CONNECT_SETTLE_MS", 200);
                    if settle_ms > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(settle_ms));
                    }
                    tracing::info!(
                        "HIXL: connected to remote engine: {} (attempt {}/{}, settle_ms={})",
                        remote_engine,
                        attempt,
                        retries,
                        settle_ms,
                    );
                    return Ok(());
                }
                Err(e) => {
                    let err = format!("{}", e);
                    tracing::warn!(
                        "HIXL: connect failed remote={} attempt={}/{} attempt_timeout_ms={} total_budget_ms={} err={}",
                        remote_engine,
                        attempt,
                        retries,
                        attempt_timeout_ms,
                        total_budget_ms,
                        err,
                    );

                    last_err = Some(err.clone());

                    if fast_fail_enabled
                        && attempt >= fast_fail_min_attempts
                        && connect_error_non_retryable(&err, &non_retryable_patterns)
                    {
                        tracing::warn!(
                            "HIXL: fast-fail connect remote={} attempt={}/{} matched non-retryable error (min_attempts={})",
                            remote_engine,
                            attempt,
                            retries,
                            fast_fail_min_attempts,
                        );
                        break;
                    }
                    if attempt < retries {
                        let factor = 1u64 << ((attempt - 1).min(6));
                        let planned_sleep_ms = base_backoff_ms.saturating_mul(factor).min(max_backoff_ms);
                        let elapsed_after_attempt_ms = connect_start.elapsed().as_millis() as u64;
                        let remaining_after_attempt_ms = total_budget_ms.saturating_sub(elapsed_after_attempt_ms);
                        if remaining_after_attempt_ms == 0 {
                            break;
                        }
                        let sleep_ms = planned_sleep_ms.min(remaining_after_attempt_ms);
                        if sleep_ms > 0 {
                            std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
                        }
                    }
                }
            }
        }

        let allow_transfer_without_connect = std::env::var("MONARCH_HIXL_ALLOW_TRANSFER_WITHOUT_CONNECT")
            .map(|v| v != "0")
            .unwrap_or(false);
        if allow_transfer_without_connect {
            tracing::warn!(
                "HIXL: explicit connect to {} failed after {} attempts (local_engine={}, acl_device={:?}); continuing with TransferSync because MONARCH_HIXL_ALLOW_TRANSFER_WITHOUT_CONNECT=1",
                remote_engine,
                total_attempts,
                self.engine_id,
                hixl_sys::get_acl_device(),
            );
            return Ok(());
        }

        Err(anyhow::anyhow!(
            "HIXL connect to {} failed after {} attempts (local_engine={}, acl_device={:?}, timeout_ms={}, total_budget_ms={}, fast_fail_enabled={}, last_error={})",
            remote_engine,
            total_attempts,
            self.engine_id,
            hixl_sys::get_acl_device(),
            timeout_ms,
            total_budget_ms,
            fast_fail_enabled,
            last_err.unwrap_or_else(|| "unknown".to_string())
        ))
    }

    fn acquire_memory(
        &mut self,
        addr: usize,
        size: usize,
        mem_type: hixl_sys::HixlMemType,
    ) -> Result<SendMemHandle> {
        if let Some((handle, refcnt)) = self.registered_by_addr.get_mut(&addr) {
            *refcnt += 1;
            return Ok(*handle);
        }

        let handle = self
            .hixl
            .register_mem(addr, size, mem_type)
            .map_err(|e| anyhow::anyhow!("HIXL register_mem failed: {}", e))?;
        let mem_handle = SendMemHandle(handle);
        self.registered_by_addr.insert(addr, (mem_handle, 1));
        Ok(mem_handle)
    }

    fn release_memory(&mut self, addr: usize) -> Result<()> {
        let Some((mem_handle, refcnt)) = self.registered_by_addr.get_mut(&addr) else {
            return Ok(());
        };

        if *refcnt > 1 {
            *refcnt -= 1;
            return Ok(());
        }

        let mem_handle = *mem_handle;
        self.registered_by_addr.remove(&addr);
        self.hixl
            .deregister_mem(mem_handle.0)
            .map_err(|e| anyhow::anyhow!("HIXL deregister_mem failed: {}", e))?;
        Ok(())
    }

    fn register_memory(
        &mut self,
        buf_id: usize,
        addr: usize,
        size: usize,
        mem_type: hixl_sys::HixlMemType,
    ) -> Result<SendMemHandle> {
        let mem_handle = self.acquire_memory(addr, size, mem_type)?;
        self.exported_buf_addrs.insert(buf_id, addr);
        Ok(mem_handle)
    }

    fn deregister_memory(&mut self, buf_id: usize) -> Result<()> {
        if let Some(addr) = self.exported_buf_addrs.remove(&buf_id) {
            self.release_memory(addr)?;
        }
        Ok(())
    }

    fn register_transfer_memory(&mut self, addr: usize, size: usize) -> Result<()> {
        if self.transfer_registered_addrs.contains(&addr) {
            return Ok(());
        }

        self.acquire_memory(addr, size, hixl_sys::HixlMemType::HIXL_MEM_DEVICE)
            .map(|_| ())?;
        self.transfer_registered_addrs.insert(addr);
        Ok(())
    }

    fn deregister_transfer_memory(&mut self, addr: usize) -> Result<()> {
        if !self.transfer_registered_addrs.remove(&addr) {
            return Ok(());
        }
        self.release_memory(addr)
    }

    fn registration_hit_status(&self, addr: usize) -> (bool, usize) {
        self.registered_by_addr
            .get(&addr)
            .map(|(_, refcnt)| (true, *refcnt))
            .unwrap_or((false, 0))
    }

    fn transfer_alignment(addr: usize, len: usize, align: usize) -> bool {
        addr % align == 0 && len % align == 0
    }

    fn transfer_connected(
        &mut self,
        remote_engine: &str,
        local_addr: usize,
        local_size: usize,
        remote_addr: usize,
        transfer_op: hixl_sys::HixlTransferOp,
        timeout_ms: i32,
    ) -> Result<()> {
        if !self.connected_peers.contains_key(remote_engine) {
            return Err(anyhow::anyhow!(
                "HIXL transfer requested before connect: local_engine={} remote_engine={}",
                self.engine_id,
                remote_engine,
            ));
        }

        let (hit_before, refcnt_before) = self.registration_hit_status(local_addr);
        let align16_ok = Self::transfer_alignment(local_addr, local_size, 16)
            && Self::transfer_alignment(remote_addr, local_size, 16);
        let align64_ok = Self::transfer_alignment(local_addr, local_size, 64)
            && Self::transfer_alignment(remote_addr, local_size, 64);
        tracing::warn!(
            "HIXL: registration hit before TransferSync local_addr={:#x} remote_addr={:#x} size={} hit={} refcnt={} remote_engine={} op={:?} align16_ok={} align64_ok={} local_mod16={} remote_mod16={} len_mod16={} local_mod64={} remote_mod64={} len_mod64={}",
            local_addr,
            remote_addr,
            local_size,
            hit_before,
            refcnt_before,
            remote_engine,
            transfer_op,
            align16_ok,
            align64_ok,
            local_addr % 16,
            remote_addr % 16,
            local_size % 16,
            local_addr % 64,
            remote_addr % 64,
            local_size % 64,
        );

        let transfer_result = match transfer_op {
            hixl_sys::HixlTransferOp::HIXL_WRITE => self
                .hixl
                .transfer_write(remote_engine, local_addr, remote_addr, local_size, timeout_ms),
            hixl_sys::HixlTransferOp::HIXL_READ => self
                .hixl
                .transfer_read(remote_engine, local_addr, remote_addr, local_size, timeout_ms),
            _ => self
                .hixl
                .transfer_sync(
                    remote_engine,
                    transfer_op,
                    &[hixl_sys::HixlTransferOpDesc {
                        local_addr,
                        remote_addr,
                        len: local_size,
                    }],
                    timeout_ms,
                ),
        }
        .map_err(|e| anyhow::anyhow!("HIXL transfer to {} failed: {}", remote_engine, e));

        let (hit_after, refcnt_after) = self.registration_hit_status(local_addr);
        tracing::warn!(
            "HIXL: registration hit after TransferSync local_addr={:#x} size={} hit={} refcnt={} remote_engine={} op={:?} transfer_ok={}",
            local_addr,
            local_size,
            hit_after,
            refcnt_after,
            remote_engine,
            transfer_op,
            transfer_result.is_ok(),
        );

        transfer_result
    }
}

fn hixl_device_id_from_env() -> i32 {
    if let Ok(v) = std::env::var("MONARCH_NPU_DEVICE") {
        if let Ok(id) = v.trim().parse::<i32>() {
            return id;
        }
    }
    if let Ok(v) = std::env::var("ASCEND_RT_VISIBLE_DEVICES") {
        let first = v.split(',').next().unwrap_or("").trim();
        if let Ok(id) = first.parse::<i32>() {
            return id;
        }
    }
    0
}

/// Initialize the process-global HIXL instance with the given engine_id.
/// Called once by `HixlManagerActor` during initialization.
fn init_process_hixl(engine_id: String, device_id: i32) -> Result<()> {
    if PROCESS_HIXL.get().is_some() {
        tracing::warn!("HIXL: PROCESS_HIXL already initialized, skipping");
        return Ok(());
    }

    let (tx, rx) = mpsc::channel::<Result<ProcessHixl, String>>();
    let init_engine_id = engine_id.clone();
    std::thread::Builder::new()
        .name(format!("hixl-init-dev{}", device_id))
        .spawn(move || {
            let init_result = (|| -> Result<ProcessHixl, String> {
                hixl_sys::set_acl_device(device_id)
                    .map_err(|e| format!("set_acl_device({}) failed: {}", device_id, e))?;

                let hixl = hixl_sys::Hixl::new()
                    .map_err(|e| format!("create HIXL instance failed: {}", e))?;

                // Keep init options stable during debugging: align with ref baseline.
                let option_storage: Vec<(String, String)> = vec![
                    (hixl_sys::HIXL_OPTION_BUFFER_POOL.to_string(), "0:0".to_string()),
                ];
                let options: Vec<(&str, &str)> = option_storage
                    .iter()
                    .map(|(key, value)| (key.as_str(), value.as_str()))
                    .collect();
                hixl.initialize(&init_engine_id, &options)
                    .map_err(|e| format!("initialize failed: {}", e))?;

                // HiXL listeners may not be immediately ready after initialize returns.
                // A short settle delay reduces first-connect race windows.
                let init_settle_ms = parse_env_u64("MONARCH_HIXL_INIT_SETTLE_MS", 500);
                if init_settle_ms > 0 {
                    tracing::info!(
                        "HIXL: init settle sleep {}ms before serving connects (engine_id={})",
                        init_settle_ms,
                        init_engine_id,
                    );
                    std::thread::sleep(std::time::Duration::from_millis(init_settle_ms));
                }

                Ok(ProcessHixl {
                    engine_id: init_engine_id,
                    hixl,
                    connected_peers: HashMap::new(),
                    registered_by_addr: HashMap::new(),
                    exported_buf_addrs: HashMap::new(),
                    transfer_registered_addrs: HashSet::new(),
                })
            })();

            let _ = tx.send(init_result);
        })
        .map_err(|e| anyhow::anyhow!("spawn hixl init thread failed: {}", e))?;

    let process_hixl = rx
        .recv()
        .map_err(|e| anyhow::anyhow!("hixl init thread channel recv failed: {}", e))?
        .map_err(|e| anyhow::anyhow!("HIXL dedicated init failed: {}", e))?;

    PROCESS_HIXL
        .set(Mutex::new(process_hixl))
        .map_err(|_| anyhow::anyhow!("HIXL PROCESS_HIXL already initialized"))?;

    tracing::info!("HIXL: global PROCESS_HIXL initialized with engine_id={}", engine_id);
    Ok(())
}

/// Get the engine_id from the process-global HIXL instance.
pub fn get_engine_id() -> Option<String> {
    PROCESS_HIXL.get().and_then(|px| {
        px.lock().ok().map(|guard| guard.engine_id.clone())
    })
}

/// Get local engine_id, waiting for PROCESS_HIXL initialization if needed.
pub fn get_engine_id_wait(timeout_ms: i32) -> Result<String> {
    wait_for_process_hixl(timeout_ms)?;
    get_engine_id().ok_or_else(|| anyhow::anyhow!("HIXL engine_id not available"))
}

/// Explicitly connect local PROCESS_HIXL to a peer engine.
pub fn hixl_connect_peer(peer_engine_id: &str, timeout_ms: i32) -> Result<()> {
    wait_for_process_hixl(timeout_ms)?;
    let process_hixl = PROCESS_HIXL
        .get()
        .ok_or_else(|| anyhow::anyhow!("HIXL not initialized"))?;
    let mut guard = process_hixl
        .lock()
        .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;
    let local_engine_id = guard.engine_id.clone();
    tracing::warn!(
        "HIXL: hixl_connect_peer start local_engine={} remote_engine={} timeout_ms={} acl_device={:?}",
        local_engine_id,
        peer_engine_id,
        timeout_ms,
        hixl_sys::get_acl_device(),
    );
    let ret = guard.ensure_connected(peer_engine_id, timeout_ms);
    match &ret {
        Ok(()) => {
            tracing::warn!(
                "HIXL: hixl_connect_peer done local_engine={} remote_engine={} acl_device={:?}",
                local_engine_id,
                peer_engine_id,
                hixl_sys::get_acl_device(),
            );
        }
        Err(e) => {
            tracing::error!(
                "HIXL: hixl_connect_peer failed local_engine={} remote_engine={} timeout_ms={} acl_device={:?} err={}",
                local_engine_id,
                peer_engine_id,
                timeout_ms,
                hixl_sys::get_acl_device(),
                e,
            );
        }
    }
    ret
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
/// 
/// IMPORTANT: The order of operations is critical for HIXL:
/// 1. RegisterMem - register local memory FIRST
/// 2. Connect - establish connection to remote engine
/// 3. TransferSync - execute the transfer
pub fn hixl_transfer_sync(
    remote_engine_id: &str,
    local_addr: usize,
    local_size: usize,
    remote_addr: usize,
    transfer_op: hixl_sys::HixlTransferOp,
    timeout_ms: i32,
) -> Result<()> {
    tracing::warn!("hixl_transfer_sync: START remote={} local_addr={:#x} size={} remote_addr={:#x} op={:?}",
        remote_engine_id, local_addr, local_size, remote_addr, transfer_op);
    
    // Wait for HIXL to be initialized (may be still initializing in background)
    tracing::warn!("hixl_transfer_sync: waiting for PROCESS_HIXL initialization");
    wait_for_process_hixl(timeout_ms)?;
    tracing::warn!("hixl_transfer_sync: PROCESS_HIXL initialized, getting lock");
    
    let process_hixl = PROCESS_HIXL
        .get()
        .ok_or_else(|| anyhow::anyhow!("HIXL not initialized - ensure HixlManagerActor is spawned first"))?;
    let mut guard = process_hixl
        .lock()
        .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;
    tracing::warn!("hixl_transfer_sync: lock acquired");

    tracing::warn!("hixl_transfer_sync: registering local memory");
    guard.register_transfer_memory(local_addr, local_size)?;
    tracing::warn!("hixl_transfer_sync: local memory registered");

    // Now connect to remote engine (after memory registration)
    guard.ensure_connected(remote_engine_id, timeout_ms)?;
    tracing::warn!("hixl_transfer_sync: connected to {}", remote_engine_id);

    tracing::warn!("hixl_transfer_sync: calling TransferSync");
    let result = match transfer_op {
        hixl_sys::HixlTransferOp::HIXL_WRITE => {
            guard
                .hixl
                .transfer_write(remote_engine_id, local_addr, remote_addr, local_size, timeout_ms)
        }
        hixl_sys::HixlTransferOp::HIXL_READ => {
            guard
                .hixl
                .transfer_read(remote_engine_id, local_addr, remote_addr, local_size, timeout_ms)
        }
        _ => guard.hixl.transfer_sync(
            remote_engine_id,
            transfer_op,
            &[hixl_sys::HixlTransferOpDesc {
                local_addr,
                remote_addr,
                len: local_size,
            }],
            timeout_ms,
        ),
    };
    tracing::warn!("hixl_transfer_sync: TransferSync returned {:?}", result);

    result.map_err(|e| anyhow::anyhow!("HIXL transfer to {} failed: {}", remote_engine_id, e))
}

pub fn hixl_register_transfer_memory(local_addr: usize, local_size: usize) -> Result<()> {
    wait_for_process_hixl(20_000)?;
    let process_hixl = PROCESS_HIXL
        .get()
        .ok_or_else(|| anyhow::anyhow!("HIXL not initialized - ensure HixlManagerActor is spawned first"))?;
    let mut guard = process_hixl
        .lock()
        .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;
    guard.register_transfer_memory(local_addr, local_size)
}

pub fn hixl_deregister_transfer_memory(local_addr: usize) -> Result<()> {
    let process_hixl = PROCESS_HIXL
        .get()
        .ok_or_else(|| anyhow::anyhow!("HIXL not initialized - ensure HixlManagerActor is spawned first"))?;
    let mut guard = process_hixl
        .lock()
        .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;
    guard.deregister_transfer_memory(local_addr)
}

pub fn hixl_transfer_connected(
    remote_engine_id: &str,
    local_addr: usize,
    local_size: usize,
    remote_addr: usize,
    transfer_op: hixl_sys::HixlTransferOp,
    timeout_ms: i32,
) -> Result<()> {
    wait_for_process_hixl(timeout_ms)?;
    let process_hixl = PROCESS_HIXL
        .get()
        .ok_or_else(|| anyhow::anyhow!("HIXL not initialized - ensure HixlManagerActor is spawned first"))?;
    let mut guard = process_hixl
        .lock()
        .map_err(|e| anyhow::anyhow!("HIXL lock poisoned: {}", e))?;
    guard.transfer_connected(
        remote_engine_id,
        local_addr,
        local_size,
        remote_addr,
        transfer_op,
        timeout_ms,
    )
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
        let device_id = hixl_device_id_from_env();
        tracing::info!(
            "HIXL: init actor with engine_id={} device_id={} acl_device_before={:?}",
            self.engine_id,
            device_id,
            hixl_sys::get_acl_device(),
        );
        init_process_hixl(self.engine_id.clone(), device_id)?;
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
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        ops: Vec<RdmaOp>,
        timeout: std::time::Duration,
    ) -> Result<()> {
        // Keep HIXL data plane on a single, bidirectionally-connected path.
        // This mirrors RdmaRemoteBuffer read/write behavior and avoids submit-only
        // fast paths that may skip remote back-connect ordering.
        let timeout_secs = timeout.as_secs().max(1);

        for op in ops {
            match op.op_type {
                RdmaOpType::ReadIntoLocal => {
                    op.remote
                        .read_into_local(cx, op.local.clone(), timeout_secs)
                        .await?;
                }
                RdmaOpType::WriteFromLocal => {
                    op.remote
                        .write_from_local(cx, op.local.clone(), timeout_secs)
                        .await?;
                }
            }
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