/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! # HIXL Manager Actor (Python-managed mode)
//!
//! HIXL engine is created and owned by Python (via ctypes). Rust only stores
//! the engine_id and buffer addresses for metadata propagation. All actual
//! HIXL operations (init, register, connect, transfer) happen from Python.
//!
//! The Python-side HIXL engine_id is passed via the `MONARCH_PYTHON_HIXL_ENGINE_ID`
//! environment variable, set during worker bootstrap.

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
use crate::RdmaTransportLevel;
use crate::backend::RdmaBackend;
use crate::rdma_manager_actor::RdmaManagerActor;

/// Messages handled by [`HixlManagerActor`].
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
}
wirevalue::register_type!(HixlManagerMessage);

// ============================================================================
// Engine-id retrieval from Python-set env var
// ============================================================================

static PYTHON_ENGINE_ID: OnceLock<String> = OnceLock::new();

pub fn global_engine_id() -> Result<String> {
    let eid = PYTHON_ENGINE_ID.get_or_init(|| {
        std::env::var("MONARCH_PYTHON_HIXL_ENGINE_ID")
            .unwrap_or_else(|_| {
                let ip = crate::rdma_manager_actor::local_ip_for_hixl();
                let port = 20000 + (std::process::id() % 40000);
                let fallback = format!("{}:{}", ip, port);
                tracing::warn!(
                    "MONARCH_PYTHON_HIXL_ENGINE_ID not set, using fallback: {}",
                    fallback
                );
                fallback
            })
    });
    Ok(eid.clone())
}

/// Kept for backward compat — Connect is now a no-op on the Rust side.
pub fn hixl_connect_peer(_peer_engine_id: &str) -> Result<()> {
    Ok(())
}

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
    owner: OnceLock<ActorHandle<RdmaManagerActor>>,
}

impl HixlManagerActor {
    pub fn new(_engine_id: String) -> Self {
        Self {
            owner: OnceLock::new(),
        }
    }
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
        tracing::warn!(
            "HIXL request_buffer (python-mode): id={} addr={:#x} size={}",
            remote_buf_id, addr, size
        );
        let engine_id = global_engine_id()?;
        Ok(Some(HixlBuffer {
            engine_id,
            addr,
            size,
        }))
    }

    async fn release_buffer(
        &mut self,
        _cx: &Context<Self>,
        remote_buf_id: usize,
    ) -> Result<(), anyhow::Error> {
        tracing::debug!("HIXL release_buffer: id={}", remote_buf_id);
        Ok(())
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
            "HIXL submit should be handled by Python ctypes path, not Rust"
        ))
    }

    fn transport_level(&self) -> RdmaTransportLevel {
        RdmaTransportLevel::Nic
    }

    fn transport_info(&self) -> Option<Self::TransportInfo> {
        None
    }
}
