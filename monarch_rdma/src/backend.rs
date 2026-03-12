/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! RDMA backend implementations.
//!
//! On GPU: ibverbs backend (rdmaxcel) for InfiniBand/RoCE.
//! On NPU: HIXL backend for Ascend RDMA/RoCE/HCCS.

#[cfg(not(feature = "hixl"))]
pub mod ibverbs;

#[cfg(feature = "hixl")]
pub mod hixl;

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde::Serialize;

use crate::RdmaOp;
use crate::RdmaTransportLevel;

/// Backend-specific context for a remote buffer.
///
/// - **Ibverbs**: native Rust-managed QP/MR transport (GPU).
/// - **Hixl**: Rust-managed HIXL transport (Ascend NPU).
#[derive(Debug, Clone)]
pub enum RdmaBackendContext {
    #[cfg(not(feature = "hixl"))]
    Ibverbs(
        hyperactor::ActorRef<ibverbs::manager_actor::IbvManagerActor>,
        Arc<tokio::sync::OnceCell<ibverbs::IbvBuffer>>,
    ),
    #[cfg(feature = "hixl")]
    Hixl(hixl::HixlBuffer),
}

impl Serialize for RdmaBackendContext {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            #[cfg(not(feature = "hixl"))]
            RdmaBackendContext::Ibverbs(actor_ref, _) => {
                serializer.serialize_newtype_variant("RdmaBackendContext", 0, "Ibverbs", actor_ref)
            }
            #[cfg(feature = "hixl")]
            RdmaBackendContext::Hixl(buf) => {
                serializer.serialize_newtype_variant("RdmaBackendContext", 1, "Hixl", buf)
            }
        }
    }
}

impl<'de> Deserialize<'de> for RdmaBackendContext {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(rename = "RdmaBackendContext")]
        enum Repr {
            #[cfg(not(feature = "hixl"))]
            Ibverbs(hyperactor::ActorRef<ibverbs::manager_actor::IbvManagerActor>),
            #[cfg(feature = "hixl")]
            Hixl(hixl::HixlBuffer),
        }

        match Repr::deserialize(deserializer)? {
            #[cfg(not(feature = "hixl"))]
            Repr::Ibverbs(actor_ref) => Ok(RdmaBackendContext::Ibverbs(
                actor_ref,
                Arc::new(tokio::sync::OnceCell::new()),
            )),
            #[cfg(feature = "hixl")]
            Repr::Hixl(buf) => Ok(RdmaBackendContext::Hixl(buf)),
        }
    }
}

/// Backend for executing RDMA operations over a specific transport.
#[async_trait]
pub trait RdmaBackend: Send + Debug {
    type TransportInfo;

    async fn submit(
        &mut self,
        cx: &(impl hyperactor::context::Actor + Send + Sync),
        ops: Vec<RdmaOp>,
        timeout: Duration,
    ) -> Result<()>;

    fn transport_level(&self) -> RdmaTransportLevel;

    fn transport_info(&self) -> Option<Self::TransportInfo>;
}
