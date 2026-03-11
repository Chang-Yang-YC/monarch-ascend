/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! HIXL-based RDMA backend for Ascend NPU single-sided communication.
//!
//! This module implements the [`RdmaBackend`] trait using the HIXL (Huawei Xfer
//! Library) which provides one-sided zero-copy data transfers over RDMA/RoCE,
//! HCCS, and other transports available on Ascend NPUs.
//!
//! # Architecture
//!
//! Unlike the ibverbs backend which manages QPs and MRs directly, the HIXL
//! backend delegates transport management to the HIXL library, which internally
//! handles connection setup, memory registration, and transfer scheduling across
//! multiple link types.

pub mod manager_actor;

use std::fmt::Debug;

use serde::Deserialize;
use serde::Serialize;
use typeuri::Named;

/// HIXL buffer descriptor, exchanged between peers to identify a
/// registered memory region on the remote side.
#[derive(Debug, Named, Clone, Serialize, Deserialize)]
pub struct HixlBuffer {
    /// Engine identifier of the owner (e.g., "192.168.1.1:50051").
    pub engine_id: String,
    /// Remote virtual address of the registered buffer.
    pub addr: usize,
    /// Size of the buffer in bytes.
    pub size: usize,
}
wirevalue::register_type!(HixlBuffer);
