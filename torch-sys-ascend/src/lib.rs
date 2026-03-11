/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// Ascend NPU runtime bindings (ACL stream/event management).
pub mod acl;
/// HCCL communicator providing collective communication on Ascend NPUs.
pub mod hccl;
