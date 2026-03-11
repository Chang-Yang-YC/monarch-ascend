/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Backend abstraction layer.
//!
//! Re-exports the communication and runtime types for the compile-time selected
//! backend (CUDA or Ascend NPU) under a unified set of names so that the rest
//! of `monarch_tensor_worker` can be backend-agnostic.

// ---- CUDA backend (default) ----
#[cfg(feature = "cuda_backend")]
pub use torch_sys_cuda::cuda::Event;
#[cfg(feature = "cuda_backend")]
pub use torch_sys_cuda::cuda::Stream;
#[cfg(feature = "cuda_backend")]
pub use torch_sys_cuda::nccl::Communicator;
#[cfg(feature = "cuda_backend")]
pub use torch_sys_cuda::nccl::NcclError as CommError;
#[cfg(feature = "cuda_backend")]
pub use torch_sys_cuda::nccl::NcclStatus as CommStatus;
#[cfg(feature = "cuda_backend")]
pub use torch_sys_cuda::nccl::ReduceOp;
#[cfg(feature = "cuda_backend")]
pub use torch_sys_cuda::nccl::UniqueId as CommId;
#[cfg(feature = "cuda_backend")]
pub use torch_sys2::CudaDevice as AccelDevice;

#[cfg(feature = "cuda_backend")]
pub fn group_start() -> Result<CommStatus, CommError> {
    torch_sys_cuda::nccl::group_start()
}
#[cfg(feature = "cuda_backend")]
pub fn group_end() -> Result<CommStatus, CommError> {
    torch_sys_cuda::nccl::group_end()
}

#[cfg(feature = "cuda_backend")]
pub fn set_device(device: AccelDevice) -> anyhow::Result<()> {
    torch_sys_cuda::cuda::set_device(device).map_err(|e| anyhow::anyhow!("{}", e))
}

// ---- Ascend NPU backend ----
#[cfg(feature = "ascend_backend")]
pub use torch_sys_ascend::acl::Event;
#[cfg(feature = "ascend_backend")]
pub use torch_sys_ascend::acl::Stream;
#[cfg(feature = "ascend_backend")]
pub use torch_sys_ascend::hccl::Communicator;
#[cfg(feature = "ascend_backend")]
pub use torch_sys_ascend::hccl::HcclError as CommError;
#[cfg(feature = "ascend_backend")]
pub use torch_sys_ascend::hccl::HcclStatus as CommStatus;
#[cfg(feature = "ascend_backend")]
pub use torch_sys_ascend::hccl::ReduceOp;
#[cfg(feature = "ascend_backend")]
pub use torch_sys_ascend::hccl::RootInfo as CommId;
#[cfg(feature = "ascend_backend")]
pub use torch_sys2::NpuDevice as AccelDevice;

#[cfg(feature = "ascend_backend")]
pub fn group_start() -> Result<CommStatus, CommError> {
    // HCCL has no group_start/end; batch operations use HcclBatchSendRecv instead.
    Ok(CommStatus::Success)
}
#[cfg(feature = "ascend_backend")]
pub fn group_end() -> Result<CommStatus, CommError> {
    Ok(CommStatus::Success)
}

#[cfg(feature = "ascend_backend")]
pub fn set_device(device: AccelDevice) -> anyhow::Result<()> {
    torch_sys_ascend::acl::set_device(device).map_err(|e| anyhow::anyhow!("{}", e))
}
