/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Low-level Rust FFI bindings for HIXL (Huawei Xfer Library).
//!
//! HIXL provides one-sided zero-copy communication for Ascend NPUs,
//! supporting RDMA over RoCE, HCCS, and other transport protocols.
//!
//! The library is linked at build time against `libcann_hixl.so` (or `libhixl.so`),
//! which must be available at runtime in `LD_LIBRARY_PATH`.

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
mod inner {
    #[cfg(cargo)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use inner::*;

use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;


/// Safe wrapper result type for HIXL operations.
pub type HixlResult<T> = Result<T, HixlError>;

#[derive(Debug, Clone)]
pub struct HixlError {
    pub status: u32,
    pub message: String,
}

impl std::fmt::Display for HixlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HIXL error {}: {}", self.status, self.message)
    }
}

impl std::error::Error for HixlError {}

/// Check HIXL status and convert to Result.
pub fn check_status(status: HixlStatus) -> HixlResult<()> {
    if status == HIXL_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = HixlGetStatusString(status);
            if ptr.is_null() {
                "unknown error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().to_string()
            }
        };
        Err(HixlError {
            status,
            message: msg,
        })
    }
}

/// RAII wrapper around an HIXL instance.
pub struct Hixl {
    handle: HixlHandle,
}

impl std::fmt::Debug for Hixl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hixl({:p})", self.handle)
    }
}

unsafe impl Send for Hixl {}
unsafe impl Sync for Hixl {}

impl Hixl {
    /// Create a new HIXL instance.
    pub fn new() -> HixlResult<Self> {
        let handle = unsafe { HixlCreate() };
        if handle.is_null() {
            return Err(HixlError {
                status: HIXL_FAILED,
                message: "Failed to create HIXL instance".to_string(),
            });
        }
        Ok(Self { handle })
    }

    /// Initialize with the given local engine ID and options.
    pub fn initialize(
        &self,
        local_engine: &str,
        options: &[(&str, &str)],
    ) -> HixlResult<()> {
        let engine_c = CString::new(local_engine).unwrap();
        let opts: Vec<(CString, CString)> = options
            .iter()
            .map(|(k, v)| (CString::new(*k).unwrap(), CString::new(*v).unwrap()))
            .collect();
        let c_opts: Vec<HixlOption> = opts
            .iter()
            .map(|(k, v)| HixlOption {
                key: k.as_ptr(),
                value: v.as_ptr(),
            })
            .collect();
        let status = unsafe {
            HixlInitialize(
                self.handle,
                engine_c.as_ptr(),
                c_opts.as_ptr(),
                c_opts.len(),
            )
        };
        check_status(status)
    }

    /// Register a memory region for RDMA transfers.
    pub fn register_mem(
        &self,
        addr: usize,
        len: usize,
        mem_type: HixlMemType,
    ) -> HixlResult<HixlMemHandle> {
        let mut mem_handle: HixlMemHandle = std::ptr::null_mut();
        let status = unsafe {
            HixlRegisterMem(self.handle, addr, len, mem_type, &mut mem_handle)
        };
        check_status(status)?;
        Ok(mem_handle)
    }

    /// Deregister a previously registered memory region.
    pub fn deregister_mem(&self, mem_handle: HixlMemHandle) -> HixlResult<()> {
        let status = unsafe { HixlDeregisterMem(self.handle, mem_handle) };
        check_status(status)
    }

    /// Connect to a remote HIXL engine.
    pub fn connect(&self, remote_engine: &str, timeout_ms: i32) -> HixlResult<()> {
        let engine_c = CString::new(remote_engine).unwrap();
        let status =
            unsafe { HixlConnect(self.handle, engine_c.as_ptr(), timeout_ms) };
        check_status(status)
    }

    /// Disconnect from a remote HIXL engine.
    pub fn disconnect(&self, remote_engine: &str, timeout_ms: i32) -> HixlResult<()> {
        let engine_c = CString::new(remote_engine).unwrap();
        let status =
            unsafe { HixlDisconnect(self.handle, engine_c.as_ptr(), timeout_ms) };
        check_status(status)
    }

    /// Synchronous memory transfer (READ or WRITE).
    pub fn transfer_sync(
        &self,
        remote_engine: &str,
        operation: HixlTransferOp,
        op_descs: &[HixlTransferOpDesc],
        timeout_ms: i32,
    ) -> HixlResult<()> {
        let engine_c = CString::new(remote_engine).unwrap();
        let status = unsafe {
            HixlTransferSync(
                self.handle,
                engine_c.as_ptr(),
                operation,
                op_descs.as_ptr(),
                op_descs.len(),
                timeout_ms,
            )
        };
        check_status(status)
    }

    /// Asynchronous memory transfer, returns a request handle.
    pub fn transfer_async(
        &self,
        remote_engine: &str,
        operation: HixlTransferOp,
        op_descs: &[HixlTransferOpDesc],
    ) -> HixlResult<HixlTransferReq> {
        let engine_c = CString::new(remote_engine).unwrap();
        let mut req: HixlTransferReq = std::ptr::null_mut();
        let status = unsafe {
            HixlTransferAsync(
                self.handle,
                engine_c.as_ptr(),
                operation,
                op_descs.as_ptr(),
                op_descs.len(),
                &mut req,
            )
        };
        check_status(status)?;
        Ok(req)
    }

    /// Query the status of an async transfer request.
    pub fn get_transfer_status(
        &self,
        req: HixlTransferReq,
    ) -> HixlResult<HixlTransferStatus> {
        let mut status = HixlTransferStatus(0);
        let ret =
            unsafe { HixlGetTransferStatus(self.handle, req, &mut status) };
        check_status(ret)?;
        Ok(status)
    }

    /// Send a notification to a remote engine.
    pub fn send_notify(
        &self,
        remote_engine: &str,
        name: &str,
        msg: &str,
        timeout_ms: i32,
    ) -> HixlResult<()> {
        let engine_c = CString::new(remote_engine).unwrap();
        let name_c = CString::new(name).unwrap();
        let msg_c = CString::new(msg).unwrap();
        let status = unsafe {
            HixlSendNotify(
                self.handle,
                engine_c.as_ptr(),
                name_c.as_ptr(),
                msg_c.as_ptr(),
                timeout_ms,
            )
        };
        check_status(status)
    }

    /// Retrieve all pending notifications.
    pub fn get_notifies(&self) -> HixlResult<Vec<(String, String)>> {
        let mut notifies: Vec<(String, String)> = Vec::new();

        unsafe extern "C" fn collect_notify(
            name: *const c_char,
            msg: *const c_char,
            user_data: *mut std::ffi::c_void,
        ) {
            let notifies =
                &mut *(user_data as *mut Vec<(String, String)>);
            let name_str = if name.is_null() {
                String::new()
            } else {
                CStr::from_ptr(name).to_string_lossy().to_string()
            };
            let msg_str = if msg.is_null() {
                String::new()
            } else {
                CStr::from_ptr(msg).to_string_lossy().to_string()
            };
            notifies.push((name_str, msg_str));
        }

        let status = unsafe {
            HixlGetNotifies(
                self.handle,
                Some(collect_notify),
                &mut notifies as *mut Vec<(String, String)> as *mut std::ffi::c_void,
            )
        };
        check_status(status)?;
        Ok(notifies)
    }

    /// Get the raw handle (for advanced usage).
    pub fn raw_handle(&self) -> HixlHandle {
        self.handle
    }
}

impl Drop for Hixl {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                HixlFinalize(self.handle);
                HixlDestroy(self.handle);
            }
        }
    }
}
