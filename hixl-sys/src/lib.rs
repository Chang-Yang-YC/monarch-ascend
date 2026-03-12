// FFI bindings to the HiXL C shim (libtest_hixl.so).
//
// The shim wraps the HiXL C++ API behind a flat C interface,
// mirroring the role that rdmaxcel-sys plays for ibverbs.

#![allow(non_camel_case_types)]

use std::os::raw::c_char;
use std::os::raw::c_int;
use std::os::raw::c_void;

unsafe extern "C" {
    /// Initialize a HiXL engine on the given NPU device.
    /// Returns an opaque context pointer, or null on failure.
    pub fn hixl_init_engine(dev: c_int, engine_id: *const c_char) -> *mut c_void;

    /// Establish a connection to a remote engine.
    /// Returns 0 on success.
    pub fn hixl_connect(ctx: *mut c_void, remote_engine_id: *const c_char) -> c_int;

    /// Register a device memory region for RDMA access.
    /// Returns 0 on success.
    pub fn hixl_register_mem(ctx: *mut c_void, addr: usize, size: usize) -> c_int;

    /// Single-sided READ: pull remote data into local buffer.
    /// Returns 0 on success.
    pub fn hixl_transfer_read(
        ctx: *mut c_void,
        remote_engine_id: *const c_char,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
    ) -> c_int;

    /// Single-sided WRITE: push local data into remote buffer.
    /// Returns 0 on success.
    pub fn hixl_transfer_write(
        ctx: *mut c_void,
        remote_engine_id: *const c_char,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
    ) -> c_int;

    /// Finalize and release engine resources.
    pub fn hixl_cleanup(ctx: *mut c_void);
}

/// Wrapper around the raw HiXL context pointer.
/// Implements Send + Sync because the underlying C library
/// serializes access through its own internal locks, and we
/// guarantee single-threaded use via the actor model.
#[derive(Debug)]
pub struct HixlEngine {
    ptr: *mut c_void,
}

unsafe impl Send for HixlEngine {}
unsafe impl Sync for HixlEngine {}

impl HixlEngine {
    pub fn new(dev: i32, engine_id: &str) -> Result<Self, String> {
        let c_eid = std::ffi::CString::new(engine_id)
            .map_err(|e| format!("invalid engine_id: {e}"))?;
        let ptr = unsafe { hixl_init_engine(dev, c_eid.as_ptr()) };
        if ptr.is_null() {
            Err(format!(
                "hixl_init_engine failed for dev={dev} engine_id={engine_id}"
            ))
        } else {
            Ok(Self { ptr })
        }
    }

    /// Wrap a pre-existing engine pointer (created externally, e.g. from Python ctypes).
    /// SAFETY: caller must ensure `ptr` is a valid HixlTestCtx* from hixl_init_engine.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        Self { ptr }
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn connect(&self, remote_engine_id: &str) -> Result<(), i32> {
        let c_eid = std::ffi::CString::new(remote_engine_id).unwrap();
        let ret = unsafe { hixl_connect(self.ptr, c_eid.as_ptr()) };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    pub fn register_mem(&self, addr: usize, size: usize) -> Result<(), i32> {
        let ret = unsafe { hixl_register_mem(self.ptr, addr, size) };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    pub fn transfer_read(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
    ) -> Result<(), i32> {
        let c_eid = std::ffi::CString::new(remote_engine_id).unwrap();
        let ret = unsafe {
            hixl_transfer_read(self.ptr, c_eid.as_ptr(), local_addr, remote_addr, len)
        };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }

    pub fn transfer_write(
        &self,
        remote_engine_id: &str,
        local_addr: usize,
        remote_addr: usize,
        len: usize,
    ) -> Result<(), i32> {
        let c_eid = std::ffi::CString::new(remote_engine_id).unwrap();
        let ret = unsafe {
            hixl_transfer_write(self.ptr, c_eid.as_ptr(), local_addr, remote_addr, len)
        };
        if ret == 0 { Ok(()) } else { Err(ret) }
    }
}

impl Drop for HixlEngine {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { hixl_cleanup(self.ptr) };
        }
    }
}
