/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Bindings for Ascend ACL runtime: streams, events, and device management.
//! Mirrors the API surface of `torch-sys-cuda/src/cuda.rs`.

use std::time::Duration;

use derive_more::Into;
use hccl_sys::aclrtStream;
use monarch_types::py_global;
use pyo3::Py;
use pyo3::PyAny;
use pyo3::prelude::*;
use thiserror::Error;
use torch_sys2::NpuDevice;

py_global!(npu_stream_class, "torch_npu.npu", "Stream");
py_global!(npu_event_class, "torch_npu.npu", "Event");
py_global!(npu_current_stream, "torch_npu.npu", "current_stream");
py_global!(npu_set_stream, "torch_npu.npu", "set_stream");
py_global!(npu_current_device, "torch_npu.npu", "current_device");
py_global!(npu_set_device_fn, "torch_npu.npu", "set_device");

/// Wrapper around an ACL / torch_npu stream.
#[derive(Debug, Into)]
#[into(ref)]
pub struct Stream {
    inner: Py<PyAny>,
}

impl Clone for Stream {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

impl Stream {
    pub fn new() -> Self {
        Python::attach(|py| {
            let stream = npu_stream_class(py).call0().unwrap();
            Self {
                inner: stream.into(),
            }
        })
    }

    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            inner: self.inner.clone_ref(py),
        }
    }

    pub fn new_with_device(device: NpuDevice) -> Self {
        Python::attach(|py| {
            let device_idx: i8 = device.index().into();
            let stream = npu_stream_class(py).call1((device_idx,)).unwrap();
            Self {
                inner: stream.into(),
            }
        })
    }

    pub fn get_current_stream() -> Self {
        Python::attach(|py| {
            let stream = npu_current_stream(py).call0().unwrap();
            Self {
                inner: stream.into(),
            }
        })
    }

    pub fn get_current_stream_on_device(device: NpuDevice) -> Self {
        Python::attach(|py| {
            let device_idx: i8 = device.index().into();
            let stream = npu_current_stream(py).call1((device_idx,)).unwrap();
            Self {
                inner: stream.into(),
            }
        })
    }

    pub fn set_current_stream(stream: &Stream) {
        Python::attach(|py| {
            let stream_obj = stream.inner.bind(py);

            let current_device = npu_current_device(py)
                .call0()
                .unwrap()
                .extract::<i64>()
                .unwrap();

            let stream_device = stream_obj
                .getattr("device_index")
                .unwrap()
                .extract::<i64>()
                .unwrap();

            if current_device != stream_device {
                npu_set_device_fn(py).call1((stream_device,)).unwrap();
            }

            npu_set_stream(py).call1((stream_obj,)).unwrap();
        })
    }

    pub fn wait_event(&self, event: &mut Event) {
        event.wait(Some(self))
    }

    pub fn wait_stream(&self, stream: &Stream) {
        self.wait_event(&mut stream.record_event(None))
    }

    pub fn record_event(&self, event: Option<Event>) -> Event {
        let mut event = event.unwrap_or(Event::new());
        event.record(Some(self));
        event
    }

    pub fn query(&self) -> bool {
        Python::attach(|py| {
            let stream_obj = self.inner.bind(py);
            stream_obj
                .call_method0("query")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        })
    }

    pub fn synchronize(&self) {
        Python::attach(|py| {
            let stream_obj = self.inner.bind(py);
            stream_obj.call_method0("synchronize").unwrap();
        })
    }

    /// Return the raw `aclrtStream` pointer for use in HCCL calls.
    ///
    /// torch_npu exposes the stream pointer via the `npu_stream` attribute,
    /// analogous to PyTorch's `cuda_stream`.
    pub fn stream(&self) -> aclrtStream {
        Python::attach(|py| {
            let stream_obj = self.inner.bind(py);
            // torch_npu stores the raw pointer in `npu_stream`
            let raw = stream_obj
                .getattr("npu_stream")
                .or_else(|_| stream_obj.getattr("stream_id"))
                .unwrap();
            let ptr = raw.extract::<usize>().unwrap();
            ptr as aclrtStream
        })
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        self.stream() == other.stream()
    }
}

/// Wrapper around an ACL / torch_npu event.
#[derive(Debug)]
pub struct Event {
    inner: Py<PyAny>,
}

impl Clone for Event {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

impl Event {
    pub fn new() -> Self {
        Python::attach(|py| {
            let event = npu_event_class(py).call0().unwrap();
            Self {
                inner: event.into(),
            }
        })
    }

    pub fn record(&mut self, stream: Option<&Stream>) {
        Python::attach(|py| {
            let event_obj = self.inner.bind(py);
            match stream {
                Some(s) => {
                    let stream_obj = s.inner.bind(py);
                    event_obj.call_method1("record", (stream_obj,)).unwrap();
                }
                None => {
                    event_obj.call_method0("record").unwrap();
                }
            }
        })
    }

    pub fn wait(&mut self, stream: Option<&Stream>) {
        Python::attach(|py| {
            let event_obj = self.inner.bind(py);
            match stream {
                Some(s) => {
                    let stream_obj = s.inner.bind(py);
                    event_obj.call_method1("wait", (stream_obj,)).unwrap();
                }
                None => {
                    event_obj.call_method0("wait").unwrap();
                }
            }
        })
    }

    pub fn query(&self) -> bool {
        Python::attach(|py| {
            let event_obj = self.inner.bind(py);
            event_obj
                .call_method0("query")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        })
    }

    pub fn elapsed_time(&self, end_event: &Event) -> Duration {
        Python::attach(|py| {
            let event_obj = self.inner.bind(py);
            let end_event_obj = end_event.inner.bind(py);
            let elapsed_ms = event_obj
                .call_method1("elapsed_time", (end_event_obj,))
                .unwrap()
                .extract::<f64>()
                .unwrap();
            Duration::from_millis(elapsed_ms as u64)
        })
    }

    pub fn synchronize(&self) {
        Python::attach(|py| {
            let event_obj = self.inner.bind(py);
            event_obj.call_method0("synchronize").unwrap();
        })
    }
}

#[derive(Debug, Error)]
pub enum AclError {
    #[error("ACL error: invalid parameter (100000)")]
    InvalidParam,
    #[error("ACL error: uninitialized (100001)")]
    Uninitialized,
    #[error("ACL error: runtime error ({0})")]
    RuntimeError(i32),
    #[error("ACL error: unknown ({0})")]
    Unknown(i32),
}

pub fn acl_check(result: i32) -> Result<(), AclError> {
    match result {
        0 => Ok(()),
        100000 => Err(AclError::InvalidParam),
        100001 => Err(AclError::Uninitialized),
        code if (100002..200000).contains(&code) => Err(AclError::RuntimeError(code)),
        code => Err(AclError::Unknown(code)),
    }
}

pub fn set_device(device: NpuDevice) -> Result<(), AclError> {
    let index: i8 = device.index().into();
    unsafe { acl_check(hccl_sys::aclrtSetDevice(index.into())) }
}
