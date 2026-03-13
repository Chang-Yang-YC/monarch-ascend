/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#![allow(unsafe_op_in_unsafe_fn)]
use std::ops::Deref;
use std::sync::Arc;

use hyperactor_mesh::ActorMesh;
use monarch_hyperactor::context::PyInstance;
use monarch_hyperactor::proc_mesh::PyProcMesh;
use monarch_hyperactor::pytokio::PyPythonTask;
use monarch_hyperactor::runtime::monarch_with_gil_blocking;
use monarch_hyperactor::runtime::signal_safe_block_on;
use monarch_rdma::RdmaLocalMemory;
use monarch_rdma::RdmaManagerActor;
use monarch_rdma::RdmaManagerMessageClient;
use monarch_rdma::RdmaRemoteBuffer;
use monarch_rdma::register_segment_scanner;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::PyTuple;
use pyo3::types::PyType;
use typeuri::Named;

// ---- GPU: ibverbs-specific imports and functions ----

#[cfg(not(feature = "hixl"))]
use monarch_rdma::rdma_supported;

#[cfg(not(feature = "hixl"))]
unsafe extern "C" fn pytorch_segment_scanner(
    segments_out: *mut monarch_rdma::rdmaxcel_sys::rdmaxcel_scanned_segment_t,
    max_segments: usize,
) -> usize {
    let result = Python::attach(|py| -> PyResult<usize> {
        let sys = py.import("sys")?;
        let modules = sys.getattr("modules")?;

        let torch = match modules.get_item("torch") {
            Ok(torch_module) => torch_module,
            Err(_) => {
                return Ok(0);
            }
        };

        let cuda_available: bool = torch
            .getattr("cuda")?
            .getattr("is_available")?
            .call0()?
            .extract()?;

        if !cuda_available {
            return Ok(0);
        }

        let snapshot = torch
            .getattr("cuda")?
            .getattr("memory")?
            .getattr("_snapshot")?
            .call0()?;

        let segments = snapshot.get_item("segments")?;
        let segments_list: Vec<Bound<'_, PyAny>> = segments.extract()?;

        let num_segments = segments_list.len();
        let segments_to_write = num_segments.min(max_segments);

        for (i, segment) in segments_list.iter().take(segments_to_write).enumerate() {
            let address: u64 = segment.get_item("address")?.extract()?;
            let total_size: usize = segment.get_item("total_size")?.extract()?;
            let device: i32 = segment.get_item("device")?.extract()?;
            let is_expandable: bool = segment.get_item("is_expandable")?.extract()?;

            let seg_info = &mut *segments_out.add(i);
            seg_info.address = address as usize;
            seg_info.size = total_size;
            seg_info.device = device;
            seg_info.is_expandable = if is_expandable { 1 } else { 0 };
        }

        Ok(num_segments)
    });

    match result {
        Ok(count) => count,
        Err(e) => {
            eprintln!("[monarch_rdma] pytorch_segment_scanner failed: {}", e);
            0
        }
    }
}

// ---- NPU: HIXL-specific functions ----

#[cfg(feature = "hixl")]
fn hixl_rdma_supported() -> bool {
    true
}

// ---- Common code ----

#[pyclass(name = "_LocalMemoryHandle", module = "monarch._rust_bindings.rdma")]
#[derive(Clone)]
pub struct PyLocalMemoryHandle {
    addr: usize,
    size: usize,
    _obj: Py<PyAny>,
}

impl std::fmt::Debug for PyLocalMemoryHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyLocalMemoryHandle")
            .field("addr", &self.addr)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl RdmaLocalMemory for PyLocalMemoryHandle {
    fn addr(&self) -> usize {
        self.addr
    }

    fn size(&self) -> usize {
        self.size
    }
}

#[pymethods]
impl PyLocalMemoryHandle {
    #[new]
    fn new(obj: Py<PyAny>, addr: usize, size: usize) -> Self {
        Self {
            addr,
            size,
            _obj: obj,
        }
    }

    #[getter]
    fn addr(&self) -> usize {
        self.addr
    }

    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    #[pyo3(name = "__repr__")]
    fn repr(&self) -> String {
        format!(
            "<LocalMemoryHandle addr={:#x} size={}>",
            self.addr, self.size
        )
    }
}

#[pyclass(name = "_RdmaBuffer", module = "monarch._rust_bindings.rdma")]
#[derive(Clone, Named)]
struct PyRdmaBuffer {
    buffer: RdmaRemoteBuffer,
}

async fn create_rdma_buffer(
    local: PyLocalMemoryHandle,
    client: PyInstance,
) -> PyResult<PyRdmaBuffer> {
    let owner_handle = RdmaManagerActor::local_handle(client.deref());

    let local: Arc<dyn RdmaLocalMemory> = Arc::new(local);
    let buffer = owner_handle
        .request_buffer(client.deref(), local)
        .await
        .map_err(|e| PyException::new_err(format!("failed to request buffer: {}", e)))?;

    Ok(PyRdmaBuffer { buffer })
}

fn is_rdma_supported() -> bool {
    #[cfg(not(feature = "hixl"))]
    {
        rdma_supported()
    }
    #[cfg(feature = "hixl")]
    {
        hixl_rdma_supported()
    }
}

#[pymethods]
impl PyRdmaBuffer {
    #[classmethod]
    fn create_rdma_buffer_nonblocking<'py>(
        _cls: &Bound<'_, PyType>,
        _py: Python<'py>,
        local: PyLocalMemoryHandle,
        client: PyInstance,
    ) -> PyResult<PyPythonTask> {
        if !is_rdma_supported() {
            return Err(PyException::new_err("RDMA is not supported on this system"));
        }
        PyPythonTask::new(create_rdma_buffer(local, client))
    }

    #[classmethod]
    fn create_rdma_buffer_blocking<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        local: PyLocalMemoryHandle,
        client: PyInstance,
    ) -> PyResult<PyRdmaBuffer> {
        if !is_rdma_supported() {
            return Err(PyException::new_err("RDMA is not supported on this system"));
        }
        signal_safe_block_on(py, create_rdma_buffer(local, client))?
    }

    #[classmethod]
    fn rdma_supported<'py>(_cls: &Bound<'_, PyType>, _py: Python<'py>) -> bool {
        is_rdma_supported()
    }

    #[pyo3(name = "__repr__")]
    fn repr(&self) -> String {
        format!("<RdmaBuffer'{:?}'>", self.buffer)
    }

    fn read_into<'py>(
        &self,
        _py: Python<'py>,
        dst: PyLocalMemoryHandle,
        client: PyInstance,
        timeout: u64,
    ) -> PyResult<PyPythonTask> {
        let buffer = self.buffer.clone();

        PyPythonTask::new(async move {
            let local_memory: Arc<dyn RdmaLocalMemory> = Arc::new(dst);

            buffer
                .read_into_local(client.deref(), local_memory, timeout)
                .await
                .map_err(|e| PyException::new_err(format!("failed to read into buffer: {}", e)))?;

            Ok(())
        })
    }

    fn write_from<'py>(
        &self,
        _py: Python<'py>,
        src: PyLocalMemoryHandle,
        client: PyInstance,
        timeout: u64,
    ) -> PyResult<PyPythonTask> {
        let buffer = self.buffer.clone();

        PyPythonTask::new(async move {
            let local_memory: Arc<dyn RdmaLocalMemory> = Arc::new(src);

            buffer
                .write_from_local(client.deref(), local_memory, timeout)
                .await
                .map_err(|e| PyException::new_err(format!("failed to write from buffer: {}", e)))?;

            Ok(())
        })
    }

    fn size(&self) -> usize {
        self.buffer.size
    }

    /// Return HIXL backend info ``(engine_id, addr)`` if present,
    /// or ``None`` when the buffer uses a native ibverbs backend.
    fn external_backend_info(&self) -> Option<(String, usize)> {
        for ctx in &self.buffer.backends {
            match ctx {
                #[cfg(feature = "hixl")]
                monarch_rdma::backend::RdmaBackendContext::Hixl(buf) => {
                    return Some((buf.engine_id.clone(), buf.addr));
                }
                #[allow(unreachable_patterns)]
                _ => {}
            }
        }
        None
    }

    /// Return the local engine_id assigned by the external transport, or None.
    #[staticmethod]
    fn local_external_engine_id() -> Option<String> {
        std::env::var("MONARCH_PYTHON_HIXL_ENGINE_ID").ok()
            .or_else(|| std::env::var("MONARCH_TRANSPORT_ENGINE_ID").ok())
    }

    /// Return (engine_ptr, engine_id) of the Rust-managed HiXL engine, for diagnostics.
    #[staticmethod]
    fn hixl_engine_diag() -> Option<(usize, String)> {
        #[cfg(feature = "hixl")]
        {
            monarch_rdma::backend::hixl::manager_actor::with_state(|s| {
                Ok((s.engine.ptr() as usize, s.engine_id.clone()))
            }).ok()
        }
        #[cfg(not(feature = "hixl"))]
        { None }
    }

    fn __reduce__(&self) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        monarch_with_gil_blocking(|py| {
            let ctor = py.get_type::<PyRdmaBuffer>().into_py_any(py)?;
            let json = serde_json::to_string(&self.buffer).map_err(|e| {
                PyErr::new::<PyValueError, _>(format!("Serialization failed: {}", e))
            })?;

            let args = PyTuple::new(py, [json])?.into_py_any(py)?;
            Ok((ctor, args))
        })
    }

    #[new]
    fn new_from_json(json: &str) -> PyResult<Self> {
        let buffer: RdmaRemoteBuffer = serde_json::from_str(json)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Deserialization failed: {}", e)))?;
        Ok(PyRdmaBuffer { buffer })
    }

    fn drop<'py>(&self, _py: Python<'py>, client: PyInstance) -> PyResult<PyPythonTask> {
        let buffer = self.buffer.clone();
        PyPythonTask::new(async move {
            buffer
                .drop_buffer(client.deref())
                .await
                .map_err(|e| PyException::new_err(format!("Failed to drop buffer: {}", e)))?;
            Ok(())
        })
    }

    fn owner_actor_id(&self) -> String {
        self.buffer.owner.actor_id().to_string()
    }
}

#[pyclass(name = "_RdmaManager", module = "monarch._rust_bindings.rdma")]
pub struct PyRdmaManager {
    #[allow(dead_code)]
    inner: ActorMesh<RdmaManagerActor>,
    device: String,
}

#[pymethods]
impl PyRdmaManager {
    #[pyo3(name = "__repr__")]
    fn repr(&self) -> String {
        format!("<RdmaManager(device='{}')>", self.device)
    }

    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    #[classmethod]
    fn create_rdma_manager_nonblocking(
        _cls: &Bound<'_, PyType>,
        proc_mesh: &Bound<'_, PyAny>,
        client: PyInstance,
    ) -> PyResult<PyPythonTask> {
        tracing::debug!("spawning RDMA manager on target proc_mesh nodes");

        let proc_mesh = proc_mesh.downcast::<PyProcMesh>()?.borrow().mesh_ref()?;
        PyPythonTask::new(async move {
            let actor_mesh: ActorMesh<RdmaManagerActor> = proc_mesh
                .spawn_service(client.deref(), "rdma_manager", &None)
                .await
                .map_err(|err| PyException::new_err(err.to_string()))?;

            let device_name = if cfg!(feature = "hixl") {
                "hixl_rdma_device"
            } else {
                "remote_rdma_device"
            };

            Ok(Some(PyRdmaManager {
                inner: actor_mesh,
                device: device_name.to_string(),
            }))
        })
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(not(feature = "hixl"))]
    {
        register_segment_scanner(Some(pytorch_segment_scanner));
    }
    #[cfg(feature = "hixl")]
    {
        register_segment_scanner(None);
    }

    module.add_class::<PyLocalMemoryHandle>()?;
    module.add_class::<PyRdmaBuffer>()?;
    module.add_class::<PyRdmaManager>()?;
    Ok(())
}
