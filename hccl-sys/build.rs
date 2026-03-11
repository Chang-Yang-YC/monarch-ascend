/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

#[cfg(target_os = "macos")]
fn main() {}

#[cfg(not(target_os = "macos"))]
fn main() {
    let ascend_config = build_utils::ascend::discover_ascend_config().expect(
        "Ascend CANN installation not found. \
         Set ASCEND_HOME or install CANN to /usr/local/Ascend/ascend-toolkit/",
    );

    let include_dir = ascend_config.include_dir.to_string_lossy().to_string();

    // Compile the bridge.cpp file
    let mut cc_builder = cc::Build::new();
    cc_builder
        .cpp(true)
        .file("src/bridge.cpp")
        .flag("-std=c++14")
        .include(&include_dir);

    cc_builder.compile("hccl_bridge");

    // Generate Rust bindings via bindgen
    let mut builder = bindgen::Builder::default()
        .header("src/bridge.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        .clang_arg(format!("-I{}", include_dir))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // HCCL functions
        .allowlist_function("HcclGetRootInfo")
        .allowlist_function("HcclGetErrorString")
        .allowlist_function("HcclCommInitRootInfo")
        .allowlist_function("HcclCommInitAll")
        .allowlist_function("HcclCommDestroy")
        .allowlist_function("HcclGetCommAsyncError")
        .allowlist_function("HcclCreateSubCommConfig")
        .allowlist_function("HcclAllReduce")
        .allowlist_function("HcclBroadcast")
        .allowlist_function("HcclReduce")
        .allowlist_function("HcclAllGather")
        .allowlist_function("HcclReduceScatter")
        .allowlist_function("HcclAlltoAll")
        .allowlist_function("HcclSend")
        .allowlist_function("HcclRecv")
        .allowlist_function("HcclBatchSendRecv")
        .allowlist_function("HcclBarrier")
        // ACL runtime functions
        .allowlist_function("aclrtSetDevice")
        .allowlist_function("aclrtCreateStream")
        .allowlist_function("aclrtDestroyStream")
        .allowlist_function("aclrtSynchronizeStream")
        .allowlist_function("aclrtCreateEvent")
        .allowlist_function("aclrtDestroyEvent")
        .allowlist_function("aclrtRecordEvent")
        .allowlist_function("aclrtSynchronizeEvent")
        .allowlist_function("aclrtStreamWaitEvent")
        // Types
        .allowlist_type("HcclResult")
        .allowlist_type("HcclReduceOp")
        .allowlist_type("HcclDataType")
        .allowlist_type("HcclSendRecvType")
        .allowlist_type("HcclSendRecvItem.*")
        .allowlist_type("aclError")
        .allowlist_type("aclrtStream")
        .allowlist_type("aclrtEvent")
        // Constants
        .allowlist_var("HCCL_ROOT_INFO_BYTES")
        // Blocklist the root info struct so we define it manually in Rust
        .blocklist_type("HcclRootInfoDef")
        .blocklist_type("HcclRootInfo")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        });

    // Include Python env dirs if available
    let python_config = match build_utils::python_env_dirs() {
        Ok(config) => config,
        Err(_) => {
            eprintln!("Warning: Failed to get Python environment directories");
            build_utils::PythonConfig {
                include_dir: None,
                lib_dir: None,
            }
        }
    };

    if let Some(inc) = &python_config.include_dir {
        builder = builder.clang_arg(format!("-I{}", inc));
    }
    if let Some(lib_dir) = &python_config.lib_dir {
        println!("cargo::rustc-link-search=native={}", lib_dir);
    }

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    builder
        .generate()
        .expect("Unable to generate HCCL bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write HCCL bindings!");

    // We dlopen libhccl.so at runtime, so no link-time dependency on HCCL.
    // Link dl for dlopen/dlsym.
    println!("cargo::rustc-link-lib=dl");
    println!("cargo::rustc-link-lib=pthread");
    println!("cargo::rustc-cfg=cargo");
    println!("cargo::rustc-check-cfg=cfg(cargo)");
}
