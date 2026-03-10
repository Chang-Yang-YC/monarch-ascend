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
    let compiler_include = find_compiler_include(&ascend_config.ascend_home);

    eprintln!("hixl-sys: ascend_home = {:?}", ascend_config.ascend_home);
    eprintln!("hixl-sys: include_dir = {:?}", ascend_config.include_dir);
    let hixl_include = find_hixl_include(&ascend_config.ascend_home);
    let hixl_lib_dir = find_hixl_lib(&ascend_config.ascend_home);

    let mut cc_builder = cc::Build::new();
    cc_builder
        .cpp(true)
        .file("src/bridge.cpp")
        .flag("-std=c++17")
        .flag("-D_GLIBCXX_USE_CXX11_ABI=0")
        .include(&include_dir);

    if let Some(ref compiler_inc) = compiler_include {
        cc_builder.include(compiler_inc);
    }
    if let Some(ref hixl_inc) = hixl_include {
        cc_builder.include(hixl_inc);
        eprintln!("hixl-sys: HIXL headers found at {}", hixl_inc);
    } else {
        eprintln!("hixl-sys: WARNING: HIXL headers not found, using stub bridge");
    }

    cc_builder.compile("hixl_bridge");

    println!("cargo::rustc-link-lib=dl");
    println!("cargo::rustc-link-lib=pthread");
    println!("cargo::rustc-link-lib=stdc++");

    // Link against libcann_hixl.so if available
    if let Some(ref lib_dir) = hixl_lib_dir {
        println!("cargo::rustc-link-search=native={}", lib_dir);
        println!("cargo::rustc-link-lib=dylib=cann_hixl");
        eprintln!("hixl-sys: Linking against libcann_hixl.so from {}", lib_dir);
    }

    // Link against ACL runtime (libascendcl.so) for device context setup
    let acl_lib_dir = ascend_config.lib_dir.to_string_lossy().to_string();
    if std::path::Path::new(&acl_lib_dir).join("libascendcl.so").exists() {
        println!("cargo::rustc-link-search=native={}", acl_lib_dir);
        println!("cargo::rustc-link-lib=dylib=ascendcl");
        eprintln!("hixl-sys: Linking against libascendcl.so from {}", acl_lib_dir);
    }

    // Generate Rust bindings via bindgen
    let mut builder = bindgen::Builder::default()
        .header("src/bridge.h")
        .clang_arg("-x")
        .clang_arg("c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("Hixl.*")
        .allowlist_type("Hixl.*")
        .allowlist_var("HIXL_.*")
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        });

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
        .expect("Unable to generate HIXL bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write HIXL bindings!");

    println!("cargo::rustc-cfg=cargo");
    println!("cargo::rustc-check-cfg=cfg(cargo)");
}

/// Find HIXL include directory containing <hixl/hixl.h>.
fn find_hixl_include(ascend_home: &std::path::Path) -> Option<String> {
    let candidates = [
        ascend_home.join("include"),
        ascend_home.join("aarch64-linux/include"),
        ascend_home.join("x86_64-linux/include"),
        ascend_home.join("../include"),
        ascend_home.join("../aarch64-linux/include"),
        ascend_home.join("../x86_64-linux/include"),
    ];
    for candidate in &candidates {
        if candidate.join("hixl/hixl.h").exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    // Walk upward
    let mut parent = ascend_home.to_path_buf();
    for _ in 0..4 {
        for arch in &["aarch64-linux", "x86_64-linux"] {
            let inc = parent.join(arch).join("include");
            if inc.join("hixl/hixl.h").exists() {
                return Some(inc.to_string_lossy().to_string());
            }
        }
        if let Some(p) = parent.parent() {
            parent = p.to_path_buf();
        } else {
            break;
        }
    }
    None
}

/// Find directory containing libcann_hixl.so.
fn find_hixl_lib(ascend_home: &std::path::Path) -> Option<String> {
    let candidates = [
        ascend_home.join("lib64"),
        ascend_home.join("../lib64"),
        ascend_home.join("aarch64-linux/lib64"),
        ascend_home.join("x86_64-linux/lib64"),
        ascend_home.join("../aarch64-linux/lib64"),
        ascend_home.join("../x86_64-linux/lib64"),
    ];
    for candidate in &candidates {
        if candidate.join("libcann_hixl.so").exists() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    None
}

/// Find CANN compiler include directory (contains external/ge_common/ and graph/ headers).
fn find_compiler_include(ascend_home: &std::path::Path) -> Option<String> {
    let candidates = [
        ascend_home.join("../compiler/include"),
        ascend_home.join("compiler/include"),
    ];

    for candidate in &candidates {
        if let Ok(canonical) = std::fs::canonicalize(candidate) {
            if canonical.join("external/ge_common/ge_api_error_codes.h").exists()
                || canonical.join("graph/ascend_string.h").exists()
            {
                return Some(canonical.to_string_lossy().to_string());
            }
        }
    }

    // Walk upward
    let mut parent = ascend_home.to_path_buf();
    for _ in 0..4 {
        let compiler_inc = parent.join("compiler/include");
        if compiler_inc.join("graph/ascend_string.h").exists() {
            return Some(compiler_inc.to_string_lossy().to_string());
        }
        if let Some(p) = parent.parent() {
            parent = p.to_path_buf();
        } else {
            break;
        }
    }

    None
}
