/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Ascend NPU (CANN) installation detection and utilities
//!
//! This module provides functionality for detecting Huawei Ascend CANN
//! installations, validating versions, and locating HCCL/ACL headers and
//! libraries.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use which::which;

use crate::BuildError;
use crate::get_env_var_with_rerun;

/// Configuration structure for Ascend CANN environment
#[derive(Debug, Clone)]
pub struct AscendConfig {
    pub ascend_home: PathBuf,
    pub include_dir: PathBuf,
    pub lib_dir: PathBuf,
    pub hccl_include_dir: PathBuf,
    pub hccl_lib_dir: PathBuf,
}

/// Validate Ascend CANN installation exists and return the toolkit root path.
///
/// Detection order:
/// 1. `ASCEND_HOME` environment variable
/// 2. `/usr/local/Ascend/ascend-toolkit/latest`
/// 3. Finding `npu-smi` in PATH and resolving upward
pub fn validate_ascend_installation() -> Result<String, BuildError> {
    // Check multiple environment variable names used by different CANN versions
    for env_name in &["ASCEND_HOME", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"] {
        if let Ok(val) = get_env_var_with_rerun(env_name) {
            let p = Path::new(&val);
            // The env var may point to the toolkit root; try arch subdirectory first
            let arch_dir = find_arch_dir(p);
            if let Some(dir) = arch_dir {
                return Ok(dir.to_string_lossy().to_string());
            }
            if p.join("include").exists() || p.join("lib64").exists() {
                return Ok(val);
            }
        }
    }

    // Walk the standard toolkit layout:
    //   /usr/local/Ascend/ascend-toolkit/<version>/<arch>-linux/
    let toolkit_base = "/usr/local/Ascend/ascend-toolkit";
    if Path::new(toolkit_base).exists() {
        // Prefer the "latest" symlink
        let latest = Path::new(toolkit_base).join("latest");
        if latest.exists() {
            if let Ok(canonical) = fs::canonicalize(&latest) {
                let arch_dir = find_arch_dir(&canonical);
                if let Some(dir) = arch_dir {
                    return Ok(dir.to_string_lossy().to_string());
                }
            }
        }

        // Fall back to scanning versioned directories
        if let Ok(entries) = fs::read_dir(toolkit_base) {
            let mut versions: Vec<_> = entries
                .filter_map(Result::ok)
                .filter(|e| e.file_name() != "latest" && e.file_name() != "set_env.sh")
                .collect();
            versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

            for entry in versions {
                let arch_dir = find_arch_dir(&entry.path());
                if let Some(dir) = arch_dir {
                    return Ok(dir.to_string_lossy().to_string());
                }
            }
        }
    }

    // Try finding npu-smi in PATH
    if let Ok(npu_smi) = which("npu-smi") {
        if let Ok(real_path) = fs::canonicalize(&npu_smi) {
            // npu-smi is typically at <ascend>/bin/npu-smi
            if let Some(ascend_root) = real_path.parent().and_then(|p| p.parent()) {
                return Ok(ascend_root.to_string_lossy().to_string());
            }
        }
    }

    Err(BuildError::PathNotFound(
        "Ascend CANN installation".to_string(),
    ))
}

/// Locate the architecture-specific subdirectory (e.g. `aarch64-linux`).
fn find_arch_dir(base: &Path) -> Option<PathBuf> {
    // The CANN toolkit puts headers/libs under <version>/<arch>-linux/
    for arch in &["aarch64-linux", "x86_64-linux"] {
        let candidate = base.join(arch);
        if candidate.join("include").exists() {
            return Some(candidate);
        }
    }
    // If `base` itself has include/, treat it as the root
    if base.join("include").exists() {
        return Some(base.to_path_buf());
    }
    None
}

/// Get CANN version from the installation.
///
/// Returns `(major, minor)` version tuple.
pub fn get_cann_version(ascend_home: &str) -> Result<(u32, u32), BuildError> {
    // Try <ascend_home>/../../version.cfg  (arch dir -> version dir)
    let home = Path::new(ascend_home);
    let candidates = [
        home.join("version.cfg"),
        home.parent()
            .map(|p| p.join("version.cfg"))
            .unwrap_or_default(),
        // ascend-toolkit level version.info
        PathBuf::from("/usr/local/Ascend/version.info"),
    ];

    for version_file in &candidates {
        if let Ok(content) = fs::read_to_string(version_file) {
            for line in content.lines() {
                // version.cfg lines look like "CANN_VERSION=8.1.RC1" or plain "8.1.RC1"
                let version_str = line
                    .strip_prefix("CANN_VERSION=")
                    .or_else(|| line.strip_prefix("version="))
                    .unwrap_or(line)
                    .trim();

                let parts: Vec<&str> = version_str.split('.').collect();
                if parts.len() >= 2 {
                    if let (Ok(major), Ok(minor)) =
                        (parts[0].parse::<u32>(), parts[1].parse::<u32>())
                    {
                        return Ok((major, minor));
                    }
                }
            }
        }
    }

    Err(BuildError::PathNotFound(
        "CANN version file".to_string(),
    ))
}

/// Discover full Ascend CANN configuration including HCCL paths.
pub fn discover_ascend_config() -> Result<AscendConfig, BuildError> {
    let ascend_home = PathBuf::from(validate_ascend_installation()?);

    let include_dir = ascend_home.join("include");
    if !include_dir.exists() {
        return Err(BuildError::PathNotFound(format!(
            "Ascend include directory at {}",
            include_dir.display()
        )));
    }

    let lib_dir = ascend_home.join("lib64");
    if !lib_dir.exists() {
        return Err(BuildError::PathNotFound(format!(
            "Ascend lib64 directory at {}",
            lib_dir.display()
        )));
    }

    let hccl_include_dir = include_dir.join("hccl");
    if !hccl_include_dir.exists() {
        return Err(BuildError::PathNotFound(format!(
            "HCCL include directory at {}",
            hccl_include_dir.display()
        )));
    }

    // HCCL libs may be at lib64/ directly or at a sibling hccl/lib64/
    let hccl_lib_dir = if lib_dir.join("libhccl.so").exists() {
        lib_dir.clone()
    } else {
        // Try sibling path: ../../hccl/lib64/
        let sibling = ascend_home
            .parent()
            .map(|p| p.join("hccl/lib64"))
            .unwrap_or_default();
        if sibling.join("libhccl.so").exists() {
            sibling
        } else {
            lib_dir.clone()
        }
    };

    Ok(AscendConfig {
        ascend_home,
        include_dir,
        lib_dir,
        hccl_include_dir,
        hccl_lib_dir,
    })
}

/// Get Ascend ACL include directory.
pub fn get_ascend_include_dir() -> Result<String, BuildError> {
    let config = discover_ascend_config()?;
    Ok(config.include_dir.to_string_lossy().to_string())
}

/// Get Ascend library directory (lib64).
pub fn get_ascend_lib_dir() -> Result<String, BuildError> {
    let config = discover_ascend_config()?;
    Ok(config.lib_dir.to_string_lossy().to_string())
}

/// Get HCCL include directory.
pub fn get_hccl_include_dir() -> Result<String, BuildError> {
    let config = discover_ascend_config()?;
    Ok(config.hccl_include_dir.to_string_lossy().to_string())
}

/// Get HCCL library directory.
pub fn get_hccl_lib_dir() -> Result<String, BuildError> {
    let config = discover_ascend_config()?;
    Ok(config.hccl_lib_dir.to_string_lossy().to_string())
}

/// Print helpful error message when Ascend CANN is not found.
pub fn print_ascend_error_help() {
    eprintln!("Error: Ascend CANN installation not found!");
    eprintln!("Please ensure CANN is installed and one of the following is true:");
    eprintln!("  1. Set ASCEND_HOME environment variable to the toolkit arch directory");
    eprintln!("     e.g. /usr/local/Ascend/ascend-toolkit/8.1.RC1/aarch64-linux");
    eprintln!("  2. Install CANN to the default location /usr/local/Ascend/ascend-toolkit/");
    eprintln!("  3. Ensure 'npu-smi' is in your PATH");
}
