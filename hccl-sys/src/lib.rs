/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Low-level Rust FFI bindings for Huawei HCCL (collective communication)
//! and a subset of the ACL runtime (streams, events, device management).
//!
//! HCCL is loaded at runtime via `dlopen`, mirroring the `nccl-sys` pattern.

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
mod inner {
    use serde::Deserialize;
    use serde::Deserializer;
    use serde::Serialize;
    use serde::Serializer;
    use serde::ser::SerializeSeq;

    #[cfg(cargo)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    /// HCCL root info used to bootstrap communicator creation.
    /// 4108 bytes, matching the C `HcclRootInfo` struct.
    #[repr(C)]
    #[derive(Debug, Copy, Clone, Serialize, Deserialize)]
    pub struct HcclRootInfo {
        #[serde(
            serialize_with = "serialize_root_info",
            deserialize_with = "deserialize_root_info"
        )]
        pub internal: [::std::os::raw::c_char; 4108usize],
    }

    fn deserialize_root_info<'de, D>(
        deserializer: D,
    ) -> Result<[::std::os::raw::c_char; 4108], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<::std::os::raw::c_char> = Deserialize::deserialize(deserializer)?;
        vec.try_into().map_err(|v: Vec<::std::os::raw::c_char>| {
            serde::de::Error::invalid_length(v.len(), &"expected an array of length 4108")
        })
    }

    fn serialize_root_info<S>(
        array: &[::std::os::raw::c_char; 4108],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(4108))?;
        for element in array {
            seq.serialize_element(element)?;
        }
        seq.end()
    }
}

pub use inner::*;

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use super::*;

    #[test]
    fn sanity() {
        unsafe {
            let mut root_info = MaybeUninit::<HcclRootInfo>::uninit();
            let result = HcclGetRootInfo(root_info.as_mut_ptr());
            assert_eq!(result.0, 0, "HcclGetRootInfo failed");
        }
    }
}
