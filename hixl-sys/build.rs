use std::env;
use std::path::PathBuf;

fn main() {
    let search_dirs: Vec<PathBuf> = vec![
        env::var("HIXL_LIB_PATH")
            .map(PathBuf::from)
            .unwrap_or_default(),
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../tests/hixl/build"),
    ];

    for dir in &search_dirs {
        if dir.join("libtest_hixl.so").exists() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=dylib=test_hixl");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
            println!("cargo:rerun-if-changed={}", dir.join("libtest_hixl.so").display());
            return;
        }
    }

    panic!(
        "Cannot find libtest_hixl.so. Searched: {:?}. \
         Set HIXL_LIB_PATH to the directory containing the library.",
        search_dirs
    );
}
