fn main() {
    if cfg!(target_os = "windows") {
        println!("cargo:rerun-if-changed=icon.ico");
        println!("cargo:rerun-if-changed=build.rs");
        let mut res = winres::WindowsResource::new();
        res.set_icon("icon.ico");
        res.compile().unwrap();

        let out_dir = std::env::var("OUT_DIR").unwrap();
        let obj_path = std::path::Path::new(&out_dir).join("resource.o");
        if obj_path.exists() {
            println!("cargo:rustc-link-arg={}", obj_path.display());
        }
    }
}
