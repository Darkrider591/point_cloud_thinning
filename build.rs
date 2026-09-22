fn main() {
    if cfg!(target_os = "windows") {
        println!("cargo:rerun-if-changed=icon.ico");
        let mut res = winres::WindowsResource::new();
        res.set_icon("icon.ico");
        res.compile().unwrap();
    }
}
