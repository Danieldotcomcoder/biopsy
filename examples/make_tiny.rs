//! Run `cargo run --example make_tiny` to create demo models without downloading any.
//! The bytes come from `biopsy::fixtures`, which the tests and `cargo verify` share.
use std::{fs, io};

fn main() -> io::Result<()> {
    fs::create_dir_all("models")?;
    for (name, bytes) in biopsy::fixtures::demo_files() {
        let path = format!("models/{name}");
        fs::write(&path, &bytes)?;
        println!("Created {path} ({} bytes)", bytes.len());
    }
    Ok(())
}
