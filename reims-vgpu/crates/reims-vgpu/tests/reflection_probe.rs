//! Off-VM reflection probe for metal2vulkan failure artifacts: prints the m2v
//! reflection for a dumped AIR blob so declared bindings can be compared
//! against the emitted SPIR-V (`spirv-dis`) and a live draw's binds.
//! No-op (skips green) when `PROBE_AIR` is unset, so it is inert in suite runs.
//! Usage: PROBE_AIR=/path/to/x.air [PROBE_STAGE=vertex] \
//!   cargo test --test reflection_probe -- --test-threads=1 --nocapture

#[test]
fn print_reflection() {
    let Some(path) = std::env::var_os("PROBE_AIR") else {
        eprintln!("PROBE_AIR unset; skipping");
        return;
    };
    let air = std::fs::read(&path).expect("read PROBE_AIR");
    let stage = match std::env::var("PROBE_STAGE").as_deref() {
        Ok("vertex") => metal2vulkan::passes::Stage::Vertex,
        _ => metal2vulkan::passes::Stage::Fragment,
    };
    let shader = reims_vgpu::runtime::m2v_cache::translate_cached_reflected(&air, stage, 9999)
        .expect("translate");
    println!(
        "stage={:?} entry bindings={}",
        stage,
        shader.reflection.bindings.len()
    );
    for b in &shader.reflection.bindings {
        println!("  {b:?}");
    }
    println!(
        "spirv_len={} words={}",
        shader.spirv.len(),
        shader.words.len()
    );
}
