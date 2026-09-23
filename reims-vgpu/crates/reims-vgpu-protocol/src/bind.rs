//! How far a plural bind reaches, and what that number is good for.
//!
//! Apple's serializer truncates every plural bind selector's `NSRange` at the
//! stage's argument-table size before writing the record, and the three
//! resource classes do not share a size. `reims-vgpu-wire` measured them; this
//! is where they are given a meaning, and the meaning is narrower than it looks.
//!
//! **These are capacity hints, not bounds.** They describe what Apple's
//! serializer emits today. A decoder must bound a bind record by the record's
//! own declared length, and a binding table must accept whatever slot the guest
//! names — because a limit raised upstream, or a guest writing its own stream,
//! turns a bound into dropped binds. This project has shipped that mistake
//! twice, with one number standing in for all three classes, and lost a
//! forty-slot texture bind whole.
//!
//! So: reserve this much, refuse nothing at it.

pub use reims_vgpu_wire::ops::bind_limit::{
    BUFFER as BUFFER_TABLE_HINT, SAMPLER as SAMPLER_TABLE_HINT, TEXTURE as TEXTURE_TABLE_HINT,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The three differ, which is the whole reason one number cannot stand in
    /// for them. The buffer size is also not a round table size — a bound
    /// written from the shape of the number rather than from the measurement
    /// would be off by one, in the direction that drops a bind.
    #[test]
    fn the_three_classes_do_not_share_a_size() {
        assert_eq!(TEXTURE_TABLE_HINT, 128);
        assert_eq!(BUFFER_TABLE_HINT, 31);
        assert_eq!(SAMPLER_TABLE_HINT, 16);
        assert_ne!(BUFFER_TABLE_HINT, 32);
    }
}
