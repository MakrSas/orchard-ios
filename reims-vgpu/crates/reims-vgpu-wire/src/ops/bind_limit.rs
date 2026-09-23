//! How far a plural bind can reach, per resource class.
//!
//! Every plural bind selector — `setTextures:withRange:`,
//! `setBuffers:offsets:withRange:`, `setSamplerStates:withRange:` and their
//! per-stage render siblings — takes an `NSRange`, and Apple's serializer
//! **truncates that range** at the stage's argument-table size before writing
//! the record. The range that arrives is not the range that was asked for.
//!
//! Three things about it, all measured on `AppleParavirtGPUMetal` 64.4.7:
//!
//! * **The three classes do not share a limit.** Asking for 200 from index 0
//!   yields 128 textures, 31 buffers and 16 samplers.
//! * **The bound is on `first + count`, not on `count`.** Twenty samplers from
//!   index 120 is not twenty — `compute_set_textures_over_bind_limit_offset`
//!   asks for 20 textures at 120 and gets **8**, so the record ends exactly at
//!   [`TEXTURE`]. A reader that took this for a cap on how *many* would expect
//!   20 and mis-size the table.
//! * **It is a property of the stage's argument table, not of an encoder.** The
//!   render encoder truncates at the same three sizes as the compute one, which
//!   is why these live here rather than in [`super::compute`].
//!
//! ## What this does *not* license
//!
//! A decoder must still bound a bind table by the **record's own declared
//! length**, never by these. They describe what Apple's serializer emits; they
//! are not a validity check on bytes arriving from a guest, and a decoder that
//! refused a record for declaring more entries than these would lose every bind
//! in it the moment Apple raised a limit or a guest wrote its own stream.
//!
//! `reims-vgpu` shipped exactly that mistake twice, and the shape is worth
//! keeping: it carried **one** `MAX_BIND_ENTRIES` per rail standing in for all
//! three classes — 128 on the compute rail and 32 on the render rail. No single
//! number can be right. 128 is correct for textures and four times too
//! permissive for buffers; 32 is above the buffer limit and a quarter of the
//! texture one, and that is the one that fired: a forty-slot texture bind was
//! refused whole. Both caps are gone and the declared length is the bound.

/// Textures, per stage. `compute_set_textures_over_bind_limit` and
/// `render_set_vertex_textures_over_bind_limit` both ask for 200 and read 128.
pub const TEXTURE: u32 = 128;

/// Buffers, per stage. `compute_set_buffers_over_bind_limit` and
/// `render_set_vertex_buffers_over_bind_limit` both ask for 200 and read 31.
///
/// Note it is 31 rather than 32 — a power of two minus one, not a round table
/// size — so a bound written from the shape of the number rather than from the
/// capture would be off by one in the direction that drops a bind.
pub const BUFFER: u32 = 31;

/// Samplers, per stage. `compute_set_samplers_over_bind_limit` asks for 200 and
/// reads 16.
pub const SAMPLER: u32 = 16;
