//! One module per operation family.
//!
//! Add a module here for each opcode as it is derived. [`texture`] is the
//! worked example; the procedure for the next one is in this crate's
//! `AGENTS.md`.

pub mod backed_texture;
pub mod bind_limit;
pub mod blit;
pub mod compute;
pub mod depth_stencil;
pub mod destroy;
pub mod event;
pub mod fence;
pub mod heap_texture;
pub mod icb;
pub mod info;
pub mod rate_map;
pub mod render;
pub mod render_pass;
pub mod sampler;
pub mod segment;
pub mod texture;
pub mod texture_view;
pub mod tile;
