//! Framing and wire decoders (batch B).

pub mod blit_spi;
pub mod compute_spi;
/// Cross-checks between the closure ledger and these decoders.
#[cfg(test)]
mod ledger;
pub mod render_spi;
pub mod resource;
