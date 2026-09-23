//! Sampler state for passes the **engine** originates.
//!
//! # Why nothing here translates a guest sampler
//!
//! It used to. Four ordinal decodes and five Vulkan spellings lived here, and
//! the rule "an unnormalized sampler may not mip, compare, or repeat" lived in
//! a sixth place beside the cache that built one. That is six tables for one
//! `MTLSamplerDescriptor`, and the only thing they could do to each other was
//! disagree about which ordinal means what.
//!
//! A guest sampler is now parsed once by `reims_vgpu_core::sampler`, which is
//! the layer allowed to say what a wire tag means, and planned once by
//! [`reims_vgpu_vulkan::sampler`], which is the layer allowed to turn a host
//! capability into a decision. Neither is reachable from here, and that is the
//! point.
//!
//! What is left is not a translation of anything the guest asked for.

use ash::vk;

/// Filter for the engine's own scaling blits on the present path.
///
/// LINEAR because a present that scales is resampling a finished image for
/// display, where point sampling shows as visible stair-stepping. A
/// non-scaling present hits the 1:1 path and never consults this.
///
/// A constant rather than three literals: it was spelled identically at three
/// separate blit sites, which is three chances for them to stop agreeing about
/// how a scaled present is filtered.
pub const PRESENT_BLIT_FILTER: vk::Filter = vk::Filter::LINEAR;
