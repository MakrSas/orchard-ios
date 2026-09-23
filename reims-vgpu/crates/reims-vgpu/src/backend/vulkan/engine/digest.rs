//! ≥128-bit content digests for cache keys (never bare DefaultHasher u64 alone).
//!
//! # A digest is a bucket, and width is not what makes it safe
//!
//! This doc used to record an asymmetry — that the Metal arm identified shaders
//! by a 64-bit fingerprint and never compared the bytes, while this arm's rule
//! was 128 bits — and to send readers to that struct for "why it was recorded
//! rather than changed". Both halves have since gone the other way, and a reader
//! who follows the old account will go looking for a hazard that is not there.
//!
//! [`crate::backend::blob`] removed the class from the Metal caches by retaining
//! the blob and comparing it, and [`crate::backend::render_pso_key`] carries the
//! same account for pipeline state. `engine::pools`'s sampled cache followed:
//! its 128-bit content fingerprint now picks a candidate and
//! `ResidentSampledSlot::content` decides the hit.
//!
//! So the rule this module's name states is a *bucketing* rule, not an identity
//! rule. Where a digest still stands alone as an identity — `ObjectCaches`
//! files `vk::ShaderModule` under a bare [`Digest128`] of the SPIR-V words and
//! retains none of them, and [`Digest128`] is the shader half of
//! `PipelineKey` and `ComputePipelineKey` — the width is all that is holding it,
//! and that is the shape the two modules above argue against. Widening is not
//! the answer there either; retaining the words is.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Two independently-seeded 64-bit hashes + byte length (≥128-bit effective).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct Digest128 {
    pub a: u64,
    pub b: u64,
    pub len: u64,
}

impl Digest128 {
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut ha = DefaultHasher::new();
        0x9e37_79b9_7f4a_7c15u64.hash(&mut ha);
        bytes.hash(&mut ha);
        let a = ha.finish();

        let mut hb = DefaultHasher::new();
        0xc2b2_ae3d_27d4_eb4fu64.hash(&mut hb);
        // reverse-mix seed so collisions in a alone do not imply collisions in b
        for chunk in bytes.chunks(8).rev() {
            chunk.hash(&mut hb);
        }
        bytes.len().hash(&mut hb);
        let b = hb.finish();

        Self {
            a,
            b,
            len: bytes.len() as u64,
        }
    }

    pub fn of_u32_words(words: &[u32]) -> Self {
        #[cfg(target_endian = "little")]
        {
            // LE hosts: the words already are the byte sequence — hash in place
            // (this runs per draw; the copy was a full-module alloc each call).
            let bytes =
                unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 4) };
            Self::of_bytes(bytes)
        }
        #[cfg(not(target_endian = "little"))]
        {
            let mut bytes = Vec::with_capacity(words.len() * 4);
            for w in words {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            Self::of_bytes(&bytes)
        }
    }

    /// Digest of a slice of hashable items and the context they belong to,
    /// without materializing their bytes.
    ///
    /// For a caller whose content is already typed — a descriptor layout's
    /// binding signatures, say — rather than a byte or word buffer. Building
    /// that buffer to reach [`Self::of_bytes`] would be a heap allocation on a
    /// path whose whole reason for wanting a digest is that it runs per draw.
    ///
    /// **A bucket, not an identity**, and here that is not a caveat: the caller
    /// this exists for retains its items and compares them on a hit, exactly as
    /// this module's header argues a digest user should. `len` is the item
    /// count, not a byte count, so two digests are comparable only within one
    /// item type.
    pub fn of_items<C: Hash, T: Hash>(context: &C, items: &[T]) -> Self {
        let mut ha = DefaultHasher::new();
        0x9e37_79b9_7f4a_7c15u64.hash(&mut ha);
        context.hash(&mut ha);
        items.hash(&mut ha);
        let a = ha.finish();

        let mut hb = DefaultHasher::new();
        0xc2b2_ae3d_27d4_eb4fu64.hash(&mut hb);
        // Reverse order, as `of_bytes` does, so a collision in `a` alone does
        // not imply one in `b`.
        for item in items.iter().rev() {
            item.hash(&mut hb);
        }
        context.hash(&mut hb);
        items.len().hash(&mut hb);
        let b = hb.finish();

        Self {
            a,
            b,
            len: items.len() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_differs_on_content() {
        let a = Digest128::of_bytes(b"hello");
        let b = Digest128::of_bytes(b"world");
        assert_ne!(a, b);
        assert_eq!(a, Digest128::of_bytes(b"hello"));
    }

    #[test]
    fn item_digest_separates_order_content_and_length() {
        let base = [1u32, 2, 3];
        let d = |items: &[u32]| Digest128::of_items(&0u8, items);
        assert_eq!(d(&base), d(&[1, 2, 3]));
        assert_ne!(d(&base), d(&[3, 2, 1]), "order");
        assert_ne!(d(&base), d(&[1, 2, 4]), "content");
        assert_ne!(d(&base), d(&[1, 2, 3, 3]), "length");
        assert_ne!(
            Digest128::of_items(&0u8, &base),
            Digest128::of_items(&1u8, &base),
            "context"
        );
    }

    #[test]
    fn digest_length_distinguishes_prefix() {
        let a = Digest128::of_bytes(b"ab");
        let b = Digest128::of_bytes(b"abc");
        assert_ne!(a, b);
    }
}
