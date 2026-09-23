//! The transfer payloads, and the memory each one declares it touches.
//!
//! # The first class where a payload produces participation
//!
//! [`crate::operation`] classifies; [`crate::stream`] orders. This is the first
//! module where a resolved operation says *which bytes*. A transfer is the
//! right place to start because its participation is entirely in its own
//! record: two refs and a geometry, with no binding table, no pipeline and no
//! descriptor set in between. Whatever [`Participation`] is derived here is
//! derived from fields the guest wrote, which is the standard the render and
//! compute classes then have to meet rather than discover.
//!
//! # Guest values are carried at guest width
//!
//! The wire's copy origins and sizes are `u64` and stay `u64`. Narrowing them
//! at parse would be a decision — "no copy is larger than 4 GiB" — that nothing
//! establishes, and a narrowing that silently truncates a size is a copy of the
//! wrong length rather than a refused one. [`reims_vgpu_protocol::extent::Extent3`]
//! is deliberately not reused: it is the dispatch grid's `u32` triple, a
//! different domain at a different width, and sharing a type between them would
//! make a grid and a copy size interchangeable.
//!
//! # The byte span a texture copy reads out of a buffer is derived, not assumed
//!
//! `copyFromBuffer:` names an offset and two pitches, and the length it reads
//! follows from them: `bytes_per_image * depth` when the image pitch is
//! declared, `bytes_per_row * height` when it is not — a 2D copy may legally
//! leave `bytes_per_image` zero. When neither determines a length the extent
//! becomes [`ParticipationExtent::Whole`], which orders against the whole
//! buffer instead of inventing a span. That is coarser and it is honest; the
//! census that prices precision rungs is what makes the coarseness visible
//! rather than permanent.

use crate::access::{
    AccessMode, ByteRange, Participation, ParticipationExtent, Participations, SubresourceRange,
};
use crate::identity::ResourceId;
pub use reims_vgpu_protocol::blit::BlitKind;

/// A copy origin, in the guest's own units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Origin3 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

/// A copy size, in the guest's own units.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Size3 {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// The two pitches a linear image layout in a buffer is described by.
///
/// `bytes_per_image` of zero is legal and means "one image", which is why this
/// is two independent fields rather than a derived stride: a caller that
/// multiplied through would read a zero-length span for every 2D copy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImagePitch {
    pub bytes_per_row: u64,
    pub bytes_per_image: u64,
}

impl ImagePitch {
    /// The byte length a copy of `size` occupies under these pitches.
    ///
    /// `None` when the pitches determine nothing — both zero, or the one that
    /// applies is zero. The caller widens to the whole buffer rather than
    /// guessing a packed layout, because a packed guess is a shorter span than
    /// the copy really touches and a short span is a missed hazard edge.
    #[must_use]
    pub const fn span_bytes(self, size: Size3) -> Option<u64> {
        if self.bytes_per_image != 0 {
            let depth = if size.depth == 0 { 1 } else { size.depth };
            return self.bytes_per_image.checked_mul(depth);
        }
        if self.bytes_per_row != 0 {
            let height = if size.height == 0 { 1 } else { size.height };
            return self.bytes_per_row.checked_mul(height);
        }
        None
    }
}

/// A texture endpoint: which subresource, and where in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexturePoint {
    pub texture: ResourceId,
    pub slice: u16,
    pub level: u16,
    pub origin: Origin3,
}

impl TexturePoint {
    /// The one level and one slice this endpoint names.
    ///
    /// See [`SubresourceRange::one`] for why the plane is zero.
    #[must_use]
    pub const fn subresource(self) -> SubresourceRange {
        SubresourceRange::one(self.slice as u32, self.level as u32)
    }
}

/// A whole-subresource texture endpoint, which names a span of both axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureSpan {
    pub texture: ResourceId,
    pub base_slice: u16,
    pub base_level: u16,
    pub slice_count: u16,
    pub level_count: u16,
}

/// Where one end of a subresource-span copy starts.
///
/// The counts are deliberately not here. The record carries **one** slice count
/// and **one** level count and both ends copy the same number of subresources
/// --- `vkCmdCopyImage` requires it of the two `layerCount`s, and a per-level
/// region built from the source's count while indexed into the destination's
/// base assumes it besides. Two counts in one operation would be two spellings
/// of one wire field, and the arithmetic that reads only one of them would be
/// right by luck. So [`BlitOp::TextureSlices`] holds them once and
/// [`Self::span`] projects an end into the [`TextureSpan`] a subresource range
/// is asked of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpanOrigin {
    pub texture: ResourceId,
    pub base_slice: u16,
    pub base_level: u16,
}

impl SpanOrigin {
    /// This end, over the operation's counts.
    #[must_use]
    pub const fn span(self, slice_count: u16, level_count: u16) -> TextureSpan {
        TextureSpan {
            texture: self.texture,
            base_slice: self.base_slice,
            base_level: self.base_level,
            slice_count,
            level_count,
        }
    }
}

impl TextureSpan {
    #[must_use]
    pub const fn subresource(self) -> SubresourceRange {
        SubresourceRange {
            base_level: self.base_level as u32,
            level_count: self.level_count as u32,
            base_slice: self.base_slice as u32,
            slice_count: self.slice_count as u32,
            plane: 0,
        }
    }
}

/// The blit options word, carried verbatim.
///
/// Not interpreted here. Its meanings — depth/stencil aspect selection, the
/// PVRTC row-linear form — are format and host questions, and an executor that
/// cannot honour one refuses with the value in hand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlitOptions(pub u32);

/// What a fill writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillPattern {
    /// `fillBuffer:range:value:` — one byte, repeated.
    Byte(u8),
    /// `fillBuffer:range:pattern4:` — a 32-bit pattern, repeated.
    Pattern4(u32),
}

/// A byte window of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferSpan {
    pub buffer: ResourceId,
    pub offset: u64,
    pub length: u64,
}

impl BufferSpan {
    #[must_use]
    pub const fn range(self) -> ByteRange {
        ByteRange {
            offset: self.offset,
            length: self.length,
        }
    }
}

/// One resolved transfer.
///
/// Every variant carries resolved [`ResourceId`]s rather than guest refs: a ref
/// is resolved once, at ingress, and an operation that still held one would be
/// an operation that could be resolved a second time — with a different answer,
/// if a slot were reused in between.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitOp {
    /// Buffer to buffer, one size for both ends.
    BufferToBuffer {
        source: ResourceId,
        source_offset: u64,
        dest: ResourceId,
        dest_offset: u64,
        size: u64,
    },
    /// A linear image in a buffer, into a texture subresource.
    BufferToTexture {
        source: ResourceId,
        source_offset: u64,
        source_pitch: ImagePitch,
        size: Size3,
        dest: TexturePoint,
        options: BlitOptions,
    },
    /// A texture subresource, out into a linear image in a buffer.
    TextureToBuffer {
        source: TexturePoint,
        size: Size3,
        dest: ResourceId,
        dest_offset: u64,
        dest_pitch: ImagePitch,
        options: BlitOptions,
    },
    /// A region of one texture into a region of another.
    TextureRegion {
        source: TexturePoint,
        dest: TexturePoint,
        size: Size3,
        options: BlitOptions,
    },
    /// Whole subresources of one texture into another, over a slice and level
    /// span.
    ///
    /// One count of each for both ends; see [`SpanOrigin`].
    TextureSlices {
        source: SpanOrigin,
        dest: SpanOrigin,
        slice_count: u16,
        level_count: u16,
    },
    /// A repeated pattern over a byte range.
    FillBuffer {
        dest: BufferSpan,
        pattern: FillPattern,
    },
    /// Every level below the top of a texture, rebuilt from the one above it.
    GenerateMipmaps { texture: ResourceId },
}

/// A transfer declares no shader stage; see [`Participation::api_stages`].
const NO_STAGES: u32 = 0;

const fn part(
    resource: ResourceId,
    extent: ParticipationExtent,
    mode: AccessMode,
) -> Participation {
    Participation {
        resource,
        extent,
        mode,
        api_stages: NO_STAGES,
    }
}

const fn buffer_extent(offset: u64, length: Option<u64>) -> ParticipationExtent {
    match length {
        Some(length) => ParticipationExtent::Range(ByteRange { offset, length }),
        None => ParticipationExtent::Whole,
    }
}

impl BlitOp {
    /// Which shape of transfer this is.
    ///
    /// Not "which opcode arrived": the two region-copy opcodes differ only in
    /// whether the record carried an options word, and a `0x130` whose options
    /// are zero executes exactly as a `0x12f` does. The payload keeps the value
    /// and not the opcode, so this reports the options form when there are
    /// options and the plain form when there are none. A census that needs to
    /// count records as the guest sent them counts opcodes at decode, where the
    /// distinction still exists.
    #[must_use]
    pub const fn kind(&self) -> BlitKind {
        match self {
            Self::BufferToBuffer { .. } => BlitKind::BufferToBuffer,
            Self::BufferToTexture { .. } => BlitKind::BufferToTexture,
            Self::TextureToBuffer { .. } => BlitKind::TextureToBuffer,
            Self::TextureRegion { options, .. } => {
                if options.0 == 0 {
                    BlitKind::TextureRegion
                } else {
                    BlitKind::TextureRegionOptions
                }
            }
            Self::TextureSlices { .. } => BlitKind::TextureSlices,
            Self::FillBuffer {
                pattern: FillPattern::Byte(_),
                ..
            } => BlitKind::FillBuffer,
            Self::FillBuffer {
                pattern: FillPattern::Pattern4(_),
                ..
            } => BlitKind::FillBufferPattern4,
            Self::GenerateMipmaps { .. } => BlitKind::GenerateMipmaps,
        }
    }

    /// The memory this transfer declares it touches.
    ///
    /// Derived only from the record's own fields. Where a field does not
    /// determine a span the answer widens to the whole resource rather than
    /// narrowing on an assumption — a span shorter than the copy is a hazard
    /// edge that does not get built, which is a race rather than a slowdown.
    #[must_use]
    pub fn participations(&self) -> Participations {
        match *self {
            Self::BufferToBuffer {
                source,
                source_offset,
                dest,
                dest_offset,
                size,
            } => Participations::two(
                part(
                    source,
                    ParticipationExtent::Range(ByteRange {
                        offset: source_offset,
                        length: size,
                    }),
                    AccessMode::Read,
                ),
                part(
                    dest,
                    ParticipationExtent::Range(ByteRange {
                        offset: dest_offset,
                        length: size,
                    }),
                    AccessMode::Write,
                ),
            ),
            Self::BufferToTexture {
                source,
                source_offset,
                source_pitch,
                size,
                dest,
                ..
            } => Participations::two(
                part(
                    source,
                    buffer_extent(source_offset, source_pitch.span_bytes(size)),
                    AccessMode::Read,
                ),
                part(
                    dest.texture,
                    ParticipationExtent::Subresource(dest.subresource()),
                    AccessMode::Write,
                ),
            ),
            Self::TextureToBuffer {
                source,
                size,
                dest,
                dest_offset,
                dest_pitch,
                ..
            } => Participations::two(
                part(
                    source.texture,
                    ParticipationExtent::Subresource(source.subresource()),
                    AccessMode::Read,
                ),
                part(
                    dest,
                    buffer_extent(dest_offset, dest_pitch.span_bytes(size)),
                    AccessMode::Write,
                ),
            ),
            Self::TextureRegion { source, dest, .. } => Participations::two(
                part(
                    source.texture,
                    ParticipationExtent::Subresource(source.subresource()),
                    AccessMode::Read,
                ),
                part(
                    dest.texture,
                    ParticipationExtent::Subresource(dest.subresource()),
                    AccessMode::Write,
                ),
            ),
            Self::TextureSlices {
                source,
                dest,
                slice_count,
                level_count,
            } => Participations::two(
                part(
                    source.texture,
                    ParticipationExtent::Subresource(
                        source.span(slice_count, level_count).subresource(),
                    ),
                    AccessMode::Read,
                ),
                part(
                    dest.texture,
                    ParticipationExtent::Subresource(
                        dest.span(slice_count, level_count).subresource(),
                    ),
                    AccessMode::Write,
                ),
            ),
            Self::FillBuffer { dest, .. } => Participations::one(part(
                dest.buffer,
                ParticipationExtent::Range(dest.range()),
                AccessMode::Write,
            )),
            // The record names the texture and nothing else. Its levels are the
            // texture's own mip count, which lives on the resource and not in
            // this record, and the operation both reads level 0 and writes the
            // rest — so it is one read-write participation over the whole
            // texture, and narrowing it would need a lookup this type does not
            // do.
            Self::GenerateMipmaps { texture } => Participations::one(part(
                texture,
                ParticipationExtent::Whole,
                AccessMode::ReadWrite,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    /// The claim [`SpanOrigin`] exists for, at the boundary that owns it: one
    /// span copies the same number of subresources at both ends.
    ///
    /// `vkCmdCopyImage` requires the two `layerCount`s to be equal, and a
    /// per-level region built from one end's count while indexed into the
    /// other end's base assumes the level count too. The wire record carries
    /// one of each, so the operation holds one of each and the two extents are
    /// projections of it rather than two fields that happen to agree.
    #[test]
    fn a_subresource_span_names_the_same_counts_at_both_ends() {
        let op = BlitOp::TextureSlices {
            source: SpanOrigin {
                texture: res(1),
                base_slice: 2,
                base_level: 1,
            },
            dest: SpanOrigin {
                texture: res(2),
                base_slice: 0,
                base_level: 3,
            },
            slice_count: 4,
            level_count: 2,
        };
        let parts = op.participations();
        let extents: Vec<_> = parts.iter().map(|p| p.extent).collect();
        assert_eq!(extents.len(), 2);
        let [read, write] = [&extents[0], &extents[1]];
        let (ParticipationExtent::Subresource(read), ParticipationExtent::Subresource(write)) =
            (read, write)
        else {
            panic!("a slice span participates by subresource");
        };
        assert_eq!(read.slice_count, write.slice_count);
        assert_eq!(read.level_count, write.level_count);
        // And they are the operation's, not a default.
        assert_eq!(read.slice_count, 4);
        assert_eq!(read.level_count, 2);
        // The bases are each end's own, which is the half that does differ.
        assert_eq!((read.base_slice, read.base_level), (2, 1));
        assert_eq!((write.base_slice, write.base_level), (0, 3));
    }

    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};
    use crate::operation::{classify, OperationClass, OperationHome};
    use reims_vgpu_protocol::closure::{Rail, LEDGER};

    fn res(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn point(slot: u32, slice: u16, level: u16) -> TexturePoint {
        TexturePoint {
            texture: res(slot),
            slice,
            level,
            origin: Origin3::default(),
        }
    }

    /// The claim the module exists to make, and the half
    /// `reims_vgpu_protocol::blit` cannot: every judged blit-rail operation the
    /// vocabulary calls a transfer has a kind, and every kind is one of those.
    #[test]
    fn the_transfer_vocabulary_is_exactly_the_judged_transfers() {
        let mut from_ledger: Vec<u32> = LEDGER
            .iter()
            .filter(|o| {
                o.rail == Rail::Blit
                    && classify(o) == Some(OperationHome::Stream(OperationClass::Blit))
            })
            .map(|o| {
                o.opcode
                    .expect("a classified blit record carries an opcode")
            })
            .collect();
        from_ledger.sort_unstable();

        let mut from_kinds: Vec<u32> = BlitKind::ALL.iter().map(|k| k.wire_opcode()).collect();
        from_kinds.sort_unstable();

        assert_eq!(
            from_ledger, from_kinds,
            "the ledger and the payload vocabulary disagree about which blit \
             opcodes are transfers"
        );
    }

    /// A copy reads one end and writes the other, and the two ends are separate
    /// participations even when they name the same resource.
    #[test]
    fn a_buffer_copy_declares_a_read_and_a_write_at_exact_ranges() {
        let op = BlitOp::BufferToBuffer {
            source: res(1),
            source_offset: 0x100,
            dest: res(2),
            dest_offset: 0x200,
            size: 0x40,
        };
        let p = op.participations();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].mode, AccessMode::Read);
        assert_eq!(
            p[0].extent,
            ParticipationExtent::Range(ByteRange {
                offset: 0x100,
                length: 0x40
            })
        );
        assert_eq!(p[1].mode, AccessMode::Write);
        assert_eq!(
            p[1].extent,
            ParticipationExtent::Range(ByteRange {
                offset: 0x200,
                length: 0x40
            })
        );
    }

    /// The image pitch decides the buffer span, and a 2D copy that leaves
    /// `bytes_per_image` zero still has one.
    #[test]
    fn a_texture_upload_spans_the_bytes_its_pitches_describe() {
        let three_d = ImagePitch {
            bytes_per_row: 256,
            bytes_per_image: 0x1000,
        };
        let size = Size3 {
            width: 64,
            height: 16,
            depth: 4,
        };
        assert_eq!(three_d.span_bytes(size), Some(0x4000));

        let two_d = ImagePitch {
            bytes_per_row: 256,
            bytes_per_image: 0,
        };
        assert_eq!(
            two_d.span_bytes(Size3 {
                width: 64,
                height: 16,
                depth: 1
            }),
            Some(256 * 16)
        );
    }

    /// A depth of zero is one image, not none. A copy whose span multiplied out
    /// to zero would declare an empty range, and an empty range conflicts with
    /// nothing — so the write would order against nothing at all.
    #[test]
    fn a_zero_depth_or_height_still_spans_one_image() {
        let pitch = ImagePitch {
            bytes_per_row: 128,
            bytes_per_image: 0x800,
        };
        assert_eq!(
            pitch.span_bytes(Size3 {
                width: 4,
                height: 4,
                depth: 0
            }),
            Some(0x800)
        );
        let rows = ImagePitch {
            bytes_per_row: 128,
            bytes_per_image: 0,
        };
        assert_eq!(
            rows.span_bytes(Size3 {
                width: 4,
                height: 0,
                depth: 0
            }),
            Some(128)
        );
    }

    /// Pitches that determine nothing widen to the whole buffer rather than
    /// producing a span the copy would outrun.
    #[test]
    fn undetermined_pitches_widen_instead_of_guessing() {
        let none = ImagePitch::default();
        assert_eq!(
            none.span_bytes(Size3 {
                width: 8,
                height: 8,
                depth: 1
            }),
            None
        );
        let op = BlitOp::BufferToTexture {
            source: res(1),
            source_offset: 0,
            source_pitch: none,
            size: Size3 {
                width: 8,
                height: 8,
                depth: 1,
            },
            dest: point(2, 0, 0),
            options: BlitOptions(0),
        };
        assert_eq!(op.participations()[0].extent, ParticipationExtent::Whole);
    }

    /// An overflowing pitch product is not a span. Wrapping it would name a
    /// short range and lose the edge; widening is the conservative answer.
    #[test]
    fn an_overflowing_span_widens_rather_than_wrapping() {
        let huge = ImagePitch {
            bytes_per_row: 0,
            bytes_per_image: u64::MAX,
        };
        assert_eq!(
            huge.span_bytes(Size3 {
                width: 1,
                height: 1,
                depth: 4
            }),
            None
        );
    }

    /// A texture endpoint names exactly one level and one slice, so two copies
    /// into different levels of one texture do not conflict.
    #[test]
    fn two_copies_into_different_levels_name_different_subresources() {
        let a = point(7, 0, 0).subresource();
        let b = point(7, 0, 1).subresource();
        assert!(!a.overlaps(b));
        assert!(a.overlaps(point(7, 0, 0).subresource()));
    }

    /// Mipmap generation reads and writes, and says so over the whole texture
    /// because the record does not carry a level count.
    #[test]
    fn mipmap_generation_is_one_read_write_over_the_whole_texture() {
        let p = BlitOp::GenerateMipmaps { texture: res(3) }.participations();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].mode, AccessMode::ReadWrite);
        assert_eq!(p[0].extent, ParticipationExtent::Whole);
    }

    /// A fill writes and reads nothing.
    #[test]
    fn a_fill_is_a_write_and_its_two_forms_are_two_kinds() {
        let dest = BufferSpan {
            buffer: res(4),
            offset: 0x10,
            length: 0x20,
        };
        let byte = BlitOp::FillBuffer {
            dest,
            pattern: FillPattern::Byte(0x5a),
        };
        let word = BlitOp::FillBuffer {
            dest,
            pattern: FillPattern::Pattern4(0x89ab_cdef),
        };
        assert_eq!(byte.kind(), BlitKind::FillBuffer);
        assert_eq!(word.kind(), BlitKind::FillBufferPattern4);
        let p = byte.participations();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].mode, AccessMode::Write);
    }

    /// Every variant declares a participation, and none declares none: a
    /// transfer that touched nothing would order against nothing and could be
    /// scheduled anywhere.
    #[test]
    fn every_transfer_declares_something() {
        let ops = [
            BlitOp::BufferToBuffer {
                source: res(1),
                source_offset: 0,
                dest: res(2),
                dest_offset: 0,
                size: 8,
            },
            BlitOp::BufferToTexture {
                source: res(1),
                source_offset: 0,
                source_pitch: ImagePitch {
                    bytes_per_row: 16,
                    bytes_per_image: 0,
                },
                size: Size3 {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                dest: point(2, 0, 0),
                options: BlitOptions(0),
            },
            BlitOp::TextureToBuffer {
                source: point(1, 0, 0),
                size: Size3 {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                dest: res(2),
                dest_offset: 0,
                dest_pitch: ImagePitch {
                    bytes_per_row: 16,
                    bytes_per_image: 0,
                },
                options: BlitOptions(0),
            },
            BlitOp::TextureRegion {
                source: point(1, 0, 0),
                dest: point(2, 0, 0),
                size: Size3 {
                    width: 4,
                    height: 4,
                    depth: 1,
                },
                options: BlitOptions(0),
            },
            BlitOp::TextureSlices {
                source: SpanOrigin {
                    texture: res(1),
                    base_slice: 0,
                    base_level: 0,
                },
                dest: SpanOrigin {
                    texture: res(2),
                    base_slice: 0,
                    base_level: 0,
                },
                slice_count: 1,
                level_count: 1,
            },
            BlitOp::FillBuffer {
                dest: BufferSpan {
                    buffer: res(1),
                    offset: 0,
                    length: 8,
                },
                pattern: FillPattern::Byte(0),
            },
            BlitOp::FillBuffer {
                dest: BufferSpan {
                    buffer: res(1),
                    offset: 0,
                    length: 8,
                },
                pattern: FillPattern::Pattern4(0),
            },
            BlitOp::GenerateMipmaps { texture: res(1) },
        ];
        for op in ops {
            let p = op.participations();
            assert!(!p.is_empty(), "{:?} declares nothing", op.kind());
            assert!(
                p.iter().any(|x| x.mode.writes()),
                "{:?} writes nothing, so it is not a transfer",
                op.kind()
            );
        }
        // Every kind but the options form of the region copy, which is the same
        // payload with a non-zero word.
        let mut kinds: Vec<BlitKind> = ops.iter().map(|o| o.kind()).collect();
        kinds.push(
            BlitOp::TextureRegion {
                source: point(1, 0, 0),
                dest: point(2, 0, 0),
                size: Size3::default(),
                options: BlitOptions(4),
            }
            .kind(),
        );
        kinds.sort_unstable();
        kinds.dedup();
        assert_eq!(kinds.len(), BlitKind::ALL.len());
    }
}
