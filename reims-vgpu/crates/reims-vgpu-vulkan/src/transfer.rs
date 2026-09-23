//! What a resolved transfer becomes here: the exact native copies, planned
//! against the resources the names resolve to and nothing else.
//!
//! # A pitch is bytes to the guest and texels to Vulkan
//!
//! `MTLBlitCommandEncoder` describes a linear image in a buffer by its
//! `bytesPerRow` and `bytesPerImage`. `VkBufferImageCopy` describes the same
//! thing by `bufferRowLength` and `bufferImageHeight`, which are **texels**.
//! The conversion is the format's, and it is exact or it is a refusal: a
//! `bytesPerRow` that is not a whole number of blocks describes a row this
//! copy cannot express, and rounding it either way copies the wrong bytes into
//! a texture that then samples plausibly wrong.
//!
//! Block-compressed formats make the difference visible — a BC row is
//! `blocks_across(width) * block_bytes`, so the texel row length is the block
//! count times the block width, and a conversion that divided bytes by a
//! bytes-per-pixel that does not exist for these formats would be off by the
//! block size.
//!
//! # A fill is four bytes at a time or it is refused
//!
//! `vkCmdFillBuffer` requires a four-byte-aligned offset and size, and the
//! guest's `fillBuffer:range:` does not. A ragged range therefore cannot
//! become one fill command, and it is refused by name rather than rounded —
//! rounding out writes bytes outside the range the guest named, and rounding
//! in leaves bytes inside it unwritten. Closing it needs a staging copy, which
//! has no alignment rule at all; [`crate::placement`] already routes one, and
//! this refusal is what says the route is needed.
//!
//! A one-byte pattern is replicated into the four, which is what the guest's
//! own `fillBuffer:range:value:` means.
//!
//! # Planned, not recorded
//!
//! Every function here produces commands and issues none, so the pitch
//! arithmetic, the region expansion and the refusals are tested with no GPU.
//! The layout each image has to be in for these copies is
//! [`crate::layout`]'s answer and is deliberately not decided here: a transfer
//! that also chose layouts would be a second place image state is tracked.

use ash::vk;
use reims_vgpu_core::blit::{BlitOp, BlitOptions, ImagePitch, Origin3, Size3, TexturePoint};
use reims_vgpu_core::blit_option::{select_aspect, OptionRefusal};
use reims_vgpu_core::pixel_format::{blit_aspect_bytes_per_pixel, block_geometry, BlitAspect};
use reims_vgpu_core::texture_shape::Texture;

use crate::buffer::BufferPlan;
use crate::resident::{Miss, Residency};
use crate::view::aspect;

/// Why a resolved transfer cannot be recorded on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A name in the transfer did not resolve.
    Unresolved { miss: Miss },
    /// `bytesPerRow` is not a whole number of the format's blocks, so the row
    /// cannot be expressed in texels.
    RowPitchNotWholeBlocks {
        bytes_per_row: u64,
        block_bytes: u32,
    },
    /// `bytesPerImage` is not a whole number of rows.
    ImagePitchNotWholeRows {
        bytes_per_image: u64,
        bytes_per_row: u64,
    },
    /// The guest's format has no block geometry this build knows, so no pitch
    /// conversion exists for it.
    UnknownFormatGeometry { format: u16 },
    /// A fill whose offset is not four-byte aligned, and whose pattern is four
    /// bytes wide.
    ///
    /// The byte-wide pattern has a ragged form — see [`plan_fill`] — because
    /// every byte of the range takes the same value and the phase of the
    /// pattern therefore does not exist. A four-byte pattern has a phase, and
    /// which byte of it lands on a range that does not *start* at a multiple of
    /// four is not a term this wire has established. Refused rather than
    /// guessed: a wrong phase writes the right number of bytes in the wrong
    /// order, everywhere, and nothing downstream can see it.
    ///
    /// A range that starts at a multiple of four and merely ends raggedly is
    /// not this refusal and never was: every byte's phase follows from the
    /// start, so the tail's bytes are the pattern's leading ones and nothing is
    /// being guessed at.
    RaggedPatternFill { offset: u64, length: u64 },
    /// The scratch memory a ragged fill needs was not available.
    NoStaging { refusal: crate::staging::Refusal },
    /// A dimension larger than the 32-bit fields a native copy carries.
    ///
    /// The guest's own values are 64-bit; this is where that stops being true,
    /// and it is checked rather than truncated because a truncated extent is a
    /// copy that succeeds and moves the wrong bytes.
    ExtentTooLarge { axis: &'static str, value: u64 },
    /// The row pitch names fewer texels than the copy is wide.
    ///
    /// `bufferRowLength` must be zero or at least `imageExtent.width`, and a
    /// smaller one is invalid usage: every row after the first would be read
    /// from, or written to, an offset that overlaps the row before it. Refused
    /// rather than widened to the extent, which would silently disagree with
    /// the stride the guest said its bytes are at.
    RowPitchShorterThanCopy { row_length: u32, width: u64 },
    /// The image pitch names fewer rows than the copy is tall, for
    /// `bufferImageHeight`'s half of the same rule.
    ImagePitchShorterThanCopy { image_height: u32, height: u64 },
    /// A slice stride with no row stride, on a copy of more than one slice.
    ///
    /// `bufferImageHeight` is texels and the only conversion from a byte
    /// slice stride runs through the byte row stride, so with no row pitch
    /// there is no conversion at all. Coercing it to "tightly packed" would
    /// drop a stride the guest named and land every slice after the first at
    /// the wrong offset --- and it is the one shape where the drop is
    /// guest-visible, because a single-slice copy never addresses a second
    /// slice. Refused rather than guessed at the packed row size, which is a
    /// term this wire has not established.
    ImagePitchWithoutRowPitch { bytes_per_image: u64, depth: u64 },
    /// A texture endpoint names a subresource, or a region of one, that the
    /// texture it resolved to does not have.
    ///
    /// A level or slice past the declared count, or an origin plus extent past
    /// the level's own size, is invalid usage on every axis it happens on --- a
    /// copy the driver may execute against memory belonging to another
    /// resource. It is checked against the *resolved* texture and not against
    /// the declaration the guest is working from, because the two can differ:
    /// a name reused across a delete resolves to a texture of a different
    /// shape, and the shape that matters is the one the copy will run on.
    ///
    /// One variant with an axis rather than five, because the report is the
    /// same sentence each time and the axis is the only thing that varies.
    OutsideTexture {
        axis: &'static str,
        named: u64,
        available: u64,
    },
    /// The buffer side of a transfer runs past the end of the buffer it names.
    ///
    /// The last byte a copy or a fill addresses, against the length the buffer
    /// was created with. Past it the driver writes into --- or reads from ---
    /// whatever the allocator put next, which for a suballocated buffer is
    /// another guest resource. Nothing downstream can see that happen.
    ///
    /// `length` for a buffer-image copy is the region's *addressed footprint*
    /// and not its texel count: a pitch wider than the copy makes the two
    /// differ, and it is the footprint that reaches memory.
    OutsideBuffer { offset: u64, length: u64, size: u64 },
    /// A copy whose source and destination are the same buffer, over windows
    /// that overlap.
    ///
    /// `vkCmdCopyBuffer` requires the regions not to overlap when `srcBuffer`
    /// and `dstBuffer` are the same (VUID-vkCmdCopyBuffer-pRegions-00117), and
    /// the guest API states the same restriction on its own caller. A copy that
    /// breaks it has no defined result: a driver copying forward and one
    /// copying backward produce different bytes from the same command, and
    /// which one ran is not observable from here.
    ///
    /// Compared on the resolved native buffer and not on the two guest names,
    /// because one buffer may answer to several: two names aliasing one
    /// allocation overlap exactly as much as one name used twice, and a check
    /// on the names alone would miss the case that is harder to notice.
    ///
    /// Refused rather than staged. A staging round-trip would define it, and
    /// it needs scratch this signature has no access to --- the same reason
    /// [`plan_fill`] is a separate door.
    ///
    /// The *exact* self-copy is not this, for the reason
    /// [`Self::OverlappingSelfImageCopy`] gives about images: a window copied
    /// onto itself leaves the buffer holding what it already held, so it is
    /// answered as nothing to record --- see [`plan`]. The undefined result
    /// above is a statement about the *shifted* case, where forward and
    /// backward orders disagree; at zero shift they agree, and there is
    /// nothing for a driver to get wrong.
    OverlappingSelfCopy {
        source_offset: u64,
        dest_offset: u64,
        size: u64,
    },
    /// A texture copy whose source and destination are the same image, over
    /// regions of one subresource that overlap without being the same region.
    ///
    /// `vkCmdCopyImage`'s source and destination regions must not overlap in
    /// memory (VUID-vkCmdCopyImage-pRegions-00124), which is the same rule
    /// [`Self::OverlappingSelfCopy`] states for buffers. Different subresources
    /// of one image are different memory, so this is about one level and one
    /// slice; the axis is what says which one.
    ///
    /// The *exact* self-copy is not this. A region copied onto itself leaves
    /// the destination holding what it already held, so it is answered as
    /// nothing to record --- see [`plan`]. This is the shifted case, which
    /// changes bytes it is also reading and needs a staging round-trip to
    /// define.
    OverlappingSelfImageCopy { level: u32, slice: u32 },
    /// The guest's `MTLBlitOption` word names no plane this device can copy.
    ///
    /// Forwarded from the layer that owns the tag, so the reason the guest
    /// gets is the reason that layer gave and not this rail's paraphrase.
    BlitOption { refusal: OptionRefusal },
    /// The option word selected a plane the texture's format does not have —
    /// the stencil of a depth-only texture, or either plane of a colour one.
    AspectNotInFormat { format: u16, aspect: &'static str },
    /// A buffer↔image copy of a combined depth-stencil texture that named no
    /// plane.
    ///
    /// `VkImageSubresourceLayers::aspectMask` in such a copy must have exactly
    /// one bit — VUID-VkBufferImageCopy-aspectMask-00212 — and no other field
    /// of the record says which of the two the guest's bytes are. Deriving the
    /// mask from the format alone produced a two-bit one and recorded an
    /// illegal copy; guessing a plane would move the right number of bytes
    /// from the wrong half of the texture.
    AmbiguousAspect { format: u16 },
    /// A blit-encoder operation that is not a copy, and so has no native
    /// transfer form at all.
    ///
    /// Named rather than absent: a caller reaching this has routed the
    /// operation to the wrong planner, and the refusal says which one it is.
    NotACopy { op: &'static str },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Unresolved { .. } => "vk_transfer_unresolved",
            Self::RowPitchNotWholeBlocks { .. } => "vk_transfer_row_pitch_not_whole_blocks",
            Self::ImagePitchNotWholeRows { .. } => "vk_transfer_image_pitch_not_whole_rows",
            Self::UnknownFormatGeometry { .. } => "vk_transfer_unknown_format_geometry",
            Self::RaggedPatternFill { .. } => "vk_transfer_ragged_pattern_fill",
            Self::NoStaging { .. } => "vk_transfer_no_staging",
            Self::ExtentTooLarge { .. } => "vk_transfer_extent_too_large",
            Self::RowPitchShorterThanCopy { .. } => "vk_transfer_row_pitch_shorter_than_copy",
            Self::ImagePitchShorterThanCopy { .. } => "vk_transfer_image_pitch_shorter_than_copy",
            Self::ImagePitchWithoutRowPitch { .. } => "vk_transfer_image_pitch_without_row_pitch",
            Self::OutsideTexture { .. } => "vk_transfer_outside_texture",
            Self::OutsideBuffer { .. } => "vk_transfer_outside_buffer",
            Self::OverlappingSelfCopy { .. } => "vk_transfer_overlapping_self_copy",
            Self::OverlappingSelfImageCopy { .. } => "vk_transfer_overlapping_self_image_copy",
            Self::BlitOption { refusal } => refusal.slug(),
            Self::AspectNotInFormat { .. } => "vk_transfer_aspect_not_in_format",
            Self::AmbiguousAspect { .. } => "vk_transfer_ambiguous_aspect",
            Self::NotACopy { .. } => "vk_transfer_not_a_copy",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolved { miss } => write!(f, "{} {miss}", self.slug()),
            Self::RowPitchNotWholeBlocks {
                bytes_per_row,
                block_bytes,
            } => write!(
                f,
                "{} bytes_per_row={bytes_per_row} block_bytes={block_bytes}",
                self.slug()
            ),
            Self::ImagePitchNotWholeRows {
                bytes_per_image,
                bytes_per_row,
            } => write!(
                f,
                "{} bytes_per_image={bytes_per_image} bytes_per_row={bytes_per_row}",
                self.slug()
            ),
            Self::UnknownFormatGeometry { format } => {
                write!(f, "{} format={format}", self.slug())
            }
            Self::RaggedPatternFill { offset, length } => {
                write!(f, "{} offset={offset} length={length}", self.slug())
            }
            Self::NoStaging { refusal } => write!(f, "{} {refusal}", self.slug()),
            Self::ExtentTooLarge { axis, value } => {
                write!(f, "{} axis={axis} value={value}", self.slug())
            }
            Self::RowPitchShorterThanCopy { row_length, width } => {
                write!(f, "{} row_length={row_length} width={width}", self.slug())
            }
            Self::ImagePitchShorterThanCopy {
                image_height,
                height,
            } => write!(
                f,
                "{} image_height={image_height} height={height}",
                self.slug()
            ),
            Self::ImagePitchWithoutRowPitch {
                bytes_per_image,
                depth,
            } => write!(
                f,
                "{} bytes_per_image={bytes_per_image} depth={depth}",
                self.slug()
            ),
            Self::OutsideTexture {
                axis,
                named,
                available,
            } => write!(
                f,
                "{} axis={axis} named={named} available={available}",
                self.slug()
            ),
            Self::OutsideBuffer {
                offset,
                length,
                size,
            } => write!(
                f,
                "{} offset={offset} length={length} size={size}",
                self.slug()
            ),
            Self::OverlappingSelfCopy {
                source_offset,
                dest_offset,
                size,
            } => write!(
                f,
                "{} source_offset={source_offset} dest_offset={dest_offset} size={size}",
                self.slug()
            ),
            Self::OverlappingSelfImageCopy { level, slice } => {
                write!(f, "{} level={level} slice={slice}", self.slug())
            }
            Self::BlitOption { refusal } => match refusal {
                OptionRefusal::UnknownBits { options } => {
                    write!(f, "{} options={options:#x}", self.slug())
                }
                OptionRefusal::RowLinearPvrtc | OptionRefusal::ConflictingAspects => {
                    f.write_str(self.slug())
                }
            },
            Self::AspectNotInFormat { format, aspect } => {
                write!(f, "{} format={format:#x} aspect={aspect}", self.slug())
            }
            Self::AmbiguousAspect { format } => {
                write!(f, "{} format={format:#x}", self.slug())
            }
            Self::NotACopy { op } => write!(f, "{} op={op}", self.slug()),
        }
    }
}

/// The regions of one native transfer, held inline when there is one of them.
///
/// Four of the five transfer shapes produce exactly one region — a whole
/// buffer copy, both directions of a buffer-image copy, and a single-region
/// texture copy. Only a slice span produces more, one per level. A `Vec` for
/// all five costs a trip into the allocator on every transfer a frame issues
/// in order to hold a value the plan already had, which is a per-record heap
/// cost the plan's structural zeros do not allow.
///
/// A borrowed slice would remove the allocation too, and it would also make
/// [`Command`] borrow the arena the plan wrote into — so a caller could hold
/// one command while planning the next only by keeping two arenas. This owns
/// what it carries and stays the value it always was.
#[derive(Clone, Debug)]
pub enum Regions<T> {
    /// The single-region case, which is every transfer but a slice span.
    One(T),
    /// A slice span's one region per level.
    Many(Vec<T>),
}

impl<T> core::ops::Deref for Regions<T> {
    type Target = [T];

    /// The regions, as the slice a native call takes.
    ///
    /// `Deref` and not a named door: every reader of this wants the slice ---
    /// the recorder hands it to `ash`, the tests index and iterate it --- and
    /// a container that spells `len`, `is_empty`, `iter` and `Index` itself is
    /// a second slice vocabulary that can disagree with the one it is holding.
    fn deref(&self) -> &[T] {
        match self {
            Self::One(region) => core::slice::from_ref(region),
            Self::Many(regions) => regions,
        }
    }
}

/// One native transfer command, with the handles it names already resolved.
///
/// Not `PartialEq`: ash's copy structures are not, and wrapping nine `u32`
/// fields per region to make them so would be a second vocabulary for
/// `VkBufferImageCopy` whose only reader is a test. The tests match and assert
/// fields instead.
#[derive(Clone, Debug)]
pub enum Command {
    CopyBuffer {
        source: vk::Buffer,
        dest: vk::Buffer,
        regions: Regions<vk::BufferCopy>,
    },
    CopyBufferToImage {
        source: vk::Buffer,
        dest: vk::Image,
        regions: Regions<vk::BufferImageCopy>,
    },
    CopyImageToBuffer {
        source: vk::Image,
        dest: vk::Buffer,
        regions: Regions<vk::BufferImageCopy>,
    },
    CopyImage {
        source: vk::Image,
        dest: vk::Image,
        regions: Regions<vk::ImageCopy>,
    },
}

impl Command {
    /// How many regions this command records.
    ///
    /// The count and not the regions: the four variants carry three different
    /// region types, and a caller that wants the regions themselves has
    /// already matched the variant that says which type they are.
    #[must_use]
    pub fn region_count(&self) -> usize {
        match self {
            Self::CopyBuffer { regions, .. } => regions.len(),
            Self::CopyBufferToImage { regions, .. } | Self::CopyImageToBuffer { regions, .. } => {
                regions.len()
            }
            Self::CopyImage { regions, .. } => regions.len(),
        }
    }
}

/// A dimension the guest gave in 64 bits, as the 32 a native copy carries.
fn narrow(axis: &'static str, value: u64) -> Result<u32, Refusal> {
    u32::try_from(value).map_err(|_| Refusal::ExtentTooLarge { axis, value })
}

/// A copy size, as the 32-bit triple a native copy carries.
///
/// # Zero on any axis is a refusal, not a copy of nothing
///
/// `VkImageCopy` and `VkBufferImageCopy` both require every component of
/// `imageExtent` to be non-zero, so a record with a zero axis has no native
/// form at all and recording one is invalid usage rather than a no-op. It is
/// refused here — the one place a guest [`Size3`] becomes a native extent, so
/// every arm that copies texels is bounded by the same sentence, the way
/// [`within_buffer`] bounds every arm that copies bytes.
///
/// A zero depth is *not* read as "one", even though [`ImagePitch::span_bytes`]
/// reads it that way. There it stands for an absent slice stride on a copy
/// whose slice count the pitch does not carry; here it is the copy's own
/// declared depth, and inventing a slice the guest did not ask for would copy
/// a plane it never named.
fn extent(size: Size3) -> Result<vk::Extent3D, Refusal> {
    debug_assert!(
        size.width != 0 && size.height != 0 && size.depth != 0,
        "a copy of no texels reaches `plan`'s own answer and never a native extent"
    );
    Ok(vk::Extent3D {
        width: narrow("width", size.width)?,
        height: narrow("height", size.height)?,
        depth: narrow("depth", size.depth)?,
    })
}

fn offset(origin: Origin3) -> Result<vk::Offset3D, Refusal> {
    Ok(vk::Offset3D {
        x: i32::try_from(origin.x).map_err(|_| Refusal::ExtentTooLarge {
            axis: "origin_x",
            value: origin.x,
        })?,
        y: i32::try_from(origin.y).map_err(|_| Refusal::ExtentTooLarge {
            axis: "origin_y",
            value: origin.y,
        })?,
        z: i32::try_from(origin.z).map_err(|_| Refusal::ExtentTooLarge {
            axis: "origin_z",
            value: origin.z,
        })?,
    })
}

/// A window of a buffer must end inside it.
///
/// The one place a buffer's own length is compared against what a transfer
/// asks of it, so every copy and every fill is bounded by the same sentence.
fn within_buffer(plan: &BufferPlan, offset: u64, length: u64) -> Result<(), Refusal> {
    let end = offset.checked_add(length).ok_or(Refusal::OutsideBuffer {
        offset,
        length,
        size: plan.size,
    })?;
    if end > plan.size {
        return Err(Refusal::OutsideBuffer {
            offset,
            length,
            size: plan.size,
        });
    }
    Ok(())
}

/// The bytes one buffer-image region addresses, from its buffer offset.
///
/// **Not the texel count and not the packed size of the copy.** Vulkan
/// addresses a region through the row and image pitches, so a copy of 8x8
/// texels described with a 1024-texel row reaches nearly seven rows further
/// into the buffer than its own texels occupy. Bounding the buffer by the
/// packed size would admit exactly that copy, and it is the shape a guest
/// producing a window into a larger image naturally sends.
///
/// The last addressed block is
/// `(depth-1)*rows_per_image*blocks_per_row + (rows-1)*blocks_per_row +
/// blocks_across`, in blocks, which is the spec's own addressing spelled in
/// this module's units. An empty extent addresses nothing.
///
/// # Errors
///
/// [`Refusal`] from the pitch conversion, or when the footprint does not fit
/// 64 bits.
fn region_bytes(pitch: ImagePitch, size: Size3, plane: Plane) -> Result<u64, Refusal> {
    let block = plane.block;
    let (row_length, image_height) = texel_pitch(pitch, size, plane)?;
    if size.width == 0 || size.height == 0 || size.depth == 0 {
        return Ok(0);
    }
    // Zero is "as wide as the copy" and "as tall as the copy", which is what
    // Vulkan reads a zero pitch as.
    let row_texels = if row_length == 0 {
        size.width
    } else {
        u64::from(row_length)
    };
    let image_rows = if image_height == 0 {
        size.height
    } else {
        u64::from(image_height)
    };
    let ceil = |value: u64, by: u32| value.div_ceil(u64::from(by));
    let blocks_per_row = ceil(row_texels, block.width);
    let rows_per_image = ceil(image_rows, block.height);
    let blocks_across = ceil(size.width, block.width);
    let rows = ceil(size.height, block.height);

    let overflow = || Refusal::ExtentTooLarge {
        axis: "footprint",
        value: u64::MAX,
    };
    let slices = rows_per_image
        .checked_mul(blocks_per_row)
        .and_then(|per_slice| per_slice.checked_mul(size.depth - 1))
        .ok_or_else(overflow)?;
    let within_slice = blocks_per_row
        .checked_mul(rows - 1)
        .and_then(|b| b.checked_add(blocks_across))
        .ok_or_else(overflow)?;
    slices
        .checked_add(within_slice)
        .and_then(|blocks| blocks.checked_mul(u64::from(block.bytes)))
        .ok_or_else(overflow)
}

/// `named` must not exceed `available`, on one axis of one texture.
fn within(axis: &'static str, named: u64, available: u64) -> Result<(), Refusal> {
    if named > available {
        return Err(Refusal::OutsideTexture {
            axis,
            named,
            available,
        });
    }
    Ok(())
}

/// One texture endpoint and the region it names, checked against the texture
/// it resolved to.
///
/// The three answers come back together because they are one decision: the
/// level chooses the extent the origin and size are bounded by, so a caller
/// given the subresource alone would have to look the extent up a second time
/// to bound the region — and the reason this check did not exist is that
/// nobody did.
///
/// # Errors
///
/// [`Refusal::OutsideTexture`] naming the first axis that does not fit, or
/// [`Refusal::ExtentTooLarge`] when a dimension does not fit the 32-bit fields
/// a native copy carries.
/// Which plane of a texture a blit addresses, and what one texel of it costs
/// on the buffer side.
///
/// # Why the format alone cannot answer this
///
/// [`crate::view::aspect`] answers "which aspects does this format have", and
/// for `MTLPixelFormatDepth24Unorm_Stencil8` and `Depth32Float_Stencil8` that
/// is two. `VkImageSubresourceLayers::aspectMask` in a buffer↔image copy must
/// name exactly one — VUID-VkBufferImageCopy-aspectMask-00212 — so the format
/// is not the whole question and reading it as if it were recorded an illegal
/// copy whenever the guest touched a combined depth-stencil texture.
///
/// The half the format does not carry is the guest's `MTLBlitOption` word,
/// which is exactly the plane selection. It also decides the buffer-side
/// texel size: the depth plane of `Depth24Unorm_Stencil8` is four bytes and
/// the stencil plane is one, where the packed cell is four — so a plane copy
/// converted through the cell's size gets a row pitch the guest never meant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Plane {
    /// The aspect the copy names.
    mask: vk::ImageAspectFlags,
    /// The addressable unit on the buffer side. A selected plane is one texel
    /// wide and its own width in bytes; the whole texel is the format's block,
    /// which for a compressed format is not one texel at all.
    block: reims_vgpu_core::extent::BlockGeometry,
}

/// Resolve the plane a blit addresses from its option word and the texture.
///
/// `single` is whether the copy has a buffer end, which is where "exactly one
/// aspect" is a Vulkan requirement rather than a preference: `vkCmdCopyImage`
/// accepts a combined mask and `vkCmdCopyBufferToImage` does not.
///
/// # Errors
///
/// [`Refusal::BlitOption`] for an option word this device cannot read as a
/// plane, [`Refusal::AspectNotInFormat`] for a plane the format does not have,
/// and [`Refusal::AmbiguousAspect`] for a buffer↔image copy of a combined
/// depth-stencil texture that named no plane — which is a copy whose bytes
/// nothing in the record identifies.
fn plane_of(texture: Texture, options: BlitOptions, single: bool) -> Result<Plane, Refusal> {
    let format = texture.pixel_format();
    let selected = select_aspect(options.0).map_err(|refusal| Refusal::BlitOption { refusal })?;
    let mask = match selected {
        BlitAspect::Full => aspect(format),
        BlitAspect::Depth => vk::ImageAspectFlags::DEPTH,
        BlitAspect::Stencil => vk::ImageAspectFlags::STENCIL,
    };
    if !aspect(format).contains(mask) {
        return Err(Refusal::AspectNotInFormat {
            format,
            aspect: match selected {
                BlitAspect::Depth => "depth",
                BlitAspect::Stencil => "stencil",
                BlitAspect::Full => "full",
            },
        });
    }
    if single
        && mask.contains(vk::ImageAspectFlags::DEPTH)
        && mask.contains(vk::ImageAspectFlags::STENCIL)
    {
        return Err(Refusal::AmbiguousAspect { format });
    }
    let block = match selected {
        BlitAspect::Full => {
            block_geometry(format).ok_or(Refusal::UnknownFormatGeometry { format })?
        }
        BlitAspect::Depth | BlitAspect::Stencil => {
            let bytes = blit_aspect_bytes_per_pixel(format, selected)
                .ok_or(Refusal::UnknownFormatGeometry { format })?;
            reims_vgpu_core::extent::BlockGeometry {
                width: 1,
                height: 1,
                bytes,
            }
        }
    };
    Ok(Plane { mask, block })
}

fn endpoint(
    texture: Texture,
    point: TexturePoint,
    size: Size3,
    plane: Plane,
) -> Result<(vk::ImageSubresourceLayers, vk::Offset3D, vk::Extent3D), Refusal> {
    within(
        "level",
        u64::from(point.level) + 1,
        u64::from(texture.mip_levels()),
    )?;
    within(
        "slice",
        u64::from(point.slice) + 1,
        u64::from(texture.layers()),
    )?;
    // Narrowed before the region is bounded: a dimension that does not fit the
    // field a native copy carries is not a copy at all, and reporting it as
    // "outside the texture" names the wrong thing about it --- every such value
    // is outside every texture, because a level extent is 32 bits wide.
    let native_offset = offset(point.origin)?;
    let native_extent = extent(size)?;
    let level = texture
        .level_extent(u32::from(point.level))
        .expect("the level count was just checked");
    // Origin plus extent, per axis, against the level's own size. A copy
    // bounded only by the whole texture reads past every level below zero.
    for (axis, origin, span, available) in [
        ("x", point.origin.x, size.width, level.x),
        ("y", point.origin.y, size.height, level.y),
        ("z", point.origin.z, size.depth, level.z),
    ] {
        // Both were narrowed above --- the origin to `i32` and the span to
        // `u32` --- so the sum is bounded well inside `u64` and cannot wrap.
        within(axis, origin + span, u64::from(available))?;
    }
    Ok((
        vk::ImageSubresourceLayers {
            aspect_mask: plane.mask,
            mip_level: u32::from(point.level),
            base_array_layer: u32::from(point.slice),
            layer_count: 1,
        },
        native_offset,
        native_extent,
    ))
}

/// The buffer-side pitch of a linear image, in texels, for a copy of `size`.
///
/// Zero is the guest saying "tightly packed", and Vulkan spells that with zero
/// too, so it passes through rather than being computed — computing it would
/// produce the same number for a tightly packed copy and a different one for a
/// copy whose extent is not the whole row.
///
/// # The copy's extent is a parameter because the pitch is only legal against it
///
/// A pitch is not well-formed on its own: `bufferRowLength` must be zero or at
/// least the copy's width and `bufferImageHeight` zero or at least its height,
/// and a byte slice stride with no byte row stride cannot be converted at all.
/// Those are three facts about a *pair*, so taking the size here is what lets
/// the pair be refused in one place instead of leaving a caller to re-derive
/// them — or, as this did, to record a copy the guest's own numbers say is
/// something else.
///
/// # Errors
///
/// [`Refusal`] when the byte pitch is not a whole number of blocks or rows,
/// when it cannot describe a copy of `size`, or when this build has no
/// geometry for the format.
fn texel_pitch(pitch: ImagePitch, size: Size3, plane: Plane) -> Result<(u32, u32), Refusal> {
    let block = plane.block;
    let block_bytes = u64::from(block.bytes);

    let row_length = if pitch.bytes_per_row == 0 {
        0
    } else {
        if block_bytes == 0 || !pitch.bytes_per_row.is_multiple_of(block_bytes) {
            return Err(Refusal::RowPitchNotWholeBlocks {
                bytes_per_row: pitch.bytes_per_row,
                block_bytes: block.bytes,
            });
        }
        // Blocks across, times the texels a block spans: for an uncompressed
        // format the block is 1x1 and this is the byte count over the texel
        // size, which is the same expression.
        narrow(
            "row_length",
            (pitch.bytes_per_row / block_bytes) * u64::from(block.width),
        )?
    };

    if row_length != 0 && u64::from(row_length) < size.width {
        return Err(Refusal::RowPitchShorterThanCopy {
            row_length,
            width: size.width,
        });
    }

    let image_height = if pitch.bytes_per_image == 0 {
        0
    } else if pitch.bytes_per_row == 0 {
        // No row stride to divide by. A copy one slice deep never addresses a
        // second slice, so the stride changes nothing and the tight answer is
        // exact; a deeper one has no answer at all.
        if size.depth > 1 {
            return Err(Refusal::ImagePitchWithoutRowPitch {
                bytes_per_image: pitch.bytes_per_image,
                depth: size.depth,
            });
        }
        0
    } else {
        if !pitch.bytes_per_image.is_multiple_of(pitch.bytes_per_row) {
            return Err(Refusal::ImagePitchNotWholeRows {
                bytes_per_image: pitch.bytes_per_image,
                bytes_per_row: pitch.bytes_per_row,
            });
        }
        narrow(
            "image_height",
            (pitch.bytes_per_image / pitch.bytes_per_row) * u64::from(block.height),
        )?
    };

    if image_height != 0 && u64::from(image_height) < size.height {
        return Err(Refusal::ImagePitchShorterThanCopy {
            image_height,
            height: size.height,
        });
    }

    Ok((row_length, image_height))
}

/// The four-byte word a fill pattern repeats.
///
/// A one-byte pattern is replicated, which is what `fillBuffer:range:value:`
/// means: every byte of the range takes the value.
#[must_use]
pub const fn fill_word(pattern: reims_vgpu_core::blit::FillPattern) -> u32 {
    match pattern {
        reims_vgpu_core::blit::FillPattern::Byte(byte) => u32::from_ne_bytes([byte; 4]),
        reims_vgpu_core::blit::FillPattern::Pattern4(word) => word,
    }
}

fn resolved<T>(result: Result<T, Miss>) -> Result<T, Refusal> {
    result.map_err(|miss| Refusal::Unresolved { miss })
}

/// Plan the native commands one resolved transfer becomes.
///
/// # `None` is a transfer that is already done, not one that was dropped
///
/// A copy whose destination is the same subresource, at the same origin, as
/// its source leaves the destination holding exactly what it already held. It
/// has no native form --- `vkCmdCopyImage`'s regions must not overlap in
/// memory, and a region overlaps itself totally --- and it needs none, so the
/// answer is that there is nothing to record. Not a refusal: an unmodified
/// guest issues `copyFromTexture:X toTexture:X` with equal origins, and a
/// refusal would drop a copy the guest treats as complete and report a failure
/// for a frame that is correct.
///
/// The shape is [`crate::layout::LayoutTracker::plan`]'s, and for the same
/// reason: "nothing to do" is the steady state of a real stream and is worth
/// saying rather than dressing as a command that does nothing.
///
/// # Errors
///
/// [`Refusal`] naming the one thing that could not be expressed.
pub fn plan(op: &BlitOp, residency: &Residency) -> Result<Option<Command>, Refusal> {
    // A transfer of no bytes is one that is already complete, not one that
    // cannot be expressed. The guest API reads a zero size and a zero-axis
    // extent as a no-op --- production's blit rail answers both on its success
    // arm --- and `vkCmdCopy*` has no form for either, so this is the same
    // answer an exact self-copy gets and for the same reason.
    //
    // Decided before any name is resolved, which is production's order too: a
    // no-op naming a stale resource is still a no-op, and refusing it would
    // report a failure for a command the guest saw succeed.
    let empty = match *op {
        BlitOp::BufferToBuffer { size, .. } => size == 0,
        BlitOp::BufferToTexture { size, .. }
        | BlitOp::TextureToBuffer { size, .. }
        | BlitOp::TextureRegion { size, .. } => {
            size.width == 0 || size.height == 0 || size.depth == 0
        }
        // A span of no levels or no slices copies no subresource. The slice
        // count is also `layerCount`, which Vulkan forbids being zero, so this
        // is the one arm where the empty case is illegal as well as pointless.
        BlitOp::TextureSlices {
            slice_count,
            level_count,
            ..
        } => level_count == 0 || slice_count == 0,
        BlitOp::FillBuffer { .. } | BlitOp::GenerateMipmaps { .. } => false,
    };
    if empty {
        return Ok(None);
    }
    Ok(Some(match *op {
        BlitOp::BufferToBuffer {
            source,
            source_offset,
            dest,
            dest_offset,
            size,
        } => {
            let from = resolved(residency.buffer(source))?;
            let to = resolved(residency.buffer(dest))?;
            within_buffer(&from.plan, source_offset, size)?;
            within_buffer(&to.plan, dest_offset, size)?;
            // Both windows are inside their buffers, so the sums below cannot
            // wrap. On the native handle rather than the two names: one
            // allocation may answer to several of them.
            if from.buffer == to.buffer {
                if source_offset == dest_offset {
                    // A window copied onto itself leaves the buffer holding
                    // what it already held, so there is nothing to record ---
                    // the same answer `TextureRegion` gives the equal-origin
                    // case below, and for the same reason. The undefined
                    // result the refusal names belongs to the *shifted* case:
                    // it is forward and backward copy orders disagreeing, and
                    // at zero shift they agree. Refusing here would report a
                    // failure for a command the guest saw succeed.
                    return Ok(None);
                }
                if source_offset < dest_offset + size && dest_offset < source_offset + size {
                    return Err(Refusal::OverlappingSelfCopy {
                        source_offset,
                        dest_offset,
                        size,
                    });
                }
            }
            Command::CopyBuffer {
                source: from.buffer,
                dest: to.buffer,
                regions: Regions::One(vk::BufferCopy {
                    src_offset: source_offset,
                    dst_offset: dest_offset,
                    size,
                }),
            }
        }
        BlitOp::BufferToTexture {
            source,
            source_offset,
            source_pitch,
            size,
            dest,
            options,
        } => {
            let from = resolved(residency.buffer(source))?;
            let image = resolved(residency.image(dest.texture))?;
            let plane = plane_of(image.texture, options, true)?;
            within_buffer(
                &from.plan,
                source_offset,
                region_bytes(source_pitch, size, plane)?,
            )?;
            Command::CopyBufferToImage {
                source: from.buffer,
                dest: image.image,
                regions: Regions::One(buffer_image_region(
                    image.texture,
                    source_offset,
                    source_pitch,
                    dest,
                    size,
                    plane,
                )?),
            }
        }
        BlitOp::TextureToBuffer {
            source,
            size,
            dest,
            dest_offset,
            dest_pitch,
            options,
        } => {
            let image = resolved(residency.image(source.texture))?;
            let to = resolved(residency.buffer(dest))?;
            let plane = plane_of(image.texture, options, true)?;
            within_buffer(
                &to.plan,
                dest_offset,
                region_bytes(dest_pitch, size, plane)?,
            )?;
            Command::CopyImageToBuffer {
                source: image.image,
                dest: to.buffer,
                regions: Regions::One(buffer_image_region(
                    image.texture,
                    dest_offset,
                    dest_pitch,
                    source,
                    size,
                    plane,
                )?),
            }
        }
        BlitOp::TextureRegion {
            source,
            dest,
            size,
            options,
        } => {
            let from = resolved(residency.image(source.texture))?;
            let to = resolved(residency.image(dest.texture))?;
            // Asked of each end, because the two textures are separately
            // declared and the option word selects a plane of whichever
            // format each one has. `single` is false: `vkCmdCopyImage` takes a
            // combined mask, so an image-to-image copy that named no plane is
            // the whole texel on both ends and not the ambiguity a buffer end
            // makes it.
            let src_plane = plane_of(from.texture, options, false)?;
            let dst_plane = plane_of(to.texture, options, false)?;
            let (src_subresource, src_offset, extent) =
                endpoint(from.texture, source, size, src_plane)?;
            let (dst_subresource, dst_offset, _) = endpoint(to.texture, dest, size, dst_plane)?;
            // One image, one subresource: the two regions are in the same
            // memory and `vkCmdCopyImage` forbids them overlapping. Different
            // levels or slices of one image are different memory, so this is
            // the only shape that can. On the native handle rather than the
            // two names, for the reason `OverlappingSelfCopy` gives.
            if from.image == to.image && source.level == dest.level && source.slice == dest.slice {
                if source.origin == dest.origin {
                    // A region copied onto itself leaves the destination
                    // holding what it already held. Answered as nothing to
                    // record rather than refused: an unmodified guest issues
                    // exactly this --- `copyFromTexture:X toTexture:X` with
                    // equal origins --- and a refusal drops a copy it treats as
                    // complete.
                    return Ok(None);
                }
                if overlaps(src_offset, dst_offset, extent) {
                    return Err(Refusal::OverlappingSelfImageCopy {
                        level: u32::from(source.level),
                        slice: u32::from(source.slice),
                    });
                }
            }
            Command::CopyImage {
                source: from.image,
                dest: to.image,
                regions: Regions::One(vk::ImageCopy {
                    src_subresource,
                    src_offset,
                    dst_subresource,
                    dst_offset,
                    extent,
                }),
            }
        }
        BlitOp::TextureSlices {
            source,
            dest,
            slice_count,
            level_count,
        } => {
            let from = resolved(residency.image(source.texture))?;
            let to = resolved(residency.image(dest.texture))?;
            // Both spans, both endpoints, before a single region is built: a
            // span that runs off the end of the destination is the same
            // invalid usage as one that runs off the source, and expanding
            // first would have found only the source's. One count of each for
            // both ends --- see `SpanOrigin` --- so the region loop below
            // cannot read a length from one end and a base from the other.
            for (texture, end) in [(from.texture, source), (to.texture, dest)] {
                within(
                    "level",
                    u64::from(end.base_level) + u64::from(level_count),
                    u64::from(texture.mip_levels()),
                )?;
                within(
                    "slice",
                    u64::from(end.base_slice) + u64::from(slice_count),
                    u64::from(texture.layers()),
                )?;
            }
            // One image, and spans that share a (level, slice) pair. Each
            // region covers its whole level, so a shared pair is a total
            // overlap and never a partial one --- there is no offset to
            // compare, only whether the two spans intersect on both axes.
            if from.image == to.image {
                let shares = |base_a: u16, base_b: u16, count: u16| {
                    let (a, b) = (u32::from(base_a), u32::from(base_b));
                    let count = u32::from(count);
                    (a < b + count && b < a + count).then_some(a.max(b))
                };
                if let (Some(level), Some(slice)) = (
                    shares(source.base_level, dest.base_level, level_count),
                    shares(source.base_slice, dest.base_slice, slice_count),
                ) {
                    if source.base_level == dest.base_level && source.base_slice == dest.base_slice
                    {
                        // The whole span onto itself: every region would read
                        // and write one subresource's own bytes. See the
                        // region arm above for why this is nothing to record
                        // rather than a refusal.
                        return Ok(None);
                    }
                    return Err(Refusal::OverlappingSelfImageCopy { level, slice });
                }
            }
            // One region per level and not per slice: `layerCount` covers the
            // slice span, and the extent is the level's own --- a mip chain
            // copied with level zero's extent reads past every level below it.
            let region = |level: u32| -> Result<vk::ImageCopy, Refusal> {
                let source_extent = from
                    .texture
                    .level_extent(u32::from(source.base_level) + level)
                    .ok_or(Refusal::ExtentTooLarge {
                        axis: "level",
                        value: u64::from(source.base_level) + u64::from(level),
                    })?;
                // And the destination level has to hold it. The span check
                // above bounds how many levels and slices each side has, not
                // how big they are, and the two textures are separately
                // declared: nothing in the record makes the level this writes
                // to the same size as the level it reads. A destination level
                // smaller than the source's is the whole-extent write past the
                // end of an image that `endpoint` refuses for its own arm.
                let dest_extent = to
                    .texture
                    .level_extent(u32::from(dest.base_level) + level)
                    .ok_or(Refusal::ExtentTooLarge {
                        axis: "level",
                        value: u64::from(dest.base_level) + u64::from(level),
                    })?;
                for (axis, named, available) in [
                    ("x", source_extent.x, dest_extent.x),
                    ("y", source_extent.y, dest_extent.y),
                    ("z", source_extent.z, dest_extent.z),
                ] {
                    within(axis, u64::from(named), u64::from(available))?;
                }
                Ok(vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: aspect(from.texture.pixel_format()),
                        mip_level: u32::from(source.base_level) + level,
                        base_array_layer: u32::from(source.base_slice),
                        layer_count: u32::from(slice_count),
                    },
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: aspect(to.texture.pixel_format()),
                        mip_level: u32::from(dest.base_level) + level,
                        base_array_layer: u32::from(dest.base_slice),
                        layer_count: u32::from(slice_count),
                    },
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width: source_extent.x,
                        height: source_extent.y,
                        depth: source_extent.z,
                    },
                })
            };
            // A one-level span is the single-region shape the other four arms
            // are, so it is held the way they are held. The count is decided
            // before the first region is built rather than after the last one,
            // because a `Vec` collected and then found to hold one has already
            // paid for itself.
            let regions = match level_count {
                // Zero was answered before any name was resolved.
                1 => Regions::One(region(0)?),
                count => Regions::Many(
                    (0..u32::from(count))
                        .map(region)
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            };
            Command::CopyImage {
                source: from.image,
                dest: to.image,
                regions,
            }
        }
        // A fill is not a copy, and a ragged one is not even one command. See
        // [`plan_fill`], which is the only thing that can answer it, because
        // it needs the scratch memory this signature has no access to.
        BlitOp::FillBuffer { .. } => return Err(Refusal::NotACopy { op: "fill_buffer" }),
        // A filtered reduction with a barrier between every pair of levels,
        // not a copy: see [`crate::mipmap`]. Refused here rather than absorbed
        // into one of the copies above, because a mipmap generation recorded
        // as a copy would produce level one and leave the rest of the chain
        // undefined. The name is still resolved first, so a generation naming
        // nothing is that refusal and not this one.
        BlitOp::GenerateMipmaps { texture } => {
            resolved(residency.image(texture))?;
            return Err(Refusal::NotACopy {
                op: "generate_mipmaps",
            });
        }
    }))
}

/// Whether two regions of one subresource, both `extent` in size, share a
/// texel.
///
/// Two axis-aligned boxes intersect exactly when they intersect on every axis,
/// which is three independent one-dimensional questions and not one
/// three-dimensional one. Both offsets were narrowed to `i32` and the extent to
/// `u32` by [`endpoint`], so the sums are inside `i64` and cannot wrap.
fn overlaps(a: vk::Offset3D, b: vk::Offset3D, extent: vk::Extent3D) -> bool {
    [
        (a.x, b.x, extent.width),
        (a.y, b.y, extent.height),
        (a.z, b.z, extent.depth),
    ]
    .into_iter()
    .all(|(a, b, span)| {
        let (a, b, span) = (i64::from(a), i64::from(b), i64::from(span));
        a < b + span && b < a + span
    })
}

/// The interior of a fill, as `vkCmdFillBuffer` takes it.
///
/// Its own type rather than a [`Command`], so a [`FillPlan`] stays comparable:
/// a plan whose parts cannot be compared is one whose split cannot be asserted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillRange {
    pub offset: u64,
    pub size: u64,
    pub data: u32,
}

/// The bytes a ragged fill's edge needs, and where they go.
///
/// The caller writes `bytes[..length]` into the window, flushes if the mapping
/// is not coherent, and copies the window into the destination buffer at
/// `dest_offset`.
///
/// The bytes are carried rather than one repeated byte because an edge is not
/// always one value: a four-byte pattern whose range ends raggedly has a tail
/// of its own leading bytes, and its phase there is known exactly — the tail
/// begins at a four-byte boundary. `length` is at most three, so the array is
/// the pattern's width and never a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagedEdge {
    pub window: crate::staging::Window,
    pub bytes: [u8; 4],
    pub dest_offset: u64,
    pub length: u64,
}

/// A fill, split into the part a native fill command can do and the edges it
/// cannot.
///
/// `head` and `tail` are at most three bytes each, so the scratch this costs
/// is six bytes whatever the size of the fill — the middle is still one
/// `vkCmdFillBuffer` over everything else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use = "a fill plan whose edges are not written leaves the range partly unfilled"]
pub struct FillPlan {
    pub dest: vk::Buffer,
    /// The unaligned bytes before the first four-byte boundary in the range.
    pub head: Option<StagedEdge>,
    /// The four-byte-aligned interior. `None` only for a range too short to
    /// contain one.
    pub middle: Option<FillRange>,
    /// The unaligned bytes after the last four-byte boundary in the range.
    pub tail: Option<StagedEdge>,
}

/// Plan a buffer fill, using scratch memory for the edges a native fill cannot
/// reach.
///
/// `vkCmdFillBuffer` requires a four-byte-aligned offset and size, and
/// `fillBuffer:range:` does not. Refusing the whole fill for that would lose a
/// legal guest command; rounding out would write bytes outside the range the
/// guest named, and rounding in would leave bytes inside it unwritten. So the
/// range is split: the interior is one fill command, and the at-most-three
/// bytes at each end are staged and copied. A copy has no alignment rule at
/// all, which is exactly why the edges go through one.
///
/// # Errors
///
/// [`Refusal`]. Scratch is taken in one allocation covering both edges, so a
/// refusal leaves the arena as it was rather than stranding a head window
/// whose tail could not be found.
pub fn plan_fill(
    span: reims_vgpu_core::blit::BufferSpan,
    pattern: reims_vgpu_core::blit::FillPattern,
    residency: &Residency,
    arena: &mut crate::staging::Arena,
) -> Result<Option<FillPlan>, Refusal> {
    use reims_vgpu_core::blit::FillPattern;

    // A fill of no bytes, answered the way [`plan`] answers a copy of none and
    // before the name is resolved. See there.
    if span.length == 0 {
        return Ok(None);
    }
    let target = resolved(residency.buffer(span.buffer))?;
    within_buffer(&target.plan, span.offset, span.length)?;
    let dest = target.buffer;

    // The bytes before the first four-byte boundary inside the range, and
    // after the last one. Both are zero for an aligned range, which is the
    // common case and costs no scratch at all.
    let head_length = (4 - span.offset % 4) % 4;
    let head_length = head_length.min(span.length);
    let remaining = span.length - head_length;
    let middle_length = remaining & !3;
    let tail_length = remaining & 3;

    if head_length == 0 && tail_length == 0 {
        return Ok(Some(FillPlan {
            dest,
            head: None,
            middle: Some(FillRange {
                offset: span.offset,
                size: span.length,
                data: fill_word(pattern),
            }),
            tail: None,
        }));
    }

    // A byte pattern has no phase: every byte of the range takes the value, so
    // splitting the range changes nothing.
    //
    // A four-byte pattern does have one, and the unestablished term is where
    // it *starts* — which of its bytes lands on a range that does not begin at
    // a multiple of four. A range that begins at one and merely ends raggedly
    // asks nothing: every byte's phase follows from the start, the tail begins
    // at a four-byte boundary, and its bytes are the pattern's leading ones.
    // Refusing that too refused a fill the guest is entitled to and the device
    // has always performed.
    if !span.offset.is_multiple_of(4) && matches!(pattern, FillPattern::Pattern4(_)) {
        return Err(Refusal::RaggedPatternFill {
            offset: span.offset,
            length: span.length,
        });
    }
    let bytes = match pattern {
        FillPattern::Byte(byte) => [byte; 4],
        FillPattern::Pattern4(word) => word.to_le_bytes(),
    };

    // One allocation for both edges, so there is one failure point and no
    // stranded window on it.
    let scratch = arena
        .allocate(head_length + tail_length, 1)
        .map_err(|refusal| Refusal::NoStaging { refusal })?;

    let head = (head_length > 0).then_some(StagedEdge {
        window: crate::staging::Window {
            chunk: scratch.chunk,
            offset: scratch.offset,
            size: head_length,
        },
        bytes,
        dest_offset: span.offset,
        length: head_length,
    });
    let tail = (tail_length > 0).then_some(StagedEdge {
        window: crate::staging::Window {
            chunk: scratch.chunk,
            offset: scratch.offset + head_length,
            size: tail_length,
        },
        bytes,
        dest_offset: span.offset + head_length + middle_length,
        length: tail_length,
    });
    let middle = (middle_length > 0).then_some(FillRange {
        offset: span.offset + head_length,
        size: middle_length,
        data: fill_word(pattern),
    });

    Ok(Some(FillPlan {
        dest,
        head,
        middle,
        tail,
    }))
}

/// One `VkBufferImageCopy`, whichever direction it goes.
///
/// Shared because the structure is the same both ways and the two directions
/// disagreeing about the pitch conversion would be a download that does not
/// match the upload that produced it.
fn buffer_image_region(
    texture: Texture,
    buffer_offset: u64,
    pitch: ImagePitch,
    point: TexturePoint,
    size: Size3,
    plane: Plane,
) -> Result<vk::BufferImageCopy, Refusal> {
    let (row_length, image_height) = texel_pitch(pitch, size, plane)?;
    let (image_subresource, image_offset, image_extent) = endpoint(texture, point, size, plane)?;
    Ok(vk::BufferImageCopy {
        buffer_offset,
        buffer_row_length: row_length,
        buffer_image_height: image_height,
        image_subresource,
        image_offset,
        image_extent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    use reims_vgpu_core::blit::{BufferSpan, FillPattern, SpanOrigin};
    use reims_vgpu_core::blit_option::{
        MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL, MTL_BLIT_OPTION_NONE,
        MTL_BLIT_OPTION_ROW_LINEAR_PVRTC, MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL,
    };
    use reims_vgpu_core::identity::{
        DeviceEpoch, ObjectListRef, ResourceId, SessionGeneration, SlotGeneration, TimelinePoint,
    };
    use reims_vgpu_core::pixel_format::{
        MTL_FORMAT_BC3_RGBA, MTL_FORMAT_DEPTH32_FLOAT, MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
        MTL_FORMAT_RGBA8_UNORM,
    };
    use reims_vgpu_core::retire::{Lifetime, NativeRetirement};
    use reims_vgpu_core::texture_shape::{TextureKind, TextureShape, TextureUsage};
    use std::collections::BTreeSet;

    use crate::buffer::{BufferPlan, EVERY_CLASS};
    use crate::image::ImagePlan;
    use crate::resident::{Native, NativeBuffer, NativeImage};

    /// The whole-texel plane of a texture, which is what every case that is
    /// not about plane selection asks for.
    fn whole(texture: Texture) -> Plane {
        plane_of(texture, BlitOptions(0), true).expect("a single-aspect format")
    }

    /// The pitch conversion, in the shape the cases below drive it: a texture
    /// rather than the plane derived from it. The plane is what the conversion
    /// actually needs — a selected depth or stencil plane has its own texel
    /// width — and every case here is a whole-texel one.
    fn texel_pitch(
        texture: Texture,
        pitch: ImagePitch,
        size: Size3,
    ) -> Result<(u32, u32), Refusal> {
        super::texel_pitch(pitch, size, whole(texture))
    }

    fn region_bytes(texture: Texture, pitch: ImagePitch, size: Size3) -> Result<u64, Refusal> {
        super::region_bytes(pitch, size, whole(texture))
    }

    const BUFFER_A: u32 = 1;
    const BUFFER_B: u32 = 2;
    const IMAGE_A: u32 = 3;
    const IMAGE_B: u32 = 4;

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration(1),
        }
    }

    fn texture(format: u16, width: u32, height: u32, levels: u32, layers: u32) -> Texture {
        TextureShape {
            kind: if layers > 1 {
                TextureKind::D2Array.ordinal()
            } else {
                TextureKind::D2.ordinal()
            },
            width,
            height,
            depth: 1,
            mipmap_level_count: levels,
            sample_count: 1,
            array_length: layers,
            pixel_format: format,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a valid declaration")
    }

    fn native_image(handle: u64, texture: Texture) -> Native {
        Native::Image(NativeImage {
            texture,
            image: vk::Image::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: ImagePlan {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: texture.extent().x,
                    height: texture.extent().y,
                    depth: 1,
                },
                mip_levels: texture.mip_levels(),
                array_layers: texture.layers(),
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            },
            sampled: std::collections::BTreeMap::from([(
                texture.pixel_format(),
                vk::ImageView::from_raw(handle),
            )]),
            attachments: Vec::new(),
        })
    }

    fn native_buffer(handle: u64) -> Native {
        Native::Buffer(NativeBuffer {
            buffer: vk::Buffer::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: BufferPlan {
                size: 1 << 20,
                usage: EVERY_CLASS,
                aliased: false,
            },
        })
    }

    /// Two buffers and two textures, published and resolvable.
    fn populated() -> Residency {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        let lifetime = Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST);
        for (slot, native) in [
            (BUFFER_A, native_buffer(0xB1)),
            (BUFFER_B, native_buffer(0xB2)),
            (
                IMAGE_A,
                native_image(0x1A, texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 4, 3)),
            ),
            (
                IMAGE_B,
                native_image(0x1B, texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 4, 3)),
            ),
        ] {
            residency
                .publish(id(slot), lifetime, native, &mut retire)
                .unwrap_or_else(|(_, e)| panic!("{e}"));
        }
        assert_eq!(retire.outstanding(), 0);
        residency
    }

    fn point(slot: u32, level: u16, slice: u16) -> TexturePoint {
        TexturePoint {
            texture: id(slot),
            slice,
            level,
            origin: Origin3 { x: 0, y: 0, z: 0 },
        }
    }

    fn size(width: u64, height: u64) -> Size3 {
        Size3 {
            width,
            height,
            depth: 1,
        }
    }

    #[test]
    fn a_buffer_copy_carries_both_offsets_and_the_size() {
        let residency = populated();
        let planned = plan(
            &BlitOp::BufferToBuffer {
                source: id(BUFFER_A),
                source_offset: 16,
                dest: id(BUFFER_B),
                dest_offset: 64,
                size: 256,
            },
            &residency,
        )
        .expect("plannable")
        .expect("native work to record");
        let Command::CopyBuffer {
            source,
            dest,
            regions,
        } = planned
        else {
            panic!("a buffer copy");
        };
        assert_eq!(source.as_raw(), 0xB1);
        assert_eq!(dest.as_raw(), 0xB2);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].src_offset, 16);
        assert_eq!(regions[0].dst_offset, 64);
        assert_eq!(regions[0].size, 256);
    }

    #[test]
    fn an_unresolved_name_refuses_with_the_miss_that_produced_it() {
        let residency = populated();
        let refusal = plan(
            &BlitOp::BufferToBuffer {
                source: id(99),
                source_offset: 0,
                dest: id(BUFFER_B),
                dest_offset: 0,
                size: 4,
            },
            &residency,
        )
        .expect_err("no such slot");
        assert_eq!(
            refusal,
            Refusal::Unresolved {
                miss: Miss::Unknown {
                    slot: ObjectListRef(99)
                }
            }
        );

        // And the kind confusion, which is the one a slot-only check misses.
        let wrong = plan(
            &BlitOp::BufferToBuffer {
                source: id(IMAGE_A),
                source_offset: 0,
                dest: id(BUFFER_B),
                dest_offset: 0,
                size: 4,
            },
            &residency,
        )
        .expect_err("a texture is not a buffer");
        assert!(matches!(
            wrong,
            Refusal::Unresolved {
                miss: Miss::WrongKind { .. }
            }
        ));
    }

    #[test]
    fn a_byte_pitch_becomes_a_texel_pitch_through_the_format() {
        // RGBA8: a 1x1 block of four bytes, so 256 bytes is 64 texels.
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 256,
                    bytes_per_image: 256 * 32,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Ok((64, 32))
        );
    }

    #[test]
    fn a_compressed_row_is_blocks_across_times_the_block_width() {
        // BC3: a 4x4 block of sixteen bytes. A 64-texel row is sixteen blocks
        // and 256 bytes, so the same byte pitch is *64* texels here too — and
        // dividing bytes by a bytes-per-pixel that does not exist for this
        // format would have produced sixteen.
        let compressed = texture(MTL_FORMAT_BC3_RGBA, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                compressed,
                ImagePitch {
                    bytes_per_row: 256,
                    bytes_per_image: 256 * 8,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            // Eight rows of blocks, each spanning four texels vertically.
            Ok((64, 32))
        );
    }

    #[test]
    fn a_tightly_packed_pitch_passes_through_as_zero() {
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 0,
                    bytes_per_image: 0,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Ok((0, 0))
        );
        // An image pitch with no row pitch has nothing to be a multiple of.
        // On a copy one slice deep the stride is never addressed, so the tight
        // answer is exact rather than a division by zero.
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 0,
                    bytes_per_image: 1024,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Ok((0, 0))
        );
    }

    #[test]
    fn a_pitch_that_is_not_whole_blocks_refuses_rather_than_rounding() {
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 258,
                    bytes_per_image: 0,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Err(Refusal::RowPitchNotWholeBlocks {
                bytes_per_row: 258,
                block_bytes: 4,
            })
        );
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 256,
                    bytes_per_image: 300,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Err(Refusal::ImagePitchNotWholeRows {
                bytes_per_image: 300,
                bytes_per_row: 256,
            })
        );
    }

    #[test]
    fn a_format_with_no_known_geometry_refuses_by_name() {
        let unknown = TextureShape {
            kind: TextureKind::D2.ordinal(),
            width: 4,
            height: 4,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            // Not a format this build has a block for.
            pixel_format: 0xFFFF,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a declaration");
        // Asked of the plane, which is where the addressable unit is decided:
        // a selected depth or stencil plane has its own texel width, and a
        // whole-texel copy has the format's block. Either way a format this
        // build has no geometry for is refused before a pitch is converted.
        assert_eq!(
            plane_of(unknown, BlitOptions(0), true),
            Err(Refusal::UnknownFormatGeometry { format: 0xFFFF })
        );
    }

    #[test]
    fn both_directions_use_one_pitch_conversion() {
        let residency = populated();
        let pitch = ImagePitch {
            bytes_per_row: 256,
            bytes_per_image: 256 * 32,
        };
        let up = plan(
            &BlitOp::BufferToTexture {
                source: id(BUFFER_A),
                source_offset: 128,
                source_pitch: pitch,
                size: size(32, 16),
                dest: point(IMAGE_A, 1, 2),
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable");
        let down = plan(
            &BlitOp::TextureToBuffer {
                source: point(IMAGE_A, 1, 2),
                size: size(32, 16),
                dest: id(BUFFER_A),
                dest_offset: 128,
                dest_pitch: pitch,
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable");

        let (
            Some(Command::CopyBufferToImage { regions: u, .. }),
            Some(Command::CopyImageToBuffer { regions: d, .. }),
        ) = (up, down)
        else {
            panic!("one of each direction");
        };
        assert_eq!(u.len(), 1);
        assert_eq!(d.len(), 1);
        // The same region both ways: a download that did not match the upload
        // that produced it is the failure this shares one function for.
        assert_eq!(u[0].buffer_offset, d[0].buffer_offset);
        assert_eq!(u[0].buffer_row_length, d[0].buffer_row_length);
        assert_eq!(u[0].buffer_image_height, d[0].buffer_image_height);
        assert_eq!(u[0].image_subresource.mip_level, 1);
        assert_eq!(u[0].image_subresource.base_array_layer, 2);
        assert_eq!(u[0].image_subresource.layer_count, 1);
        assert_eq!(
            u[0].image_subresource.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        // Level one of a 64x32 texture is 32x16, and the row pitch describes the
        // wider image the copy is a window into --- the ordinary reason a pitch
        // is named at all.
        assert_eq!(u[0].image_extent.width, 32);
        assert_eq!(u[0].buffer_row_length, 64);
    }

    #[test]
    fn a_depth_texture_transfers_on_its_own_aspect() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(
                id(IMAGE_A),
                Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST),
                native_image(0x1A, texture(MTL_FORMAT_DEPTH32_FLOAT, 8, 8, 1, 1)),
                &mut retire,
            )
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency
            .publish(
                id(BUFFER_A),
                Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST),
                native_buffer(0xB1),
                &mut retire,
            )
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        let planned = plan(
            &BlitOp::TextureToBuffer {
                source: point(IMAGE_A, 0, 0),
                size: size(8, 8),
                dest: id(BUFFER_A),
                dest_offset: 0,
                dest_pitch: ImagePitch {
                    bytes_per_row: 32,
                    bytes_per_image: 0,
                },
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable")
        .expect("native work to record");
        let Command::CopyImageToBuffer { regions, .. } = planned else {
            panic!("a download");
        };
        assert_eq!(
            regions[0].image_subresource.aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
    }

    /// A combined depth-stencil texture with a buffer at the other end, and
    /// the three answers the option word decides between.
    ///
    /// `VkImageSubresourceLayers::aspectMask` must have exactly one bit here
    /// (VUID-VkBufferImageCopy-aspectMask-00212). The format alone answers
    /// "depth and stencil", so a planner reading only the format recorded an
    /// illegal copy for every guest that touched one of these textures, and
    /// dropped the guest's plane selection while doing it.
    #[test]
    fn a_combined_depth_stencil_copy_takes_its_aspect_from_the_option_word() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        residency
            .publish(
                id(IMAGE_A),
                Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST),
                native_image(0x1A, texture(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, 8, 8, 1, 1)),
                &mut retire,
            )
            .unwrap_or_else(|(_, e)| panic!("{e}"));
        residency
            .publish(
                id(BUFFER_A),
                Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST),
                native_buffer(0xB1),
                &mut retire,
            )
            .unwrap_or_else(|(_, e)| panic!("{e}"));

        let download = |options: u32, bytes_per_row: u64| {
            plan(
                &BlitOp::TextureToBuffer {
                    source: point(IMAGE_A, 0, 0),
                    size: size(8, 8),
                    dest: id(BUFFER_A),
                    dest_offset: 0,
                    dest_pitch: ImagePitch {
                        bytes_per_row,
                        bytes_per_image: 0,
                    },
                    options: BlitOptions(options),
                },
                &residency,
            )
        };

        // No plane named. Nothing in the record says which half of the texel
        // the guest's bytes are, so there is nothing to derive and the copy is
        // refused by name rather than recorded with a two-bit mask.
        assert_eq!(
            download(MTL_BLIT_OPTION_NONE, 32).expect_err("no plane is named"),
            Refusal::AmbiguousAspect {
                format: MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
            }
        );

        // The depth plane: four bytes a texel, so a row of eight is 32.
        let depth = download(MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL, 32)
            .expect("plannable")
            .expect("native work");
        let Command::CopyImageToBuffer { regions, .. } = depth else {
            panic!("a download");
        };
        assert_eq!(
            regions[0].image_subresource.aspect_mask,
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            regions[0].buffer_row_length, 8,
            "32 bytes of four-byte depth"
        );

        // The stencil plane: one byte a texel, so the same 32-byte row is
        // thirty-two texels wide. Converted through the packed cell it would
        // have been four, which is the pitch the guest never meant.
        let stencil = download(MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL, 32)
            .expect("plannable")
            .expect("native work");
        let Command::CopyImageToBuffer { regions, .. } = stencil else {
            panic!("a download");
        };
        assert_eq!(
            regions[0].image_subresource.aspect_mask,
            vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            regions[0].buffer_row_length, 32,
            "32 bytes of one-byte stencil"
        );
    }

    /// The option word's own refusals reach the guest under the name the layer
    /// that owns the tag gave them, and a plane the format does not have is
    /// refused rather than silently widened.
    #[test]
    fn an_option_word_this_device_cannot_read_refuses_by_its_owners_name() {
        let residency = populated();
        let download = |options: u32| {
            plan(
                &BlitOp::TextureToBuffer {
                    source: point(IMAGE_A, 0, 0),
                    size: size(8, 8),
                    dest: id(BUFFER_A),
                    dest_offset: 0,
                    dest_pitch: ImagePitch {
                        bytes_per_row: 0,
                        bytes_per_image: 0,
                    },
                    options: BlitOptions(options),
                },
                &residency,
            )
            .expect_err("refused")
        };

        assert_eq!(
            download(1 << 20),
            Refusal::BlitOption {
                refusal: OptionRefusal::UnknownBits { options: 1 << 20 }
            }
        );
        assert_eq!(
            download(MTL_BLIT_OPTION_ROW_LINEAR_PVRTC),
            Refusal::BlitOption {
                refusal: OptionRefusal::RowLinearPvrtc
            }
        );
        assert_eq!(
            download(
                MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL
                    | MTL_BLIT_OPTION_STENCIL_FROM_DEPTH_STENCIL
            ),
            Refusal::BlitOption {
                refusal: OptionRefusal::ConflictingAspects
            }
        );
        // `IMAGE_A` is a colour texture, which has neither plane.
        assert_eq!(
            download(MTL_BLIT_OPTION_DEPTH_FROM_DEPTH_STENCIL),
            Refusal::AspectNotInFormat {
                format: MTL_FORMAT_RGBA8_UNORM,
                aspect: "depth",
            }
        );
    }

    #[test]
    fn a_slice_span_copies_one_region_per_level_at_that_levels_extent() {
        let residency = populated();
        let planned = plan(
            &BlitOp::TextureSlices {
                source: SpanOrigin {
                    texture: id(IMAGE_A),
                    base_slice: 0,
                    base_level: 1,
                },
                dest: SpanOrigin {
                    texture: id(IMAGE_B),
                    base_slice: 0,
                    base_level: 0,
                },
                slice_count: 3,
                level_count: 3,
            },
            &residency,
        )
        .expect("plannable")
        .expect("native work to record");
        let Command::CopyImage { regions, .. } = planned else {
            panic!("an image copy");
        };
        assert_eq!(regions.len(), 3);
        // 64x32 at level 0, so levels one through three are 32x16, 16x8, 8x4.
        // Copying all three with level zero's extent reads past every level
        // below the first.
        let extents: Vec<(u32, u32)> = regions
            .iter()
            .map(|r| (r.extent.width, r.extent.height))
            .collect();
        assert_eq!(extents, [(32, 16), (16, 8), (8, 4)]);
        for (index, region) in regions.iter().enumerate() {
            assert_eq!(region.src_subresource.mip_level, 1 + index as u32);
            assert_eq!(region.dst_subresource.mip_level, index as u32);
            // The slice span is the layer count, not a region each.
            assert_eq!(region.src_subresource.layer_count, 3);
            assert_eq!(region.dst_subresource.layer_count, 3);
        }
    }

    /// A transfer of no bytes is complete, not refused.
    ///
    /// The guest API reads a zero size and a zero-axis extent as a no-op, and
    /// production's blit rail answers both on its success arm --- `BlitStatus::
    /// ZeroExtent` reports no refusal and its dispatch site treats it exactly
    /// as `Ok`. Refusing it here would report a failure for a command the
    /// guest saw succeed.
    ///
    /// Every arm, and every axis of every arm, because "no bytes" is spelled
    /// five different ways across the five shapes.
    #[test]
    fn a_transfer_of_no_bytes_is_complete_rather_than_refused() {
        let residency = populated();
        let flat = ImagePitch {
            bytes_per_row: 0,
            bytes_per_image: 0,
        };
        let mut empty: Vec<BlitOp> = vec![BlitOp::BufferToBuffer {
            source: id(BUFFER_A),
            source_offset: 0,
            dest: id(BUFFER_B),
            dest_offset: 0,
            size: 0,
        }];
        for axis in 0..3 {
            let size = Size3 {
                width: if axis == 0 { 0 } else { 4 },
                height: if axis == 1 { 0 } else { 4 },
                depth: if axis == 2 { 0 } else { 1 },
            };
            empty.push(BlitOp::BufferToTexture {
                source: id(BUFFER_A),
                source_offset: 0,
                source_pitch: flat,
                size,
                dest: point(IMAGE_A, 0, 0),
                options: Default::default(),
            });
            empty.push(BlitOp::TextureToBuffer {
                source: point(IMAGE_A, 0, 0),
                size,
                dest: id(BUFFER_A),
                dest_offset: 0,
                dest_pitch: flat,
                options: Default::default(),
            });
            empty.push(BlitOp::TextureRegion {
                source: point(IMAGE_A, 0, 0),
                dest: point(IMAGE_B, 0, 0),
                size,
                options: Default::default(),
            });
        }
        // A span of no levels, and one of no slices --- the slice count is
        // also `layerCount`, which Vulkan forbids being zero, so the second is
        // illegal as well as pointless.
        for (level_count, slice_count) in [(0, 1), (1, 0)] {
            let origin = |texture| SpanOrigin {
                texture,
                base_slice: 0,
                base_level: 0,
            };
            empty.push(BlitOp::TextureSlices {
                source: origin(id(IMAGE_A)),
                dest: origin(id(IMAGE_B)),
                slice_count,
                level_count,
            });
        }
        for op in empty {
            let kind = op.kind();
            assert!(
                plan(&op, &residency)
                    .unwrap_or_else(|e| panic!("{kind:?}: {e}"))
                    .is_none(),
                "{kind:?} of no bytes"
            );
        }
    }

    /// And it is answered before the names are resolved, which is production's
    /// order too: a no-op naming a resource that is gone is still a no-op, and
    /// refusing it would report a failure for a command the guest completed.
    #[test]
    fn a_transfer_of_no_bytes_does_not_need_its_names_to_resolve() {
        let residency = Residency::new();
        assert!(plan(
            &BlitOp::BufferToBuffer {
                source: id(BUFFER_A),
                source_offset: 0,
                dest: id(BUFFER_B),
                dest_offset: 0,
                size: 0,
            },
            &residency,
        )
        .expect("nothing to resolve")
        .is_none());
        // The same names with one byte to move do not resolve, so the empty
        // answer above is the emptiness and not the fixture.
        assert!(plan(
            &BlitOp::BufferToBuffer {
                source: id(BUFFER_A),
                source_offset: 0,
                dest: id(BUFFER_B),
                dest_offset: 0,
                size: 1,
            },
            &residency,
        )
        .is_err());
    }

    #[test]
    fn a_one_byte_fill_pattern_is_replicated_into_the_word() {
        assert_eq!(fill_word(FillPattern::Byte(0xAB)), 0xABAB_ABAB);
        assert_eq!(fill_word(FillPattern::Pattern4(0x1234_5678)), 0x1234_5678);
    }

    fn arena() -> crate::staging::Arena {
        crate::staging::Arena::adopt(
            256,
            64,
            vec![vk::Buffer::from_raw(0xFF)],
            vec![vk::DeviceMemory::from_raw(0xFF)],
            vec![std::ptr::null_mut()],
        )
    }

    fn span(offset: u64, length: u64) -> BufferSpan {
        BufferSpan {
            buffer: id(BUFFER_A),
            offset,
            length,
        }
    }

    #[test]
    fn a_fill_is_not_a_copy_and_the_copy_planner_says_so() {
        let residency = populated();
        assert_eq!(
            plan(
                &BlitOp::FillBuffer {
                    dest: span(0, 16),
                    pattern: FillPattern::Byte(0),
                },
                &residency,
            )
            .err(),
            Some(Refusal::NotACopy { op: "fill_buffer" })
        );
    }

    #[test]
    fn an_aligned_fill_is_one_command_and_costs_no_scratch() {
        let residency = populated();
        let mut arena = arena();
        let planned = plan_fill(
            span(64, 256),
            FillPattern::Byte(0xFF),
            &residency,
            &mut arena,
        )
        .expect("plannable")
        .expect("bytes to fill");
        assert_eq!(planned.dest.as_raw(), 0xB1);
        assert_eq!(planned.head, None);
        assert_eq!(planned.tail, None);
        assert_eq!(
            planned.middle,
            Some(FillRange {
                offset: 64,
                size: 256,
                data: 0xFFFF_FFFF,
            })
        );
        // The common case must not touch the arena at all.
        assert_eq!(arena.census().allocated, 0);
    }

    #[test]
    fn a_ragged_fill_stages_both_edges_and_fills_the_interior() {
        let residency = populated();
        let mut arena = arena();
        // 65..=263: one byte before the boundary at 68, one after the last at
        // 260.
        let planned = plan_fill(
            span(65, 198),
            FillPattern::Byte(0xAB),
            &residency,
            &mut arena,
        )
        .expect("plannable")
        .expect("bytes to fill");

        let head = planned.head.expect("a head");
        let tail = planned.tail.expect("a tail");
        let middle = planned.middle.expect("an interior");

        // Exactly the guest's range, once: no byte outside it and none inside
        // it left out.
        assert_eq!(head.dest_offset, 65);
        assert_eq!(head.length, 3);
        assert_eq!(middle.offset, 68);
        assert_eq!(middle.size, 192);
        assert_eq!(tail.dest_offset, 260);
        assert_eq!(tail.length, 3);
        assert_eq!(head.dest_offset + head.length, middle.offset);
        assert_eq!(middle.offset + middle.size, tail.dest_offset);
        assert_eq!(tail.dest_offset + tail.length, 65 + 198);

        assert_eq!(head.bytes, [0xAB; 4]);
        assert_eq!(tail.bytes, [0xAB; 4]);
        assert_eq!(middle.data, 0xABAB_ABAB);

        // Six bytes of scratch whatever the size of the fill, in one
        // allocation so there is one failure point.
        assert_eq!(arena.census().allocated, 1);
        assert_eq!(head.window.chunk, tail.window.chunk);
        assert_eq!(head.window.end(), tail.window.offset);
        assert_eq!(head.window.size + tail.window.size, 6);
    }

    #[test]
    fn a_fill_too_short_to_reach_a_boundary_is_all_edge() {
        let residency = populated();
        let mut arena = arena();
        let planned = plan_fill(span(1, 2), FillPattern::Byte(7), &residency, &mut arena)
            .expect("plannable")
            .expect("bytes to fill");
        let head = planned.head.expect("a head");
        assert_eq!(head.dest_offset, 1);
        assert_eq!(head.length, 2);
        assert_eq!(planned.middle, None);
        assert_eq!(planned.tail, None);
    }

    #[test]
    fn a_ragged_four_byte_pattern_refuses_because_its_phase_is_not_established() {
        let residency = populated();
        let mut arena = arena();
        assert_eq!(
            plan_fill(
                span(1, 16),
                FillPattern::Pattern4(0x1234_5678),
                &residency,
                &mut arena,
            )
            .err(),
            Some(Refusal::RaggedPatternFill {
                offset: 1,
                length: 16,
            })
        );
        // Refused before anything was taken.
        assert_eq!(arena.census().allocated, 0);
        // And the aligned form of the same pattern is fine, which is what
        // makes this a phase question and not a pattern one.
        assert!(plan_fill(
            span(4, 16),
            FillPattern::Pattern4(0x1234_5678),
            &residency,
            &mut arena,
        )
        .is_ok());
    }

    /// The other half of the same phase question, which the refusal used to
    /// swallow: a range that *starts* aligned and ends raggedly asks nothing.
    ///
    /// Its tail begins at a four-byte boundary, so the bytes there are the
    /// pattern's leading ones and no term is being guessed at. The device this
    /// replaces fills exactly this range — it refuses on the start alone — so
    /// refusing it here would have dropped a legal fill and left the guest's
    /// buffer holding stale bytes with only a slug to show for it.
    #[test]
    fn a_four_byte_pattern_ending_raggedly_fills_its_tail_from_the_patterns_front() {
        let residency = populated();
        let mut arena = arena();
        let plan = plan_fill(
            span(4, 6),
            FillPattern::Pattern4(0x1234_5678),
            &residency,
            &mut arena,
        )
        .expect("an aligned start asks nothing")
        .expect("six bytes are filled");

        assert!(plan.head.is_none(), "the range starts aligned");
        let middle = plan.middle.expect("four bytes a native fill can do");
        assert_eq!(middle.offset, 4);
        assert_eq!(middle.size, 4);
        assert_eq!(middle.data, 0x1234_5678);

        let tail = plan.tail.expect("two bytes it cannot");
        assert_eq!(tail.dest_offset, 8, "at the boundary the middle ended on");
        assert_eq!(tail.length, 2);
        assert_eq!(
            &tail.bytes[..2],
            &0x1234_5678u32.to_le_bytes()[..2],
            "the phase at a four-byte boundary is the pattern's own front"
        );
    }

    #[test]
    fn a_fill_of_no_bytes_and_one_naming_nothing_refuse_before_the_arena() {
        let residency = populated();
        let mut arena = arena();
        // A fill of no bytes is complete rather than refused --- see
        // `a_transfer_of_no_bytes_is_complete_rather_than_refused` --- and it
        // is answered before the name is resolved, so it holds for a span
        // naming nothing too.
        for buffer in [id(BUFFER_A), id(99)] {
            assert!(plan_fill(
                BufferSpan {
                    buffer,
                    offset: 0,
                    length: 0,
                },
                FillPattern::Byte(0),
                &residency,
                &mut arena,
            )
            .expect("nothing to fill")
            .is_none());
        }
        assert_eq!(
            plan_fill(
                BufferSpan {
                    buffer: id(99),
                    offset: 1,
                    length: 8,
                },
                FillPattern::Byte(0),
                &residency,
                &mut arena,
            )
            .err(),
            Some(Refusal::Unresolved {
                miss: Miss::Unknown {
                    slot: ObjectListRef(99)
                }
            })
        );
        assert_eq!(arena.census().allocated, 0);
    }

    #[test]
    fn a_ragged_fill_with_no_scratch_left_refuses_with_the_arenas_own_reason() {
        let residency = populated();
        let mut arena = arena();
        let _ = arena.allocate(256, 1).expect("the whole chunk");
        arena.submitted(TimelinePoint(1));
        let refusal = plan_fill(span(1, 16), FillPattern::Byte(0), &residency, &mut arena)
            .expect_err("nothing left to stage in");
        assert!(matches!(
            refusal,
            Refusal::NoStaging {
                refusal: crate::staging::Refusal::Exhausted { .. }
            }
        ));
        assert!(refusal.to_string().contains("vk_staging_exhausted"));
    }

    #[test]
    fn a_mipmap_generation_is_sent_to_its_own_planner_rather_than_recorded_here() {
        let residency = populated();
        assert_eq!(
            plan(
                &BlitOp::GenerateMipmaps {
                    texture: id(IMAGE_A)
                },
                &residency
            )
            .err(),
            Some(Refusal::NotACopy {
                op: "generate_mipmaps"
            })
        );
        // And it still resolves the name first, so a generation of a resource
        // that does not exist is that refusal and not this one.
        assert!(matches!(
            plan(&BlitOp::GenerateMipmaps { texture: id(99) }, &residency),
            Err(Refusal::Unresolved { .. })
        ));
    }

    #[test]
    fn an_extent_wider_than_a_native_field_refuses_rather_than_truncating() {
        let residency = populated();
        let refusal = plan(
            &BlitOp::TextureRegion {
                source: point(IMAGE_A, 0, 0),
                dest: point(IMAGE_B, 0, 0),
                size: Size3 {
                    width: u64::from(u32::MAX) + 1,
                    height: 1,
                    depth: 1,
                },
                options: Default::default(),
            },
            &residency,
        )
        .expect_err("wider than a u32");
        assert_eq!(
            refusal,
            Refusal::ExtentTooLarge {
                axis: "width",
                value: u64::from(u32::MAX) + 1,
            }
        );
    }

    #[test]
    fn a_region_copy_carries_both_origins_and_both_subresources() {
        let residency = populated();
        let planned = plan(
            &BlitOp::TextureRegion {
                source: TexturePoint {
                    origin: Origin3 { x: 4, y: 4, z: 0 },
                    ..point(IMAGE_A, 2, 1)
                },
                dest: TexturePoint {
                    origin: Origin3 { x: 1, y: 2, z: 0 },
                    ..point(IMAGE_B, 0, 2)
                },
                size: size(8, 4),
                options: Default::default(),
            },
            &residency,
        )
        .expect("plannable")
        .expect("native work to record");
        let Command::CopyImage {
            source,
            dest,
            regions,
        } = planned
        else {
            panic!("an image copy");
        };
        assert_eq!(source.as_raw(), 0x1A);
        assert_eq!(dest.as_raw(), 0x1B);
        assert_eq!(regions[0].src_offset.x, 4);
        // Level two is 16x8, and the window ends exactly on its edge.
        assert_eq!(regions[0].src_offset.y, 4);
        assert_eq!(regions[0].dst_offset.x, 1);
        assert_eq!(regions[0].src_subresource.mip_level, 2);
        assert_eq!(regions[0].src_subresource.base_array_layer, 1);
        assert_eq!(regions[0].dst_subresource.mip_level, 0);
        assert_eq!(regions[0].dst_subresource.base_array_layer, 2);
        assert_eq!(regions[0].extent.width, 8);
        assert_eq!(regions[0].extent.height, 4);
    }

    /// The three arms of "this copy changes nothing" answer alike. The buffer
    /// one used to refuse, which would report a failure for a command the
    /// guest saw succeed --- and its refusal's own reason, that forward and
    /// backward copy orders disagree, is a statement about a shift.
    #[test]
    fn a_buffer_window_copied_onto_itself_records_nothing_rather_than_refusing() {
        let residency = populated();
        assert!(plan(
            &BlitOp::BufferToBuffer {
                source: id(BUFFER_A),
                source_offset: 64,
                dest: id(BUFFER_A),
                dest_offset: 64,
                size: 32,
            },
            &residency,
        )
        .expect("a copy onto itself is complete, not invalid")
        .is_none());
    }

    /// And the shift is still refused, at one byte either way, so the arm
    /// above did not widen into the case that has no defined result.
    #[test]
    fn a_buffer_window_that_overlaps_without_being_the_same_window_still_refuses() {
        let residency = populated();
        for (source_offset, dest_offset) in [(64, 65), (65, 64)] {
            let refusal = plan(
                &BlitOp::BufferToBuffer {
                    source: id(BUFFER_A),
                    source_offset,
                    dest: id(BUFFER_A),
                    dest_offset,
                    size: 32,
                },
                &residency,
            )
            .expect_err("a shifted self-copy has no defined result");
            assert_eq!(
                refusal,
                Refusal::OverlappingSelfCopy {
                    source_offset,
                    dest_offset,
                    size: 32,
                }
            );
        }
        // Disjoint windows of one buffer are an ordinary copy.
        assert!(plan(
            &BlitOp::BufferToBuffer {
                source: id(BUFFER_A),
                source_offset: 0,
                dest: id(BUFFER_A),
                dest_offset: 32,
                size: 32,
            },
            &residency,
        )
        .expect("disjoint")
        .is_some());
    }

    #[test]
    fn a_stale_resource_lifetime_does_not_survive_into_a_transfer() {
        let mut residency = populated();
        let mut retire = NativeRetirement::new();
        residency
            .delete(id(BUFFER_B), &mut retire)
            .expect("a live name");
        let _ = retire.reached(DeviceEpoch::FIRST, TimelinePoint(1));

        assert!(matches!(
            plan(
                &BlitOp::BufferToBuffer {
                    source: id(BUFFER_A),
                    source_offset: 0,
                    dest: id(BUFFER_B),
                    dest_offset: 0,
                    size: 4,
                },
                &residency,
            ),
            Err(Refusal::Unresolved {
                miss: Miss::Unknown { .. }
            })
        ));
    }

    #[test]
    fn every_refusal_names_itself() {
        let refusals = [
            Refusal::Unresolved {
                miss: Miss::Unknown {
                    slot: ObjectListRef(1),
                },
            },
            Refusal::RowPitchNotWholeBlocks {
                bytes_per_row: 3,
                block_bytes: 4,
            },
            Refusal::ImagePitchNotWholeRows {
                bytes_per_image: 3,
                bytes_per_row: 2,
            },
            Refusal::UnknownFormatGeometry { format: 1 },
            Refusal::RaggedPatternFill {
                offset: 1,
                length: 1,
            },
            Refusal::NoStaging {
                refusal: crate::staging::Refusal::BadAlignment { alignment: 3 },
            },
            Refusal::ExtentTooLarge {
                axis: "width",
                value: 1,
            },
            Refusal::RowPitchShorterThanCopy {
                row_length: 1,
                width: 2,
            },
            Refusal::ImagePitchShorterThanCopy {
                image_height: 1,
                height: 2,
            },
            Refusal::ImagePitchWithoutRowPitch {
                bytes_per_image: 1,
                depth: 2,
            },
            Refusal::OutsideTexture {
                axis: "level",
                named: 2,
                available: 1,
            },
            Refusal::OutsideBuffer {
                offset: 1,
                length: 2,
                size: 2,
            },
            Refusal::NotACopy { op: "x" },
        ];
        let slugs: BTreeSet<&str> = refusals.iter().map(|r| r.slug()).collect();
        assert_eq!(slugs.len(), refusals.len());
        for refusal in refusals {
            assert!(refusal.to_string().starts_with(refusal.slug()));
            assert!(refusal.slug().starts_with("vk_transfer_"));
        }
    }

    /// The failure this exists to prevent: a `bufferRowLength` shorter than the
    /// copy is wide is invalid usage, and every row after the first lands
    /// overlapping the one before it. The guest's own numbers say the copy is
    /// something other than what a widened pitch would record.
    #[test]
    fn a_row_pitch_shorter_than_the_copy_is_wide_refuses() {
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 128,
                    bytes_per_image: 0,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Err(Refusal::RowPitchShorterThanCopy {
                row_length: 32,
                width: 64,
            })
        );
        // Exactly the width is legal, and so is a wider row --- a copy of part
        // of a wider image is the ordinary reason a pitch exists.
        for bytes_per_row in [256, 512] {
            assert!(texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row,
                    bytes_per_image: 0,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            )
            .is_ok());
        }
    }

    /// `bufferImageHeight`'s half of the same rule.
    #[test]
    fn an_image_pitch_shorter_than_the_copy_is_tall_refuses() {
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        assert_eq!(
            texel_pitch(
                flat,
                ImagePitch {
                    bytes_per_row: 256,
                    bytes_per_image: 256 * 16,
                },
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Err(Refusal::ImagePitchShorterThanCopy {
                image_height: 16,
                height: 32,
            })
        );
    }

    /// A slice stride the conversion cannot reach, on the one copy shape where
    /// dropping it is visible. Coerced to "tightly packed" this planned a copy
    /// whose second slice lands wherever the packed size happens to put it,
    /// while the semantic model's own `ImagePitch::span_bytes` reserved
    /// `bytes_per_image * depth` for the same operation --- two layers
    /// disagreeing about one guest field.
    #[test]
    fn a_slice_stride_with_no_row_stride_refuses_once_a_second_slice_exists() {
        let volume = TextureShape {
            kind: TextureKind::D3.ordinal(),
            width: 64,
            height: 32,
            depth: 4,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            pixel_format: MTL_FORMAT_RGBA8_UNORM,
            usage: TextureUsage::SHADER_READ,
        }
        .checked()
        .expect("a valid declaration");
        let pitch = ImagePitch {
            bytes_per_row: 0,
            bytes_per_image: 256 * 32,
        };
        assert_eq!(
            texel_pitch(
                volume,
                pitch,
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 4,
                },
            ),
            Err(Refusal::ImagePitchWithoutRowPitch {
                bytes_per_image: 256 * 32,
                depth: 4,
            })
        );
        // One slice deep, the stride is never addressed and the tight answer
        // is exact. Refusing here would lose a copy that is entirely correct.
        assert_eq!(
            texel_pitch(
                volume,
                pitch,
                Size3 {
                    width: 64,
                    height: 32,
                    depth: 1,
                },
            ),
            Ok((0, 0))
        );
    }

    /// The failure this exists to prevent: a copy naming a level, a slice or a
    /// region the resolved texture does not have. Every one of them is invalid
    /// usage a driver may carry out anyway, against memory that belongs to
    /// something else.
    #[test]
    fn a_copy_outside_the_texture_it_names_refuses_on_the_axis_that_does_not_fit() {
        let residency = populated();
        // 64x32, four levels, three slices. Level three is 8x4.
        let cases = [
            (point(IMAGE_A, 4, 0), size(1, 1), "level", 5_u64, 4_u64),
            (point(IMAGE_A, 0, 3), size(1, 1), "slice", 4, 3),
            (point(IMAGE_A, 3, 0), size(9, 1), "x", 9, 8),
            (point(IMAGE_A, 3, 0), size(8, 5), "y", 5, 4),
            (
                TexturePoint {
                    origin: Origin3 { x: 4, y: 0, z: 0 },
                    ..point(IMAGE_A, 3, 0)
                },
                size(8, 4),
                "x",
                12,
                8,
            ),
            (
                point(IMAGE_A, 0, 0),
                Size3 {
                    width: 1,
                    height: 1,
                    depth: 2,
                },
                "z",
                2,
                1,
            ),
        ];
        for (dest, size, axis, named, available) in cases {
            assert_eq!(
                plan(
                    &BlitOp::BufferToTexture {
                        source: id(BUFFER_A),
                        source_offset: 0,
                        source_pitch: ImagePitch {
                            bytes_per_row: 0,
                            bytes_per_image: 0,
                        },
                        size,
                        dest,
                        options: Default::default(),
                    },
                    &residency,
                )
                .expect_err("outside the texture"),
                Refusal::OutsideTexture {
                    axis,
                    named,
                    available
                },
                "{axis} of {dest:?} {size:?}"
            );
        }
    }

    /// The same rule for the endpoint that is only ever a destination. A copy
    /// checked on its source alone passes whatever the destination cannot hold.
    #[test]
    fn a_region_copy_checks_the_destination_and_not_only_the_source() {
        let residency = populated();
        assert_eq!(
            plan(
                &BlitOp::TextureRegion {
                    // Level zero is 64x32 and holds this window.
                    source: point(IMAGE_A, 0, 0),
                    // Level three is 8x4 and does not.
                    dest: point(IMAGE_B, 3, 0),
                    size: size(16, 4),
                    options: Default::default(),
                },
                &residency,
            )
            .expect_err("outside the destination"),
            Refusal::OutsideTexture {
                axis: "x",
                named: 16,
                available: 8,
            }
        );
    }

    /// A level or slice *span* that runs off the end, on either endpoint,
    /// before a single region is built --- expanding first would have found
    /// only the source's.
    #[test]
    fn a_slice_span_that_runs_off_either_end_refuses_before_any_region_is_built() {
        let residency = populated();
        let top = SpanOrigin {
            texture: id(IMAGE_A),
            base_level: 0,
            base_slice: 0,
        };
        assert!(plan(
            &BlitOp::TextureSlices {
                source: top,
                dest: SpanOrigin {
                    texture: id(IMAGE_B),
                    ..top
                },
                level_count: 4,
                slice_count: 3,
            },
            &residency,
        )
        .is_ok());
        // The counts are shared, so an end runs off by starting too high ---
        // which is the only way it can, and is why one count reaching both
        // ends cannot hide a destination that does not fit.
        for (source, dest, level_count, slice_count, axis, named, available) in [
            (
                SpanOrigin {
                    base_level: 2,
                    ..top
                },
                SpanOrigin {
                    texture: id(IMAGE_B),
                    ..top
                },
                4_u16,
                3_u16,
                "level",
                6_u64,
                4_u64,
            ),
            (
                top,
                SpanOrigin {
                    texture: id(IMAGE_B),
                    base_slice: 1,
                    ..top
                },
                4,
                3,
                "slice",
                4,
                3,
            ),
        ] {
            assert_eq!(
                plan(
                    &BlitOp::TextureSlices {
                        source,
                        dest,
                        level_count,
                        slice_count,
                    },
                    &residency
                )
                .expect_err("a span off the end"),
                Refusal::OutsideTexture {
                    axis,
                    named,
                    available
                }
            );
        }
    }

    /// A destination texture whose level zero is `width` by `height`, so a
    /// slice span can be aimed at a level too small to hold what it copies.
    fn with_small_dest(width: u32, height: u32) -> Residency {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        let lifetime = Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST);
        for (slot, native) in [
            (
                IMAGE_A,
                native_image(0x1A, texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 4, 3)),
            ),
            (
                IMAGE_B,
                native_image(0x1B, texture(MTL_FORMAT_RGBA8_UNORM, width, height, 4, 3)),
            ),
        ] {
            residency
                .publish(id(slot), lifetime, native, &mut retire)
                .unwrap_or_else(|(_, e)| panic!("{e}"));
        }
        residency
    }

    /// The failure this exists to prevent: a slice span copies at the *source*
    /// level's extent, and the span check bounds only how many levels and
    /// slices each side has. Two separately declared textures can differ in
    /// size at the same level, so the destination has to be measured too ---
    /// otherwise the copy writes a 64-wide row into a 32-wide image, which is
    /// the same write past the end of an image that a region copy refuses.
    #[test]
    fn a_slice_span_into_a_smaller_destination_level_refuses() {
        let top = SpanOrigin {
            texture: id(IMAGE_A),
            base_level: 0,
            base_slice: 0,
        };
        let span = |residency: &Residency| {
            plan(
                &BlitOp::TextureSlices {
                    source: top,
                    dest: SpanOrigin {
                        texture: id(IMAGE_B),
                        ..top
                    },
                    level_count: 4,
                    slice_count: 3,
                },
                residency,
            )
        };
        assert!(
            span(&with_small_dest(64, 32)).is_ok(),
            "the same shape on both sides is what this copy is for"
        );
        assert!(
            span(&with_small_dest(128, 64)).is_ok(),
            "a destination larger than the source is a copy into part of it"
        );
        assert_eq!(
            span(&with_small_dest(32, 32)).expect_err("half as wide"),
            Refusal::OutsideTexture {
                axis: "x",
                named: 64,
                available: 32,
            }
        );
        assert_eq!(
            span(&with_small_dest(64, 16)).expect_err("half as tall"),
            Refusal::OutsideTexture {
                axis: "y",
                named: 32,
                available: 16,
            }
        );
    }

    /// The check is per level and not only on the first one: a destination
    /// whose chain is shorter than its extent suggests --- here the mismatch
    /// only appears below level zero --- is still a write past the end.
    #[test]
    fn a_slice_span_measures_every_level_it_copies() {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        let lifetime = Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST);
        for (slot, native) in [
            (
                IMAGE_A,
                native_image(0x1A, texture(MTL_FORMAT_RGBA8_UNORM, 64, 8, 4, 1)),
            ),
            (
                IMAGE_B,
                native_image(0x1B, texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 4, 1)),
            ),
        ] {
            residency
                .publish(id(slot), lifetime, native, &mut retire)
                .unwrap_or_else(|(_, e)| panic!("{e}"));
        }
        // Level zero fits --- 64x8 into 64x32 --- and so do levels one and
        // two. Level three is 8x1 in the source and 8x4 in the destination, so
        // a check that stopped at the first level would pass this and a check
        // that runs on all four still passes it.
        let ok = SpanOrigin {
            texture: id(IMAGE_A),
            base_level: 0,
            base_slice: 0,
        };
        assert!(plan(
            &BlitOp::TextureSlices {
                source: ok,
                dest: SpanOrigin {
                    texture: id(IMAGE_B),
                    ..ok
                },
                level_count: 4,
                slice_count: 1,
            },
            &residency,
        )
        .is_ok());
        // Aim the source's level zero at the destination's level two: 64x8
        // into 16x8, which level zero's own comparison never sees.
        assert_eq!(
            plan(
                &BlitOp::TextureSlices {
                    source: ok,
                    dest: SpanOrigin {
                        texture: id(IMAGE_B),
                        base_level: 2,
                        ..ok
                    },
                    level_count: 2,
                    slice_count: 1,
                },
                &residency,
            )
            .expect_err("level zero into level two"),
            Refusal::OutsideTexture {
                axis: "x",
                named: 64,
                available: 16,
            }
        );
    }

    /// A buffer whose length is exactly the fixture's, so a bound can be
    /// written against a number instead of against `1 << 20`.
    fn sized_buffer(handle: u64, size: u64) -> Native {
        Native::Buffer(NativeBuffer {
            buffer: vk::Buffer::from_raw(handle),
            memory: vk::DeviceMemory::from_raw(handle),
            plan: BufferPlan {
                size,
                usage: EVERY_CLASS,
                aliased: false,
            },
        })
    }

    fn with_small_buffer(size: u64) -> Residency {
        let mut residency = Residency::new();
        let mut retire = NativeRetirement::new();
        let lifetime = Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST);
        for (slot, native) in [
            (BUFFER_A, sized_buffer(0xB1, size)),
            (BUFFER_B, native_buffer(0xB2)),
            (
                IMAGE_A,
                native_image(0x1A, texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 4, 3)),
            ),
        ] {
            residency
                .publish(id(slot), lifetime, native, &mut retire)
                .unwrap_or_else(|(_, e)| panic!("{e}"));
        }
        residency
    }

    /// The failure this exists to prevent: a copy whose buffer window ends past
    /// the buffer. Past it the driver reaches whatever the allocator put next,
    /// which for a suballocated buffer is another guest resource.
    #[test]
    fn a_buffer_copy_past_the_end_of_its_buffer_refuses() {
        let residency = with_small_buffer(256);
        let copy = |source_offset, size| {
            plan(
                &BlitOp::BufferToBuffer {
                    source: id(BUFFER_A),
                    source_offset,
                    dest: id(BUFFER_B),
                    dest_offset: 0,
                    size,
                },
                &residency,
            )
        };
        assert!(copy(192, 64).is_ok(), "ending exactly on the end is inside");
        assert_eq!(
            copy(192, 65).expect_err("one byte past"),
            Refusal::OutsideBuffer {
                offset: 192,
                length: 65,
                size: 256,
            }
        );
        // The wrap a plain addition would have turned into a copy that fits.
        assert_eq!(
            copy(u64::MAX, 8).expect_err("wrapped"),
            Refusal::OutsideBuffer {
                offset: u64::MAX,
                length: 8,
                size: 256,
            }
        );
    }

    /// A fill is not a copy and had no bound at all --- a `fillBuffer:range:`
    /// past the end writes the pattern into whatever follows.
    #[test]
    fn a_fill_past_the_end_of_its_buffer_refuses() {
        let residency = with_small_buffer(256);
        let mut arena = arena();
        assert_eq!(
            plan_fill(
                BufferSpan {
                    buffer: id(BUFFER_A),
                    offset: 128,
                    length: 256,
                },
                FillPattern::Byte(0xAB),
                &residency,
                &mut arena,
            )
            .expect_err("past the end"),
            Refusal::OutsideBuffer {
                offset: 128,
                length: 256,
                size: 256,
            }
        );
        assert!(plan_fill(
            BufferSpan {
                buffer: id(BUFFER_A),
                offset: 128,
                length: 128,
            },
            FillPattern::Byte(0xAB),
            &residency,
            &mut arena,
        )
        .is_ok());
    }

    /// A buffer-image copy reaches as far as its *pitch* says, not as far as
    /// its texels occupy. Bounding by the packed size would admit a copy whose
    /// row pitch walks it off the end --- which is the shape a guest sending a
    /// window into a larger image naturally produces.
    #[test]
    fn a_buffer_image_copy_is_bounded_by_the_footprint_the_pitch_addresses() {
        let flat = texture(MTL_FORMAT_RGBA8_UNORM, 64, 32, 1, 1);
        // Eight rows of 8 texels each, described with a 64-texel row: the last
        // row starts 7*256 bytes in and is 32 bytes long.
        let windowed = ImagePitch {
            bytes_per_row: 256,
            bytes_per_image: 0,
        };
        assert_eq!(region_bytes(flat, windowed, size(8, 8)), Ok(7 * 256 + 32));
        // The same copy packed is eight rows of 32 bytes.
        assert_eq!(
            region_bytes(
                flat,
                ImagePitch {
                    bytes_per_row: 0,
                    bytes_per_image: 0,
                },
                size(8, 8),
            ),
            Ok(8 * 32)
        );
        // BC3 is a 4x4 block of sixteen bytes: an 8x8 copy is two rows of two
        // blocks, and a 64-texel row is sixteen blocks.
        let compressed = texture(MTL_FORMAT_BC3_RGBA, 64, 32, 1, 1);
        assert_eq!(
            region_bytes(compressed, windowed, size(8, 8)),
            Ok(16 * 16 + 2 * 16)
        );

        let residency = with_small_buffer(7 * 256 + 32);
        let copy = |offset| {
            plan(
                &BlitOp::BufferToTexture {
                    source: id(BUFFER_A),
                    source_offset: offset,
                    source_pitch: windowed,
                    size: size(8, 8),
                    dest: point(IMAGE_A, 0, 0),
                    options: Default::default(),
                },
                &residency,
            )
        };
        assert!(copy(0).is_ok(), "the footprint is exactly the buffer");
        assert_eq!(
            copy(1).expect_err("one byte past"),
            Refusal::OutsideBuffer {
                offset: 1,
                length: 7 * 256 + 32,
                size: 7 * 256 + 32,
            }
        );
    }

    // ---- The bounds, driven against an independently shaped shadow -------
    //
    // Every check `plan` makes about whether a transfer fits is arithmetic, and
    // arithmetic that agrees with itself is the failure mode a hand-written
    // case cannot see. So the footprint is checked against a *brute force* ---
    // every block the region addresses, enumerated, and the furthest one taken
    // --- and acceptance is checked against a shadow that re-derives the bound
    // from the texture's declared shape rather than from anything `endpoint`
    // returned.

    struct Rng(u64);

    impl Rng {
        const fn new(seed: u64) -> Self {
            Self(seed ^ 0x9E37_79B9_7F4A_7C15)
        }

        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> u64 {
            if bound == 0 {
                return 0;
            }
            self.next() % bound
        }
    }

    /// The last byte a buffer-image region addresses, found by visiting every
    /// block it names rather than by a closed form.
    ///
    /// This is the spec's addressing read literally: block `(x, y, z)` of the
    /// copy sits at `((z * rows_per_image + y) * blocks_per_row + x)` blocks
    /// from the buffer offset. Slow, obviously right, and shaped nothing like
    /// [`region_bytes`].
    fn brute_footprint(
        block_width: u32,
        block_height: u32,
        block_bytes: u32,
        row_texels: u64,
        image_rows: u64,
        size: Size3,
    ) -> u64 {
        if size.width == 0 || size.height == 0 || size.depth == 0 {
            return 0;
        }
        let ceil = |value: u64, by: u32| value.div_ceil(u64::from(by));
        let blocks_per_row = ceil(row_texels, block_width);
        let rows_per_image = ceil(image_rows, block_height);
        let mut furthest = 0;
        for z in 0..size.depth {
            for y in 0..ceil(size.height, block_height) {
                for x in 0..ceil(size.width, block_width) {
                    let block = (z * rows_per_image + y) * blocks_per_row + x;
                    furthest = furthest.max((block + 1) * u64::from(block_bytes));
                }
            }
        }
        furthest
    }

    #[test]
    fn the_footprint_closed_form_is_the_furthest_block_the_region_addresses() {
        let mut rng = Rng::new(7);
        let mut nonzero_pitches = 0_u32;
        for _ in 0..4_000 {
            let (format, block_width, block_height, block_bytes) = if rng.below(2) == 0 {
                (MTL_FORMAT_RGBA8_UNORM, 1, 1, 4)
            } else {
                (MTL_FORMAT_BC3_RGBA, 4, 4, 16)
            };
            // Wide enough that a level exists for every copy below.
            let texture = texture(format, 64, 64, 1, 1);
            let size = Size3 {
                width: rng.below(17),
                height: rng.below(17),
                depth: 1 + rng.below(3),
            };
            // A byte pitch is only convertible in whole blocks, so it is built
            // out of them rather than filtered for afterwards.
            let row_blocks = rng.below(9);
            let bytes_per_row = row_blocks * u64::from(block_bytes);
            let image_block_rows = rng.below(9);
            let bytes_per_image = image_block_rows * bytes_per_row;
            let pitch = ImagePitch {
                bytes_per_row,
                bytes_per_image,
            };
            let Ok(bytes) = region_bytes(texture, pitch, size) else {
                continue;
            };
            let (row_length, image_height) =
                texel_pitch(texture, pitch, size).expect("region_bytes just converted it");
            if row_length != 0 {
                nonzero_pitches += 1;
            }
            let row_texels = if row_length == 0 {
                size.width
            } else {
                u64::from(row_length)
            };
            let image_rows = if image_height == 0 {
                size.height
            } else {
                u64::from(image_height)
            };
            assert_eq!(
                bytes,
                brute_footprint(
                    block_width,
                    block_height,
                    block_bytes,
                    row_texels,
                    image_rows,
                    size
                ),
                "format={format:x} {size:?} {pitch:?}"
            );
        }
        assert!(nonzero_pitches > 500, "{nonzero_pitches}");
    }

    /// What the shadow believes about one texture, derived from the declared
    /// shape alone.
    fn level_of(texture: Texture, level: u16) -> Option<(u64, u64, u64)> {
        texture
            .level_extent(u32::from(level))
            .map(|e| (u64::from(e.x), u64::from(e.y), u64::from(e.z)))
    }

    /// A driven sweep of every copy shape against a shadow that re-derives both
    /// bounds. `plan` accepts exactly the transfers the shadow finds nothing
    /// wrong with, and refuses the rest by the reason the shadow names.
    #[test]
    fn a_transfer_is_planned_exactly_when_both_of_its_ends_hold_it() {
        let mut tally = (0_u32, 0_u32, 0_u32, 0_u32, 0_u32);
        for seed in 0..1_500_u64 {
            let mut rng = Rng::new(seed);
            let shape = texture(MTL_FORMAT_RGBA8_UNORM, 32, 16, 3, 2);
            // Varied per seed: a buffer large enough for every copy never
            // drives the buffer bound, and one too small for any never drives
            // the accepted path.
            let buffer_bytes = 128 + rng.below(2_048);
            let mut residency = Residency::new();
            let mut retire = NativeRetirement::new();
            let lifetime = Lifetime::new(SessionGeneration::FIRST, DeviceEpoch::FIRST);
            for (slot, native) in [
                (BUFFER_A, sized_buffer(0xB1, buffer_bytes)),
                (IMAGE_A, native_image(0x1A, shape)),
            ] {
                residency
                    .publish(id(slot), lifetime, native, &mut retire)
                    .unwrap_or_else(|(_, e)| panic!("{e}"));
            }

            let level = rng.below(4) as u16;
            let slice = rng.below(3) as u16;
            // Draw the window inside the level three times in four. Unsteered,
            // the sweep is nine refusals to one accepted copy and the accepted
            // path --- the one the bounds must not break --- goes undriven.
            let (lx, ly) = level_of(shape, level).map_or((32, 16), |(x, y, _)| (x, y));
            let mut window = |available: u64, wide: u64| {
                if rng.below(4) == 0 {
                    (rng.below(wide), rng.below(wide))
                } else {
                    let origin = rng.below(available + 1);
                    (origin, rng.below(available - origin + 1))
                }
            };
            let (origin_x, width) = window(lx, 34);
            let (origin_y, height) = window(ly, 18);
            let point = TexturePoint {
                texture: id(IMAGE_A),
                slice,
                level,
                origin: Origin3 {
                    x: origin_x,
                    y: origin_y,
                    z: 0,
                },
            };
            let size = Size3 {
                width,
                height,
                depth: 1,
            };
            // A row pitch that reaches the copy most of the time, so the
            // footprint is exercised rather than the pitch refusal.
            let pitch = ImagePitch {
                bytes_per_row: match rng.below(4) {
                    0 => 0,
                    1 => rng.below(9) * 4,
                    _ => (width + rng.below(4)) * 4,
                },
                bytes_per_image: 0,
            };
            let buffer_offset = if rng.below(4) == 0 {
                rng.below(buffer_bytes + 64)
            } else {
                rng.below(512)
            };

            let planned = plan(
                &BlitOp::BufferToTexture {
                    source: id(BUFFER_A),
                    source_offset: buffer_offset,
                    source_pitch: pitch,
                    size,
                    dest: point,
                    options: Default::default(),
                },
                &residency,
            );

            // The shadow: the texture's own declared shape, and the buffer's
            // own declared length. Nothing here calls `endpoint`, `texel_pitch`
            // or `within_buffer`.
            let fits_texture = level_of(shape, level).is_some_and(|(x, y, z)| {
                u64::from(slice) < u64::from(shape.layers())
                    && point.origin.x + size.width <= x
                    && point.origin.y + size.height <= y
                    && point.origin.z + size.depth <= z
            });
            let row_texels = if pitch.bytes_per_row == 0 {
                size.width
            } else {
                pitch.bytes_per_row / 4
            };
            let footprint = brute_footprint(1, 1, 4, row_texels, size.height, size);
            let fits_buffer = buffer_offset + footprint <= buffer_bytes;
            let pitch_reaches = pitch.bytes_per_row == 0 || row_texels >= size.width;
            // A copy with a zero axis has no native extent, so it is neither
            // inside the texture nor outside it --- it is not a copy.
            let nonempty = size.width != 0 && size.height != 0 && size.depth != 0;

            match planned {
                Ok(None) => {
                    assert!(!nonempty, "answered a copy that has texels as done");
                    tally.4 += 1;
                }
                Ok(Some(_)) => {
                    assert!(nonempty, "planned a copy of no texels");
                    assert!(fits_texture, "planned a copy outside the texture");
                    assert!(fits_buffer, "planned a copy outside the buffer");
                    assert!(
                        pitch_reaches,
                        "planned a copy its row pitch cannot describe"
                    );
                    tally.0 += 1;
                }
                Err(Refusal::OutsideTexture { .. }) => {
                    assert!(!fits_texture, "refused a copy the texture holds");
                    tally.1 += 1;
                }
                Err(Refusal::OutsideBuffer { .. }) => {
                    assert!(!fits_buffer, "refused a copy the buffer holds");
                    // The texture is checked after the buffer only because the
                    // buffer needs the footprint the pitch gives; either
                    // refusal is a correct answer when both hold.
                    tally.2 += 1;
                }
                Err(Refusal::RowPitchShorterThanCopy { .. }) => {
                    assert!(!pitch_reaches);
                    tally.3 += 1;
                }
                Err(other) => panic!("{other}"),
            }
        }
        let (ok, outside_texture, outside_buffer, short_pitch, empty) = tally;
        // Floors per path, not one aggregate: the first driver here reached
        // the accepted path fifteen times in fifteen hundred, which reads as
        // covered and is not.
        assert!(ok > 200, "{ok}");
        assert!(outside_texture > 400, "{outside_texture}");
        assert!(outside_buffer > 100, "{outside_buffer}");
        assert!(short_pitch > 100, "{short_pitch}");
        assert!(empty > 20, "{empty}");
    }
}
