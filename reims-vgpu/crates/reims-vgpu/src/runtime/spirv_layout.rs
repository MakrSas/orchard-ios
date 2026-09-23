//! Structural repair for LLVM/AIR aggregate layout lost during SPIR-V emission.
//!
//! LLVM vectors have a store size and an allocation size.  A three-lane vector
//! can therefore store three bytes while occupying four bytes according to the
//! module data layout (`v24:32:32`).  metal2vulkan currently advances the next
//! struct member by the store size.  This pass reconciles existing SPIR-V
//! `Offset` decorations with the source module's vector allocation alignment.
//! It uses only type structure and the LLVM data-layout contract; shader names,
//! pipeline ids, resource ids, dimensions, and content never participate.

use std::collections::HashMap;

use crate::observe::Decline;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_STRUCT: u16 = 30;
const OP_MEMBER_DECORATE: u16 = 72;
const DECORATION_OFFSET: u32 = 35;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LayoutRepairStats {
    pub structs: usize,
    pub members: usize,
}

/// A structural refusal while reconciling AIR vector allocation layout with
/// emitted SPIR-V member offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpirvLayoutDecline {
    DataLayoutVectorAlignmentMissing,
    SpirvLengthMisaligned {
        len: usize,
    },
    SpirvHeaderInvalid {
        words: usize,
        magic: Option<u32>,
    },
    VectorWidthOverflow {
        type_id: u32,
        component_bits: u32,
        count: u32,
    },
    TypeVectorAlignmentMissing {
        type_id: u32,
        width: u32,
    },
    AllocationRoundUpOverflow {
        type_id: u32,
        store_size: u32,
        alignment: u32,
    },
    InstructionMalformed {
        word_index: usize,
        word_count: usize,
        words: usize,
    },
    DuplicateMemberOffset {
        struct_id: u32,
        member: u32,
    },
    InitialMemberOffsetOverflow {
        struct_id: u32,
        member: usize,
    },
    FollowingMemberOffsetOverflow {
        struct_id: u32,
        member: usize,
    },
}

impl Decline for SpirvLayoutDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::DataLayoutVectorAlignmentMissing => {
                "spirv_layout_datalayout_vector_alignment_missing"
            }
            Self::SpirvLengthMisaligned { .. } => "spirv_layout_length_misaligned",
            Self::SpirvHeaderInvalid { .. } => "spirv_layout_header_invalid",
            Self::VectorWidthOverflow { .. } => "spirv_layout_vector_width_overflow",
            Self::TypeVectorAlignmentMissing { .. } => "spirv_layout_type_vector_alignment_missing",
            Self::AllocationRoundUpOverflow { .. } => "spirv_layout_allocation_round_up_overflow",
            Self::InstructionMalformed { .. } => "spirv_layout_instruction_malformed",
            Self::DuplicateMemberOffset { .. } => "spirv_layout_duplicate_member_offset",
            Self::InitialMemberOffsetOverflow { .. } => {
                "spirv_layout_initial_member_offset_overflow"
            }
            Self::FollowingMemberOffsetOverflow { .. } => {
                "spirv_layout_following_member_offset_overflow"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::DataLayoutVectorAlignmentMissing => Vec::new(),
            Self::SpirvLengthMisaligned { len } => vec![("len", len.to_string())],
            Self::SpirvHeaderInvalid { words, magic } => vec![
                ("words", words.to_string()),
                (
                    "magic",
                    magic.map_or_else(|| "none".to_string(), |value| format!("{value:#x}")),
                ),
            ],
            Self::VectorWidthOverflow {
                type_id,
                component_bits,
                count,
            } => vec![
                ("type_id", type_id.to_string()),
                ("component_bits", component_bits.to_string()),
                ("count", count.to_string()),
            ],
            Self::TypeVectorAlignmentMissing { type_id, width } => vec![
                ("type_id", type_id.to_string()),
                ("width", width.to_string()),
            ],
            Self::AllocationRoundUpOverflow {
                type_id,
                store_size,
                alignment,
            } => vec![
                ("type_id", type_id.to_string()),
                ("store_size", store_size.to_string()),
                ("alignment", alignment.to_string()),
            ],
            Self::InstructionMalformed {
                word_index,
                word_count,
                words,
            } => vec![
                ("word_index", word_index.to_string()),
                ("word_count", word_count.to_string()),
                ("words", words.to_string()),
            ],
            Self::DuplicateMemberOffset { struct_id, member } => vec![
                ("struct_id", struct_id.to_string()),
                ("member", member.to_string()),
            ],
            Self::InitialMemberOffsetOverflow { struct_id, member }
            | Self::FollowingMemberOffsetOverflow { struct_id, member } => vec![
                ("struct_id", struct_id.to_string()),
                ("member", member.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(SpirvLayoutDecline);

impl std::error::Error for SpirvLayoutDecline {}

/// Parse vector ABI alignments straight from the raw `target datalayout` VALUE (the string inside
/// the quotes, e.g. `e-p:32:32-...-v24:32:32-...`). This is what `metal2vulkan`'s
/// `ShaderReflection.datalayout` already carries — the single source of truth — so a consumer never
/// re-reads the source `.ll` from disk to recover it.
fn data_layout_vector_alignments_from_value(
    layout: &str,
) -> Result<HashMap<u32, u32>, SpirvLayoutDecline> {
    let alignments: HashMap<u32, u32> = layout
        .split('-')
        .filter_map(|field| {
            let spec = field.strip_prefix('v')?;
            let mut parts = spec.split(':');
            let width = parts.next()?.parse::<u32>().ok()?;
            let abi_align_bits = parts.next()?.parse::<u32>().ok()?;
            (width > 0 && abi_align_bits > 0 && abi_align_bits % 8 == 0)
                .then_some((width, abi_align_bits / 8))
        })
        .collect();
    if alignments.is_empty() {
        return Err(SpirvLayoutDecline::DataLayoutVectorAlignmentMissing);
    }
    Ok(alignments)
}

fn words_from_bytes(bytes: &[u8]) -> Result<Vec<u32>, SpirvLayoutDecline> {
    if !bytes.len().is_multiple_of(4) {
        return Err(SpirvLayoutDecline::SpirvLengthMisaligned { len: bytes.len() });
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words.len() < HEADER_WORDS || words[0] != SPIRV_MAGIC {
        return Err(SpirvLayoutDecline::SpirvHeaderInvalid {
            words: words.len(),
            magic: words.first().copied(),
        });
    }
    Ok(words)
}

fn bytes_from_words(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for word in words {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

fn type_alloc_size(
    ty: u32,
    scalar_widths: &HashMap<u32, u32>,
    vectors: &HashMap<u32, (u32, u32)>,
    vector_alignments: &HashMap<u32, u32>,
) -> Result<Option<u32>, SpirvLayoutDecline> {
    if let Some(width) = scalar_widths.get(&ty).copied() {
        return Ok((width % 8 == 0).then_some(width / 8));
    }
    let Some((component, count)) = vectors.get(&ty).copied() else {
        return Ok(None);
    };
    let Some(component_bits) = scalar_widths.get(&component).copied() else {
        return Ok(None);
    };
    let Some(vector_bits) = component_bits.checked_mul(count) else {
        return Err(SpirvLayoutDecline::VectorWidthOverflow {
            type_id: ty,
            component_bits,
            count,
        });
    };
    if vector_bits % 8 != 0 {
        return Ok(None);
    }
    let store_size = vector_bits / 8;
    let Some(align) = vector_alignments.get(&vector_bits).copied() else {
        return Err(SpirvLayoutDecline::TypeVectorAlignmentMissing {
            type_id: ty,
            width: vector_bits,
        });
    };
    let allocation =
        store_size
            .checked_add(align - 1)
            .ok_or(SpirvLayoutDecline::AllocationRoundUpOverflow {
                type_id: ty,
                store_size,
                alignment: align,
            })?;
    Ok(Some(allocation / align * align))
}

/// Repair struct-member offsets that advance past an LLVM vector by its store
/// size instead of its allocation size, using the raw `target datalayout` VALUE
/// (`metal2vulkan::reflect::ShaderReflection.datalayout`) as the single source of
/// truth for vector ABI alignment — no source `.ll` on disk to re-read.
pub fn repair_llvm_vector_alloc_offsets_from_datalayout(
    datalayout: &str,
    spirv: &[u8],
) -> Result<(Vec<u8>, LayoutRepairStats), SpirvLayoutDecline> {
    repair_with_alignments(data_layout_vector_alignments_from_value(datalayout)?, spirv)
}

fn repair_with_alignments(
    vector_alignments: HashMap<u32, u32>,
    spirv: &[u8],
) -> Result<(Vec<u8>, LayoutRepairStats), SpirvLayoutDecline> {
    let mut words = words_from_bytes(spirv)?;
    let mut scalar_widths = HashMap::<u32, u32>::new();
    let mut vectors = HashMap::<u32, (u32, u32)>::new();
    let mut structs = HashMap::<u32, Vec<u32>>::new();
    let mut offsets = HashMap::<(u32, u32), (u32, usize)>::new();

    let mut i = HEADER_WORDS;
    while i < words.len() {
        let word0 = words[i];
        let word_count = (word0 >> 16) as usize;
        let opcode = (word0 & 0xffff) as u16;
        if word_count == 0 || i + word_count > words.len() {
            return Err(SpirvLayoutDecline::InstructionMalformed {
                word_index: i,
                word_count,
                words: words.len(),
            });
        }
        match opcode {
            OP_TYPE_INT | OP_TYPE_FLOAT if word_count >= 3 => {
                scalar_widths.insert(words[i + 1], words[i + 2]);
            }
            OP_TYPE_VECTOR if word_count >= 4 => {
                vectors.insert(words[i + 1], (words[i + 2], words[i + 3]));
            }
            OP_TYPE_STRUCT if word_count >= 2 => {
                structs.insert(words[i + 1], words[i + 2..i + word_count].to_vec());
            }
            OP_MEMBER_DECORATE if word_count >= 5 && words[i + 3] == DECORATION_OFFSET => {
                let key = (words[i + 1], words[i + 2]);
                if offsets.insert(key, (words[i + 4], i + 4)).is_some() {
                    return Err(SpirvLayoutDecline::DuplicateMemberOffset {
                        struct_id: key.0,
                        member: key.1,
                    });
                }
            }
            _ => {}
        }
        i += word_count;
    }

    let mut stats = LayoutRepairStats::default();
    for (struct_id, member_types) in structs {
        if member_types.len() < 2 {
            continue;
        }
        let original: Vec<Option<(u32, usize)>> = (0..member_types.len())
            .map(|member| offsets.get(&(struct_id, member as u32)).copied())
            .collect();
        // Ordinary SSA structs have no Offset decorations.  Partially laid-out
        // structs are left to spirv-val/translation failure rather than guessed.
        if original.iter().any(Option::is_none) {
            continue;
        }

        let mut desired: Vec<u32> = original
            .iter()
            .map(|offset| offset.expect("checked above").0)
            .collect();
        for member in 0..member_types.len() - 1 {
            let Some((component, count)) = vectors.get(&member_types[member]).copied() else {
                continue;
            };
            let Some(component_bits) = scalar_widths.get(&component).copied() else {
                continue;
            };
            let Some(vector_bits) = component_bits.checked_mul(count) else {
                continue;
            };
            if vector_bits % 8 != 0 {
                continue;
            }
            let store_size = vector_bits / 8;
            let Some(alloc_size) = type_alloc_size(
                member_types[member],
                &scalar_widths,
                &vectors,
                &vector_alignments,
            )?
            else {
                continue;
            };
            let old_offset = original[member].expect("checked above").0;
            let next_offset = original[member + 1].expect("checked above").0;
            if alloc_size <= store_size || next_offset != old_offset.saturating_add(store_size) {
                continue;
            }

            let mut required = desired[member]
                .checked_add(alloc_size)
                .ok_or(SpirvLayoutDecline::InitialMemberOffsetOverflow { struct_id, member })?;
            for next in member + 1..member_types.len() {
                // A later explicit/aligned offset can absorb the missing byte;
                // do not shift it (or anything after it) a second time.
                if desired[next] >= required {
                    break;
                }
                desired[next] = required;
                let Some(size) = type_alloc_size(
                    member_types[next],
                    &scalar_widths,
                    &vectors,
                    &vector_alignments,
                )?
                else {
                    break;
                };
                required = desired[next].checked_add(size).ok_or(
                    SpirvLayoutDecline::FollowingMemberOffsetOverflow {
                        struct_id,
                        member: next,
                    },
                )?;
            }
        }

        let mut changed = 0usize;
        for (member, new_offset) in desired.into_iter().enumerate() {
            let (old_offset, word_index) = original[member].expect("checked above");
            if new_offset != old_offset {
                words[word_index] = new_offset;
                changed += 1;
            }
        }
        if changed != 0 {
            stats.structs += 1;
            stats.members += changed;
        }
    }

    Ok((bytes_from_words(&words), stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: u16, operands: &[u32], words: &mut Vec<u32>) {
        words.push((((operands.len() + 1) as u32) << 16) | u32::from(op));
        words.extend_from_slice(operands);
    }

    fn module(offsets: &[u32]) -> Vec<u8> {
        let mut words = vec![SPIRV_MAGIC, 0x0001_0600, 0, 32, 0];
        inst(OP_TYPE_INT, &[1, 8, 0], &mut words);
        inst(OP_TYPE_VECTOR, &[2, 1, 3], &mut words);
        inst(OP_TYPE_STRUCT, &[3, 1, 1, 1, 1, 2, 1, 1, 1], &mut words);
        for (member, offset) in offsets.iter().copied().enumerate() {
            inst(
                OP_MEMBER_DECORATE,
                &[3, member as u32, DECORATION_OFFSET, offset],
                &mut words,
            );
        }
        bytes_from_words(&words)
    }

    fn member_offsets(bytes: &[u8]) -> Vec<u32> {
        let words = words_from_bytes(bytes).unwrap();
        let mut out = vec![];
        let mut i = HEADER_WORDS;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            if (words[i] & 0xffff) as u16 == OP_MEMBER_DECORATE {
                out.push(words[i + 4]);
            }
            i += wc;
        }
        out
    }

    #[test]
    fn repairs_v3i8_following_members_from_llvm_allocation_size() {
        let layout = "e-p:64:64-v24:32:32-n8:16:32:64";
        let input = module(&[0, 1, 2, 3, 4, 7, 8, 9]);
        let (output, stats) =
            repair_llvm_vector_alloc_offsets_from_datalayout(layout, &input).unwrap();
        assert_eq!(member_offsets(&output), vec![0, 1, 2, 3, 4, 8, 9, 10]);
        assert_eq!(
            stats,
            LayoutRepairStats {
                structs: 1,
                members: 3
            }
        );
    }

    #[test]
    fn leaves_already_allocated_vector_and_unrelated_layout_unchanged() {
        let input = module(&[0, 1, 2, 3, 4, 8, 9, 10]);
        let (output, stats) =
            repair_llvm_vector_alloc_offsets_from_datalayout("e-p:64:64-v24:32:32", &input)
                .unwrap();
        assert_eq!(output, input);
        assert_eq!(stats, LayoutRepairStats::default());

        // A datalayout with no vector spec at all fails visibly (no guess).
        assert_eq!(
            repair_llvm_vector_alloc_offsets_from_datalayout("e-p:64:64", &input).unwrap_err(),
            SpirvLayoutDecline::DataLayoutVectorAlignmentMissing
        );
        // A datalayout whose vector spec is for a different width also fails visibly.
        assert_eq!(
            repair_llvm_vector_alloc_offsets_from_datalayout("e-p:64:64-v32:32:32", &input)
                .unwrap_err(),
            SpirvLayoutDecline::TypeVectorAlignmentMissing {
                type_id: 2,
                width: 24
            }
        );

        let scalar_widths = HashMap::from([(1, u32::MAX)]);
        let vectors = HashMap::from([(2, (1, 3))]);
        assert_eq!(
            type_alloc_size(2, &scalar_widths, &vectors, &HashMap::new()).unwrap_err(),
            SpirvLayoutDecline::VectorWidthOverflow {
                type_id: 2,
                component_bits: u32::MAX,
                count: 3
            }
        );
    }

    #[test]
    fn layout_declines_have_distinct_log_safe_reasons_and_fields() {
        let cases = [
            SpirvLayoutDecline::DataLayoutVectorAlignmentMissing,
            SpirvLayoutDecline::SpirvLengthMisaligned { len: 7 },
            SpirvLayoutDecline::SpirvHeaderInvalid {
                words: 1,
                magic: Some(0),
            },
            SpirvLayoutDecline::VectorWidthOverflow {
                type_id: 2,
                component_bits: u32::MAX,
                count: 3,
            },
            SpirvLayoutDecline::TypeVectorAlignmentMissing {
                type_id: 2,
                width: 24,
            },
            SpirvLayoutDecline::AllocationRoundUpOverflow {
                type_id: 2,
                store_size: u32::MAX,
                alignment: 4,
            },
            SpirvLayoutDecline::InstructionMalformed {
                word_index: 5,
                word_count: 9,
                words: 8,
            },
            SpirvLayoutDecline::DuplicateMemberOffset {
                struct_id: 3,
                member: 1,
            },
            SpirvLayoutDecline::InitialMemberOffsetOverflow {
                struct_id: 3,
                member: 1,
            },
            SpirvLayoutDecline::FollowingMemberOffsetOverflow {
                struct_id: 3,
                member: 2,
            },
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in cases {
            assert!(slugs.insert(decline.slug()), "duplicate {}", decline.slug());
            assert!(decline
                .slug()
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_'));
            for (_, value) in decline.fields() {
                assert!(!value.contains(char::is_whitespace));
            }
        }
    }

    #[test]
    fn malformed_spirv_fires_the_typed_parse_sites() {
        assert_eq!(
            words_from_bytes(&[0; 3]).unwrap_err(),
            SpirvLayoutDecline::SpirvLengthMisaligned { len: 3 }
        );
        assert_eq!(
            words_from_bytes(&0u32.to_le_bytes()).unwrap_err(),
            SpirvLayoutDecline::SpirvHeaderInvalid {
                words: 1,
                magic: Some(0)
            }
        );

        let mut malformed = vec![SPIRV_MAGIC, 0, 0, 0, 0];
        malformed.push((8u32 << 16) | u32::from(OP_TYPE_INT));
        assert_eq!(
            repair_with_alignments(HashMap::from([(24, 4)]), &bytes_from_words(&malformed))
                .unwrap_err(),
            SpirvLayoutDecline::InstructionMalformed {
                word_index: HEADER_WORDS,
                word_count: 8,
                words: HEADER_WORDS + 1
            }
        );
    }

    #[test]
    fn later_alignment_gap_absorbs_vector_allocation_delta() {
        let input = module(&[0, 1, 2, 3, 4, 7, 8, 16]);
        let (output, stats) =
            repair_llvm_vector_alloc_offsets_from_datalayout("e-p:64:64-v24:32:32", &input)
                .unwrap();
        assert_eq!(member_offsets(&output), vec![0, 1, 2, 3, 4, 8, 9, 16]);
        assert_eq!(
            stats,
            LayoutRepairStats {
                structs: 1,
                members: 2
            }
        );
    }
}
