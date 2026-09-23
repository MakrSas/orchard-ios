//! The present packet's trailer: which of the three present commands a packet
//! is, and where its words are.
//!
//! # Three commands, not one shape with a variant flag
//!
//! They present the same frame to the same host window, and it is tempting to
//! fold them. Their trailers say otherwise, and say it in a way that punishes a
//! fold:
//!
//! - `CmdDisplayTransaction2_DEPRECATED` — `[pipe][surface][task]`, 12 bytes.
//! - `CmdDisplayTransaction3` — `[pipe][task][surface][gamma…]`, 36 bytes. The
//!   first two words are swapped relative to the form above, and a gamma table
//!   follows. They are separate commands with separate handlers, not one
//!   command with a variant flag, which is why the trailer differs at all.
//! - `CmdDisplaySwapMapping` — `[display][_][mapping]`, 12 bytes. This one names
//!   a single mapping instead of serializing a transaction, so its target word
//!   is at offset 8, *not* at the first form's offset 4, and **it has no task
//!   field at all**. Reading one at the first form's slot returns the
//!   unidentified word between the display index and the mapping.
//!
//! So the form travels with the packet and the offsets come from the form. A
//! single decoder with an opcode-keyed offset table is that fold with extra
//! steps; this is the table, stated once, with the reason each entry differs.
//!
//! # There is no plane list on the wire
//!
//! A present is an `IOAccelDisplayPipeTransaction2` on the guest side — a
//! per-frame list of planes carrying source, destination and dirty rects — and
//! only a single target id reaches the wire. That reads like a truncation and
//! is not one: the guest's display pipe serializes the transaction by taking
//! **plane 0's** surface and writing that one id into a fixed-size command. One
//! target is the whole contract rather than a first approximation of it.
//!
//! One consequence worth stating because it is easy to go looking for: the
//! guest's damage rects never reach this device. They exist in the transaction
//! and the serializer drops them. Anything that wants per-frame damage has to
//! get it from somewhere other than the present path.
//!
//! # The tail past the trailer is reported, not read
//!
//! A payload longer than the emitting command declares means either that the
//! guest grew the command or that this decode has become a truncation, and the
//! difference matters. [`Trailer::undecoded_tail`] is how many bytes followed
//! the words named here, so the layer with a failure channel can say so. This
//! layer does not read them and does not guess what they are.

use crate::packets::Channel;

/// Which of the three present commands a packet is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PresentForm {
    /// The superseded form. Its trailer keeps the target before the task.
    Transaction2,
    /// The current form. Task before target, and a gamma table after both.
    Transaction3,
    /// The form that names a single mapping rather than serializing a
    /// transaction, and therefore carries no task.
    SwapMapping,
}

impl PresentForm {
    /// The present command a packet is, or `None` if it is not a present.
    ///
    /// The three are child-channel commands. A root-channel packet at the same
    /// opcode is a different command entirely — one flat opcode space, two
    /// dispatch tables — so the channel is part of the key here as it is
    /// everywhere else in [`crate::packets`].
    #[must_use]
    pub const fn of(channel: Channel, opcode: u16) -> Option<Self> {
        match (channel, opcode) {
            (Channel::Child, 0x06) => Some(Self::Transaction2),
            (Channel::Child, 0x07) => Some(Self::Transaction3),
            (Channel::Child, 0x08) => Some(Self::SwapMapping),
            _ => None,
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transaction2 => "display_transaction2",
            Self::Transaction3 => "display_transaction3",
            Self::SwapMapping => "display_swap_mapping",
        }
    }

    /// Bytes this command declares its trailer to be.
    ///
    /// The gamma form's extra 24 bytes are the gamma table. Nothing here reads
    /// it; what the length buys is knowing where the *undecoded* tail starts,
    /// which is a different fact from where the target word sits.
    #[must_use]
    pub const fn trailer_len(self) -> usize {
        match self {
            Self::Transaction2 | Self::SwapMapping => 0x0c,
            Self::Transaction3 => 0x24,
        }
    }

    /// Byte offset of the word naming what to present.
    ///
    /// A surface id for the two transaction forms and a mapping id for
    /// [`Self::SwapMapping`]. One namespace, three positions — which is exactly
    /// why the position comes from the form and not from a shared constant.
    #[must_use]
    pub const fn target_offset(self) -> usize {
        match self {
            Self::Transaction2 => 0x04,
            Self::Transaction3 | Self::SwapMapping => 0x08,
        }
    }

    /// Byte offset of the submitting task's id, for the forms that carry one.
    ///
    /// `None` for [`Self::SwapMapping`], whose word at this position is
    /// unidentified. A reader that assumed every present names a task would
    /// read that word and call it a task.
    #[must_use]
    pub const fn task_offset(self) -> Option<usize> {
        match self {
            Self::Transaction2 => Some(0x08),
            Self::Transaction3 => Some(0x04),
            Self::SwapMapping => None,
        }
    }

    /// Whether this form's trailer names the task that owns what it presents.
    #[must_use]
    pub const fn names_a_task(self) -> bool {
        self.task_offset().is_some()
    }
}

/// A present packet's trailer, decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trailer {
    pub form: PresentForm,
    /// Word zero: the display index for [`PresentForm::SwapMapping`] and the
    /// pipe index for the other two. Decoded rather than skipped because a
    /// guest field with no name is the one nobody notices being ignored.
    pub pipe: u32,
    /// What to present: a surface id, or a mapping id for
    /// [`PresentForm::SwapMapping`].
    pub target: u32,
    /// The submitting task, for the forms that name one. **Not** a completion
    /// stamp — the packet's own stamp is in its FIFO header.
    pub task: Option<u32>,
    /// Bytes that followed the declared trailer. Zero on a payload of exactly
    /// the declared size. See the module docs for why this is reported rather
    /// than read.
    pub undecoded_tail: usize,
}

/// Why a present payload could not be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The payload is shorter than the emitting command's own trailer.
    ///
    /// Refused rather than clamped: presenting mapping zero and completing the
    /// packet in silence is a frame the guest believes it showed.
    Short { have: usize, need: usize },
}

impl Refusal {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Short { .. } => "display_present_short",
        }
    }
}

/// Decode a present packet's trailer.
///
/// # Errors
///
/// [`Refusal::Short`] when the payload cannot hold the command's own trailer.
pub fn trailer(form: PresentForm, payload: &[u8]) -> Result<Trailer, Refusal> {
    let need = form.trailer_len();
    if payload.len() < need {
        return Err(Refusal::Short {
            have: payload.len(),
            need,
        });
    }
    Ok(Trailer {
        form,
        pipe: word(payload, 0),
        target: word(payload, form.target_offset()),
        task: form.task_offset().map(|offset| word(payload, offset)),
        undecoded_tail: payload.len() - need,
    })
}

/// One little-endian word at a byte offset the trailer length has already
/// bounded.
fn word(payload: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        payload[offset],
        payload[offset + 1],
        payload[offset + 2],
        payload[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packets::LEDGER;

    const FORMS: [PresentForm; 3] = [
        PresentForm::Transaction2,
        PresentForm::Transaction3,
        PresentForm::SwapMapping,
    ];

    /// A payload of `len` bytes whose word `n` is `0xA0 + n`, so a decoder that
    /// read the wrong slot cannot accidentally read the right value.
    fn payload(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| {
                if i % 4 == 0 {
                    0xA0 + u8::try_from(i / 4).expect("small")
                } else {
                    0
                }
            })
            .collect()
    }

    /// Every word this layer names sits inside the trailer it named. A slot
    /// past the declared length is an offset table that disagrees with itself.
    #[test]
    fn every_named_word_is_inside_the_trailer_it_belongs_to() {
        // The widths themselves, pinned: the tail reading is only meaningful at
        // the right one, and a payload that did carry an inline plane list
        // would still look trailer-only if the width were the other command's.
        assert_eq!(PresentForm::Transaction2.trailer_len(), 0x0c);
        assert_eq!(PresentForm::Transaction3.trailer_len(), 0x24);
        assert_eq!(PresentForm::SwapMapping.trailer_len(), 0x0c);
        for form in FORMS {
            assert!(
                form.target_offset() + 4 <= form.trailer_len(),
                "{}",
                form.name()
            );
            if let Some(task) = form.task_offset() {
                assert!(task + 4 <= form.trailer_len(), "{}", form.name());
                assert_ne!(task, form.target_offset(), "{}", form.name());
            }
            assert_ne!(form.target_offset(), 0, "word zero is the pipe index");
        }
    }

    /// The reading that costs a frame when it is wrong: each form's target word
    /// is at its own slot, and reading one at another's returns a different
    /// number.
    #[test]
    fn each_form_reads_its_target_from_its_own_slot() {
        let p = payload(0x24);
        assert_eq!(
            trailer(PresentForm::Transaction2, &p)
                .expect("long enough")
                .target,
            0xA1
        );
        assert_eq!(
            trailer(PresentForm::Transaction3, &p)
                .expect("long enough")
                .target,
            0xA2
        );
        assert_eq!(
            trailer(PresentForm::SwapMapping, &p)
                .expect("long enough")
                .target,
            0xA2
        );
        // And the two transaction forms swap their target and task words, which
        // is the whole reason they are two commands here.
        let two = trailer(PresentForm::Transaction2, &p).expect("long enough");
        let three = trailer(PresentForm::Transaction3, &p).expect("long enough");
        assert_eq!((two.target, two.task), (0xA1, Some(0xA2)));
        assert_eq!((three.target, three.task), (0xA2, Some(0xA1)));
    }

    /// The swap form has no task, and the word where the other forms keep one
    /// is not reported as one.
    #[test]
    fn the_swap_form_names_no_task() {
        let p = payload(0x0c);
        let t = trailer(PresentForm::SwapMapping, &p).expect("long enough");
        assert_eq!(t.task, None);
        assert!(!PresentForm::SwapMapping.names_a_task());
        assert!(PresentForm::Transaction2.names_a_task());
        assert!(PresentForm::Transaction3.names_a_task());
    }

    /// A short payload is refused rather than clamped, at exactly one byte
    /// under.
    #[test]
    fn a_payload_one_byte_under_its_trailer_is_refused() {
        for form in FORMS {
            let need = form.trailer_len();
            let p = payload(need);
            assert!(trailer(form, &p).is_ok(), "{}", form.name());
            assert_eq!(
                trailer(form, &p[..need - 1]),
                Err(Refusal::Short {
                    have: need - 1,
                    need
                }),
                "{}",
                form.name()
            );
        }
    }

    /// The tail is counted, not read. A guest that grew the command grew it by
    /// exactly this much.
    #[test]
    fn bytes_past_the_declared_trailer_are_counted() {
        for form in FORMS {
            let p = payload(form.trailer_len() + 28);
            let t = trailer(form, &p).expect("long enough");
            assert_eq!(t.undecoded_tail, 28, "{}", form.name());
            let exact = payload(form.trailer_len());
            assert_eq!(
                trailer(form, &exact).expect("long enough").undecoded_tail,
                0,
                "{}",
                form.name()
            );
        }
    }

    /// The forms and the ledger's present rows are the same three packets.
    #[test]
    fn the_forms_are_exactly_the_ledgers_present_packets() {
        let mut seen = 0;
        for p in LEDGER {
            if PresentForm::of(p.channel, p.opcode).is_some() {
                seen += 1;
            }
        }
        assert_eq!(seen, 3);
        assert_eq!(PresentForm::of(Channel::Root, 0x07), None);
        assert_eq!(PresentForm::of(Channel::Child, 0x09), None);
    }
}
