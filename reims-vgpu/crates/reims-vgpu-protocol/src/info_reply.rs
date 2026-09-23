//! The key/value reply table an info query is answered with.
//!
//! # One layout, two commands, and three bounds that are not the same bound
//!
//! The device-info and compute-info queries are answered the same way: a run of
//! `(key, value)` pairs, each two little-endian 32-bit words, written at a
//! destination the request names. Three separate numbers bound what may be
//! written, they come from three different places, and conflating any two of
//! them is a real defect:
//!
//! 1. **`key_table_len`** — the guest's own parser's table length, sent with the
//!    request. It is **exclusive**: the guest writes one past the highest key
//!    its walker has an arm for, so a key equal to it is one the guest discards.
//!    Reading it inclusively spends a pair slot on an answer nobody reads.
//! 2. **`count`** — how many pairs the guest is prepared to consume.
//! 3. **the destination's size** — how many pairs actually fit. A guest may ask
//!    for more pairs than its own buffer holds, and writing past the end is not
//!    a truncated reply but a write into whatever is next.
//!
//! # Truncation is reported, never silent
//!
//! [`encode`] returns how many answers it could not carry. Every answer it
//! drops is a capability the guest spends the rest of its run without — it asks
//! once and there is no larger re-ask — so the caller has something to name on
//! the failure channel. A function that simply wrote what fit would make a
//! capability disappear with no evidence that it had.
//!
//! # The terminator
//!
//! A guest that asked for more pairs than were answered gets one zero pair
//! after the last real one, which is how its walker stops. A terminator that
//! ran off the end would be the same overrun the size bound exists to prevent,
//! so it is never written past the destination — and the caller is told the
//! exact byte count rather than assuming one.
//!
//! Where the guest's `count` is more than the destination holds, the last slot
//! is **kept** for the terminator rather than filled with an answer. Without
//! that reservation a full destination carried no stop word at all, and the
//! guest's walker — which stops at a zero pair or at `count`, whichever comes
//! first — read past the last real pair into whatever the destination already
//! held. The trade is one capability against a parser walking into bytes
//! nothing wrote, and the capability is the cheaper of the two: it is counted
//! in [`Written::dropped`], so it reaches the failure channel by the same
//! route every other truncation does, while the bytes the walker would have
//! read have no name at all.

use crate::endian::st32;

/// Bytes in one `(key, value)` pair.
pub const PAIR_LEN: usize = 8;

/// The bounds a request carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplyBounds {
    /// One past the highest key the guest's parser has an arm for. Exclusive.
    pub key_table_len: u32,
    /// How many pairs the guest is prepared to consume.
    pub count: u32,
}

impl ReplyBounds {
    /// Whether the guest's own parser has an arm for `key`.
    ///
    /// The one place the exclusivity of [`Self::key_table_len`] is spelled.
    /// It was spelled three times before — once inside [`encode`] and once in
    /// each of the two reply builders that answer these queries — and the three
    /// had to agree about a polarity whose whole hazard is that both readings
    /// look right. A caller that needs to know *which* answers the guest
    /// reaches, in order to census the ones it does not, asks here rather than
    /// writing the comparison a fourth time.
    #[must_use]
    pub const fn parses(self, key: u32) -> bool {
        key < self.key_table_len
    }
}

/// What [`encode`] wrote, and what it could not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[must_use = "a dropped answer is a capability the guest never learns about"]
pub struct Written {
    /// Real answers written.
    pub pairs: u32,
    /// Bytes written, including the terminator if there was one.
    pub bytes: usize,
    /// Whether the terminating zero pair was written.
    pub terminated: bool,
    /// Answers the guest's own table says it parses and that did not fit.
    /// Zero on every healthy reply; a non-zero one is a defect with a name.
    pub dropped: u32,
}

/// Encode the answers this device has for a query into the guest's reply
/// destination.
///
/// `answers` are `(key, value)` pairs in the order they should appear. Keys the
/// guest's table does not reach are skipped before anything is counted, so they
/// are not "dropped" — the guest asked not to be told.
pub fn encode(bounds: ReplyBounds, answers: &[(u32, u32)], out: &mut [u8]) -> Written {
    let capacity = out.len() / PAIR_LEN;
    // The stop word needs a slot of its own, and a destination filled to the
    // last pair has none left to give it. So where the guest may read further
    // than this destination reaches, the last slot is reserved. Conditioned on
    // `count` and not on how many answers there are: a reply that fills the
    // destination exactly and answers the whole count needs no terminator, and
    // reserving there would drop an answer for nothing.
    let usable = if bounds.count as usize > capacity {
        capacity.saturating_sub(1)
    } else {
        capacity
    };
    let mut written = Written::default();
    let mut parsed = 0u32;
    for &(key, value) in answers {
        // The guest's table is exclusive; a key at or above it is discarded on
        // arrival, so answering it would spend a slot for nothing.
        if !bounds.parses(key) {
            continue;
        }
        parsed = parsed.saturating_add(1);
        if written.pairs >= bounds.count || written.pairs as usize >= usable {
            continue;
        }
        let at = written.pairs as usize * PAIR_LEN;
        st32(&mut out[at..at + 4], key);
        st32(&mut out[at + 4..at + 8], value);
        written.pairs += 1;
    }
    written.dropped = parsed - written.pairs;
    written.bytes = written.pairs as usize * PAIR_LEN;
    // The guest asked for more than it got, so it needs the stop word — and
    // only if the room for it is really there.
    if written.pairs < bounds.count && (written.pairs as usize) < capacity {
        let at = written.bytes;
        out[at..at + PAIR_LEN].fill(0);
        written.bytes += PAIR_LEN;
        written.terminated = true;
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(out: &[u8], i: usize) -> (u32, u32) {
        let at = i * PAIR_LEN;
        (
            u32::from_le_bytes(out[at..at + 4].try_into().expect("four bytes")),
            u32::from_le_bytes(out[at + 4..at + 8].try_into().expect("four bytes")),
        )
    }

    #[test]
    fn answers_are_pairs_of_little_endian_words() {
        let mut out = [0xaau8; 64];
        let w = encode(
            ReplyBounds {
                key_table_len: 8,
                count: 8,
            },
            &[(1, 1024), (3, 32)],
            &mut out,
        );
        assert_eq!(w.pairs, 2);
        assert_eq!(w.dropped, 0);
        assert!(w.terminated, "the guest asked for eight and got two");
        assert_eq!(w.bytes, 3 * PAIR_LEN);
        assert_eq!(pair(&out, 0), (1, 1024));
        assert_eq!(pair(&out, 1), (3, 32));
        assert_eq!(pair(&out, 2), (0, 0), "the stop word");
        assert_eq!(out[w.bytes], 0xaa, "and nothing past it");
    }

    /// The bound that is easiest to get wrong. A key equal to the table length
    /// is one the guest's walker has no arm for.
    #[test]
    fn the_guests_table_length_is_exclusive() {
        let mut out = [0u8; 64];
        let w = encode(
            ReplyBounds {
                key_table_len: 3,
                count: 8,
            },
            &[(2, 7), (3, 9), (4, 11)],
            &mut out,
        );
        assert_eq!(w.pairs, 1, "only key 2 is below the table length");
        assert_eq!(
            w.dropped, 0,
            "a key the guest does not parse is not a dropped answer"
        );
        assert_eq!(pair(&out, 0), (2, 7));
    }

    #[test]
    fn a_full_reply_carries_no_terminator() {
        let mut out = [0xaau8; 64];
        let w = encode(
            ReplyBounds {
                key_table_len: 8,
                count: 2,
            },
            &[(1, 10), (2, 20)],
            &mut out,
        );
        assert_eq!(w.pairs, 2);
        assert!(!w.terminated, "the guest got exactly what it asked for");
        assert_eq!(w.bytes, 2 * PAIR_LEN);
    }

    /// A guest may ask for more pairs than its own destination holds. The
    /// destination wins, what did not fit is named, and the reply is still
    /// terminated.
    ///
    /// The regression: the destination was filled to its last pair and the
    /// stop word had nowhere to go, so a walker that stops at a zero pair or
    /// at `count` read the one real pair and then three pairs of whatever the
    /// destination already held. Reserving the last slot costs one more
    /// capability, and a dropped capability is counted and reported while
    /// bytes nobody wrote are not.
    #[test]
    fn the_destination_bounds_the_reply_and_the_loss_is_reported() {
        let mut out = [0xaau8; PAIR_LEN];
        let w = encode(
            ReplyBounds {
                key_table_len: 8,
                count: 4,
            },
            &[(1, 10), (2, 20), (3, 30)],
            &mut out,
        );
        assert_eq!(w.pairs, 0, "the one slot there is belongs to the stop word");
        assert!(w.terminated);
        assert_eq!(w.bytes, PAIR_LEN);
        assert_eq!(out, [0u8; PAIR_LEN], "and the stop word was written");
        assert_eq!(w.dropped, 3, "three capabilities the guest never learns");
    }

    /// Two pairs of room and a guest that will read four: one answer and the
    /// stop word, never two answers and a walk into the third slot.
    #[test]
    fn a_truncated_reply_keeps_its_last_slot_for_the_stop_word() {
        let mut out = [0xaau8; 2 * PAIR_LEN];
        let w = encode(
            ReplyBounds {
                key_table_len: 8,
                count: 4,
            },
            &[(1, 10), (2, 20), (3, 30)],
            &mut out,
        );
        assert_eq!(w.pairs, 1);
        assert!(w.terminated);
        assert_eq!(w.bytes, 2 * PAIR_LEN);
        assert_eq!(pair(&out, 0), (1, 10));
        assert_eq!(pair(&out, 1), (0, 0));
        assert_eq!(w.dropped, 2);
    }

    /// And no slot is reserved where none is needed: a destination that holds
    /// exactly what the guest will read needs no stop word, and giving one a
    /// slot would drop an answer for nothing.
    #[test]
    fn a_reply_that_answers_the_whole_count_reserves_nothing() {
        let mut out = [0xaau8; 2 * PAIR_LEN];
        let w = encode(
            ReplyBounds {
                key_table_len: 8,
                count: 2,
            },
            &[(1, 10), (2, 20), (3, 30)],
            &mut out,
        );
        assert_eq!(w.pairs, 2, "both slots carry an answer");
        assert!(!w.terminated, "the guest reads exactly `count` and stops");
        assert_eq!(w.bytes, 2 * PAIR_LEN);
        assert_eq!(pair(&out, 0), (1, 10));
        assert_eq!(pair(&out, 1), (2, 20));
        assert_eq!(w.dropped, 1);
    }

    #[test]
    fn a_reply_with_no_room_at_all_writes_nothing() {
        let mut out = [0u8; 0];
        let w = encode(
            ReplyBounds {
                key_table_len: 8,
                count: 4,
            },
            &[(1, 10)],
            &mut out,
        );
        assert_eq!(
            w,
            Written {
                pairs: 0,
                bytes: 0,
                terminated: false,
                dropped: 1,
            }
        );
    }

    /// A guest that says it will consume nothing gets nothing, not a lone
    /// terminator written into a destination it never sized.
    #[test]
    fn a_zero_count_writes_nothing() {
        let mut out = [0xaau8; 32];
        let w = encode(
            ReplyBounds {
                key_table_len: 8,
                count: 0,
            },
            &[(1, 10)],
            &mut out,
        );
        assert_eq!(w.pairs, 0);
        assert!(!w.terminated);
        assert_eq!(w.bytes, 0);
        assert_eq!(w.dropped, 1);
        assert_eq!(out[0], 0xaa);
    }
}
