//! `MTLStorageMode`, and what this wire's use of it does and does not license.
//!
//! # The three values this wire has been seen to carry
//!
//! The mode arrives as `resource_options[7:4]` — the ordinal shifted left by
//! four, which is `MTLResourceOptions`' documented storage-mode shift rather
//! than a bare mode field. Three ordinals are pinned by fixtures that moved the
//! nibble alone: shared, managed, private. The public API declares a fourth,
//! memoryless, and this wire has never been observed carrying it, so it is a
//! refusal here rather than a variant: a memoryless resource's backing
//! contract on *this* interface is not established, and a route chosen for it
//! would be chosen from the SDK's description of a different transport.
//!
//! # Private is an announcement, not an access contract
//!
//! This is the reading that is easy to get backwards and expensive to get
//! wrong, so it is stated where the type is rather than at a call site.
//!
//! `Private` looks like a licence to stop keeping guest pages current for a
//! resource the guest has declared GPU-only. It is not one on this wire, and
//! the emitting serializer says why: backing is allocated mode-blind, so a
//! private texture still gets page-rounded guest backing exactly as a shared one
//! does; the guest still CPU-touches it, because create-with-contents,
//! region-replace and get-bytes each copy through that mapping with no
//! storage-mode check on the path; and what the mode actually gates is the
//! *announcement* — the modified-range notification is emitted only for
//! `Managed`.
//!
//! So private means "I will not tell you when I write this", not "I will not
//! write this". Treating it as the latter turns silence into a guarantee of
//! inaction, and the resulting stale-page defect is invisible at every counter
//! because its only symptom is wrong content.
//!
//! [`StorageMode::announces_writes`] is therefore the question this vocabulary
//! answers, and there is deliberately no `skips_guest_coherence`. What is not
//! established is the *receiving* side: absence of a mode-dependent branch has
//! been read on the emitting serializer only, so a future host deserializer
//! reading is what would reopen this.

/// The storage mode a resource descriptor declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageMode {
    /// `MTLStorageModeShared`. CPU and GPU share one allocation.
    Shared,
    /// `MTLStorageModeManaged`. Two copies, with explicit synchronisation, and
    /// the only mode whose writes are announced.
    Managed,
    /// `MTLStorageModePrivate`. GPU-only *by declaration*; see the module docs
    /// for what that does and does not mean here.
    Private,
}

impl StorageMode {
    /// The ordinal shift inside `MTLResourceOptions`.
    pub const OPTIONS_SHIFT: u32 = 4;
    /// The nibble the mode occupies.
    pub const OPTIONS_MASK: u32 = 0xf;

    /// Whether a guest write to this resource is announced to the device.
    ///
    /// Only `Managed` announces. The other two mean the device learns about a
    /// write some other way or not at all — which is a statement about
    /// notification and never about whether the write happens.
    #[must_use]
    pub const fn announces_writes(self) -> bool {
        matches!(self, Self::Managed)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Managed => "managed",
            Self::Private => "private",
        }
    }

    /// The ordinal this mode is spelled with.
    #[must_use]
    pub const fn ordinal(self) -> u8 {
        match self {
            Self::Shared => 0,
            Self::Managed => 1,
            Self::Private => 2,
        }
    }
}

/// A storage-mode ordinal this wire has not been observed carrying.
///
/// Named rather than folded into a nearby mode: the API declares a memoryless
/// mode whose backing contract on this interface is not established, and
/// choosing a route for it from the SDK's description of a different transport
/// is the guess this refusal exists instead of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnobservedStorageMode {
    pub ordinal: u8,
}

impl UnobservedStorageMode {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        "storage_mode_unobserved"
    }
}

/// The mode a raw ordinal names.
///
/// # Errors
///
/// If the ordinal is not one of the three this wire carries.
pub const fn storage_mode(ordinal: u8) -> Result<StorageMode, UnobservedStorageMode> {
    match ordinal {
        0 => Ok(StorageMode::Shared),
        1 => Ok(StorageMode::Managed),
        2 => Ok(StorageMode::Private),
        other => Err(UnobservedStorageMode { ordinal: other }),
    }
}

/// The mode a whole `MTLResourceOptions` word declares.
///
/// # Errors
///
/// As [`storage_mode`].
pub const fn from_resource_options(options: u32) -> Result<StorageMode, UnobservedStorageMode> {
    storage_mode(((options >> StorageMode::OPTIONS_SHIFT) & StorageMode::OPTIONS_MASK) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three fixtures that moved the nibble alone.
    #[test]
    fn the_observed_options_words_read_as_their_modes() {
        assert_eq!(from_resource_options(0x0000), Ok(StorageMode::Shared));
        assert_eq!(from_resource_options(0x0010), Ok(StorageMode::Managed));
        assert_eq!(from_resource_options(0x0020), Ok(StorageMode::Private));
    }

    /// The cache-mode and hazard-tracking fields share the word and must not
    /// reach the storage nibble.
    #[test]
    fn the_neighbouring_fields_do_not_change_the_mode() {
        // Write-combined cache with shared storage.
        assert_eq!(from_resource_options(0x0001), Ok(StorageMode::Shared));
        // Tracked hazard mode with private storage.
        assert_eq!(from_resource_options(0x0220), Ok(StorageMode::Private));
        // Untracked with private.
        assert_eq!(from_resource_options(0x0120), Ok(StorageMode::Private));
    }

    /// Memoryless is a real API mode whose contract on this wire is not
    /// established. Folding it into a neighbour would pick a route from another
    /// transport's description.
    #[test]
    fn an_unobserved_ordinal_is_refused_by_number() {
        assert_eq!(
            storage_mode(3),
            Err(UnobservedStorageMode { ordinal: 3 }),
            "memoryless"
        );
        for ordinal in 4..=15u8 {
            assert_eq!(
                storage_mode(ordinal),
                Err(UnobservedStorageMode { ordinal })
            );
        }
        assert_eq!(
            from_resource_options(0x0030),
            Err(UnobservedStorageMode { ordinal: 3 })
        );
    }

    /// The question this vocabulary answers. There is deliberately no
    /// `skips_guest_coherence`; see the module docs.
    #[test]
    fn only_managed_announces_its_writes() {
        assert!(StorageMode::Managed.announces_writes());
        assert!(!StorageMode::Shared.announces_writes());
        assert!(
            !StorageMode::Private.announces_writes(),
            "and that is a statement about notification, not about writing"
        );
    }

    #[test]
    fn every_mode_names_itself_and_its_ordinal_once() {
        let modes = [
            StorageMode::Shared,
            StorageMode::Managed,
            StorageMode::Private,
        ];
        for m in modes {
            assert_eq!(storage_mode(m.ordinal()), Ok(m));
        }
        let mut names: Vec<&str> = modes.iter().map(|m| m.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }
}
