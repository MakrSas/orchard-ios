//! Typed failures at the Vulkan engine façade and host-window presenter seam.
//!
//! These checks are neither malformed draw/compute requests nor failed Vulkan
//! calls. They reject an engine entry point because the façade's tracked state
//! disappeared or disagreed with the caller — the named resident is absent, is
//! at the wrong generation, or is not yet content-ready.

use super::compute_execution::residency_fields;
use crate::model::ComputeStorageResidencyKey;
use crate::observe::Decline;

/// A specific engine façade or host-window presenter state failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineFacadeDecline {
    WindowPresenterNotAttached,
    StorageReadResidentAbsent {
        identity: ComputeStorageResidencyKey,
    },
    StorageReadGenerationMismatch {
        identity: ComputeStorageResidencyKey,
        actual_generation: u32,
        expected_generation: u32,
    },
    /// The device's rail slot is held by a different rail, so this rail's
    /// object caches for it cannot be reached.
    ///
    /// Unreachable in any live build — `backend::select` latches one rail per
    /// process — and refused by name anyway, because the alternative is a
    /// second cache owning handles the first one also owns. A caller that fell
    /// back to one would be running two owners of the same `VkPipeline`.
    DeviceCachesUnreachable,
}

impl Decline for EngineFacadeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::WindowPresenterNotAttached => "vk_engine_window_presenter_not_attached",
            Self::StorageReadResidentAbsent { .. } => "vk_engine_storage_read_resident_absent",
            Self::StorageReadGenerationMismatch { .. } => {
                "vk_engine_storage_read_generation_mismatch"
            }
            Self::DeviceCachesUnreachable => "vk_engine_device_caches_unreachable",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::WindowPresenterNotAttached => Vec::new(),
            Self::StorageReadResidentAbsent { identity } => residency_fields(identity),
            Self::StorageReadGenerationMismatch {
                identity,
                actual_generation,
                expected_generation,
            } => {
                let mut fields = residency_fields(identity);
                fields.extend([
                    ("actual_generation", actual_generation.to_string()),
                    ("expected_generation", expected_generation.to_string()),
                ]);
                fields
            }
            Self::DeviceCachesUnreachable => Vec::new(),
        }
    }
}

crate::observe::decline_display!(EngineFacadeDecline);

#[cfg(test)]
mod tests {
    use super::*;

    fn residency() -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey {
            mapping_id: 7,
            map_generation: 8,
            surface_offset: 0x9000,
            surface_bpr: 256,
            span_end: 4096,
            width: 64,
            height: 32,
            pixel_format: 80,
            texture_ref: 11,
        }
    }

    fn all() -> Vec<EngineFacadeDecline> {
        vec![
            EngineFacadeDecline::WindowPresenterNotAttached,
            EngineFacadeDecline::StorageReadResidentAbsent {
                identity: residency(),
            },
            EngineFacadeDecline::StorageReadGenerationMismatch {
                identity: residency(),
                actual_generation: 8,
                expected_generation: 9,
            },
        ]
    }

    #[test]
    fn every_engine_facade_check_has_a_unique_log_safe_slug() {
        let mut slugs: Vec<_> = all().iter().map(Decline::slug).collect();
        for slug in &slugs {
            assert!(slug.starts_with("vk_engine_"), "{slug}");
            assert!(
                slug.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                "{slug}"
            );
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 3, "the engine façade reason census moved");
        assert_eq!(before, slugs.len(), "duplicate engine façade slug");
    }

    #[test]
    fn engine_facade_fields_are_structured_and_log_safe() {
        for decline in all() {
            let line = crate::observe::Emit::decline("engine_facade_test", &decline).render();
            assert!(line.starts_with(&format!("engine_facade_test reason={}", decline.slug())));
            for field in line.split(' ').skip(1) {
                assert!(!field.is_empty(), "empty field in {line:?}");
                assert!(
                    !field.contains(char::is_whitespace),
                    "non-token field in {line:?}"
                );
            }
        }
    }
}
