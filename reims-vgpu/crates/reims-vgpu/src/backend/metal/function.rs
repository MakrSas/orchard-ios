//! MTLB sole-function load with content-hash cache.

use crate::backend::blob::BlobKey;
use crate::backend::metal::cache::{fn_cache_insert, fn_cache_lookup};
use crate::backend::metal::util::{set_err, ErrOut, Status};
use metal::{Device, Function};

pub fn load_only_function(
    device: &Device,
    mtlb: &[u8],
    label: &str,
    err: ErrOut<'_>,
) -> Result<Function, Status> {
    validate_mtlb(mtlb, label, err)?;
    let key = BlobKey::new(mtlb);
    if let Some(hit) = fn_cache_lookup(&key) {
        return Ok(hit);
    }
    let function = load_only_function_uncached(device, mtlb, label, err)?;
    Ok(fn_cache_insert(&key, function))
}

fn validate_mtlb(mtlb: &[u8], label: &str, err: ErrOut<'_>) -> Result<(), Status> {
    if mtlb.is_empty() {
        set_err(err, format!("{label} MTLB is empty"));
        Err(Status::args("metal_function_mtlb_empty"))
    } else {
        Ok(())
    }
}

fn load_only_function_uncached(
    device: &Device,
    mtlb: &[u8],
    label: &str,
    err: ErrOut<'_>,
) -> Result<Function, Status> {
    let library = match device.new_library_with_data(mtlb) {
        Ok(lib) => lib,
        Err(e) => match load_restamped(device, mtlb, &e) {
            Some(lib) => lib,
            None => {
                let header = MtlbHeader::read(mtlb);
                // The driver's own words, which a Status cannot carry: once
                // per blob, since a refused blob never reaches the cache and
                // every draw that needs it asks again.
                if keep_refused(mtlb) {
                    crate::observe::fail(format!(
                        "metal_library_refused platform={} os={}.{} mtlb_len={} error={}",
                        header.platform(),
                        header.os_major(),
                        header.os_minor(),
                        mtlb.len(),
                        compact(&e)
                    ));
                }
                set_err(err, format!("{label} newLibraryWithData failed: {e}"));
                return Err(Status::execute("metal_function_library_create_failed")
                    .field("mtlb_len", mtlb.len())
                    .field("mtlb_platform", header.platform())
                    .field("mtlb_os_major", header.os_major()));
            }
        },
    };
    let names = library.function_names();
    if names.len() != 1 {
        set_err(
            err,
            format!("{label} exposed {} functions, expected one", names.len()),
        );
        return Err(Status::args("metal_function_count_not_one")
            .field("count", names.len())
            .field("expected", 1usize));
    }
    match library.get_function(&names[0], None) {
        Ok(f) => Ok(f),
        Err(_) => {
            set_err(err, format!("{label} function lookup failed"));
            Err(Status::execute("metal_function_lookup_failed"))
        }
    }
}

/// The few bytes of an MTLB container header that say what it was built for.
///
/// Read off a survey of the metallibs shipped with macOS, iOS and the iOS
/// simulator (tools/ios-metallib-probe in the Orchard tree): 0x0B separates
/// the platforms (0x81 macOS, 0x82 iOS, 0x87 iOS Simulator), bit 7 of 0x05
/// tracks the same split, and 0x0C/0x0E hold the target OS major and minor.
/// Inferred from samples, not documented; used to describe a blob a driver
/// refused and, on iOS only, to retry it — never to decide what a guest meant.
struct MtlbHeader<'a>(&'a [u8]);

const MTLB_MAGIC: &[u8; 4] = b"MTLB";
/// Only the iOS retry rewrites these two.
#[cfg(target_os = "ios")]
const MTLB_FLAGS: usize = 0x05;
const MTLB_PLATFORM: usize = 0x0B;
#[cfg(target_os = "ios")]
const MTLB_PLATFORM_IOS: u8 = 0x82;

impl<'a> MtlbHeader<'a> {
    fn read(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }
    fn valid(&self) -> bool {
        self.0.len() >= 0x10 && &self.0[..4] == MTLB_MAGIC
    }
    fn platform(&self) -> &'static str {
        if !self.valid() {
            return "not_mtlb";
        }
        match self.0[MTLB_PLATFORM] {
            0x81 => "macos",
            0x82 => "ios",
            0x87 => "ios_simulator",
            _ => "unknown",
        }
    }
    fn u16_at(&self, i: usize) -> u16 {
        if self.valid() {
            u16::from_le_bytes([self.0[i], self.0[i + 1]])
        } else {
            0
        }
    }
    fn os_major(&self) -> u16 {
        self.u16_at(0x0C)
    }
    fn os_minor(&self) -> u16 {
        self.u16_at(0x0E)
    }
}

/// A driver's error on one line, for the failure channel: its `key=value`
/// fields are split on spaces, so the message keeps none.
fn compact(e: &impl std::fmt::Display) -> String {
    let mut s: String = e
        .to_string()
        .chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect();
    s.truncate(240);
    s
}

/// iOS only: a blob the driver refused, retried with its platform stamp
/// rewritten to iOS.
///
/// A macOS guest compiles its shaders for macOS, and an iOS host's Metal
/// driver is the one asked to load them. The same GPU family runs either
/// way; what differs is the stamp. Whether the retry was needed, and whether
/// it worked, goes on the failure channel once per blob — the cache keeps a
/// loaded function, so a blob is only ever retried once.
#[cfg(target_os = "ios")]
fn load_restamped(
    device: &Device,
    mtlb: &[u8],
    first: &impl std::fmt::Display,
) -> Option<metal::Library> {
    let header = MtlbHeader::read(mtlb);
    if !header.valid() || mtlb[MTLB_PLATFORM] == MTLB_PLATFORM_IOS {
        return None;
    }
    let mut stamped = mtlb.to_vec();
    stamped[MTLB_PLATFORM] = MTLB_PLATFORM_IOS;
    stamped[MTLB_FLAGS] &= 0x7F;
    let outcome = device.new_library_with_data(&stamped);
    crate::observe::fail(format!(
        "metal_library_restamp platform={} os={}.{} mtlb_len={} result={} first_error={}{}",
        header.platform(),
        header.os_major(),
        header.os_minor(),
        mtlb.len(),
        if outcome.is_ok() { "loaded" } else { "refused" },
        compact(first),
        outcome
            .as_ref()
            .err()
            .map(|e| format!(" restamped_error={}", compact(e)))
            .unwrap_or_default(),
    ));
    outcome.ok()
}

#[cfg(not(target_os = "ios"))]
fn load_restamped(_: &Device, _: &[u8], _: &impl std::fmt::Display) -> Option<metal::Library> {
    None
}

/// iOS only: keep one copy of each blob the driver refused, in the process's
/// temporary directory, so its header and contents can be read off the
/// device. Bounded by what a guest compiles; each blob is refused once.
/// Returns whether this blob was refused for the first time.
#[cfg(target_os = "ios")]
fn keep_refused(mtlb: &[u8]) -> bool {
    let hash = mtlb.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    });
    let dir = std::env::temp_dir().join("reims-vgpu-mtlb");
    let path = dir.join(format!("{hash:016x}-{}.mtlb", mtlb.len()));
    if path.exists() {
        return false;
    }
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(path, mtlb);
    true
}

/// Everywhere but iOS the refusal is reported every time, as it always was.
#[cfg(not(target_os = "ios"))]
fn keep_refused(_: &[u8]) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::Refusal;
    use std::ffi::CStr;
    use std::os::raw::c_char;

    #[test]
    fn empty_mtlb_is_rejected_before_device_access_with_a_reason() {
        let mut error = [0 as c_char; 64];
        let status = validate_mtlb(&[], "vertex", (error.as_mut_ptr(), error.len()))
            .expect_err("empty input must fail");
        assert_eq!(status.refusal(), Some("metal_function_mtlb_empty"));
        assert!(status.is_args());
        let message = unsafe { CStr::from_ptr(error.as_ptr()) };
        assert_eq!(message.to_str().unwrap(), "vertex MTLB is empty");
    }

    #[test]
    fn nonempty_mtlb_passes_the_pre_device_validation() {
        assert_eq!(
            validate_mtlb(&[1], "fragment", (std::ptr::null_mut(), 0)),
            Ok(())
        );
    }
}
