//! Sampled textures kept across draws, keyed by their content.
//!
//! # Why
//!
//! Every sampled bind on this rail was a fresh `MTLTexture`: the guest texture's
//! pages walked, every texel decoded, and the result uploaded, per draw. Measured
//! on an iPhone running a macOS guest at its login window: 48 binds a second
//! moved **1.3 GB** — the wallpaper, redrawn — with `metal_sampled_load_us` at
//! 436 ms and `metal_sampled_tex_upload_us` at 185 ms of every second. That is
//! most of a host core, on a host whose two fast cores are what the guest's
//! emulated CPUs run on.
//!
//! # Why the key is the content, and why that is safe
//!
//! This device does not see the guest's CPU writes to its own texture memory, so
//! nothing here can know that a texture is unchanged without looking at it. The
//! caller reads the texture's bytes exactly as the decode would — the same rows,
//! through the same translation — and hashes them together with everything else
//! the decode depends on (geometry, row pitch, both formats). Equal keys are equal
//! inputs to a deterministic decode, so a hit hands back the texture that decode
//! produced. A stale texture is impossible short of a 64-bit collision.
//!
//! What a hit saves is the decode and the upload; the read is paid either way.
//! On the measurement above that is about two thirds of the bar.
//!
//! # Eviction is lawful
//!
//! Every entry is derived from guest pages that are still there, so dropping one
//! costs the next bind a decode and loses nothing — a cache bound in the sense
//! `AGENTS.md` allows. Least recently used goes first.

use metal::Texture;
use parking_lot::Mutex;

/// Bytes of textures kept. On a phone the whole app lives under a 3 GB limit
/// the guest's RAM takes two of; elsewhere there is room for a desktop's worth
/// of layers.
///
/// On iOS it has to hold one texture above all: the login window samples the
/// Ventura wallpaper at its full 6016x6016, 138 MiB in RGBA8. At 96 MB that
/// was refused on every bind — measured, 136 misses carrying 7.4 GB while
/// every smaller texture hit — and each refusal decoded 138 MiB into a buffer
/// and uploaded it into a second texture of the same size, both transient,
/// both counted against the limit. Keeping the one texture costs less than
/// making it every frame.
#[cfg(target_os = "ios")]
const MAX_BYTES: u64 = 160 << 20;
#[cfg(not(target_os = "ios"))]
const MAX_BYTES: u64 = 256 << 20;

struct Entry {
    key: u64,
    texture: Texture,
    bytes: u64,
    last_use: u64,
}

struct Cache {
    entries: Vec<Entry>,
    bytes: u64,
    tick: u64,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache {
    entries: Vec::new(),
    bytes: 0,
    tick: 0,
});

/// The texture decoded from these inputs before, if it is still kept.
pub fn get(key: u64) -> Option<Texture> {
    if key == 0 {
        return None;
    }
    let mut cache = CACHE.lock();
    cache.tick += 1;
    let tick = cache.tick;
    let entry = cache.entries.iter_mut().find(|e| e.key == key)?;
    entry.last_use = tick;
    Some(entry.texture.clone())
}

/// Keep `texture` as the decode of `key`. A texture bigger than the whole
/// budget is not kept: it would evict everything and still not fit.
pub fn insert(key: u64, texture: &Texture, bytes: u64) {
    if key == 0 || bytes > MAX_BYTES {
        return;
    }
    let mut cache = CACHE.lock();
    if cache.entries.iter().any(|e| e.key == key) {
        return;
    }
    while cache.bytes + bytes > MAX_BYTES {
        let Some(oldest) = cache
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_use)
            .map(|(i, _)| i)
        else {
            break;
        };
        let gone = cache.entries.swap_remove(oldest);
        cache.bytes -= gone.bytes;
        crate::runtime::drain::note_store_route("metal_sampled_cache_evicted");
    }
    cache.tick += 1;
    let last_use = cache.tick;
    cache.bytes += bytes;
    cache.entries.push(Entry {
        key,
        texture: texture.clone(),
        bytes,
        last_use,
    });
}

/// Entries and bytes kept, for the census.
pub fn levels() -> (usize, u64) {
    let cache = CACHE.lock();
    (cache.entries.len(), cache.bytes)
}

/// A streaming 64-bit hash over the bytes a decode reads, fed in pieces.
///
/// Four independent multiply-xor lanes, so the loop is bound by memory rather
/// than by one multiply chain; not cryptographic, only well mixed. Zero is
/// reserved for "not cacheable", so a finished hash never returns it.
pub struct ContentHash {
    lanes: [u64; 4],
    words: u64,
    tail: u64,
    tail_len: u32,
    total: u64,
}

const K: [u64; 4] = [
    0x9e37_79b9_7f4a_7c15,
    0xc2b2_ae3d_27d4_eb4f,
    0x1656_67b1_9e37_79f9,
    0xd6e8_feb8_6659_fd93,
];

impl ContentHash {
    pub fn new(seed: &[u64]) -> Self {
        let mut hash = Self {
            lanes: K,
            words: 0,
            tail: 0,
            tail_len: 0,
            total: 0,
        };
        for word in seed {
            hash.word(*word);
        }
        hash
    }

    /// One 8-byte word, into the lane its position picks. Position rather than
    /// call boundaries decides the lane, so the result does not depend on how
    /// the caller split its input — a texture read in rows of one pitch must
    /// hash the same as the same bytes read any other way.
    #[inline(always)]
    fn word(&mut self, word: u64) {
        let lane = (self.words & 3) as usize;
        self.lanes[lane] = mix(self.lanes[lane], word, K[lane]);
        self.words += 1;
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.total = self.total.wrapping_add(bytes.len() as u64);
        let mut rest = bytes;
        // Finish a word the previous piece left partial.
        while self.tail_len != 0 && !rest.is_empty() {
            self.byte(rest[0]);
            rest = &rest[1..];
        }
        let (words, remainder) = rest.as_chunks::<8>();
        for word in words {
            self.word(u64::from_le_bytes(*word));
        }
        for &b in remainder {
            self.byte(b);
        }
    }

    #[inline(always)]
    fn byte(&mut self, b: u8) {
        self.tail |= u64::from(b) << (8 * self.tail_len);
        self.tail_len += 1;
        if self.tail_len == 8 {
            let tail = self.tail;
            self.tail = 0;
            self.tail_len = 0;
            self.word(tail);
        }
    }

    pub fn finish(self) -> u64 {
        let mut h = self.total.wrapping_mul(K[1]);
        for lane in self.lanes {
            h = mix(h, lane, K[2]);
        }
        h = mix(h, self.tail ^ u64::from(self.tail_len), K[3]);
        h ^= h >> 32;
        if h == 0 {
            1
        } else {
            h
        }
    }
}

#[inline(always)]
fn mix(acc: u64, word: u64, k: u64) -> u64 {
    let x = (acc ^ word).wrapping_mul(k);
    x ^ (x >> 29)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(seed: &[u64], pieces: &[&[u8]]) -> u64 {
        let mut h = ContentHash::new(seed);
        for p in pieces {
            h.update(p);
        }
        h.finish()
    }

    /// The key must not depend on how the rows were split, or a texture read in
    /// rows of one pitch would miss against itself.
    #[test]
    fn the_hash_does_not_depend_on_how_the_input_was_split() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + 3) as u8).collect();
        let whole = hash(&[1, 2], &[&data]);
        assert_eq!(whole, hash(&[1, 2], &[&data[..1], &data[1..]]));
        // Split on a 32-byte boundary and elsewhere.
        assert_eq!(
            whole,
            hash(&[1, 2], &[&data[..64], &data[64..500], &data[500..]])
        );
        assert_eq!(
            whole,
            hash(&[1, 2], &[&data[..33], &data[33..999], &data[999..]])
        );
    }

    /// Any change to the content or to what the decode depends on is a
    /// different key.
    #[test]
    fn a_changed_byte_or_parameter_is_a_different_key() {
        let data = vec![0x5au8; 4096];
        let base = hash(&[640, 480, 2560], &[&data]);
        let mut changed = data.clone();
        changed[4000] ^= 1;
        assert_ne!(base, hash(&[640, 480, 2560], &[&changed]));
        assert_ne!(base, hash(&[480, 640, 2560], &[&data]));
        assert_ne!(base, hash(&[640, 480, 2560], &[&data[..4095]]));
        assert_ne!(base, 0);
    }
}
