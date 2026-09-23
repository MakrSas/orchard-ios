//! COM1 (0x3F8) scrape rail for QEMU `-serial file:`.
//!
//! Option-ROM load runs before SerialIo is useful; ConOut may already be
//! graphical. Polled UART is the reliable product boot gate.

const COM1: u16 = 0x3F8;
const LSR: u16 = COM1 + 5;

/// Write one line (CRLF) to QEMU isa-serial / COM1.
pub fn line(msg: &str) {
    unsafe {
        let put = |b: u8| {
            for _ in 0..100_000 {
                let lsr: u8;
                core::arch::asm!(
                    "in al, dx",
                    in("dx") LSR,
                    out("al") lsr,
                    options(nomem, nostack, preserves_flags)
                );
                if lsr & 0x20 != 0 {
                    break;
                }
            }
            core::arch::asm!(
                "out dx, al",
                in("dx") COM1,
                in("al") b,
                options(nomem, nostack, preserves_flags)
            );
        };
        for &b in msg.as_bytes() {
            put(b);
        }
        put(b'\r');
        put(b'\n');
    }
}
