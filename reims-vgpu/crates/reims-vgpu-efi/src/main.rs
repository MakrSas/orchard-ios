//! UEFI GOP for product PCI `reims-vgpu-pci` (0x106B:0xEEEE) only.
//!
//! Same device as host BAR1 (linear BGRA8). Not a second QEMU display.
//! Loaded as PCI option ROM (`romfile=`) by OVMF when our VGA-class function appears.
//!
//! Protocol structures are allocated from boot-services pool so they remain valid
//! after the option-ROM image returns from StartImage.
//!
//! Paint logic lives in [`reims_vgpu_efi::paint`] (same code host unit tests run).

#![no_main]
#![no_std]

extern crate alloc;

mod gop;
mod pci;
mod serial;

use log::info;
use reims_vgpu_efi::paint;
use uefi::boot;
use uefi::prelude::*;

/// Success line scraped by product boot gates (must stay stable).
const SUCCESS_TAG: &str = "reims-vgpu efi-gop: GOP installed";

#[entry]
fn main() -> Status {
    if let Err(e) = uefi::helpers::init() {
        return e.status();
    }
    serial::line("reims-vgpu efi-gop: start");

    let base = match pci::find_bar1() {
        Ok(b) => b,
        Err(e) => {
            info!("reims-vgpu efi-gop: 0x106B:0xEEEE BAR1 not found ({e:?})");
            serial::line("reims-vgpu efi-gop: BAR1 not found");
            return e;
        }
    };

    let gop_ptr = match gop::build_ctx(base) {
        Ok(p) => p as *mut core::ffi::c_void,
        Err(e) => {
            info!("reims-vgpu efi-gop: allocate_pool / build_ctx failed {e:?}");
            serial::line("reims-vgpu efi-gop: allocate_pool failed");
            return e;
        }
    };

    // Program Apple EFI FB regs (BAR0 +0x1210/+0x1228) so host + boot.efi/kernel
    // agree on the early console GPA (same BAR1 base as GOP FrameBufferBase).
    // Live freeze class: kernel later "console relocated" off BAR1; host follows
    // efi_fb_start when the guest rewrites 0x1210 (contract), else BAR1.
    if let Err(e) = pci::program_efi_fb_regs(base, paint::FB_W, paint::FB_H) {
        info!("reims-vgpu efi-gop: program EFI FB regs failed {e:?}");
        serial::line("reims-vgpu efi-gop: program EFI FB regs failed");
        // Non-fatal: GOP still installed; host falls back to BAR1 copy.
    }

    let st = unsafe {
        boot::install_protocol_interface(Some(boot::image_handle()), &gop::GOP_GUID, gop_ptr)
    };
    match st {
        Ok(_) => {
            let msg = alloc::format!(
                "{SUCCESS_TAG} BAR1={base:#x} {}x{} BGRA8",
                paint::FB_W,
                paint::FB_H
            );
            info!("{msg}");
            serial::line(&msg);
            Status::SUCCESS
        }
        Err(e) => {
            info!("reims-vgpu efi-gop: install GOP failed {e:?}");
            serial::line("reims-vgpu efi-gop: install GOP failed");
            e.status()
        }
    }
}
