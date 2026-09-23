//! Locate product `reims-vgpu-pci` (0x106B:0xEEEE) and return BAR1 base.

use uefi::prelude::*;
use uefi::proto::unsafe_protocol;
use uefi::{boot, Identify};

const VENDOR_ID: u16 = 0x106B;
const DEVICE_ID: u16 = 0xEEEE;

const PCI_IO_WIDTH_UINT16: u32 = 1;
const PCI_IO_WIDTH_UINT32: u32 = 2;
const PCI_ATTR_OP_ENABLE: u32 = 2;
const PCI_ATTR_MEM_IO_BM: u64 = 0x0600 | 0x0100 | 0x0400;

#[repr(C)]
struct PciIoAccess {
    read: unsafe extern "efiapi" fn(
        this: *mut PciIoFull,
        width: u32,
        bar_index: u8,
        offset: u64,
        count: usize,
        buffer: *mut u8,
    ) -> Status,
    write: unsafe extern "efiapi" fn(
        this: *mut PciIoFull,
        width: u32,
        bar_index: u8,
        offset: u64,
        count: usize,
        buffer: *mut u8,
    ) -> Status,
}

#[repr(C)]
struct PciIoCfg {
    read: unsafe extern "efiapi" fn(
        this: *mut PciIoFull,
        width: u32,
        offset: u32,
        count: usize,
        buffer: *mut u8,
    ) -> Status,
    write: unsafe extern "efiapi" fn(
        this: *mut PciIoFull,
        width: u32,
        offset: u32,
        count: usize,
        buffer: *mut u8,
    ) -> Status,
}

#[repr(C)]
struct PciIoFull {
    poll_mem: usize,
    poll_io: usize,
    mem: PciIoAccess,
    io: PciIoAccess,
    pci: PciIoCfg,
    copy_mem: usize,
    map: usize,
    unmap: usize,
    allocate_buffer: usize,
    free_buffer: usize,
    flush: usize,
    get_location: usize,
    attributes: unsafe extern "efiapi" fn(
        this: *mut PciIoFull,
        operation: u32,
        attributes: u64,
        result: *mut u64,
    ) -> Status,
    get_bar_attributes: usize,
    set_bar_attributes: usize,
    rom_size: u64,
    rom_image: *mut u8,
}

#[unsafe_protocol("4cf5b200-68b8-4ca5-9eec-b23e3f50029a")]
#[repr(transparent)]
struct RawPciIo(PciIoFull);

fn cfg_u16(pci: *mut PciIoFull, off: u32) -> Result<u16, Status> {
    let mut v = 0u16;
    let st = unsafe {
        ((*pci).pci.read)(
            pci,
            PCI_IO_WIDTH_UINT16,
            off,
            1,
            &mut v as *mut _ as *mut u8,
        )
    };
    if st.is_error() {
        Err(st)
    } else {
        Ok(v)
    }
}

fn cfg_u32(pci: *mut PciIoFull, off: u32) -> Result<u32, Status> {
    let mut v = 0u32;
    let st = unsafe {
        ((*pci).pci.read)(
            pci,
            PCI_IO_WIDTH_UINT32,
            off,
            1,
            &mut v as *mut _ as *mut u8,
        )
    };
    if st.is_error() {
        Err(st)
    } else {
        Ok(v)
    }
}

fn bar1_base(pci: *mut PciIoFull) -> Result<u64, Status> {
    // BAR1 is at config offset 0x14 (after BAR0 at 0x10).
    let bar = cfg_u32(pci, 0x14)?;
    if bar & 1 != 0 {
        return Err(Status::UNSUPPORTED);
    }
    let mut addr = (bar & 0xffff_fff0) as u64;
    if (bar >> 1) & 3 == 2 {
        addr |= (cfg_u32(pci, 0x18)? as u64) << 32;
    }
    if addr == 0 {
        Err(Status::NOT_FOUND)
    } else {
        Ok(addr)
    }
}

fn with_product_pci<F, T>(f: F) -> Result<T, Status>
where
    F: FnOnce(*mut PciIoFull) -> Result<T, Status>,
{
    use uefi::boot::{OpenProtocolAttributes, OpenProtocolParams, SearchType};

    let handles = boot::locate_handle_buffer(SearchType::ByProtocol(&RawPciIo::GUID))
        .map_err(|e| e.status())?;
    for &handle in handles.iter() {
        let opened = unsafe {
            boot::open_protocol::<RawPciIo>(
                OpenProtocolParams {
                    handle,
                    agent: boot::image_handle(),
                    controller: None,
                },
                OpenProtocolAttributes::GetProtocol,
            )
        };
        let Ok(io) = opened else {
            continue;
        };
        let pci = (&*io as *const RawPciIo as *mut RawPciIo) as *mut PciIoFull;
        let Ok(vid) = cfg_u16(pci, 0) else {
            continue;
        };
        let Ok(did) = cfg_u16(pci, 2) else {
            continue;
        };
        if vid != VENDOR_ID || did != DEVICE_ID {
            continue;
        }
        let mut attrs = 0u64;
        unsafe {
            let _ = ((*pci).attributes)(pci, PCI_ATTR_OP_ENABLE, PCI_ATTR_MEM_IO_BM, &mut attrs);
        }
        return f(pci);
    }
    Err(Status::NOT_FOUND)
}

/// Enable MEM/IO/BM on 0x106B:0xEEEE and return its BAR1 physical base.
pub fn find_bar1() -> Result<u64, Status> {
    with_product_pci(bar1_base)
}

fn mem_write_u32(pci: *mut PciIoFull, bar: u8, offset: u64, val: u32) -> Result<(), Status> {
    let mut v = val;
    let st = unsafe {
        ((*pci).mem.write)(
            pci,
            PCI_IO_WIDTH_UINT32,
            bar,
            offset,
            1,
            &mut v as *mut _ as *mut u8,
        )
    };
    if st.is_error() {
        Err(st)
    } else {
        Ok(())
    }
}

/// Program Apple EFI early-display regs on BAR0.
///
/// Host pre-boundary console prefers `efi_fb_start` (0x1210) when non-zero so it
/// can follow the kernel video console if the guest relocates off BAR1 into
/// system RAM (live: `console relocated to 0xf1000000` vs VMware stay-on-BAR).
pub fn program_efi_fb_regs(bar1_base: u64, width: usize, height: usize) -> Result<(), Status> {
    const BAR0: u8 = 0;
    // Control block is at BAR0 + 0x1000; EFI FB start absolute offset 0x1210.
    const OFF_FB_START: u64 = 0x1210;
    const OFF_FB_LENGTH: u64 = 0x1214;
    const OFF_FB_DEPTH: u64 = 0x1218;
    const OFF_FB_STRIDE: u64 = 0x1228;
    let stride = (width * 4) as u32;
    let len = (width * height * 4) as u32;
    with_product_pci(|pci| {
        // 32-bit FB bases under 4G (live BAR1=0x80000000; relocate observed at 0xf1000000).
        mem_write_u32(pci, BAR0, OFF_FB_START, bar1_base as u32)?;
        mem_write_u32(pci, BAR0, OFF_FB_LENGTH, len)?;
        mem_write_u32(pci, BAR0, OFF_FB_DEPTH, 32)?;
        mem_write_u32(pci, BAR0, OFF_FB_STRIDE, stride)?;
        Ok(())
    })
}
