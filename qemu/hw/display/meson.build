hw_display_modules = {}

system_ss.add(when: 'CONFIG_DDC', if_true: files('i2c-ddc.c'))
system_ss.add(when: 'CONFIG_EDID', if_true: files('edid-generate.c', 'edid-region.c'))

system_ss.add(when: 'CONFIG_FW_CFG_DMA', if_true: files('ramfb.c'))
stub_ss.add(files('ramfb-stubs.c'))
system_ss.add(when: 'CONFIG_FW_CFG_DMA', if_true: files('ramfb-standalone.c'))

system_ss.add(when: 'CONFIG_VGA_CIRRUS', if_true: files('cirrus_vga.c'))
system_ss.add(when: ['CONFIG_VGA_CIRRUS', 'CONFIG_VGA_ISA'], if_true: files('cirrus_vga_isa.c'))
system_ss.add(when: 'CONFIG_G364FB', if_true: files('g364fb.c'))
system_ss.add(when: 'CONFIG_JAZZ_LED', if_true: files('jazz_led.c'))
system_ss.add(when: 'CONFIG_PL110', if_true: files('pl110.c'))
system_ss.add(when: 'CONFIG_IMX6UL_LCDIF', if_true: files('imx6ul_lcdif.c'))
system_ss.add(when: 'CONFIG_SII9022', if_true: files('sii9022.c'))
system_ss.add(when: 'CONFIG_SSD0303', if_true: files('ssd0303.c'))
system_ss.add(when: 'CONFIG_SSD0323', if_true: files('ssd0323.c'))
system_ss.add(when: 'CONFIG_XEN_BUS', if_true: files('xenfb.c'))

system_ss.add(when: 'CONFIG_VGA_PCI', if_true: files('vga-pci.c'))
system_ss.add(when: 'CONFIG_VGA_ISA', if_true: files('vga-isa.c'))
system_ss.add(when: 'CONFIG_VGA_MMIO', if_true: files('vga-mmio.c'))
system_ss.add(when: 'CONFIG_VMWARE_VGA', if_true: files('vmware_vga.c'))
system_ss.add(when: 'CONFIG_BOCHS_DISPLAY', if_true: files('bochs-display.c'))

system_ss.add(when: 'CONFIG_EXYNOS4', if_true: files('exynos4210_fimd.c'))
system_ss.add(when: 'CONFIG_FRAMEBUFFER', if_true: files('framebuffer.c'))

system_ss.add(when: 'CONFIG_RASPI', if_true: files('bcm2835_fb.c'))
system_ss.add(when: 'CONFIG_SM501', if_true: files('sm501.c'))
system_ss.add(when: 'CONFIG_TCX', if_true: files('tcx.c'))
system_ss.add(when: 'CONFIG_CG3', if_true: files('cg3.c'))
system_ss.add(when: 'CONFIG_MACFB', if_true: files('macfb.c'))
system_ss.add(when: 'CONFIG_NEXTCUBE', if_true: files('next-fb.c'))

system_ss.add(when: 'CONFIG_VGA', if_true: files('vga.c'))
system_ss.add(when: 'CONFIG_VIRTIO', if_true: files('virtio-dmabuf.c'))
system_ss.add(when: 'CONFIG_DM163', if_true: files('dm163.c'))
system_ss.add(when: 'CONFIG_VHOST_USER_MEDIA', if_true: files('vhost-user-media.c'))
system_ss.add(when: ['CONFIG_VHOST_USER_MEDIA', 'CONFIG_VIRTIO_PCI'],
  if_true: files('vhost-user-media-pci.c'))

stub_ss.add([files('acpi-vga-stub.c'), pixman])
if (config_all_devices.has_key('CONFIG_VGA_CIRRUS') or
    config_all_devices.has_key('CONFIG_VGA_PCI') or
    config_all_devices.has_key('CONFIG_VMWARE_VGA') or
    config_all_devices.has_key('CONFIG_ATI_VGA')
   )
  system_ss.add(when: 'CONFIG_ACPI', if_true: files('acpi-vga.c'))
endif

if config_all_devices.has_key('CONFIG_QXL')
  qxl_ss = ss.source_set()
  qxl_ss.add(when: 'CONFIG_QXL', if_true: [files('qxl.c', 'qxl-logger.c', 'qxl-render.c'),
                                           pixman, spice])
  qxl_ss.add(when: 'CONFIG_ACPI', if_true: files('acpi-vga.c'))
  hw_display_modules += {'qxl': qxl_ss}
endif

system_ss.add(when: 'CONFIG_DPCD', if_true: files('dpcd.c'))
system_ss.add(when: 'CONFIG_XLNX_DISPLAYPORT', if_true: files('xlnx_dp.c'))

system_ss.add(when: 'CONFIG_ARTIST', if_true: files('artist.c'))

system_ss.add(when: 'CONFIG_ATI_VGA', if_true: [files('ati.c', 'ati_2d.c', 'ati_dbg.c'), pixman])

system_ss.add(when: [pvg, 'CONFIG_MAC_PVG_PCI'],     if_true: [files('apple-gfx.m', 'apple-gfx-pci.m')])
system_ss.add(when: [pvg, 'CONFIG_MAC_PVG_MMIO'],    if_true: [files('apple-gfx.m', 'apple-gfx-mmio.m')])

if config_all_devices.has_key('CONFIG_VIRTIO_GPU')
  virtio_gpu_ss = ss.source_set()
  virtio_gpu_ss.add(when: 'CONFIG_VIRTIO_GPU',
                    if_true: [files('virtio-gpu-base.c', 'virtio-gpu.c'), pixman])
  if host_os == 'linux'
    virtio_gpu_ss.add(files('virtio-gpu-udmabuf.c'))
  else
    virtio_gpu_ss.add(files('virtio-gpu-udmabuf-stubs.c'))
  endif
  virtio_gpu_ss.add(when: 'CONFIG_VHOST_USER_GPU', if_true: files('vhost-user-gpu.c'))
  hw_display_modules += {'virtio-gpu': virtio_gpu_ss}

  if virgl.found() and opengl.found()
    virtio_gpu_gl_ss = ss.source_set()
    virtio_gpu_gl_ss.add(when: ['CONFIG_VIRTIO_GPU', virgl, opengl],
                         if_true: [files('virtio-gpu-gl.c', 'virtio-gpu-virgl.c'), pixman, virgl])
    hw_display_modules += {'virtio-gpu-gl': virtio_gpu_gl_ss}
  endif

  if rutabaga.found()
    virtio_gpu_rutabaga_ss = ss.source_set()
    virtio_gpu_rutabaga_ss.add(when: ['CONFIG_VIRTIO_GPU', rutabaga],
                               if_true: [files('virtio-gpu-rutabaga.c'), pixman])
    hw_display_modules += {'virtio-gpu-rutabaga': virtio_gpu_rutabaga_ss}
  endif
endif

if config_all_devices.has_key('CONFIG_VIRTIO_PCI')
  virtio_gpu_pci_ss = ss.source_set()
  virtio_gpu_pci_ss.add(when: ['CONFIG_VIRTIO_GPU', 'CONFIG_VIRTIO_PCI'],
                        if_true: [files('virtio-gpu-pci.c'), pixman])
  virtio_gpu_pci_ss.add(when: ['CONFIG_VHOST_USER_GPU', 'CONFIG_VIRTIO_PCI'],
                        if_true: files('vhost-user-gpu-pci.c'))
  hw_display_modules += {'virtio-gpu-pci': virtio_gpu_pci_ss}

  if virgl.found() and opengl.found()
    virtio_gpu_pci_gl_ss = ss.source_set()
    virtio_gpu_pci_gl_ss.add(when: ['CONFIG_VIRTIO_GPU', 'CONFIG_VIRTIO_PCI', virgl, opengl],
                             if_true: [files('virtio-gpu-pci-gl.c'), pixman])
    hw_display_modules += {'virtio-gpu-pci-gl': virtio_gpu_pci_gl_ss}
  endif
  if rutabaga.found()
    virtio_gpu_pci_rutabaga_ss = ss.source_set()
    virtio_gpu_pci_rutabaga_ss.add(when: ['CONFIG_VIRTIO_GPU', 'CONFIG_VIRTIO_PCI', rutabaga],
                                   if_true: [files('virtio-gpu-pci-rutabaga.c'), pixman])
    hw_display_modules += {'virtio-gpu-pci-rutabaga': virtio_gpu_pci_rutabaga_ss}
  endif
endif

if config_all_devices.has_key('CONFIG_VIRTIO_VGA')
  virtio_vga_ss = ss.source_set()
  virtio_vga_ss.add(when: 'CONFIG_VIRTIO_VGA',
                    if_true: [files('virtio-vga.c'), pixman])
  virtio_vga_ss.add(when: 'CONFIG_VHOST_USER_VGA',
                    if_true: files('vhost-user-vga.c'))
  virtio_vga_ss.add(when: 'CONFIG_ACPI', if_true: files('acpi-vga.c'))
  hw_display_modules += {'virtio-vga': virtio_vga_ss}

  if virgl.found() and opengl.found()
    virtio_vga_gl_ss = ss.source_set()
    virtio_vga_gl_ss.add(when: ['CONFIG_VIRTIO_VGA', virgl, opengl],
                         if_true: [files('virtio-vga-gl.c'), pixman])
    virtio_vga_gl_ss.add(when: 'CONFIG_ACPI', if_true: files('acpi-vga.c'))
    hw_display_modules += {'virtio-vga-gl': virtio_vga_gl_ss}
  endif

  if rutabaga.found()
    virtio_vga_rutabaga_ss = ss.source_set()
    virtio_vga_rutabaga_ss.add(when: ['CONFIG_VIRTIO_VGA', rutabaga],
                               if_true: [files('virtio-vga-rutabaga.c'), pixman])
    virtio_vga_rutabaga_ss.add(when: 'CONFIG_ACPI', if_true: files('acpi-vga.c'))
    hw_display_modules += {'virtio-vga-rutabaga': virtio_vga_rutabaga_ss}
  endif
endif

system_ss.add(when: 'CONFIG_OMAP', if_true: files('omap_lcdc.c'))


if config_all_devices.has_key('CONFIG_REIMS_VGPU')
  # The Rust device lives beside this checkout by default. Override with
  # `configure --reims-vgpu-dir=/path` (or `meson configure -Dreims_vgpu_dir=`);
  # this is resolved when the build is configured, not when it runs, so an
  # environment variable set at `ninja` time would be silently ignored.
  reims_vgpu_repo_dir = get_option('reims_vgpu_dir')
  if reims_vgpu_repo_dir == ''
    reims_vgpu_repo_dir = meson.project_source_root() / '..' / 'reims-vgpu'
  endif
  reims_vgpu_abi_header_dir = reims_vgpu_repo_dir / 'crates' / 'reims-vgpu' / 'include'
  cargo = find_program('cargo')
  reims_vgpu_rust_sys_libs = ['-lm', '-ldl', '-lpthread']
  reims_vgpu_cargo_features = '--no-default-features --features backend-vulkan,host-window'
  reims_vgpu_link_args = reims_vgpu_rust_sys_libs
  reims_vgpu_built = reims_vgpu_repo_dir / 'target' / 'release' / 'libreims_vgpu.a'
  reims_vgpu_a = custom_target('reims-vgpu-staticlib',
    output: 'libreims_vgpu.a',
    console: true,
    build_always_stale: true,
    command: [find_program('sh'), '-c',
      '"' + cargo.full_path() + '" build --release --manifest-path "' +
      reims_vgpu_repo_dir + '/Cargo.toml" -p reims-vgpu ' + reims_vgpu_cargo_features +
      ' && ( cmp -s "' + reims_vgpu_built + '" "@OUTPUT@" || cp "' +
      reims_vgpu_built + '" "@OUTPUT@" )'])
  reims_vgpu_header_dep = declare_dependency(
    compile_args: ['-I' + reims_vgpu_abi_header_dir])
  reims_vgpu_dep = declare_dependency(
    link_with: reims_vgpu_a,
    link_args: reims_vgpu_link_args,
    compile_args: ['-I' + reims_vgpu_abi_header_dir])
  system_ss.add(when: 'CONFIG_REIMS_VGPU',
    if_true: [files('reims-vgpu-dirty.c', 'reims-vgpu-shim.c'), reims_vgpu_header_dep])
  specific_ss.add(when: 'CONFIG_REIMS_VGPU',
    if_true: [files('reims-vgpu-mmio.c'), reims_vgpu_dep])
endif

modules += { 'hw-display': hw_display_modules }
