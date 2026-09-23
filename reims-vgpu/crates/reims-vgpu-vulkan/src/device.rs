//! The `VkDevice` and everything that dies with it.
//!
//! # What an epoch is, and why it is a boundary rather than a field
//!
//! A `VkDevice` can be lost, and when it is, every handle made from it becomes
//! unusable at the same instant: queues, timeline semaphores, command pools,
//! descriptor pools, allocations, pipelines. The architecture makes that set an
//! object — the device epoch — precisely so no code has to remember which
//! handles those were. There is one active epoch per live session, and an old
//! one is retained only until the host work it already issued is retired or
//! abandoned.
//!
//! Guest semantic validity and Vulkan handle validity are different lifetimes,
//! and a native lease carries both: a session generation says whether the guest
//! still means the object, and [`EpochId`] says whether its handles may still
//! be touched. Collapsing them makes a guest reset destroy live GPU work, or a
//! device loss leave semantic state claiming to own dead handles.
//!
//! That identity is [`reims_vgpu_core::identity::DeviceEpoch`] and is *given*
//! to [`DeviceEpoch::create`] rather than minted here. The session owns which
//! incarnation this is — it is the value [`reims_vgpu_core::retire`] compares a
//! queued destruction against — and a rail that numbered its own would be a
//! second answer to the one question a stale lease asks.
//!
//! # The enabled set is derived from the census and from nothing else
//!
//! [`Enabled::for_census`] is the one place a feature or extension is asked
//! for, and it can only ask for what [`Census::take`] already admitted. That is
//! the difference between a rail that uses a capability and one that hopes for
//! it: enabling a feature the device did not report is undefined behaviour that
//! usually works, right up to the driver where it does not.
//!
//! The floor is enabled unconditionally because a device that could not supply
//! it never became a census in the first place.
//!
//! # This module creates the device and stops
//!
//! It owns no command pool, no descriptor pool, no allocation and no pipeline.
//! Those are per-worker or per-owner and are handed the device rather than
//! living in it — see [`crate::pools`] and [`crate::descriptor`]. What is here
//! is what device creation itself decides: which features are on, which
//! extensions are requested, and which queues exist.

use crate::census::{Census, DeviceExtensions};
use crate::queues::QueuePlan;
use ash::vk;
use reims_vgpu_core::identity::DeviceEpoch as EpochId;
use std::ffi::CString;

/// What device creation turns on.
///
/// Derived from a [`Census`] and never assembled by hand, so nothing can be
/// enabled that was not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Enabled {
    pub extensions: DeviceExtensions,
    /// Always true. It is the support floor, and a device that lacked it never
    /// became a census.
    pub timeline_semaphore: bool,
    /// The capability is on. *Which structure asks for it* is
    /// [`Self::core_promotions`]: below 1.3 it is the extension's own feature
    /// struct, and at 1.3 and above it is `VkPhysicalDeviceVulkan13Features`.
    ///
    /// A promoted feature is still a feature. It is not on because the device
    /// supports the version — it is on because it was requested, and a device
    /// created without requesting it answers `vkCmdPipelineBarrier2` with
    /// undefined behaviour rather than with a refusal.
    pub synchronization2: bool,
    pub dynamic_rendering: bool,
    /// `vkCmdSetPrimitiveTopology`, which decides whether one pipeline can
    /// serve more than one primitive type — see [`crate::topology`].
    ///
    /// The one promoted capability here that is *not* a feature at its core
    /// version: 1.3 made these commands mandatory and
    /// `VkPhysicalDeviceVulkan13Features` has no field for them. So unlike
    /// `synchronization2` beside it, there is nothing to request at 1.3 and
    /// above, and below it the extension's own structure is the only asker.
    pub extended_dynamic_state: bool,
    /// The two instance-divisor features, which a constant-rate and a
    /// divisor-above-one vertex binding respectively need — see
    /// [`crate::vertex`]. Requested together because they arrive in one
    /// structure and the census admitted each on its own.
    pub vertex_attribute_divisor: bool,
    pub vertex_attribute_zero_divisor: bool,
    /// The two `VK_EXT_extended_dynamic_state3` members this rail sets, which
    /// let a pipeline take its fill mode and its depth clamp per draw instead
    /// of baking one pipeline per value — see [`crate::raster`].
    ///
    /// Two fields and not one, unlike `dynamic_cull_and_winding` beside them:
    /// they are two feature bits of one structure and a device may report
    /// either without the other, so the census admits each on its own and the
    /// request has to be able to say so.
    ///
    /// Never promoted, so there is no `core_promotions` route: the extension's
    /// own structure is the only asker.
    pub extended_dynamic_state3_polygon_mode: bool,
    pub extended_dynamic_state3_depth_clamp_enable: bool,
    pub mesh_shader: bool,
    pub descriptor_buffer: bool,
    /// The 1.0 boolean block, requested through
    /// `VkDeviceCreateInfo::pEnabledFeatures`.
    pub depth_clamp: bool,
    pub fill_mode_non_solid: bool,
    /// `VkPhysicalDeviceFeatures::wideLines`, the 1.0 block again.
    ///
    /// `vkCmdSetLineWidth` with anything but 1.0 on a device without it is
    /// invalid (VUID-vkCmdSetLineWidth-lineWidth-00788), and
    /// [`crate::raster::line_width`] refuses one against the same census cell
    /// --- so what is requested here and what a width is checked against are
    /// the same fact.
    pub wide_lines: bool,
    /// `VkPhysicalDeviceFeatures::multiViewport`, the 1.0 block again.
    ///
    /// A pipeline whose `viewportCount` is above one without it is invalid
    /// (VUID-VkPipelineViewportStateCreateInfo-viewportCount-01216), and
    /// [`crate::raster::viewport_slots`] refuses one against the same census
    /// cell --- so what is requested here and what a count is checked against
    /// are the same fact.
    pub multi_viewport: bool,
    /// Also the 1.0 block. A sampler whose plan sets `anisotropyEnable` on a
    /// device that never requested this feature is undefined behaviour, so the
    /// census cell [`crate::sampler::plan`] reads and the feature requested
    /// here are the same fact and are derived from the same place.
    pub sampler_anisotropy: bool,
    /// `VkPhysicalDeviceFeatures::dualSrcBlend`, the 1.0 block again. A
    /// pipeline naming a `SRC1_*` factor without it is invalid, and
    /// [`crate::blend::plan`] refuses one against the same census cell.
    pub dual_src_blend: bool,
    /// `VkPhysicalDeviceFeatures::independentBlend`.
    pub independent_blend: bool,
    /// `VkPhysicalDeviceVulkan12Features::samplerMirrorClampToEdge`. A
    /// promoted feature is still a feature: the address mode enumerant exists
    /// in core 1.2 whether or not it was enabled, and using it unenabled is
    /// exactly the failure the enumerant's presence hides.
    pub sampler_mirror_clamp_to_edge: bool,
    /// Whether the promoted capabilities are asked for through
    /// `VkPhysicalDeviceVulkan13Features` rather than each extension's own
    /// structure.
    ///
    /// Not derivable from the extension list: a 1.3 device need not enumerate
    /// `VK_KHR_synchronization2` at all, so an empty extension list there
    /// means "core", and on a 1.2 device it means "absent".
    pub core_promotions: bool,
}

impl Enabled {
    /// Everything the census admitted, and nothing else.
    #[must_use]
    pub fn for_census(census: &Census) -> Self {
        Self {
            extensions: census.extensions(),
            timeline_semaphore: true,
            synchronization2: census.synchronization2(),
            dynamic_rendering: census.passes().dynamic_rendering,
            extended_dynamic_state: census.topology().dynamic,
            vertex_attribute_divisor: census.vertex().instance_rate_divisor,
            vertex_attribute_zero_divisor: census.vertex().zero_divisor,
            extended_dynamic_state3_polygon_mode: census.raster().dynamic_polygon_mode,
            extended_dynamic_state3_depth_clamp_enable: census.raster().dynamic_depth_clamp,
            mesh_shader: census.stages().mesh_shader,
            descriptor_buffer: census.descriptors().descriptor_buffer,
            depth_clamp: census.raster().depth_clamp,
            fill_mode_non_solid: census.raster().fill_mode_non_solid,
            wide_lines: census.line_widths().wide_lines,
            multi_viewport: census.viewports().multi_viewport,
            sampler_anisotropy: census.samplers().anisotropy,
            dual_src_blend: census.blend().dual_source,
            independent_blend: census.blend().independent,
            sampler_mirror_clamp_to_edge: census.samplers().mirror_clamp_to_edge,
            core_promotions: census.api().at_least(1, 3),
        }
    }
}

/// Why the device could not be created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceFailure {
    CreateDevice {
        result: vk::Result,
    },
    /// The chosen family reported queues and then handed none out. A driver
    /// contradiction rather than a capability answer, so it is its own reason.
    NoQueue {
        family: u32,
    },
}

impl DeviceFailure {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::CreateDevice { .. } => "vk_device_create",
            Self::NoQueue { .. } => "vk_device_no_queue",
        }
    }
}

impl std::fmt::Display for DeviceFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateDevice { result } => write!(f, "{} result={result:?}", self.slug()),
            Self::NoQueue { family } => write!(f, "{} family={family}", self.slug()),
        }
    }
}

/// One `VkDevice`, its identity, and the queue ledger for it.
///
/// Not `Clone`: its `Drop` destroys the device, and a second copy would be a
/// second destruction. Everything made from the device has to be retired before
/// it is dropped, which is what [`Self::wait_idle`] is for.
pub struct DeviceEpoch {
    id: EpochId,
    device: ash::Device,
    census: Census,
    enabled: Enabled,
    queues: QueuePlan,
}

impl DeviceEpoch {
    /// Create the device from a host's catalog.
    ///
    /// `epoch` is the incarnation identity the session assigned; see the module
    /// doc for why it is not minted here.
    ///
    /// One queue is requested from the chosen family. More would be a
    /// concurrency decision, and the architecture puts that behind measurement
    /// rather than behind "the family reported a count".
    ///
    /// # Errors
    ///
    /// [`DeviceFailure`] when the driver refuses creation or hands back no
    /// queue.
    pub fn create(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        census: Census,
        epoch: EpochId,
    ) -> Result<Self, DeviceFailure> {
        let enabled = Enabled::for_census(&census);
        let family = census.queues().universal().index;

        // SAFETY: `physical` belongs to `instance`, and every structure and
        // array the create info points at lives for the closure's whole body.
        let device = with_create_info(&enabled, family, |create| unsafe {
            instance.create_device(physical, create, None)
        })
        .map_err(|result| DeviceFailure::CreateDevice { result })?;

        // SAFETY: the family and index were requested above, so the queue
        // exists. A driver that returns a null handle here is the
        // contradiction `NoQueue` names.
        let queue = unsafe { device.get_device_queue(family, 0) };
        if queue == vk::Queue::null() {
            // SAFETY: nothing was created from the device.
            unsafe { device.destroy_device(None) };
            return Err(DeviceFailure::NoQueue { family });
        }

        Ok(Self {
            id: epoch,
            device,
            census,
            enabled,
            queues: QueuePlan::adopt(census.queues()),
        })
    }
}

/// Build the `VkDeviceCreateInfo` this rail asks for and hand it to `f`.
///
/// A closure for the reason [`crate::pipeline::Build::with_create_info`] takes
/// one: every feature structure is a local that the chain points at, so the
/// chain cannot outlive them and cannot be returned. What it *can* do is be
/// walked --- which is the only way a test can say that an extension in the
/// list has its feature structure on the chain, rather than a reader saying so.
fn with_create_info<R>(
    enabled: &Enabled,
    family: u32,
    f: impl FnOnce(&vk::DeviceCreateInfo) -> R,
) -> R {
    let names: Vec<CString> = enabled
        .extensions
        .names()
        .into_iter()
        .map(|n| CString::new(n).expect("an extension name has no interior NUL"))
        .collect();
    // `c_char` is i8 on x86-64 Linux but u8 on aarch64 Linux; spelling the
    // platform type keeps the aarch64 cross-build working.
    let pointers: Vec<*const std::ffi::c_char> =
        names.iter().map(|n| n.as_ptr()).collect();

    let priorities = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priorities)];

    // Chained only where the capability was admitted, for the reason the
    // census module gives: a feature struct for something the driver never
    // reported is a question it has no answer to.
    let mut vulkan12 = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .sampler_mirror_clamp_to_edge(enabled.sampler_mirror_clamp_to_edge);
    // The promoted pair, asked for through the version's own structure.
    // Chained only at 1.3 and above, where that structure is legal at all.
    let mut vulkan13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(enabled.synchronization2)
        .dynamic_rendering(enabled.dynamic_rendering);
    let mut synchronization2 =
        vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    let mut dynamic_rendering =
        vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
    let mut extended_dynamic_state =
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default().extended_dynamic_state(true);
    // One structure whichever route the capability arrived by, and each
    // half requested only where the census admitted it: a device that
    // reports the divisor and not the zero divisor is asked for exactly
    // what it reported.
    let mut divisor = vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR::default()
        .vertex_attribute_instance_rate_divisor(enabled.vertex_attribute_divisor)
        .vertex_attribute_instance_rate_zero_divisor(enabled.vertex_attribute_zero_divisor);
    // Each member requested only where the census admitted it, like the
    // divisor pair above: a device that offers the polygon mode and not
    // the depth clamp is asked for exactly what it offers.
    let mut extended_dynamic_state_3 =
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
            .extended_dynamic_state3_polygon_mode(enabled.extended_dynamic_state3_polygon_mode)
            .extended_dynamic_state3_depth_clamp_enable(
                enabled.extended_dynamic_state3_depth_clamp_enable,
            );
    let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true);
    let mut descriptor_buffer =
        vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
    // The 1.0 block. Two states a guest sets need these, and a device
    // created without them clips where the guest asked to clamp and fills
    // where it asked for lines — see [`crate::raster`].
    let core_features = vk::PhysicalDeviceFeatures::default()
        .depth_clamp(enabled.depth_clamp)
        .fill_mode_non_solid(enabled.fill_mode_non_solid)
        .wide_lines(enabled.wide_lines)
        .multi_viewport(enabled.multi_viewport)
        .sampler_anisotropy(enabled.sampler_anisotropy)
        .dual_src_blend(enabled.dual_src_blend)
        .independent_blend(enabled.independent_blend);

    let mut create = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&pointers)
        .enabled_features(&core_features)
        .push_next(&mut vulkan12);
    if enabled.core_promotions {
        create = create.push_next(&mut vulkan13);
    } else {
        // Below 1.3 each capability is asked for through its own
        // extension's structure, and only where the extension is in the
        // list above — a feature structure for an extension that was not
        // enabled is a question the driver has no answer to.
        if enabled.extensions.synchronization2 {
            create = create.push_next(&mut synchronization2);
        }
        if enabled.extensions.dynamic_rendering {
            create = create.push_next(&mut dynamic_rendering);
        }
        if enabled.extensions.extended_dynamic_state {
            create = create.push_next(&mut extended_dynamic_state);
        }
    }
    // Outside the `core_promotions` split: 1.4 promoted this and the
    // baseline is 1.2, so a device on any route asks through the same
    // structure. Chained only where something was admitted, so a device
    // that reported neither half is not handed a structure asking for
    // both to be off.
    if enabled.vertex_attribute_divisor || enabled.vertex_attribute_zero_divisor {
        create = create.push_next(&mut divisor);
    }
    // Outside the `core_promotions` split for the opposite reason to the
    // divisor's: nothing ever promoted this extension, so there is no
    // version at which its structure stops being the one to ask through.
    // Chained on the extension being enabled, which
    // `DeviceExtensions::extended_dynamic_state_3` already makes mean
    // "and a member this rail sets came back with it".
    if enabled.extensions.extended_dynamic_state_3 {
        create = create.push_next(&mut extended_dynamic_state_3);
    }
    if enabled.mesh_shader {
        create = create.push_next(&mut mesh);
    }
    if enabled.descriptor_buffer {
        create = create.push_next(&mut descriptor_buffer);
    }

    f(&create)
}

impl DeviceEpoch {
    #[must_use]
    pub const fn id(&self) -> EpochId {
        self.id
    }

    #[must_use]
    pub const fn device(&self) -> &ash::Device {
        &self.device
    }

    /// The catalog this epoch was created against.
    ///
    /// A copy of the host's, not a second reading: a decision made here and a
    /// decision made at selection see one snapshot.
    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    #[must_use]
    pub const fn enabled(&self) -> &Enabled {
        &self.enabled
    }

    /// The ledger of which of this device's queues have an owner.
    ///
    /// `&mut` because claiming is the only thing anyone does with it, and two
    /// claimants would be two owners of one externally-synchronized `VkQueue`.
    pub fn queues(&mut self) -> &mut QueuePlan {
        &mut self.queues
    }

    /// Wait for every submission on this device to finish.
    ///
    /// Only correct at epoch shutdown, after ingress and submissions have
    /// stopped. Routine destruction never comes here: it goes through timeline
    /// retirement, because a device-wide idle in a resource path serialises
    /// every worker behind the slowest one.
    ///
    /// # Errors
    ///
    /// The driver's result, including `VK_ERROR_DEVICE_LOST` — which is not a
    /// reason to skip destruction, since the handles still have to be given
    /// back.
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        // SAFETY: the device is live and this owner is the only one that
        // submits, so no queue is being used concurrently.
        unsafe { self.device.device_wait_idle() }
    }
}

impl Drop for DeviceEpoch {
    fn drop(&mut self) {
        // The architecture's rule: a final clean epoch shutdown may idle the
        // device, and only after ingress and submissions have stopped. A loss
        // makes the wait fail and destruction still has to happen — Vulkan's
        // finite-return rules mean the handles are ours to give back either
        // way.
        let _ = self.wait_idle();
        // SAFETY: this type owns the device, is not `Clone`, and the wait above
        // has completed every submission that could still name a child object.
        //
        // The second half is the owner's and not this type's: `vkDestroyDevice`
        // also requires every child object to have been destroyed already
        // (VUID-vkDestroyDevice-device-05137), and this epoch creates none ---
        // pools, arenas, semaphores, swapchains and natives are all created by
        // whoever holds this. Rust's drop order is what enforces it, so a value
        // holding both must declare this field *after* them.
        unsafe { self.device.destroy_device(None) };
    }
}

impl std::fmt::Debug for DeviceEpoch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceEpoch")
            .field("id", &self.id)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::{extension, ApiVersion, Reported};
    use crate::host::VulkanHost;
    use crate::memory::fixtures as mem;

    fn families() -> [vk::QueueFamilyProperties; 1] {
        [vk::QueueFamilyProperties {
            queue_flags: vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            queue_count: 4,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        }]
    }

    /// A 1.2 device that reports the floor and nothing optional, for the
    /// sweeps that vary one fact at a time through `..`.
    fn bare<'a>(
        memory: &'a vk::PhysicalDeviceMemoryProperties,
        queue_families: &'a [vk::QueueFamilyProperties],
    ) -> Reported<'a> {
        Reported {
            api_version: vk::make_api_version(0, 1, 2, 0),
            extensions: &[extension::SWAPCHAIN],
            timeline_semaphore: true,
            synchronization2: false,
            dynamic_rendering: false,
            depth_clamp: false,
            fill_mode_non_solid: false,
            wide_lines: false,
            line_width_range: [1.0, 1.0],
            multi_viewport: false,
            max_viewports: 1,
            sampler_anisotropy: false,
            max_sampler_anisotropy: 1.0,
            extended_dynamic_state: false,
            dynamic_primitive_topology_unrestricted: None,
            extended_dynamic_state3_polygon_mode: false,
            extended_dynamic_state3_depth_clamp_enable: false,
            vertex_attribute_instance_rate_divisor: false,
            vertex_attribute_instance_rate_zero_divisor: false,
            max_vertex_attrib_divisor: 0,
            vertex_formats: crate::vertex::VertexFormatSupport::NONE,
            dual_src_blend: false,
            independent_blend: false,
            sampler_mirror_clamp_to_edge: false,
            mesh_shader: false,
            descriptor_buffer: false,
            max_push_descriptors: 0,
            max_buffer_size: None,
            host_pointer_importable: false,
            min_imported_host_pointer_alignment: 0,
            memory,
            queue_families,
        }
    }

    fn census_of(api: (u32, u32), extensions: &[&str], features: (bool, bool, bool)) -> Census {
        let memory = mem::apple_m3_max();
        let families = families();
        Census::take(Reported {
            api_version: vk::make_api_version(0, api.0, api.1, 0),
            extensions,
            timeline_semaphore: true,
            synchronization2: features.0,
            dynamic_rendering: false,
            depth_clamp: false,
            fill_mode_non_solid: false,
            wide_lines: false,
            line_width_range: [1.0, 1.0],
            multi_viewport: false,
            max_viewports: 1,
            sampler_anisotropy: false,
            max_sampler_anisotropy: 1.0,
            extended_dynamic_state: false,
            dynamic_primitive_topology_unrestricted: None,
            extended_dynamic_state3_polygon_mode: false,
            extended_dynamic_state3_depth_clamp_enable: false,
            vertex_attribute_instance_rate_divisor: false,
            vertex_attribute_instance_rate_zero_divisor: false,
            max_vertex_attrib_divisor: 0,
            vertex_formats: crate::vertex::VertexFormatSupport::NONE,
            dual_src_blend: false,
            independent_blend: false,
            sampler_mirror_clamp_to_edge: false,
            mesh_shader: features.1,
            descriptor_buffer: features.2,
            max_push_descriptors: 32,
            max_buffer_size: None,
            host_pointer_importable: true,
            min_imported_host_pointer_alignment: 4096,
            memory: &memory,
            queue_families: &families,
        })
        .expect("admitted")
    }

    #[test]
    fn the_floor_is_always_enabled() {
        let bare = census_of((1, 2), &[extension::SWAPCHAIN], (false, false, false));
        let enabled = Enabled::for_census(&bare);
        assert!(enabled.timeline_semaphore);
        assert!(!enabled.synchronization2);
        assert!(!enabled.mesh_shader);
        assert!(!enabled.descriptor_buffer);
        assert_eq!(enabled.extensions.names(), vec![extension::SWAPCHAIN]);
    }

    /// The claim that keeps this from being a wish list.
    #[test]
    fn nothing_is_enabled_that_the_census_did_not_admit() {
        // Every combination of enumerated-extension and reported-feature, at
        // three API versions.
        let mut checked = 0;
        for (major, minor) in [(1u32, 2u32), (1, 3), (1, 4)] {
            for mesh_ext in [false, true] {
                for mesh_feature in [false, true] {
                    for buffer_ext in [false, true] {
                        for buffer_feature in [false, true] {
                            let mut names = vec![extension::SWAPCHAIN];
                            if mesh_ext {
                                names.push(extension::MESH_SHADER);
                            }
                            if buffer_ext {
                                names.push(extension::DESCRIPTOR_BUFFER);
                            }
                            let census = census_of(
                                (major, minor),
                                &names,
                                (false, mesh_feature, buffer_feature),
                            );
                            let enabled = Enabled::for_census(&census);

                            assert_eq!(enabled.mesh_shader, mesh_ext && mesh_feature);
                            assert_eq!(enabled.descriptor_buffer, buffer_ext && buffer_feature);
                            // And the extension list agrees with the features.
                            assert_eq!(enabled.extensions.mesh_shader, enabled.mesh_shader);
                            assert_eq!(
                                enabled.extensions.descriptor_buffer,
                                enabled.descriptor_buffer
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 48);
        // Non-vacuity: the sweep contained a case where each was on.
        let both = census_of(
            (1, 3),
            &[
                extension::SWAPCHAIN,
                extension::MESH_SHADER,
                extension::DESCRIPTOR_BUFFER,
            ],
            (false, true, true),
        );
        let enabled = Enabled::for_census(&both);
        assert!(enabled.mesh_shader && enabled.descriptor_buffer);
    }

    /// A promoted capability is still requested; only the structure that asks
    /// for it changes.
    ///
    /// This test previously asserted the opposite — that a core capability
    /// needs neither a name nor a feature — and the belief cost a device whose
    /// `synchronization2` was never enabled while `vkCmdPipelineBarrier2` was
    /// being recorded into it. A promoted feature is not on because the device
    /// supports the version; it is on because it was asked for.
    #[test]
    fn a_promoted_capability_is_requested_through_the_version_and_not_the_name() {
        let promoted = census_of((1, 3), &[extension::SWAPCHAIN], (false, false, false));
        assert!(promoted.synchronization2(), "core at 1.3");
        let enabled = Enabled::for_census(&promoted);
        assert!(enabled.synchronization2, "still requested");
        assert!(
            enabled.core_promotions,
            "through VkPhysicalDeviceVulkan13Features"
        );
        assert!(
            !enabled.extensions.synchronization2,
            "and not by a name the device need not have enumerated"
        );

        // Through the extension below its promotion, the name is requested and
        // the version structure is not the route.
        let extended = census_of(
            (1, 2),
            &[extension::SWAPCHAIN, extension::SYNCHRONIZATION_2],
            (true, false, false),
        );
        let enabled = Enabled::for_census(&extended);
        assert!(enabled.synchronization2);
        assert!(enabled.extensions.synchronization2);
        assert!(!enabled.core_promotions);

        // A 1.2 device with neither the name nor the feature asks for nothing.
        let bare = census_of((1, 2), &[extension::SWAPCHAIN], (false, false, false));
        let enabled = Enabled::for_census(&bare);
        assert!(!enabled.synchronization2);
        assert!(!enabled.extensions.synchronization2);
        assert!(!enabled.core_promotions);
    }

    /// Every extension this rail enables that publishes a feature structure
    /// has that structure on the device's `pNext` chain, and no extension it
    /// did not enable does.
    ///
    /// The law was previously true only by reading, and it was not true:
    /// `VK_EXT_extended_dynamic_state3` was enumerated with no
    /// `VkPhysicalDeviceExtendedDynamicState3FeaturesEXT` chained anywhere.
    /// The chain is walkable now, so the law is a test --- and a new extension
    /// whose structure nobody remembers to chain fails here rather than on a
    /// host nobody ran.
    #[test]
    fn every_enabled_extension_that_has_features_has_them_on_the_chain() {
        /// Every structure type this rail may chain, and the extension whose
        /// features it carries. `None` is a structure that belongs to a core
        /// version rather than to an extension in the list.
        const CARRIERS: &[(vk::StructureType, Option<&str>)] = &[
            (vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, None),
            (vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, None),
            (
                vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                Some(extension::SYNCHRONIZATION_2),
            ),
            (
                vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
                Some(extension::DYNAMIC_RENDERING),
            ),
            (
                vk::StructureType::PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
                Some(extension::EXTENDED_DYNAMIC_STATE),
            ),
            (
                vk::StructureType::PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
                Some(extension::EXTENDED_DYNAMIC_STATE_3),
            ),
            (
                vk::StructureType::PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
                Some(extension::MESH_SHADER),
            ),
            (
                vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
                Some(extension::DESCRIPTOR_BUFFER),
            ),
            // The one structure with no single owning name: the capability
            // arrives by either spelling of the extension or by 1.4 core, and
            // one structure asks on every route. Checked by its own assertion
            // below rather than against a name.
            (
                vk::StructureType::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR,
                None,
            ),
        ];
        /// The extensions this rail enables that publish no feature structure
        /// at all, so there is nothing for the chain to carry.
        const FEATURELESS: &[&str] = &[
            extension::SWAPCHAIN,
            extension::PUSH_DESCRIPTOR,
            extension::EXTERNAL_MEMORY_HOST,
        ];

        let memory = mem::apple_m3_max();
        let families = families();
        let every = &[
            extension::SWAPCHAIN,
            extension::PUSH_DESCRIPTOR,
            extension::EXTERNAL_MEMORY_HOST,
            extension::SYNCHRONIZATION_2,
            extension::DYNAMIC_RENDERING,
            extension::EXTENDED_DYNAMIC_STATE,
            extension::EXTENDED_DYNAMIC_STATE_3,
            extension::MESH_SHADER,
            extension::DESCRIPTOR_BUFFER,
            extension::VERTEX_ATTRIBUTE_DIVISOR,
        ];

        // Two hosts, at the two ends of what this rail admits: one that
        // reports nothing optional, and one that reports everything. The chain
        // has to match the list on both.
        for (label, reported) in [
            ("nothing optional", bare(&memory, &families)),
            (
                "everything",
                Reported {
                    extensions: every,
                    synchronization2: true,
                    dynamic_rendering: true,
                    extended_dynamic_state: true,
                    extended_dynamic_state3_polygon_mode: true,
                    extended_dynamic_state3_depth_clamp_enable: true,
                    vertex_attribute_instance_rate_divisor: true,
                    vertex_attribute_instance_rate_zero_divisor: true,
                    mesh_shader: true,
                    descriptor_buffer: true,
                    host_pointer_importable: true,
                    min_imported_host_pointer_alignment: 4096,
                    max_push_descriptors: 32,
                    ..bare(&memory, &families)
                },
            ),
        ] {
            let census = Census::take(reported).expect("admitted");
            let enabled = Enabled::for_census(&census);
            let listed = enabled.extensions.names();

            let chained = with_create_info(&enabled, 0, |create| {
                let mut seen = Vec::new();
                let mut next = create.p_next;
                while !next.is_null() {
                    // SAFETY: every structure on this chain begins with the
                    // `VkBaseOutStructure` prefix, and the chain is alive for
                    // the whole closure.
                    let base = unsafe { &*next.cast::<vk::BaseOutStructure<'_>>() };
                    seen.push(base.s_type);
                    next = base.p_next.cast::<std::ffi::c_void>();
                }
                seen
            });

            // Nothing unknown rode along.
            for s_type in &chained {
                assert!(
                    CARRIERS.iter().any(|(known, _)| known == s_type),
                    "{label}: an unrecognized structure {s_type:?} is on the chain"
                );
            }
            // And every extension's structure is present exactly where its
            // extension is.
            for (s_type, owner) in CARRIERS {
                let Some(name) = owner else {
                    continue;
                };
                assert_eq!(
                    chained.contains(s_type),
                    listed.contains(name),
                    "{label}: {name} and {s_type:?} disagree"
                );
            }
            // The divisor structure, on the capability rather than on a name.
            assert_eq!(
                chained.contains(
                    &vk::StructureType::PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR
                ),
                enabled.vertex_attribute_divisor || enabled.vertex_attribute_zero_divisor,
                "{label}: the divisor structure and the divisor capability disagree"
            );
            // The featureless ones have no structure to be missing, which is
            // what keeps the loop above from being vacuously satisfiable by an
            // empty carrier table.
            for name in FEATURELESS {
                assert!(
                    !CARRIERS.iter().any(|(_, owner)| owner == &Some(*name)),
                    "{name} was given a feature structure it does not have"
                );
            }
        }
    }

    /// The two `VK_EXT_extended_dynamic_state3` members are requested exactly
    /// where the census admitted them, and the extension is asked for exactly
    /// where a member was.
    ///
    /// The regression: neither was requested at all. The extension went into
    /// `ppEnabledExtensionNames` and no
    /// `VkPhysicalDeviceExtendedDynamicState3FeaturesEXT` was ever chained, so
    /// `raster::plan` placeheld the pipeline's `polygonMode` and
    /// `depthClampEnable` and declared them dynamic on a device that had
    /// enabled neither feature --- `vkCmdSetPolygonModeEXT` as invalid use,
    /// and on a driver that ignores it a pipeline that silently rasterizes the
    /// placeholder instead of the guest's wireframe fill.
    #[test]
    fn the_two_dynamic_state_3_members_are_requested_one_at_a_time() {
        let memory = mem::apple_m3_max();
        let families = families();
        for polygon_mode in [false, true] {
            for depth_clamp_enable in [false, true] {
                let census = Census::take(Reported {
                    extensions: &[extension::SWAPCHAIN, extension::EXTENDED_DYNAMIC_STATE_3],
                    extended_dynamic_state3_polygon_mode: polygon_mode,
                    extended_dynamic_state3_depth_clamp_enable: depth_clamp_enable,
                    ..bare(&memory, &families)
                })
                .expect("admitted");
                let enabled = Enabled::for_census(&census);

                // Each member on its own, and each is what the raster planner
                // will spend.
                assert_eq!(enabled.extended_dynamic_state3_polygon_mode, polygon_mode);
                assert_eq!(
                    enabled.extended_dynamic_state3_depth_clamp_enable,
                    depth_clamp_enable
                );
                assert_eq!(census.raster().dynamic_polygon_mode, polygon_mode);
                assert_eq!(census.raster().dynamic_depth_clamp, depth_clamp_enable);

                // And the extension travels with them: enabled where a member
                // was admitted, absent where neither was, so no structure is
                // ever chained for an extension that is not in the list and no
                // member is ever spent without one.
                let wanted = polygon_mode || depth_clamp_enable;
                assert_eq!(enabled.extensions.extended_dynamic_state_3, wanted);
                assert_eq!(
                    enabled
                        .extensions
                        .names()
                        .contains(&extension::EXTENDED_DYNAMIC_STATE_3),
                    wanted
                );
            }
        }

        // A device that does not enumerate the extension admits neither member
        // however it answers, because there is no structure to have answered
        // through.
        let census = Census::take(Reported {
            extended_dynamic_state3_polygon_mode: true,
            extended_dynamic_state3_depth_clamp_enable: true,
            ..bare(&memory, &families)
        })
        .expect("admitted");
        let enabled = Enabled::for_census(&census);
        assert!(!enabled.extended_dynamic_state3_polygon_mode);
        assert!(!enabled.extended_dynamic_state3_depth_clamp_enable);
        assert!(!enabled.extensions.extended_dynamic_state_3);
    }

    #[test]
    fn the_optional_raster_features_are_enabled_exactly_when_reported() {
        let memory = mem::apple_m3_max();
        let families = families();
        for depth_clamp in [false, true] {
            for fill_mode_non_solid in [false, true] {
                for multi_viewport in [false, true] {
                    let census = Census::take(Reported {
                        depth_clamp,
                        fill_mode_non_solid,
                        multi_viewport,
                        max_viewports: if multi_viewport { 16 } else { 1 },
                        ..bare(&memory, &families)
                    })
                    .expect("admitted");
                    let enabled = Enabled::for_census(&census);
                    assert_eq!(enabled.depth_clamp, depth_clamp);
                    assert_eq!(enabled.fill_mode_non_solid, fill_mode_non_solid);
                    // And the census agrees, so nothing can be enabled that the
                    // raster planner would not see.
                    assert_eq!(census.raster().depth_clamp, depth_clamp);
                    assert_eq!(census.raster().fill_mode_non_solid, fill_mode_non_solid);
                    // `multiViewport` is the same shape and the same hazard: a
                    // count is admitted against the census cell, so requesting the
                    // feature and admitting the count have to be one fact.
                    assert_eq!(enabled.multi_viewport, multi_viewport);
                    // `wideLines` shares the shape and the hazard, and the bare
                    // cell above it is what every device reports: the one width
                    // that needs no feature.
                    assert!(!enabled.wide_lines);
                    assert_eq!(census.line_widths(), crate::raster::LineWidthCell::NARROW);
                    assert_eq!(census.viewports().multi_viewport, multi_viewport);
                    assert_eq!(
                        crate::raster::viewport_slots(2, census.viewports()).is_ok(),
                        multi_viewport,
                        "a second viewport is admitted exactly where the feature is"
                    );
                }
            }
        }
    }

    /// The two sampler features are on the same footing as the raster pair:
    /// the census cell `sampler::plan` reads and the feature this module asks
    /// for are one fact. A device that enabled neither and planned with a cell
    /// that said both would be the undefined-behaviour case, and it is the one
    /// this asserts cannot arise.
    #[test]
    fn the_sampler_features_are_enabled_exactly_when_reported() {
        let memory = mem::apple_m3_max();
        let families = families();
        for anisotropy in [false, true] {
            for mirror in [false, true] {
                let census = Census::take(Reported {
                    sampler_anisotropy: anisotropy,
                    sampler_mirror_clamp_to_edge: mirror,
                    ..Reported {
                        api_version: vk::make_api_version(0, 1, 2, 0),
                        extensions: &[extension::SWAPCHAIN],
                        timeline_semaphore: true,
                        synchronization2: false,
                        dynamic_rendering: false,
                        depth_clamp: false,
                        fill_mode_non_solid: false,
                        wide_lines: false,
                        line_width_range: [1.0, 1.0],
                        multi_viewport: false,
                        max_viewports: 1,
                        sampler_anisotropy: false,
                        max_sampler_anisotropy: 16.0,
                        extended_dynamic_state: false,
                        dynamic_primitive_topology_unrestricted: None,
                        extended_dynamic_state3_polygon_mode: false,
                        extended_dynamic_state3_depth_clamp_enable: false,
                        vertex_attribute_instance_rate_divisor: false,
                        vertex_attribute_instance_rate_zero_divisor: false,
                        max_vertex_attrib_divisor: 0,
                        vertex_formats: crate::vertex::VertexFormatSupport::NONE,
                        dual_src_blend: false,
                        independent_blend: false,
                        sampler_mirror_clamp_to_edge: false,
                        mesh_shader: false,
                        descriptor_buffer: false,
                        max_push_descriptors: 0,
                        max_buffer_size: None,
                        host_pointer_importable: false,
                        min_imported_host_pointer_alignment: 0,
                        memory: &memory,
                        queue_families: &families,
                    }
                })
                .expect("admitted");
                let enabled = Enabled::for_census(&census);
                assert_eq!(enabled.sampler_anisotropy, anisotropy);
                assert_eq!(enabled.sampler_mirror_clamp_to_edge, mirror);
                assert_eq!(census.samplers().anisotropy, anisotropy);
                assert_eq!(census.samplers().mirror_clamp_to_edge, mirror);
            }
        }
    }

    /// Same footing again: a pipeline naming `SRC1_*` without `dualSrcBlend`
    /// is invalid, and `crate::blend::plan` refuses one against the very cell
    /// this module's request is derived from.
    #[test]
    fn the_blend_features_are_enabled_exactly_when_reported() {
        let memory = mem::apple_m3_max();
        let families = families();
        for dual in [false, true] {
            for independent in [false, true] {
                let census = Census::take(Reported {
                    dual_src_blend: dual,
                    independent_blend: independent,
                    ..Reported {
                        api_version: vk::make_api_version(0, 1, 2, 0),
                        extensions: &[extension::SWAPCHAIN],
                        timeline_semaphore: true,
                        synchronization2: false,
                        dynamic_rendering: false,
                        depth_clamp: false,
                        fill_mode_non_solid: false,
                        wide_lines: false,
                        line_width_range: [1.0, 1.0],
                        multi_viewport: false,
                        max_viewports: 1,
                        sampler_anisotropy: false,
                        max_sampler_anisotropy: 16.0,
                        extended_dynamic_state: false,
                        dynamic_primitive_topology_unrestricted: None,
                        extended_dynamic_state3_polygon_mode: false,
                        extended_dynamic_state3_depth_clamp_enable: false,
                        vertex_attribute_instance_rate_divisor: false,
                        vertex_attribute_instance_rate_zero_divisor: false,
                        max_vertex_attrib_divisor: 0,
                        vertex_formats: crate::vertex::VertexFormatSupport::NONE,
                        dual_src_blend: false,
                        independent_blend: false,
                        sampler_mirror_clamp_to_edge: false,
                        mesh_shader: false,
                        descriptor_buffer: false,
                        max_push_descriptors: 0,
                        max_buffer_size: None,
                        host_pointer_importable: false,
                        min_imported_host_pointer_alignment: 0,
                        memory: &memory,
                        queue_families: &families,
                    }
                })
                .expect("admitted");
                let enabled = Enabled::for_census(&census);
                assert_eq!(enabled.dual_src_blend, dual);
                assert_eq!(enabled.independent_blend, independent);
                assert_eq!(census.blend().dual_source, dual);
                assert_eq!(census.blend().independent, independent);
            }
        }
    }

    /// The identity is the session's, not this module's — which is the whole
    /// point: it is the value a queued destruction is compared against.
    #[test]
    fn the_epoch_identity_is_the_one_the_session_assigned() {
        let a = EpochId::FIRST;
        let b = a.next();
        assert_ne!(a, b);
        assert!(b > a, "an identity has to say which incarnation is later");
    }

    #[test]
    fn a_failure_names_itself() {
        let refused = DeviceFailure::CreateDevice {
            result: vk::Result::ERROR_INITIALIZATION_FAILED,
        };
        assert_eq!(refused.slug(), "vk_device_create");
        assert!(refused.to_string().contains("INITIALIZATION"));
        assert!(DeviceFailure::NoQueue { family: 3 }
            .to_string()
            .contains("family=3"));
    }

    /// Bring up a real device: the part purity cannot reach is whether a driver
    /// accepts the exact feature chain and extension list this census produced.
    /// Every combination that reaches `create` is one the census admitted, so a
    /// refusal here would mean the admission rule is wrong.
    #[test]
    fn a_real_device_accepts_the_set_its_census_admitted() {
        let Ok(host) = VulkanHost::open("reims-vgpu-vulkan device test") else {
            println!("no real device: nothing to create");
            return;
        };
        let census = host.census();
        let epoch = DeviceEpoch::create(
            host.instance(),
            host.physical_device(),
            census,
            EpochId::FIRST,
        )
        .expect("the driver refused a set its own census admitted");

        println!("real device: {:?} {:?}", epoch.id(), epoch.enabled());
        assert_eq!(epoch.census(), census, "one snapshot, not a second reading");
        assert!(epoch.enabled().timeline_semaphore);

        // A second incarnation on the same device carries the identity the
        // session gave it, so a lease from the first is recognisably stale.
        let second = DeviceEpoch::create(
            host.instance(),
            host.physical_device(),
            census,
            EpochId::FIRST.next(),
        )
        .expect("a second device");
        assert_ne!(epoch.id(), second.id());

        // The queue ledger is this epoch's, and hands each queue out once.
        let mut epoch = epoch;
        let family = census.queues().universal().index;
        let owned = epoch.queues().claim_in(family, 0).expect("queue zero");
        assert!(
            epoch.queues().claim_in(family, 0).is_none(),
            "a VkQueue has exactly one owner"
        );
        epoch.queues().release(owned);

        assert_eq!(epoch.wait_idle(), Ok(()));
        drop(second);
        drop(epoch);
    }

    #[test]
    fn a_below_floor_version_never_reaches_device_creation() {
        // Not a device test: it asserts that the type this module takes can
        // only have come from an admitted census, so `create` has no
        // below-floor case to handle.
        let memory = mem::apple_m3_max();
        let families = families();
        let refused = Census::take(Reported {
            api_version: vk::make_api_version(0, 1, 1, 0),
            extensions: &[extension::SWAPCHAIN],
            timeline_semaphore: true,
            synchronization2: false,
            dynamic_rendering: false,
            depth_clamp: false,
            fill_mode_non_solid: false,
            wide_lines: false,
            line_width_range: [1.0, 1.0],
            multi_viewport: false,
            max_viewports: 1,
            sampler_anisotropy: false,
            max_sampler_anisotropy: 1.0,
            extended_dynamic_state: false,
            dynamic_primitive_topology_unrestricted: None,
            extended_dynamic_state3_polygon_mode: false,
            extended_dynamic_state3_depth_clamp_enable: false,
            vertex_attribute_instance_rate_divisor: false,
            vertex_attribute_instance_rate_zero_divisor: false,
            max_vertex_attrib_divisor: 0,
            vertex_formats: crate::vertex::VertexFormatSupport::NONE,
            dual_src_blend: false,
            independent_blend: false,
            sampler_mirror_clamp_to_edge: false,
            mesh_shader: false,
            descriptor_buffer: false,
            max_push_descriptors: 0,
            max_buffer_size: None,
            host_pointer_importable: false,
            min_imported_host_pointer_alignment: 0,
            memory: &memory,
            queue_families: &families,
        });
        assert_eq!(
            refused,
            Err(crate::census::Floor::ApiTooOld {
                reported: ApiVersion { major: 1, minor: 1 }
            })
        );
    }
}
