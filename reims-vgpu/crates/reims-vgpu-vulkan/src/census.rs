//! What this physical device offers, taken once, and the floor it has to clear
//! to be used at all.
//!
//! # One snapshot, because two queries of one fact can disagree
//!
//! Every capability decision this rail makes — which memory class an allocation
//! gets, which queue it submits to, which mechanism carries a descriptor,
//! whether a barrier can name the mesh stage — reads from a [`Census`] taken
//! once at device bring-up and never re-queried. A rail that asked the driver
//! again at each decision would be one where a lost device, a re-enumerated
//! extension list or a differently-populated feature chain could make the same
//! guest stream mean two things in one run.
//!
//! [`Census`] is therefore immutable, `Copy`, and has no method that asks the
//! driver anything. It is the *catalog* the architecture allows to be
//! process-global: no command pools, no queues, no resources, no submission
//! state, no mutable policy.
//!
//! # A name is not a capability, and this type cannot hold one
//!
//! The rule is that gates are on structural capabilities and never on a driver
//! name, a vendor id, an API implementation, or `VK_KHR_portability_subset`.
//! That rule is enforced here by absence: [`Reported`] has no field for a
//! vendor id, a device id, a device name, or a driver id, so a gate downstream
//! of this module has nothing to branch on even if somebody wanted to. What
//! comes out is a set of measured facts, and the census is the only thing the
//! rest of the crate is given.
//!
//! # An extension in the list is not a feature
//!
//! `vkEnumerateDeviceExtensionProperties` says the entry points can be
//! resolved. It does not say the feature bit is set, and a rail that used a
//! feature it did not enable is one whose failures are validation errors on
//! somebody else's machine. So every optional capability here needs both: the
//! extension present *and* the feature reported. [`Census::take`] is where the
//! two are joined, and it is the only place.
//!
//! # The floor is queried, not implied
//!
//! Timeline semaphores are a support floor for this rail, and Vulkan 1.2 being
//! the baseline does not supply them: `timelineSemaphore` is a feature that
//! must be queried and enabled. So a device that reports 1.2 with the feature
//! off is refused with [`Floor::NoTimelineSemaphores`] rather than quietly
//! taking a blocking drain path — the architecture is explicit that no such
//! compatibility path is installed, because installing one makes the fast path
//! untested on exactly the hosts that need it.

use crate::barrier::StageSupport;
use crate::descriptor::DescriptorCell;
use crate::memory::{classify_memory, MemoryProfile};
use crate::placement::HostCell;
use crate::queues::{self, QueueChoice};
use ash::vk;

/// A Vulkan API version, as the two numbers a floor is expressed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ApiVersion {
    pub major: u32,
    pub minor: u32,
}

impl ApiVersion {
    /// The version the device reports in `VkPhysicalDeviceProperties::apiVersion`.
    ///
    /// The patch and variant fields are dropped deliberately: no decision this
    /// rail makes is allowed to turn on a patch level, and keeping them would
    /// make `ApiVersion`'s ordering compare things no gate may consult.
    #[must_use]
    pub const fn decode(packed: u32) -> Self {
        Self {
            major: vk::api_version_major(packed),
            minor: vk::api_version_minor(packed),
        }
    }

    #[must_use]
    pub const fn at_least(self, major: u32, minor: u32) -> bool {
        self.major > major || (self.major == major && self.minor >= minor)
    }
}

impl std::fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Extension names this census asks about.
///
/// Spelled once here rather than at each comparison, so a typo is a compile
/// error in one place instead of a capability that is silently always absent.
///
/// `VK_KHR_timeline_semaphore` is deliberately absent: 1.2 is the baseline and
/// promoted it, so the extension name is never the question — the feature bit
/// is, and that is a field on [`Reported`].
pub mod extension {
    pub const PUSH_DESCRIPTOR: &str = "VK_KHR_push_descriptor";
    pub const DESCRIPTOR_BUFFER: &str = "VK_EXT_descriptor_buffer";
    pub const MESH_SHADER: &str = "VK_EXT_mesh_shader";
    pub const EXTERNAL_MEMORY_HOST: &str = "VK_EXT_external_memory_host";
    pub const SYNCHRONIZATION_2: &str = "VK_KHR_synchronization2";
    pub const SWAPCHAIN: &str = "VK_KHR_swapchain";
    /// Carries `maxBufferSize`, which 1.3 promoted to core.
    pub const MAINTENANCE_4: &str = "VK_KHR_maintenance4";
    pub const DYNAMIC_RENDERING: &str = "VK_KHR_dynamic_rendering";
    /// Carries `vkCmdSetPrimitiveTopology`, which 1.3 promoted to core.
    pub const EXTENDED_DYNAMIC_STATE: &str = "VK_EXT_extended_dynamic_state";
    /// Carries `dynamicPrimitiveTopologyUnrestricted`, which is a *property*
    /// and has no core promotion — a 1.4 device that does not enumerate this
    /// extension has not said it is unrestricted.
    ///
    /// Deliberately absent from [`super::DeviceExtensions`]: a physical-device
    /// property is readable from a *supported* extension and nothing here
    /// calls a command this one adds, so requesting it at device creation
    /// would be asking for something this rail does not use.
    pub const EXTENDED_DYNAMIC_STATE_3: &str = "VK_EXT_extended_dynamic_state3";
    /// Carries the two instance-divisor features and `maxVertexAttribDivisor`,
    /// which 1.4 promoted to core.
    pub const VERTEX_ATTRIBUTE_DIVISOR: &str = "VK_KHR_vertex_attribute_divisor";
    /// The same capability under its earlier vendor name. A device may
    /// enumerate either, and the feature is the question rather than which
    /// spelling it arrived under.
    pub const VERTEX_ATTRIBUTE_DIVISOR_EXT: &str = "VK_EXT_vertex_attribute_divisor";
}

/// What the driver said, as the caller read it off the device.
///
/// A borrowed view rather than an owned struct: the caller already holds the
/// enumerated extension list and the property structures, and copying them
/// would create a second place a fact lives.
///
/// Deliberately has no vendor id, device id, device name or driver id. See the
/// module doc.
#[derive(Clone, Copy, Debug)]
pub struct Reported<'a> {
    /// `VkPhysicalDeviceProperties::apiVersion`, packed.
    pub api_version: u32,
    /// Device extension names, exactly as enumerated.
    pub extensions: &'a [&'a str],
    /// `VkPhysicalDeviceVulkan12Features::timelineSemaphore`.
    pub timeline_semaphore: bool,
    /// `VkPhysicalDeviceSynchronization2Features::synchronization2`.
    pub synchronization2: bool,
    /// `VkPhysicalDeviceDynamicRenderingFeatures::dynamicRendering`.
    pub dynamic_rendering: bool,
    /// `VkPhysicalDeviceFeatures::depthClamp`.
    pub depth_clamp: bool,
    /// `VkPhysicalDeviceFeatures::fillModeNonSolid`.
    pub fill_mode_non_solid: bool,
    /// `VkPhysicalDeviceFeatures::wideLines`.
    pub wide_lines: bool,
    /// `VkPhysicalDeviceLimits::lineWidthRange`, `[min, max]`.
    ///
    /// A limit rather than a feature, so it is always reported --- and it is
    /// required to contain 1.0 on every device, which is why a guest that
    /// never sets a width needs no capability. See
    /// [`crate::raster::LineWidthCell`].
    pub line_width_range: [f32; 2],
    /// `VkPhysicalDeviceFeatures::multiViewport`.
    pub multi_viewport: bool,
    /// `VkPhysicalDeviceLimits::maxViewports`.
    ///
    /// A limit rather than a feature, so it is always reported --- and it is
    /// one exactly where the feature above is off, which is why both travel:
    /// see [`crate::raster::ViewportCell`].
    pub max_viewports: u32,
    /// `VkPhysicalDeviceFeatures::samplerAnisotropy`.
    pub sampler_anisotropy: bool,
    /// `VkPhysicalDeviceLimits::maxSamplerAnisotropy`.
    ///
    /// A limit rather than a feature, so it is always reported; it is only
    /// *meaningful* where `sampler_anisotropy` is set, and the cell it lands
    /// in says so.
    pub max_sampler_anisotropy: f32,
    /// `VkPhysicalDeviceExtendedDynamicStateFeaturesEXT::extendedDynamicState`.
    pub extended_dynamic_state: bool,
    /// `VkPhysicalDeviceExtendedDynamicState3PropertiesEXT::dynamicPrimitiveTopologyUnrestricted`,
    /// when this device reported one.
    ///
    /// `None` and `Some(false)` mean the same thing to a decision and are
    /// still different facts: the first is a device that was never asked, and
    /// substituting one for the other here would make the census say a driver
    /// answered when it did not.
    pub dynamic_primitive_topology_unrestricted: Option<bool>,
    /// `…ExtendedDynamicState3FeaturesEXT::extendedDynamicState3PolygonMode`.
    ///
    /// A *feature*, unlike the property above: the two live in different
    /// structures of the same extension, and a device may report the property
    /// while offering neither dynamic member.
    pub extended_dynamic_state3_polygon_mode: bool,
    /// `…ExtendedDynamicState3FeaturesEXT::extendedDynamicState3DepthClampEnable`.
    pub extended_dynamic_state3_depth_clamp_enable: bool,
    /// `…VertexAttributeDivisorFeatures::vertexAttributeInstanceRateDivisor`.
    pub vertex_attribute_instance_rate_divisor: bool,
    /// `…VertexAttributeDivisorFeatures::vertexAttributeInstanceRateZeroDivisor`.
    pub vertex_attribute_instance_rate_zero_divisor: bool,
    /// `…VertexAttributeDivisorProperties::maxVertexAttribDivisor`. Meaningless
    /// without the feature above, and never read without it.
    pub max_vertex_attrib_divisor: u32,
    /// Which formats reported `VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT`.
    ///
    /// Measured by the caller because it is one
    /// `vkGetPhysicalDeviceFormatProperties` per format, and the census asks
    /// no driver anything.
    pub vertex_formats: crate::vertex::VertexFormatSupport,
    /// `VkPhysicalDeviceFeatures::dualSrcBlend`.
    pub dual_src_blend: bool,
    /// `VkPhysicalDeviceFeatures::independentBlend`.
    pub independent_blend: bool,
    /// `VkPhysicalDeviceVulkan12Features::samplerMirrorClampToEdge`.
    ///
    /// 1.2 is the baseline and promoted the extension, so the name is never
    /// the question here — the feature bit is. Same shape as
    /// `timeline_semaphore` above, and for the same reason.
    pub sampler_mirror_clamp_to_edge: bool,
    /// `VkPhysicalDeviceMeshShaderFeaturesEXT::meshShader`.
    pub mesh_shader: bool,
    /// `VkPhysicalDeviceDescriptorBufferFeaturesEXT::descriptorBuffer`.
    pub descriptor_buffer: bool,
    /// `VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors`.
    pub max_push_descriptors: u32,
    /// `VkPhysicalDeviceMaintenance4Properties::maxBufferSize`, when this
    /// device reported one. See [`crate::buffer::BufferLimits`] for why the
    /// absence is carried rather than substituted.
    pub max_buffer_size: Option<u64>,
    /// Whether `vkGetPhysicalDeviceExternalBufferProperties` reported
    /// `VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT` for
    /// `HOST_ALLOCATION_EXT`.
    ///
    /// `VK_EXT_external_memory_host` has no feature structure, so this is the
    /// only thing that plays the part one would: the extension name says the
    /// entry points resolve, and this says the handle type can actually be
    /// imported. Measured by the caller because the census asks no driver
    /// anything, exactly like `vertex_formats`.
    pub host_pointer_importable: bool,
    /// `VkPhysicalDeviceExternalMemoryHostPropertiesEXT::minImportedHostPointerAlignment`.
    ///
    /// The granularity a host pointer and an allocation size must both meet.
    /// It is a device fact and not a page size — MoltenVK reports Apple's and
    /// a Linux driver may report more — so it is asked for rather than
    /// assumed. Zero means the device never filled it in.
    pub min_imported_host_pointer_alignment: u64,
    pub memory: &'a vk::PhysicalDeviceMemoryProperties,
    pub queue_families: &'a [vk::QueueFamilyProperties],
}

impl Reported<'_> {
    /// Whether the device enumerated an extension.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.extensions.contains(&name)
    }
}

/// Why this device cannot be used at all.
///
/// A floor failure is a typed refusal and never a degraded mode. Each variant
/// carries what was reported, so the refusal says which fact was missing rather
/// than that a device "was not suitable".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Floor {
    /// Vulkan 1.2 is the baseline on every supported host.
    ApiTooOld { reported: ApiVersion },
    /// The device reports a new enough API but has `timelineSemaphore` off.
    ///
    /// Distinct from [`Self::ApiTooOld`] precisely because it is the case the
    /// version number does not cover; see the module doc.
    NoTimelineSemaphores { reported: ApiVersion },
    /// No queue family can both draw and dispatch.
    NoUsableQueue { decline: queues::Decline },
    /// The device cannot present.
    NoSwapchain,
    /// The device's extension list could not be read, so no fact about it was
    /// ever reported.
    ///
    /// Every other variant names a fact the device reported and this rail
    /// needs. This one names the absence of the report itself: a judgement
    /// built on an empty list would refuse the device for missing whichever
    /// extension is checked first, which is a reason the device never gave.
    /// See the module doc.
    Unenumerable { result: vk::Result },
}

impl Floor {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ApiTooOld { .. } => "vk_census_api_too_old",
            Self::NoTimelineSemaphores { .. } => "vk_census_no_timeline_semaphores",
            Self::NoUsableQueue { .. } => "vk_census_no_usable_queue",
            Self::NoSwapchain => "vk_census_no_swapchain",
            Self::Unenumerable { .. } => "vk_census_unenumerable",
        }
    }
}

impl std::fmt::Display for Floor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiTooOld { reported } | Self::NoTimelineSemaphores { reported } => {
                write!(f, "{} reported={reported}", self.slug())
            }
            Self::NoUsableQueue { decline } => write!(f, "{} {decline}", self.slug()),
            Self::NoSwapchain => f.write_str(self.slug()),
            Self::Unenumerable { result } => write!(f, "{} result={result:?}", self.slug()),
        }
    }
}

/// Which extension names this rail has to ask for at device creation.
///
/// Separate from the capabilities themselves, because a capability and the
/// route it arrived by are different facts. Push descriptors are core in 1.4
/// and `synchronization2` is core in 1.3: on such a device the capability is
/// present and the extension may not be enumerated at all, so a device
/// creation that requested it by name would be refused for asking about
/// something the driver never offered.
///
/// So each field here means "enumerated, wanted, and not already core at this
/// device's version" — which is exactly the list `ppEnabledExtensionNames`
/// takes, and nothing else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceExtensions {
    pub swapchain: bool,
    pub push_descriptor: bool,
    pub descriptor_buffer: bool,
    pub mesh_shader: bool,
    pub external_memory_host: bool,
    pub synchronization2: bool,
    pub dynamic_rendering: bool,
    pub extended_dynamic_state: bool,
    /// Enabled only where a dynamic member this rail actually sets came back
    /// with it. The topology *property* is readable without enabling anything,
    /// so a device that reports it and no feature does not enumerate here.
    pub extended_dynamic_state_3: bool,
    /// Whichever spelling this device enumerated, when the capability did not
    /// arrive through core.
    pub vertex_attribute_divisor: Option<&'static str>,
}

impl DeviceExtensions {
    /// The names, in a stable order.
    #[must_use]
    pub fn names(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        for (wanted, name) in [
            (self.swapchain, extension::SWAPCHAIN),
            (self.push_descriptor, extension::PUSH_DESCRIPTOR),
            (self.descriptor_buffer, extension::DESCRIPTOR_BUFFER),
            (self.mesh_shader, extension::MESH_SHADER),
            (self.external_memory_host, extension::EXTERNAL_MEMORY_HOST),
            (self.synchronization2, extension::SYNCHRONIZATION_2),
            (self.dynamic_rendering, extension::DYNAMIC_RENDERING),
            (
                self.extended_dynamic_state,
                extension::EXTENDED_DYNAMIC_STATE,
            ),
            (
                self.extended_dynamic_state_3,
                extension::EXTENDED_DYNAMIC_STATE_3,
            ),
        ] {
            if wanted {
                names.push(name);
            }
        }
        if let Some(name) = self.vertex_attribute_divisor {
            names.push(name);
        }
        names
    }
}

/// Whether descriptor buffers have been qualified on this host.
///
/// The architecture asks for descriptor buffers where the driver reports
/// support *and* validation and performance tests pass. The second half is a
/// measurement, not something a driver reports, so it cannot come out of
/// [`Census::take`] — and it is a separate type rather than a `bool` so that
/// "nobody has measured this" is a state, distinct from "measured and failed".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DescriptorBufferProbe {
    /// No qualification has run here. The conservative state, and the one every
    /// freshly taken census is in.
    #[default]
    NotRun,
    Passed,
    Failed,
}

/// The measured facts about one physical device.
///
/// Immutable and `Copy`. Every accessor is a projection of what was taken; none
/// asks the driver anything.
/// `Eq` is deliberately absent: [`crate::sampler::SamplerCell`] carries
/// `maxSamplerAnisotropy`, which is an `f32`. Every other cell here is `Eq`,
/// and a census still compares — it just compares by the rules a float has.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Census {
    api: ApiVersion,
    memory: MemoryProfile,
    queues: QueueChoice,
    stages: StageSupport,
    descriptors: DescriptorCell,
    passes: crate::pass::PassCell,
    raster: crate::raster::RasterCell,
    buffers: crate::buffer::BufferLimits,
    samplers: crate::sampler::SamplerCell,
    blend: crate::blend::BlendCell,
    topology: crate::topology::TopologyCell,
    vertex: crate::vertex::VertexCell,
    viewports: crate::raster::ViewportCell,
    line_widths: crate::raster::LineWidthCell,
    host_pointer_import: bool,
    synchronization2: bool,
    can_present: bool,
    extensions: DeviceExtensions,
}

impl Census {
    /// Take the snapshot, or refuse the device.
    ///
    /// The one place an extension name and a feature bit are joined, and the
    /// one place the support floor is checked.
    ///
    /// # Errors
    ///
    /// [`Floor`] when the device cannot serve this rail at all.
    pub fn take(reported: Reported<'_>) -> Result<Self, Floor> {
        let api = ApiVersion::decode(reported.api_version);
        if !api.at_least(1, 2) {
            return Err(Floor::ApiTooOld { reported: api });
        }
        // The floor, queried. 1.2 promoted the extension, so a 1.2 device need
        // not enumerate `VK_KHR_timeline_semaphore` — but it must still report
        // the feature, and that is the fact checked.
        if !reported.timeline_semaphore {
            return Err(Floor::NoTimelineSemaphores { reported: api });
        }
        if !reported.has(extension::SWAPCHAIN) {
            return Err(Floor::NoSwapchain);
        }

        // Whether the divisor capability has any route onto this device at
        // all, named once because three decisions below read it.
        let divisor_route = api.at_least(1, 4)
            || reported.has(extension::VERTEX_ATTRIBUTE_DIVISOR)
            || reported.has(extension::VERTEX_ATTRIBUTE_DIVISOR_EXT);

        // Whether the second dynamic-state extension is present at all, named
        // once because three decisions below read it.
        let dynamic_state_3 = reported.has(extension::EXTENDED_DYNAMIC_STATE_3);

        let families = queues::families(reported.queue_families);
        let queues = QueueChoice::from_families(&families)
            .map_err(|decline| Floor::NoUsableQueue { decline })?;

        // Both halves, every time: the entry points resolve *and* the feature
        // is on. Either alone is a capability this rail must not use.
        let mesh_shader = reported.has(extension::MESH_SHADER) && reported.mesh_shader;
        let descriptor_buffer =
            reported.has(extension::DESCRIPTOR_BUFFER) && reported.descriptor_buffer;
        // Push descriptors have no feature bit of their own in the extension;
        // the property is the capability, and a limit of zero is checked where
        // the tier is chosen rather than here, so the refusal can say which of
        // the two facts was missing.
        let push_descriptor = reported.has(extension::PUSH_DESCRIPTOR) || api.at_least(1, 4);

        // `VK_EXT_external_memory_host` publishes no feature structure, so the
        // extension list is the whole of what a name can say about it --- and
        // the module's rule is that a name is not a capability. Two measured
        // facts stand in for the missing feature bit, and both are needed:
        //
        // - the device reports the host-allocation handle type as importable
        //   at all, which is a `vkGetPhysicalDeviceExternalBufferProperties`
        //   answer and not an entry-point question;
        // - it states an alignment a pointer can actually meet. Zero is a
        //   device that never filled the property in, and a value that is not
        //   a power of two is one no pointer is aligned to and no span mask
        //   can round to.
        //
        // Without these this rail would offer `Route::DirectAlias` on the
        // strength of a name, and the refusal would arrive at
        // `vkAllocateMemory` --- after placement had already promised the
        // guest's pages *are* the resource, which is the one route with no
        // copy to fall back to.
        let host_pointer_import = reported.has(extension::EXTERNAL_MEMORY_HOST)
            && reported.host_pointer_importable
            && reported.min_imported_host_pointer_alignment != 0
            && reported
                .min_imported_host_pointer_alignment
                .is_power_of_two();

        Ok(Self {
            api,
            memory: classify_memory(reported.memory),
            queues,
            stages: StageSupport { mesh_shader },
            descriptors: DescriptorCell {
                push_descriptor,
                max_push_descriptors: reported.max_push_descriptors,
                descriptor_buffer,
                // Never reported by a driver. See [`DescriptorBufferProbe`].
                descriptor_buffer_qualified: false,
            },
            // Core from 1.3, and both halves below it — for the reason
            // `synchronization2` needs both.
            passes: crate::pass::PassCell {
                dynamic_rendering: api.at_least(1, 3)
                    || (reported.has(extension::DYNAMIC_RENDERING) && reported.dynamic_rendering),
            },
            raster: crate::raster::RasterCell {
                depth_clamp: reported.depth_clamp,
                fill_mode_non_solid: reported.fill_mode_non_solid,
                // Core from 1.3, and both halves below it — for the reason
                // `synchronization2` needs both.
                dynamic_cull_and_winding: api.at_least(1, 3)
                    || (reported.has(extension::EXTENDED_DYNAMIC_STATE)
                        && reported.extended_dynamic_state),
                // Never promoted, so these two are only ever the extension's,
                // and each needs its own feature: the extension groups thirty
                // dynamic members and a device may offer any subset of them.
                dynamic_polygon_mode: dynamic_state_3
                    && reported.extended_dynamic_state3_polygon_mode,
                dynamic_depth_clamp: dynamic_state_3
                    && reported.extended_dynamic_state3_depth_clamp_enable,
            },
            viewports: crate::raster::ViewportCell {
                multi_viewport: reported.multi_viewport,
                max_viewports: reported.max_viewports,
            },
            // Carried as reported, like `max_sampler_anisotropy`: the range a
            // device states is the range a width is admitted against, and a
            // floor substituted here would hide a driver that reported one
            // not containing 1.0.
            line_widths: crate::raster::LineWidthCell {
                wide_lines: reported.wide_lines,
                range: reported.line_width_range,
            },
            buffers: crate::buffer::BufferLimits {
                max_buffer_size: reported.max_buffer_size,
            },
            vertex: crate::vertex::VertexCell {
                formats: reported.vertex_formats,
                // Either spelling of the extension, or 1.4 core, and the
                // feature in every case: a capability that arrived by
                // promotion is still one that has to be reported.
                instance_rate_divisor: divisor_route
                    && reported.vertex_attribute_instance_rate_divisor,
                zero_divisor: divisor_route && reported.vertex_attribute_instance_rate_zero_divisor,
                max_divisor: reported.max_vertex_attrib_divisor,
            },
            topology: crate::topology::TopologyCell {
                // Core from 1.3, and both halves below it — for the reason
                // `synchronization2` needs both.
                dynamic: api.at_least(1, 3)
                    || (reported.has(extension::EXTENDED_DYNAMIC_STATE)
                        && reported.extended_dynamic_state),
                // Never promoted, so the property is only ever what the
                // extension reported. A device that was not asked is
                // restricted, which is the conservative rung.
                unrestricted: dynamic_state_3
                    && reported
                        .dynamic_primitive_topology_unrestricted
                        .unwrap_or(false),
            },
            blend: crate::blend::BlendCell {
                dual_source: reported.dual_src_blend,
                independent: reported.independent_blend,
            },
            samplers: crate::sampler::SamplerCell {
                mirror_clamp_to_edge: reported.sampler_mirror_clamp_to_edge,
                anisotropy: reported.sampler_anisotropy,
                // Carried as reported. The clamp a plan applies raises it to
                // at least 1.0 at the point of use, which is where the guest's
                // request is also known; substituting a floor here would hide
                // a driver that reported less than one.
                max_anisotropy: reported.max_sampler_anisotropy,
            },
            host_pointer_import,
            // 1.3 promoted it to core, so a 1.3 device has it whether or not it
            // enumerates the extension. Below that both facts are needed, for
            // the reason above.
            synchronization2: api.at_least(1, 3)
                || (reported.has(extension::SYNCHRONIZATION_2) && reported.synchronization2),
            can_present: true,
            // Enumerated, wanted, and not already core here. A capability that
            // arrived through core is used through core and never re-requested
            // by a name the driver may not have offered.
            extensions: DeviceExtensions {
                swapchain: true,
                push_descriptor: reported.has(extension::PUSH_DESCRIPTOR) && !api.at_least(1, 4),
                descriptor_buffer,
                mesh_shader,
                // The qualified answer, like `descriptor_buffer` and
                // `mesh_shader` above: an extension enabled for a capability
                // this device does not offer is a name in
                // `ppEnabledExtensionNames` that no decision here will ever
                // read.
                external_memory_host: host_pointer_import,
                synchronization2: reported.has(extension::SYNCHRONIZATION_2)
                    && reported.synchronization2
                    && !api.at_least(1, 3),
                dynamic_rendering: reported.has(extension::DYNAMIC_RENDERING)
                    && reported.dynamic_rendering
                    && !api.at_least(1, 3),
                extended_dynamic_state: reported.has(extension::EXTENDED_DYNAMIC_STATE)
                    && reported.extended_dynamic_state
                    && !api.at_least(1, 3),
                // No version clause, because nothing promoted this one. It is
                // enumerated only where a member this rail sets came back with
                // it: enabling an extension whose every feature is off buys
                // nothing and still has to be reported as enabled.
                extended_dynamic_state_3: dynamic_state_3
                    && (reported.extended_dynamic_state3_polygon_mode
                        || reported.extended_dynamic_state3_depth_clamp_enable),
                // Only where a divisor capability was actually admitted, and
                // under the name this device enumerated. The KHR spelling wins
                // where both are present, because it is the one core promoted.
                vertex_attribute_divisor: if api.at_least(1, 4)
                    || !(reported.vertex_attribute_instance_rate_divisor
                        || reported.vertex_attribute_instance_rate_zero_divisor)
                {
                    None
                } else if reported.has(extension::VERTEX_ATTRIBUTE_DIVISOR) {
                    Some(extension::VERTEX_ATTRIBUTE_DIVISOR)
                } else if reported.has(extension::VERTEX_ATTRIBUTE_DIVISOR_EXT) {
                    Some(extension::VERTEX_ATTRIBUTE_DIVISOR_EXT)
                } else {
                    None
                },
            },
        })
    }

    /// Record a descriptor-buffer qualification result.
    ///
    /// Consuming and returning `Self` rather than mutating: a census is one
    /// snapshot, and a probe result taken later produces a new one rather than
    /// changing what an earlier decision was made against.
    ///
    /// Only [`DescriptorBufferProbe::Passed`] can raise the tier, and it cannot
    /// raise it past what the device reported — a probe that passed on a device
    /// without the extension leaves the cell exactly as it was, because
    /// [`Self::take`] already decided that fact.
    #[must_use]
    pub const fn with_descriptor_buffer_probe(mut self, probe: DescriptorBufferProbe) -> Self {
        self.descriptors.descriptor_buffer_qualified =
            self.descriptors.descriptor_buffer && matches!(probe, DescriptorBufferProbe::Passed);
        self
    }

    #[must_use]
    pub const fn api(&self) -> ApiVersion {
        self.api
    }

    #[must_use]
    pub const fn memory(&self) -> MemoryProfile {
        self.memory
    }

    /// The immutable half of the queue decision. A device epoch turns it into
    /// a [`crate::queues::QueuePlan`] with [`QueuePlan::adopt`]; the catalog
    /// holds no ownership state.
    ///
    /// [`QueuePlan::adopt`]: crate::queues::QueuePlan::adopt
    #[must_use]
    pub const fn queues(&self) -> QueueChoice {
        self.queues
    }

    #[must_use]
    pub const fn stages(&self) -> StageSupport {
        self.stages
    }

    #[must_use]
    pub const fn descriptors(&self) -> DescriptorCell {
        self.descriptors
    }

    /// The cell [`crate::pass::select`] chooses a carrier from.
    #[must_use]
    pub const fn passes(&self) -> crate::pass::PassCell {
        self.passes
    }

    /// The optional features [`crate::raster`] needs for two of the states a
    /// guest sets.
    #[must_use]
    pub const fn raster(&self) -> crate::raster::RasterCell {
        self.raster
    }

    /// The bound [`crate::buffer::plan`] checks a length against.
    #[must_use]
    pub const fn buffers(&self) -> crate::buffer::BufferLimits {
        self.buffers
    }

    /// The cell [`crate::vertex::attribute`] and [`crate::vertex::binding`]
    /// decide a format substitution and a divisor from.
    #[must_use]
    pub const fn vertex(&self) -> crate::vertex::VertexCell {
        self.vertex
    }

    /// The cell [`crate::topology::key`] reads to decide what a built
    /// pipeline can serve.
    #[must_use]
    pub const fn topology(&self) -> crate::topology::TopologyCell {
        self.topology
    }

    /// The cell [`crate::blend::plan`] checks a dual-source factor against,
    /// and [`crate::blend::independent`] a whole attachment list.
    #[must_use]
    pub const fn blend(&self) -> crate::blend::BlendCell {
        self.blend
    }

    /// The cell [`crate::sampler::plan`] translates an address mode and
    /// clamps an anisotropy request against.
    #[must_use]
    pub const fn samplers(&self) -> crate::sampler::SamplerCell {
        self.samplers
    }

    /// The cell [`crate::placement`] decides a route from.
    #[must_use]
    pub const fn host_cell(&self) -> HostCell {
        HostCell {
            topology: self.memory.topology,
            host_pointer_import: self.host_pointer_import,
        }
    }

    /// Whether barriers may use the `synchronization2` commands, or have to go
    /// through [`crate::barrier::BarrierPlan::legacy`].
    #[must_use]
    pub const fn synchronization2(&self) -> bool {
        self.synchronization2
    }

    /// How many viewports a pipeline built here may declare. See
    /// [`crate::raster::viewport_slots`].
    #[must_use]
    pub const fn viewports(&self) -> crate::raster::ViewportCell {
        self.viewports
    }

    /// What width a draw that rasterizes lines may be given. See
    /// [`crate::raster::line_width`].
    #[must_use]
    pub const fn line_widths(&self) -> crate::raster::LineWidthCell {
        self.line_widths
    }

    #[must_use]
    pub const fn can_present(&self) -> bool {
        self.can_present
    }

    /// The extension names device creation has to ask for.
    #[must_use]
    pub const fn extensions(&self) -> DeviceExtensions {
        self.extensions
    }

    /// One line naming every fact a decision here is allowed to read.
    ///
    /// The architecture asks for the exact capability census to be recorded
    /// with every run, and this is it. It names no device, because there is no
    /// device name in the census to name.
    #[must_use]
    pub fn report_line(&self) -> String {
        format!(
            "vk_census api={} topology={} signal={} import={} sync2={} mesh={} push={} \
             push_max={} desc_buffer={} desc_qualified={} queue_family={} compute={} \
             mirror_clamp={} aniso={} aniso_max={} dual_src={} independent_blend={} \
             dyn_topology={} topology_unrestricted={} vertex_formats={} \
             vertex_divisor={} vertex_zero_divisor={} vertex_max_divisor={} \
             depth_clamp={} fill_non_solid={} dyn_cull_winding={} dyn_polygon={} \
             dyn_depth_clamp={} multi_viewport={} max_viewports={} wide_lines={} \
             line_width_max={}",
            self.api,
            self.memory.topology.slug(),
            self.memory.signal.slug(),
            self.host_pointer_import,
            self.synchronization2,
            self.stages.mesh_shader,
            self.descriptors.push_descriptor,
            self.descriptors.max_push_descriptors,
            self.descriptors.descriptor_buffer,
            self.descriptors.descriptor_buffer_qualified,
            self.queues.universal().index,
            self.queues.compute(),
            self.samplers.mirror_clamp_to_edge,
            self.samplers.anisotropy,
            self.samplers.max_anisotropy,
            self.blend.dual_source,
            self.blend.independent,
            self.topology.dynamic,
            self.topology.unrestricted,
            self.vertex.formats.count(),
            self.vertex.instance_rate_divisor,
            self.vertex.zero_divisor,
            self.vertex.max_divisor,
            self.raster.depth_clamp,
            self.raster.fill_mode_non_solid,
            self.raster.dynamic_cull_and_winding,
            self.raster.dynamic_polygon_mode,
            self.raster.dynamic_depth_clamp,
            self.viewports.multi_viewport,
            self.viewports.max_viewports,
            self.line_widths.wide_lines,
            self.line_widths.range[1],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{select, Narrowing, Tier};
    use crate::memory::{fixtures as mem, MemoryTopology};

    /// Queue families as the driver reports them, rather than as
    /// [`crate::queues::Family`] — the census takes what the driver said.
    fn family(flags: vk::QueueFlags, count: u32) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: count,
            timestamp_valid_bits: 64,
            min_image_transfer_granularity: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
        }
    }

    /// One universal family plus a dedicated copy engine.
    fn discrete_families() -> Vec<vk::QueueFamilyProperties> {
        vec![
            family(
                vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
                16,
            ),
            family(vk::QueueFlags::TRANSFER, 2),
        ]
    }

    /// One universal family, which is most integrated parts.
    fn integrated_families() -> Vec<vk::QueueFamilyProperties> {
        vec![family(
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER,
            1,
        )]
    }

    fn packed(major: u32, minor: u32) -> u32 {
        vk::make_api_version(0, major, minor, 0)
    }

    const BASELINE: &[&str] = &[extension::SWAPCHAIN];

    fn reported<'a>(
        api: u32,
        extensions: &'a [&'a str],
        memory: &'a vk::PhysicalDeviceMemoryProperties,
        families: &'a [vk::QueueFamilyProperties],
    ) -> Reported<'a> {
        Reported {
            api_version: api,
            extensions,
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
            // The two facts that qualify the import capability, at the values
            // an import-capable device reports. A test that wants the
            // extension present and the capability absent clears one of them.
            host_pointer_importable: true,
            min_imported_host_pointer_alignment: 4096,
            memory,
            queue_families: families,
        }
    }

    #[test]
    fn a_baseline_device_is_admitted_and_takes_the_bottom_rung_of_everything() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let census = Census::take(reported(packed(1, 2), BASELINE, &memory, &families))
            .expect("1.2 with timeline semaphores is the baseline");

        assert_eq!(census.api(), ApiVersion { major: 1, minor: 2 });
        assert_eq!(census.memory().topology, MemoryTopology::Discrete);
        assert!(!census.synchronization2(), "1.2 has no core sync2");
        assert!(!census.stages().mesh_shader);
        assert!(!census.host_cell().host_pointer_import);
        assert_eq!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::PooledSets
        );
    }

    #[test]
    fn a_device_below_the_baseline_is_refused_by_version() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let floor = Census::take(reported(packed(1, 1), BASELINE, &memory, &families))
            .expect_err("1.1 is below the baseline");
        assert_eq!(
            floor,
            Floor::ApiTooOld {
                reported: ApiVersion { major: 1, minor: 1 }
            }
        );
        assert_eq!(floor.slug(), "vk_census_api_too_old");
    }

    /// The floor the version number does not cover.
    #[test]
    fn a_new_enough_device_with_the_feature_off_is_still_refused() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let mut r = reported(packed(1, 3), BASELINE, &memory, &families);
        r.timeline_semaphore = false;
        let floor = Census::take(r).expect_err("the feature is the floor, not the version");
        assert_eq!(
            floor,
            Floor::NoTimelineSemaphores {
                reported: ApiVersion { major: 1, minor: 3 }
            }
        );
        assert!(floor.to_string().contains("1.3"));
    }

    #[test]
    fn a_device_that_cannot_present_is_refused_before_anything_is_measured() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let floor = Census::take(reported(packed(1, 2), &[], &memory, &families))
            .expect_err("no swapchain");
        assert_eq!(floor, Floor::NoSwapchain);
    }

    #[test]
    fn a_device_with_no_drawing_queue_is_refused_and_says_which_decline() {
        let memory = mem::nvidia_discrete();
        let families: Vec<vk::QueueFamilyProperties> = Vec::new();
        let floor = Census::take(reported(packed(1, 2), BASELINE, &memory, &families))
            .expect_err("no queue family at all");
        assert!(matches!(floor, Floor::NoUsableQueue { .. }));
        assert_eq!(floor.slug(), "vk_census_no_usable_queue");
    }

    /// The claim the module exists to make structural: the entry points
    /// resolving is not the feature being on.
    #[test]
    fn an_extension_without_its_feature_is_not_a_capability() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let extensions = &[
            extension::SWAPCHAIN,
            extension::MESH_SHADER,
            extension::DESCRIPTOR_BUFFER,
        ];

        let mut r = reported(packed(1, 2), extensions, &memory, &families);
        r.mesh_shader = false;
        r.descriptor_buffer = false;
        let without = Census::take(r).expect("admitted");
        assert!(!without.stages().mesh_shader);
        assert!(!without.descriptors().descriptor_buffer);

        r.mesh_shader = true;
        r.descriptor_buffer = true;
        let with = Census::take(r).expect("admitted");
        assert!(with.stages().mesh_shader);
        assert!(with.descriptors().descriptor_buffer);
    }

    /// And the mirror: a feature bit set with no extension enumerated is not a
    /// capability either.
    #[test]
    fn a_feature_without_its_extension_is_not_a_capability() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let mut r = reported(packed(1, 2), BASELINE, &memory, &families);
        r.mesh_shader = true;
        r.descriptor_buffer = true;
        r.synchronization2 = true;
        let census = Census::take(r).expect("admitted");
        assert!(!census.stages().mesh_shader);
        assert!(!census.descriptors().descriptor_buffer);
        assert!(!census.synchronization2());
    }

    #[test]
    fn a_promoted_capability_needs_no_extension_at_its_core_version() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        // 1.3 has synchronization2 in core; the extension is not enumerated and
        // the feature struct is not chained.
        let census =
            Census::take(reported(packed(1, 3), BASELINE, &memory, &families)).expect("admitted");
        assert!(census.synchronization2());
        // And below it, the extension route reaches the same answer.
        let mut r = reported(
            packed(1, 2),
            &[extension::SWAPCHAIN, extension::SYNCHRONIZATION_2],
            &memory,
            &families,
        );
        r.synchronization2 = true;
        assert!(Census::take(r).expect("admitted").synchronization2());
    }

    /// The M-series shape, measured rather than named: unified memory, host
    /// pointer import, push descriptors, no descriptor buffers.
    #[test]
    fn a_unified_import_capable_host_reaches_the_push_rung_and_the_direct_route() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let mut r = reported(
            packed(1, 2),
            &[
                extension::SWAPCHAIN,
                extension::PUSH_DESCRIPTOR,
                extension::EXTERNAL_MEMORY_HOST,
            ],
            &memory,
            &families,
        );
        r.max_push_descriptors = 32;
        let census = Census::take(r).expect("admitted");

        assert_eq!(census.memory().topology, MemoryTopology::Unified);
        assert!(census.host_cell().host_pointer_import);
        assert_eq!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::PushDescriptor { max: 32 }
        );
    }

    /// The extension name is not the capability, and this is the one
    /// capability with no feature bit to say so.
    ///
    /// The regression: `host_pointer_import` was `has(EXTERNAL_MEMORY_HOST)`
    /// alone, so a device that enumerates the extension and declines the
    /// handle type --- or states an alignment no pointer can meet --- was
    /// offered `Route::DirectAlias`, which is the one route with no copy to
    /// fall back to once the allocation refuses.
    #[test]
    fn the_import_capability_needs_the_two_facts_the_extension_name_does_not_carry() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let named = &[extension::SWAPCHAIN, extension::EXTERNAL_MEMORY_HOST];
        let taken = |mutate: &dyn Fn(&mut Reported<'_>)| {
            let mut r = reported(packed(1, 2), named, &memory, &families);
            mutate(&mut r);
            Census::take(r).expect("admitted")
        };

        // All three facts: the capability, and the extension asked for.
        let whole = taken(&|_| {});
        assert!(whole.host_cell().host_pointer_import);
        assert!(whole
            .extensions()
            .names()
            .contains(&extension::EXTERNAL_MEMORY_HOST));

        // Each fact alone withdraws it, and withdraws the extension with it:
        // a name enabled for a capability this device does not offer is one
        // no decision here would read.
        /// A named way for one reported fact to fall short.
        type Withdraws<'a> = (&'a str, &'a dyn Fn(&mut Reported<'_>));
        let cases: [Withdraws<'_>; 3] = [
            ("the handle type is not importable", &|r| {
                r.host_pointer_importable = false;
            }),
            ("the device stated no alignment", &|r| {
                r.min_imported_host_pointer_alignment = 0;
            }),
            ("the alignment is not a power of two", &|r| {
                r.min_imported_host_pointer_alignment = 3072;
            }),
        ];
        for (what, mutate) in cases {
            let census = taken(mutate);
            assert!(
                !census.host_cell().host_pointer_import,
                "admitted when {what}"
            );
            assert!(
                !census
                    .extensions()
                    .names()
                    .contains(&extension::EXTERNAL_MEMORY_HOST),
                "enabled the extension when {what}"
            );
        }

        // And the two facts do not admit it without the extension: there are
        // no entry points to import through.
        let mut unnamed = reported(packed(1, 2), BASELINE, &memory, &families);
        unnamed.host_pointer_importable = true;
        unnamed.min_imported_host_pointer_alignment = 4096;
        assert!(
            !Census::take(unnamed)
                .expect("admitted")
                .host_cell()
                .host_pointer_import
        );
    }

    /// The viewport limits reach the cell `raster::viewport_slots` is asked
    /// against, both halves, unchanged.
    ///
    /// A limit and its feature that the census did not carry is a check no
    /// caller could have made: the census is the only thing the rest of the
    /// crate is given.
    #[test]
    fn the_viewport_limits_reach_the_cell_that_admits_a_count() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let mut r = reported(packed(1, 2), BASELINE, &memory, &families);
        r.multi_viewport = true;
        r.max_viewports = 16;
        let census = Census::take(r).expect("admitted");
        assert_eq!(
            census.viewports(),
            crate::raster::ViewportCell {
                multi_viewport: true,
                max_viewports: 16,
            }
        );
        assert!(crate::raster::viewport_slots(16, census.viewports()).is_ok());
        assert!(crate::raster::viewport_slots(17, census.viewports()).is_err());

        // And a device that reports neither admits exactly one, which is the
        // cell every device answers.
        let bare =
            Census::take(reported(packed(1, 2), BASELINE, &memory, &families)).expect("admitted");
        assert_eq!(bare.viewports(), crate::raster::ViewportCell::SINGLE);
        assert!(crate::raster::viewport_slots(2, bare.viewports()).is_err());
    }

    /// The line-width facts reach the cell `raster::line_width` is asked
    /// against, and a device that reports neither admits exactly the width
    /// that needs no feature.
    #[test]
    fn the_line_width_limits_reach_the_cell_that_admits_a_width() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let mut r = reported(packed(1, 2), BASELINE, &memory, &families);
        r.wide_lines = true;
        r.line_width_range = [0.5, 8.0];
        let census = Census::take(r).expect("admitted");
        assert!(census.line_widths().wide_lines);
        assert_eq!(census.line_widths().range, [0.5, 8.0]);
        assert_eq!(
            crate::raster::line_width(Some(4.0), true, census.line_widths()),
            Ok(4.0)
        );
        assert!(crate::raster::line_width(Some(9.0), true, census.line_widths()).is_err());

        // A device that reports no `wideLines` rasterizes lines at exactly the
        // one width every device carries, and refuses any other.
        let bare =
            Census::take(reported(packed(1, 2), BASELINE, &memory, &families)).expect("admitted");
        assert_eq!(bare.line_widths(), crate::raster::LineWidthCell::NARROW);
        assert_eq!(
            crate::raster::line_width(Some(1.0), true, bare.line_widths()),
            Ok(1.0)
        );
        assert!(crate::raster::line_width(Some(2.0), true, bare.line_widths()).is_err());
    }

    #[test]
    fn a_probe_cannot_qualify_what_the_device_does_not_report() {
        let memory = mem::intel_igpu();
        let families = integrated_families();
        let census = Census::take(reported(packed(1, 2), BASELINE, &memory, &families))
            .expect("admitted")
            .with_descriptor_buffer_probe(DescriptorBufferProbe::Passed);
        assert!(
            !census.descriptors().descriptor_buffer_qualified,
            "a probe result cannot conjure the extension"
        );
        assert_ne!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::DescriptorBuffer
        );
    }

    #[test]
    fn a_probe_is_what_raises_a_reported_device_to_the_top_rung() {
        let memory = mem::intel_igpu();
        let families = integrated_families();
        let mut r = reported(
            packed(1, 3),
            &[extension::SWAPCHAIN, extension::DESCRIPTOR_BUFFER],
            &memory,
            &families,
        );
        r.descriptor_buffer = true;
        let census = Census::take(r).expect("admitted");

        // Reported, unmeasured: not the top rung.
        assert_eq!(
            select(census.descriptors(), Narrowing::default()).tier,
            Tier::PooledSets
        );
        for probe in [DescriptorBufferProbe::NotRun, DescriptorBufferProbe::Failed] {
            let tried = census.with_descriptor_buffer_probe(probe);
            assert!(
                !tried.descriptors().descriptor_buffer_qualified,
                "{probe:?}"
            );
        }
        let qualified = census.with_descriptor_buffer_probe(DescriptorBufferProbe::Passed);
        assert_eq!(
            select(qualified.descriptors(), Narrowing::default()).tier,
            Tier::DescriptorBuffer
        );
    }

    #[test]
    fn the_report_line_names_every_fact_and_no_device() {
        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let mut r = reported(
            packed(1, 2),
            &[
                extension::SWAPCHAIN,
                extension::PUSH_DESCRIPTOR,
                extension::EXTERNAL_MEMORY_HOST,
            ],
            &memory,
            &families,
        );
        r.max_push_descriptors = 32;
        r.sampler_anisotropy = true;
        r.max_sampler_anisotropy = 16.0;
        let line = Census::take(r).expect("admitted").report_line();

        for fact in [
            "api=1.2",
            "topology=unified",
            "import=true",
            "sync2=false",
            "mesh=false",
            "push=true",
            "push_max=32",
            "desc_buffer=false",
            "desc_qualified=false",
            "mirror_clamp=false",
            "aniso=true",
            "aniso_max=16",
            "dual_src=false",
            "independent_blend=false",
            "dyn_topology=false",
            "topology_unrestricted=false",
            "vertex_formats=0",
            "vertex_divisor=false",
            "vertex_zero_divisor=false",
            "vertex_max_divisor=0",
            "depth_clamp=false",
            "fill_non_solid=false",
            "multi_viewport=false",
            "max_viewports=1",
            "wide_lines=false",
            "line_width_max=1",
            "dyn_cull_winding=false",
            "dyn_polygon=false",
            "dyn_depth_clamp=false",
        ] {
            assert!(line.contains(fact), "{fact} missing from {line}");
        }
    }

    /// The three dynamic-rasterizer bits, and the three different ways they
    /// are established.
    ///
    /// The first has a core route and the other two never will, so a version
    /// bump must move exactly one of them. The two extension-only bits need
    /// their own feature each, because they are two members of a
    /// thirty-member structure and a device may offer either alone — which is
    /// also why the extension is enumerated for the union and not for the
    /// presence of the extension string.
    #[test]
    fn the_dynamic_rasterizer_bits_each_have_their_own_route() {
        let memory = mem::intel_igpu();
        let families = integrated_families();
        let base = |api| reported(api, BASELINE, &memory, &families);

        // 1.2 with nothing: every rasterizer state is baked.
        let bare = Census::take(base(packed(1, 2))).expect("admitted");
        assert_eq!(bare.raster(), crate::raster::RasterCell::default());
        assert!(!bare.extensions().extended_dynamic_state_3);

        // The pair arrives by version, and the other two do not follow it.
        let promoted = Census::take(base(packed(1, 3))).expect("admitted");
        assert!(promoted.raster().dynamic_cull_and_winding);
        assert!(!promoted.raster().dynamic_polygon_mode);
        assert!(!promoted.raster().dynamic_depth_clamp);
        // Nothing to enumerate: 1.3 made those commands core.
        assert!(!promoted.extensions().extended_dynamic_state);

        // The pair below 1.3 needs the extension *and* the feature.
        let mut r = base(packed(1, 2));
        let with_eds = [extension::SWAPCHAIN, extension::EXTENDED_DYNAMIC_STATE];
        r.extensions = &with_eds;
        assert!(
            !Census::take(r)
                .expect("admitted")
                .raster()
                .dynamic_cull_and_winding
        );
        let mut r = base(packed(1, 2));
        r.extensions = &with_eds;
        r.extended_dynamic_state = true;
        let classed = Census::take(r).expect("admitted");
        assert!(classed.raster().dynamic_cull_and_winding);
        assert!(classed.extensions().extended_dynamic_state);

        // Each ext3 member on its own, and neither implied by the other.
        let with_eds3 = [extension::SWAPCHAIN, extension::EXTENDED_DYNAMIC_STATE_3];
        for (polygon, clamp) in [(true, false), (false, true), (true, true)] {
            let mut r = base(packed(1, 4));
            r.extensions = &with_eds3;
            r.extended_dynamic_state3_polygon_mode = polygon;
            r.extended_dynamic_state3_depth_clamp_enable = clamp;
            let census = Census::take(r).expect("admitted");
            assert_eq!(census.raster().dynamic_polygon_mode, polygon);
            assert_eq!(census.raster().dynamic_depth_clamp, clamp);
            // Enumerated for the union, so the device is created with the
            // extension whenever either member will be commanded.
            assert!(census.extensions().extended_dynamic_state_3);
            assert!(census
                .extensions()
                .names()
                .contains(&extension::EXTENDED_DYNAMIC_STATE_3));
        }

        // The extension string with neither feature enumerates nothing: an
        // extension whose every member is off buys nothing and would still
        // have to be reported as enabled.
        let mut r = base(packed(1, 4));
        r.extensions = &with_eds3;
        let neither = Census::take(r).expect("admitted");
        assert!(!neither.raster().dynamic_polygon_mode);
        assert!(!neither.raster().dynamic_depth_clamp);
        assert!(!neither.extensions().extended_dynamic_state_3);

        // A feature reported without the extension is not a route. Nothing
        // promoted this one, so there is no version that makes it one.
        let mut r = base(packed(1, 4));
        r.extended_dynamic_state3_polygon_mode = true;
        r.extended_dynamic_state3_depth_clamp_enable = true;
        let lying = Census::take(r).expect("admitted");
        assert!(!lying.raster().dynamic_polygon_mode);
        assert!(!lying.raster().dynamic_depth_clamp);
    }

    /// The two facts a sampler plan reads reach it verbatim, and the limit is
    /// carried rather than substituted.
    ///
    /// `maxSamplerAnisotropy` is a limit, so the driver always reports one; a
    /// census that floored it at 1.0 would make a device claiming less than
    /// one indistinguishable from a conformant one, and the clamp belongs
    /// where the guest's request is also known.
    #[test]
    fn the_sampler_cell_carries_what_was_reported_and_gates_the_mode_that_needs_a_feature() {
        use crate::sampler::{plan, Refusal};
        use reims_vgpu_core::sampler::{
            SamplerShape, MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
            MTL_SAMPLER_ADDRESS_MODE_REPEAT, MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            MTL_SAMPLER_MIN_MAG_FILTER_LINEAR, MTL_SAMPLER_MIP_FILTER_LINEAR,
        };

        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let mut bare = reported(packed(1, 2), BASELINE, &memory, &families);
        bare.max_sampler_anisotropy = 0.5;
        let bare = Census::take(bare).expect("admitted");
        assert!(!bare.samplers().anisotropy);
        assert!(!bare.samplers().mirror_clamp_to_edge);
        // Carried, not floored.
        assert!((bare.samplers().max_anisotropy - 0.5).abs() < f32::EPSILON);

        let mut rich = reported(packed(1, 2), BASELINE, &memory, &families);
        rich.sampler_anisotropy = true;
        rich.max_sampler_anisotropy = 4.0;
        rich.sampler_mirror_clamp_to_edge = true;
        let rich = Census::take(rich).expect("admitted");

        let shape = |address: u32, anisotropy: u32| SamplerShape {
            min_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mag_filter: MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: MTL_SAMPLER_MIP_FILTER_LINEAR,
            s_address: address,
            t_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            r_address: MTL_SAMPLER_ADDRESS_MODE_REPEAT,
            max_anisotropy: anisotropy,
            lod_min_clamp: 0.0,
            lod_max_clamp: 8.0,
            compare_function: 0,
            compare_enabled: false,
            border_color: MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
            normalized_coordinates: true,
        };
        let checked = |address, anisotropy| {
            shape(address, anisotropy)
                .checked()
                .expect("a declaration the guest API admits")
        };

        // The address mode is a core 1.2 enumerant whether or not the feature
        // was enabled, so nothing but this cell stops it being used.
        assert_eq!(
            plan(
                checked(MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, 1),
                bare.samplers()
            ),
            Err(Refusal::NoMirrorClampToEdge)
        );
        assert!(plan(
            checked(MTL_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE, 1),
            rich.samplers()
        )
        .is_ok());

        // The device's limit is the one a request is clamped against, and it
        // arrived here through the census.
        let over = plan(
            checked(MTL_SAMPLER_ADDRESS_MODE_REPEAT, 16),
            rich.samplers(),
        )
        .expect("a repeat sampler needs no feature");
        assert!(over.anisotropy_enable);
        assert!((over.max_anisotropy - 4.0).abs() < f32::EPSILON);

        // And with the feature off it is not merely clamped, it is not asked
        // for — enabling it unrequested is the undefined case.
        let off = plan(
            checked(MTL_SAMPLER_ADDRESS_MODE_REPEAT, 16),
            bare.samplers(),
        )
        .expect("a repeat sampler needs no feature");
        assert!(!off.anisotropy_enable);
        assert!((off.max_anisotropy - 1.0).abs() < f32::EPSILON);
    }

    /// The blend cell reaches the planner, and both halves of it decide
    /// something a guest can ask for.
    #[test]
    fn the_blend_cell_carries_what_was_reported_and_gates_what_needs_a_feature() {
        use crate::blend::{independent, plan, Refusal};
        use reims_vgpu_core::blend::{
            ColorAttachmentShape, ColorAttachmentState, ColorWriteMask, MTL_BLEND_FACTOR_ONE,
            MTL_BLEND_FACTOR_SOURCE_1_COLOR, MTL_BLEND_FACTOR_ZERO, MTL_BLEND_OPERATION_ADD,
        };

        let memory = mem::apple_m3_max();
        let families = integrated_families();
        let bare =
            Census::take(reported(packed(1, 2), BASELINE, &memory, &families)).expect("admitted");
        assert!(!bare.blend().dual_source);
        assert!(!bare.blend().independent);

        let mut rich = reported(packed(1, 2), BASELINE, &memory, &families);
        rich.dual_src_blend = true;
        rich.independent_blend = true;
        let rich = Census::take(rich).expect("admitted");

        let dual = ColorAttachmentShape {
            blending_enabled: true,
            src_rgb: MTL_BLEND_FACTOR_SOURCE_1_COLOR,
            dst_rgb: MTL_BLEND_FACTOR_ZERO,
            op_rgb: MTL_BLEND_OPERATION_ADD,
            src_alpha: MTL_BLEND_FACTOR_ONE,
            dst_alpha: MTL_BLEND_FACTOR_ZERO,
            op_alpha: MTL_BLEND_OPERATION_ADD,
            write_mask: ColorWriteMask::ALL,
        }
        .checked()
        .expect("a declaration the guest API admits");
        assert!(matches!(
            plan(&dual, bare.blend()),
            Err(Refusal::NoDualSource { .. })
        ));
        let planned = plan(&dual, rich.blend()).expect("the device reports it");

        let opaque = plan(&ColorAttachmentState::OPAQUE, rich.blend()).expect("nothing to refuse");
        assert!(matches!(
            independent(&[planned, opaque], bare.blend()),
            Err(Refusal::NoIndependentBlend { .. })
        ));
        assert!(independent(&[planned, opaque], rich.blend()).is_ok());
    }

    /// The three rungs of the topology cell, and the two facts that make them.
    ///
    /// The property has no core promotion, so a 1.4 device that never
    /// enumerated `VK_EXT_extended_dynamic_state3` is restricted — which is
    /// the case a version check would have got wrong.
    #[test]
    fn the_topology_cell_has_three_rungs_and_the_property_never_arrives_by_version() {
        use crate::topology::{key, TopologyKey};
        use reims_vgpu_core::topology::{PrimitiveType, TopologyClass};

        let memory = mem::intel_igpu();
        let families = integrated_families();

        // 1.2 with neither: one pipeline per primitive type.
        let bare =
            Census::take(reported(packed(1, 2), BASELINE, &memory, &families)).expect("admitted");
        assert!(!bare.topology().dynamic);
        assert!(!bare.topology().unrestricted);
        assert_eq!(
            key(PrimitiveType::LineStrip, bare.topology()),
            TopologyKey::Exact(PrimitiveType::LineStrip)
        );

        // 1.2 with the extension and its feature: dynamic within a class.
        let classed = &[extension::SWAPCHAIN, extension::EXTENDED_DYNAMIC_STATE];
        let mut r = reported(packed(1, 2), classed, &memory, &families);
        r.extended_dynamic_state = true;
        let classed = Census::take(r).expect("admitted");
        assert!(classed.topology().dynamic);
        assert!(!classed.topology().unrestricted);
        assert_eq!(
            key(PrimitiveType::LineStrip, classed.topology()),
            TopologyKey::Class(TopologyClass::Line)
        );
        // The extension is asked for by name, because 1.2 is below the
        // promotion.
        assert!(classed.extensions().extended_dynamic_state);

        // The extension without its feature is not a capability.
        let mut r = reported(
            packed(1, 2),
            &[extension::SWAPCHAIN, extension::EXTENDED_DYNAMIC_STATE],
            &memory,
            &families,
        );
        r.extended_dynamic_state = false;
        assert!(!Census::take(r).expect("admitted").topology().dynamic);

        // 1.3 has it in core and never asks for the name.
        let promoted =
            Census::take(reported(packed(1, 3), BASELINE, &memory, &families)).expect("admitted");
        assert!(promoted.topology().dynamic);
        assert!(!promoted.extensions().extended_dynamic_state);

        // A 1.4 device that never enumerated the property extension is still
        // restricted, and one that enumerated it and answered yes is not.
        let mut r = reported(packed(1, 4), BASELINE, &memory, &families);
        r.dynamic_primitive_topology_unrestricted = Some(true);
        let lying = Census::take(r).expect("admitted");
        assert!(
            !lying.topology().unrestricted,
            "a property from an extension this device did not enumerate is not an answer"
        );

        let named = &[extension::SWAPCHAIN, extension::EXTENDED_DYNAMIC_STATE_3];
        let mut r = reported(packed(1, 4), named, &memory, &families);
        r.dynamic_primitive_topology_unrestricted = Some(true);
        let free = Census::take(r).expect("admitted");
        assert!(free.topology().dynamic);
        assert!(free.topology().unrestricted);
        assert_eq!(key(PrimitiveType::Point, free.topology()), TopologyKey::Any);
        // And the extension is never requested, because only its property was
        // read.
        assert!(!free
            .extensions()
            .names()
            .contains(&extension::EXTENDED_DYNAMIC_STATE_3));

        // Enumerated but answered no is the restricted rung, and is a
        // different fact from never having been asked.
        let mut r = reported(packed(1, 4), named, &memory, &families);
        r.dynamic_primitive_topology_unrestricted = Some(false);
        assert!(!Census::take(r).expect("admitted").topology().unrestricted);
    }

    /// The divisor capability has three routes and one feature question, and
    /// the extension is requested under the name this device enumerated.
    #[test]
    fn the_divisor_capability_takes_any_route_and_still_needs_the_feature() {
        let memory = mem::intel_igpu();
        let families = integrated_families();

        // 1.2 with neither name: no route, so the feature bits are ignored and
        // nothing is requested.
        let mut r = reported(packed(1, 2), BASELINE, &memory, &families);
        r.vertex_attribute_instance_rate_divisor = true;
        r.vertex_attribute_instance_rate_zero_divisor = true;
        let none = Census::take(r).expect("admitted");
        assert!(!none.vertex().instance_rate_divisor);
        assert!(!none.vertex().zero_divisor);
        assert_eq!(none.extensions().vertex_attribute_divisor, None);

        // 1.2 with the EXT name and the feature: admitted, and asked for under
        // that name.
        let ext = &[
            extension::SWAPCHAIN,
            extension::VERTEX_ATTRIBUTE_DIVISOR_EXT,
        ];
        let mut r = reported(packed(1, 2), ext, &memory, &families);
        r.vertex_attribute_instance_rate_divisor = true;
        r.max_vertex_attrib_divisor = 8;
        let by_ext = Census::take(r).expect("admitted");
        assert!(by_ext.vertex().instance_rate_divisor);
        // The two halves are separate facts, and only one was reported.
        assert!(!by_ext.vertex().zero_divisor);
        assert_eq!(by_ext.vertex().max_divisor, 8);
        assert_eq!(
            by_ext.extensions().vertex_attribute_divisor,
            Some(extension::VERTEX_ATTRIBUTE_DIVISOR_EXT)
        );
        assert!(by_ext
            .extensions()
            .names()
            .contains(&extension::VERTEX_ATTRIBUTE_DIVISOR_EXT));

        // Both names present: the promoted spelling is the one requested.
        let both = &[
            extension::SWAPCHAIN,
            extension::VERTEX_ATTRIBUTE_DIVISOR,
            extension::VERTEX_ATTRIBUTE_DIVISOR_EXT,
        ];
        let mut r = reported(packed(1, 2), both, &memory, &families);
        r.vertex_attribute_instance_rate_zero_divisor = true;
        let by_khr = Census::take(r).expect("admitted");
        assert!(by_khr.vertex().zero_divisor);
        assert_eq!(
            by_khr.extensions().vertex_attribute_divisor,
            Some(extension::VERTEX_ATTRIBUTE_DIVISOR)
        );

        // The name without the feature is not a capability, and nothing is
        // requested for a capability that was not admitted.
        let bare = Census::take(reported(packed(1, 2), ext, &memory, &families)).expect("admitted");
        assert!(!bare.vertex().instance_rate_divisor);
        assert!(!bare.vertex().zero_divisor);
        assert_eq!(bare.extensions().vertex_attribute_divisor, None);

        // 1.4 has it in core and never asks for a name.
        let mut r = reported(packed(1, 4), BASELINE, &memory, &families);
        r.vertex_attribute_instance_rate_divisor = true;
        r.vertex_attribute_instance_rate_zero_divisor = true;
        let promoted = Census::take(r).expect("admitted");
        assert!(promoted.vertex().instance_rate_divisor);
        assert!(promoted.vertex().zero_divisor);
        assert_eq!(promoted.extensions().vertex_attribute_divisor, None);
    }

    /// The measured vertex formats reach the planner, and a declined one is
    /// the case the widening substitute exists for.
    #[test]
    fn the_measured_vertex_formats_reach_the_planner() {
        use crate::vertex::{attribute, Refusal, ShaderInput, VertexFormatSupport};
        use reims_vgpu_core::vertex_format::VertexFormat;

        let memory = mem::intel_igpu();
        let families = integrated_families();
        let mut r = reported(packed(1, 2), BASELINE, &memory, &families);
        r.vertex_formats = VertexFormatSupport::all().without(VertexFormat::Short3);
        let census = Census::take(r).expect("admitted");

        assert!(!census.vertex().formats.has(VertexFormat::Short3));
        assert!(census.vertex().formats.has(VertexFormat::Short4));

        // The declined format widens where the shader reads three.
        let widened = attribute(
            0,
            0,
            VertexFormat::Short3,
            0,
            8,
            census.vertex().formats,
            || ShaderInput::Channels(3),
        )
        .expect("the wider sibling is mandatory");
        assert_eq!(widened.widened_from, Some(VertexFormat::Short3));

        // A device that declined everything has nothing to substitute with.
        let mut r = reported(packed(1, 2), BASELINE, &memory, &families);
        r.vertex_formats = VertexFormatSupport::NONE;
        let empty = Census::take(r).expect("admitted");
        assert_eq!(empty.vertex().formats.count(), 0);
        assert_eq!(
            attribute(
                0,
                0,
                VertexFormat::Short3,
                0,
                8,
                empty.vertex().formats,
                || { ShaderInput::Channels(3) }
            ),
            Err(Refusal::NoFormat {
                guest: VertexFormat::Short3
            })
        );
    }

    /// The rule a device creation gets refused for breaking: a capability that
    /// is core at this version is used through core and never asked for by a
    /// name the driver may not have enumerated.
    #[test]
    fn a_promoted_capability_is_never_requested_as_an_extension() {
        let memory = mem::intel_igpu();
        let families = integrated_families();
        let all = &[
            extension::SWAPCHAIN,
            extension::PUSH_DESCRIPTOR,
            extension::SYNCHRONIZATION_2,
        ];

        // Below both promotions: both names are requested.
        let mut r = reported(packed(1, 2), all, &memory, &families);
        r.synchronization2 = true;
        let below = Census::take(r).expect("admitted").extensions();
        assert!(below.push_descriptor);
        assert!(below.synchronization2);

        // 1.3 promoted synchronization2; 1.4 promoted push descriptors.
        r.api_version = packed(1, 3);
        let at13 = Census::take(r).expect("admitted").extensions();
        assert!(at13.push_descriptor, "not core until 1.4");
        assert!(!at13.synchronization2, "core at 1.3");

        r.api_version = packed(1, 4);
        let at14 = Census::take(r).expect("admitted").extensions();
        assert!(!at14.push_descriptor, "core at 1.4");
        assert!(!at14.synchronization2);

        // And the capabilities themselves are still there — only the request
        // route changed.
        let census = Census::take(r).expect("admitted");
        assert!(census.descriptors().push_descriptor);
        assert!(census.synchronization2());
    }

    /// Nothing reaches the extension list that the capability side refused.
    #[test]
    fn an_unadmitted_capability_is_never_in_the_extension_list() {
        let memory = mem::intel_igpu();
        let families = integrated_families();
        let mut r = reported(
            packed(1, 2),
            &[
                extension::SWAPCHAIN,
                extension::MESH_SHADER,
                extension::DESCRIPTOR_BUFFER,
            ],
            &memory,
            &families,
        );
        // Enumerated with the features off, which is not a capability.
        r.mesh_shader = false;
        r.descriptor_buffer = false;
        let extensions = Census::take(r).expect("admitted").extensions();
        assert!(!extensions.mesh_shader);
        assert!(!extensions.descriptor_buffer);
        assert_eq!(extensions.names(), vec![extension::SWAPCHAIN]);

        r.mesh_shader = true;
        r.descriptor_buffer = true;
        let names = Census::take(r).expect("admitted").extensions().names();
        assert_eq!(
            names,
            vec![
                extension::SWAPCHAIN,
                extension::DESCRIPTOR_BUFFER,
                extension::MESH_SHADER,
            ]
        );
    }

    #[test]
    fn a_version_orders_by_major_then_minor_and_ignores_the_patch() {
        let a = ApiVersion::decode(vk::make_api_version(0, 1, 2, 198));
        let b = ApiVersion::decode(vk::make_api_version(0, 1, 2, 0));
        assert_eq!(a, b, "no gate may turn on a patch level");
        assert!(a.at_least(1, 2));
        assert!(!a.at_least(1, 3));
        assert!(a.at_least(1, 0));
        assert!(ApiVersion::decode(vk::make_api_version(0, 2, 0, 0)).at_least(1, 9));
    }

    #[test]
    fn a_reported_buffer_maximum_reaches_the_planner_and_an_absent_one_stays_absent() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let absent = Census::take(reported(packed(1, 2), BASELINE, &memory, &families))
            .expect("the baseline");
        assert_eq!(absent.buffers().max_buffer_size, None);

        let stated = Census::take(Reported {
            max_buffer_size: Some(1 << 30),
            ..reported(packed(1, 3), BASELINE, &memory, &families)
        })
        .expect("the baseline");
        assert_eq!(stated.buffers().max_buffer_size, Some(1 << 30));
    }

    #[test]
    fn dynamic_rendering_needs_both_halves_below_the_version_that_promoted_it() {
        let memory = mem::nvidia_discrete();
        let families = discrete_families();
        let with = |api: (u32, u32), extensions: &[&str], feature: bool| {
            Census::take(Reported {
                dynamic_rendering: feature,
                ..reported(packed(api.0, api.1), extensions, &memory, &families)
            })
            .expect("the baseline")
            .passes()
            .dynamic_rendering
        };

        // 1.3 promoted it, so the extension name is never the question there.
        assert!(with((1, 3), BASELINE, false));
        // Below it, the extension alone is not enough and the feature alone is
        // not either.
        let named = &[extension::SWAPCHAIN, extension::DYNAMIC_RENDERING];
        assert!(!with((1, 2), named, false));
        assert!(!with((1, 2), BASELINE, true));
        assert!(with((1, 2), named, true));
    }
}
