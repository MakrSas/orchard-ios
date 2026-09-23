//! Where the primitive type is a draw argument on one side and pipeline state
//! on the other, and what that costs.
//!
//! # Metal passes it to the draw; Vulkan builds it into the pipeline
//!
//! `drawPrimitives:` takes an `MTLPrimitiveType`, so a guest may draw
//! triangles and then lines from one render pipeline state without rebuilding
//! anything. `VkGraphicsPipelineCreateInfo` names the topology, so on a bare
//! 1.2 baseline that guest needs a pipeline per type it draws with — the
//! topology is part of the cache key, and a guest that alternates pays a
//! compile the first time it does.
//!
//! `VK_EXT_extended_dynamic_state` — core in 1.3 — makes it
//! `vkCmdSetPrimitiveTopology`, and then one pipeline serves several types.
//! Not all of them: the spec permits changing topology only within one
//! [`TopologyClass`] unless the device also reports
//! `VkPhysicalDeviceExtendedDynamicState3PropertiesEXT::dynamicPrimitiveTopologyUnrestricted`.
//! So the capability has three rungs and they produce three different cache
//! keys, which is what [`key`] returns. Nothing else in this module decides a
//! pipeline's identity, and nothing outside it re-derives one.
//!
//! Getting that wrong is not a validation error on this host — a driver that
//! happens to allow a cross-class change will run it — which is exactly why
//! the rung is asked for structurally rather than assumed.
//!
//! # Metal has no primitive restart
//!
//! There is no restart index in Metal: an index equal to `0xFFFF` or
//! `0xFFFFFFFF` is a vertex like any other. `primitiveRestartEnable` is
//! therefore false for every plan here, including the strips — which are the
//! only topologies it would have affected, and the only ones where leaving it
//! on would silently cut a strip the guest expected continuous.
//!
//! # Planned, not created
//!
//! Nothing here builds a pipeline. Every mapping is tested with no GPU.

use ash::vk;
use reims_vgpu_core::topology::{PrimitiveType, TopologyClass};

/// What this host offers for changing topology without a rebuild.
///
/// Two bits rather than one, because the second is meaningless without the
/// first and the two rungs it makes are different cache keys.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TopologyCell {
    /// `VK_EXT_extended_dynamic_state`'s `extendedDynamicState`, or 1.3 core.
    pub dynamic: bool,
    /// `VkPhysicalDeviceExtendedDynamicState3PropertiesEXT::dynamicPrimitiveTopologyUnrestricted`.
    ///
    /// A *property*, not a feature: a device reports whether it is restricted
    /// rather than being asked to lift the restriction.
    pub unrestricted: bool,
}

/// `MTLPrimitiveType` → `VkPrimitiveTopology`. Total.
///
/// The two enumerations agree numerically across all five, and that is
/// precisely why this is written out: the agreement is a coincidence of two
/// independent specifications rather than a contract, and they already diverge
/// where Metal's set ends — Vulkan continues into `TRIANGLE_FAN`, the four
/// adjacency topologies and `PATCH_LIST`, none of which Metal names. A cast
/// would carry that coincidence forward and turn a future divergence into
/// silently wrong rasterization instead of a compile error.
#[must_use]
pub const fn topology(guest: PrimitiveType) -> vk::PrimitiveTopology {
    match guest {
        PrimitiveType::Point => vk::PrimitiveTopology::POINT_LIST,
        PrimitiveType::Line => vk::PrimitiveTopology::LINE_LIST,
        PrimitiveType::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        PrimitiveType::Triangle => vk::PrimitiveTopology::TRIANGLE_LIST,
        PrimitiveType::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

/// What a built pipeline is identified by, as far as topology is concerned.
///
/// The three rungs of [`TopologyCell`], as three different things a cache may
/// key on. Comparing two keys answers "can this pipeline serve that draw"
/// without anybody re-reading the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TopologyKey {
    /// The baseline: one pipeline per primitive type.
    Exact(PrimitiveType),
    /// Dynamic within a class: one pipeline serves a list and its strip.
    Class(TopologyClass),
    /// Dynamic across classes: one pipeline serves every type.
    Any,
}

/// The key a pipeline drawing `guest` is cached under on this host.
#[must_use]
pub const fn key(guest: PrimitiveType, cell: TopologyCell) -> TopologyKey {
    if !cell.dynamic {
        TopologyKey::Exact(guest)
    } else if cell.unrestricted {
        TopologyKey::Any
    } else {
        TopologyKey::Class(guest.class())
    }
}

/// A pipeline's input-assembly state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputAssemblyPlan {
    pub topology: vk::PrimitiveTopology,
    /// Always false. See the module doc.
    pub primitive_restart_enable: bool,
    /// Whether `vkCmdSetPrimitiveTopology` supplies the topology per draw, in
    /// which case the field above is the pipeline's declared one and the draw
    /// may move within what [`key`] allows.
    pub dynamic: bool,
}

impl InputAssemblyPlan {
    /// The dynamic states a pipeline built from this plan must declare.
    ///
    /// Read off the plan rather than off the cell, for the reason
    /// `raster::RasterizationState::dynamic` is read off the pipeline key: the
    /// state list and the stand-in topology it explains are two readings of
    /// one decision, and a second derivation could disagree. A pipeline that
    /// declares a stand-in without declaring the state draws that stand-in —
    /// silently, on a host with no validation layers.
    #[must_use]
    pub const fn states(self) -> &'static [vk::DynamicState] {
        if self.dynamic {
            &[vk::DynamicState::PRIMITIVE_TOPOLOGY]
        } else {
            &[]
        }
    }

    pub const fn native(self) -> vk::PipelineInputAssemblyStateCreateInfo<'static> {
        vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            topology: self.topology,
            primitive_restart_enable: vk::FALSE,
            _marker: core::marker::PhantomData,
        }
    }
}

impl TopologyKey {
    /// The input assembly a pipeline under this key is built with.
    ///
    /// Derived from the key rather than from the guest's own primitive type,
    /// and that is the whole reason this is a method on the key: two draws
    /// that share a key must produce a byte-identical pipeline, so the
    /// declared topology has to be a function of what they share. A `Class`
    /// pipeline built from a triangle strip and one built from a triangle list
    /// would otherwise declare different topologies while claiming to be the
    /// same cache entry.
    ///
    /// `Class` declares [`TopologyClass::list`] — the class's one non-strip
    /// member, which exists for exactly this. `Any` declares a triangle list;
    /// under `dynamicPrimitiveTopologyUnrestricted` the declared topology
    /// constrains nothing, so the only thing that matters is that every `Any`
    /// pipeline picks the same one.
    #[must_use]
    pub const fn input_assembly(self) -> InputAssemblyPlan {
        InputAssemblyPlan {
            topology: topology(self.declares()),
            primitive_restart_enable: false,
            dynamic: !matches!(self, Self::Exact(_)),
        }
    }

    /// The primitive type a pipeline under this key declares.
    ///
    /// Always a type this key itself serves — see
    /// [`Self::input_assembly`] — so on the restricted rung the declared
    /// topology is always in the class the encoder may move within.
    #[must_use]
    pub const fn declares(self) -> PrimitiveType {
        match self {
            Self::Exact(guest) => guest,
            Self::Class(class) => class.list(),
            Self::Any => PrimitiveType::Triangle,
        }
    }
}

/// The topology the encoder sets before a draw of `guest`, or `None` where
/// this host baked it into the pipeline instead.
///
/// `Some` exactly when [`TopologyKey::input_assembly`] reported `dynamic`, so
/// a caller that sets every `Some` has reproduced the guest's draw and a
/// caller that sets none of them is a host that bakes the topology.
#[must_use]
pub const fn dynamic(guest: PrimitiveType, cell: TopologyCell) -> Option<vk::PrimitiveTopology> {
    if cell.dynamic {
        Some(topology(guest))
    } else {
        None
    }
}

/// Whether a pipeline built for `built` can draw `wanted` on this host.
///
/// The whole point of [`TopologyKey`]: a caller asks the key, not the cell.
#[must_use]
pub fn serves(built: PrimitiveType, wanted: PrimitiveType, cell: TopologyCell) -> bool {
    key(built, cell) == key(wanted, cell)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    const STATIC: TopologyCell = TopologyCell {
        dynamic: false,
        unrestricted: false,
    };
    const CLASSED: TopologyCell = TopologyCell {
        dynamic: true,
        unrestricted: false,
    };
    const FREE: TopologyCell = TopologyCell {
        dynamic: true,
        unrestricted: true,
    };

    #[test]
    fn every_primitive_type_maps_to_a_distinct_topology() {
        let mapped: BTreeSet<i32> = PrimitiveType::ALL
            .iter()
            .map(|p| topology(*p).as_raw())
            .collect();
        assert_eq!(mapped.len(), PrimitiveType::ALL.len());
        assert_eq!(
            topology(PrimitiveType::LineStrip),
            vk::PrimitiveTopology::LINE_STRIP
        );
        assert_eq!(
            topology(PrimitiveType::TriangleStrip),
            vk::PrimitiveTopology::TRIANGLE_STRIP
        );

        // The five agree numerically today, which is the coincidence the
        // written-out mapping exists to survive: asserted so that a future
        // divergence shows up here as a changed expectation rather than
        // nowhere.
        for guest in PrimitiveType::ALL {
            assert_eq!(topology(guest).as_raw(), guest.ordinal() as i32);
        }
        // And the sets are not the same set: Vulkan continues past Metal's
        // last value, so the agreement is a prefix and nothing more.
        assert_eq!(PrimitiveType::parse(5), None);
        assert_eq!(vk::PrimitiveTopology::TRIANGLE_FAN.as_raw(), 5);
    }

    /// Metal has no restart index, so a strip is never cut. This is the one
    /// place it could have been turned on and the one topology it would have
    /// affected.
    #[test]
    fn no_plan_enables_primitive_restart_including_the_strips() {
        for guest in PrimitiveType::ALL {
            for cell in [STATIC, CLASSED, FREE] {
                let plan = key(guest, cell).input_assembly();
                assert!(!plan.primitive_restart_enable);
                assert_eq!(plan.native().primitive_restart_enable, vk::FALSE);
            }
        }
        // Non-vacuous: two of those five were strips.
        assert_eq!(
            PrimitiveType::ALL.iter().filter(|p| p.is_strip()).count(),
            2
        );
    }

    /// Without dynamic topology every type is its own pipeline; that is the
    /// baseline cost, asserted rather than assumed.
    #[test]
    fn the_baseline_needs_a_pipeline_per_primitive_type() {
        let keys: BTreeSet<TopologyKey> =
            PrimitiveType::ALL.iter().map(|p| key(*p, STATIC)).collect();
        assert_eq!(keys.len(), 5);
        for a in PrimitiveType::ALL {
            for b in PrimitiveType::ALL {
                assert_eq!(serves(a, b, STATIC), a == b);
            }
        }
    }

    /// With `extendedDynamicState` and nothing more, a pipeline serves its
    /// class and no further. Five types collapse to three pipelines.
    #[test]
    fn dynamic_topology_serves_a_class_and_stops_at_its_edge() {
        let keys: BTreeSet<TopologyKey> = PrimitiveType::ALL
            .iter()
            .map(|p| key(*p, CLASSED))
            .collect();
        assert_eq!(keys.len(), 3);
        assert!(serves(
            PrimitiveType::Line,
            PrimitiveType::LineStrip,
            CLASSED
        ));
        assert!(serves(
            PrimitiveType::Triangle,
            PrimitiveType::TriangleStrip,
            CLASSED
        ));
        // The edge. A driver that happens to allow this would run it, which is
        // why the restriction is asked for rather than discovered.
        assert!(!serves(
            PrimitiveType::Triangle,
            PrimitiveType::Line,
            CLASSED
        ));
        assert!(!serves(PrimitiveType::Point, PrimitiveType::Line, CLASSED));
        for guest in PrimitiveType::ALL {
            assert!(key(guest, CLASSED).input_assembly().dynamic);
        }
    }

    /// With the unrestricted property one pipeline serves everything.
    #[test]
    fn an_unrestricted_host_needs_one_pipeline() {
        let keys: BTreeSet<TopologyKey> =
            PrimitiveType::ALL.iter().map(|p| key(*p, FREE)).collect();
        assert_eq!(keys, BTreeSet::from([TopologyKey::Any]));
        for a in PrimitiveType::ALL {
            for b in PrimitiveType::ALL {
                assert!(serves(a, b, FREE));
            }
        }
    }

    /// The property means nothing without the feature: a device reporting
    /// `unrestricted` while `extendedDynamicState` is off is still on the
    /// baseline rung, because there is no dynamic state for it to unrestrict.
    #[test]
    fn the_property_alone_does_not_lift_anything() {
        let confused = TopologyCell {
            dynamic: false,
            unrestricted: true,
        };
        assert_eq!(
            key(PrimitiveType::Line, confused),
            TopologyKey::Exact(PrimitiveType::Line)
        );
        assert!(!serves(
            PrimitiveType::Line,
            PrimitiveType::LineStrip,
            confused
        ));
        assert!(!key(PrimitiveType::Line, confused).input_assembly().dynamic);
        assert_eq!(dynamic(PrimitiveType::Line, confused), None);
        assert!(key(PrimitiveType::Line, confused)
            .input_assembly()
            .states()
            .is_empty());
    }

    #[test]
    fn the_native_state_carries_the_plan() {
        let native = TopologyKey::Exact(PrimitiveType::TriangleStrip)
            .input_assembly()
            .native();
        assert_eq!(native.topology, vk::PrimitiveTopology::TRIANGLE_STRIP);
        assert_eq!(native.primitive_restart_enable, vk::FALSE);
        assert_eq!(
            native.s_type,
            vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
        );
    }

    /// The claim a cache rests on: every guest type that shares a key builds
    /// the *same* pipeline. If the declared topology were read from the guest
    /// instead of from the key, a triangle list and a triangle strip would be
    /// one cache entry describing two different pipelines — the first one
    /// created would silently serve the other.
    #[test]
    fn one_key_declares_one_input_assembly_whatever_guest_type_reached_it() {
        for cell in [STATIC, CLASSED, FREE] {
            let mut per_key: BTreeMap<TopologyKey, InputAssemblyPlan> = BTreeMap::new();
            for guest in PrimitiveType::ALL {
                let k = key(guest, cell);
                let asm = k.input_assembly();
                if let Some(seen) = per_key.get(&k) {
                    assert_eq!(*seen, asm, "{k:?} declares two different pipelines");
                } else {
                    per_key.insert(k, asm);
                }
                // And the declared topology is one the key admits, so a draw
                // whose dynamic topology is set can lawfully reach it.
                assert_eq!(key(k.declares(), cell), k);
            }
        }
    }

    /// A member is dynamic in exactly one of the two halves: the encoder is
    /// given the guest's own type precisely where the pipeline declared a
    /// stand-in, and given nothing where the pipeline declared the guest's.
    #[test]
    fn the_guests_type_is_in_exactly_one_half() {
        for cell in [STATIC, CLASSED, FREE] {
            for guest in PrimitiveType::ALL {
                let asm = key(guest, cell).input_assembly();
                let dyn_value = dynamic(guest, cell);
                assert_eq!(dyn_value.is_some(), asm.dynamic);
                assert_eq!(
                    asm.states().contains(&vk::DynamicState::PRIMITIVE_TOPOLOGY),
                    asm.dynamic
                );
                if let Some(set) = dyn_value {
                    assert_eq!(set, topology(guest));
                } else {
                    assert_eq!(asm.topology, topology(guest));
                }
            }
        }
    }

    /// Every primitive type lands on the Vulkan topology of the same name.
    ///
    /// Metal names the primitive and leaves the grouping implicit; Vulkan names
    /// both, so a guest type that does not say `Strip` is a list. That is the
    /// whole difference, and it is the one a pipeline once lost --- a `Class`
    /// key declaring a stand-in topology rasterized every triangle strip as a
    /// list. A check that reads the guest name rather than the table cannot
    /// lose it the same way twice.
    #[test]
    fn every_primitive_lands_on_the_topology_of_the_same_name() {
        for guest in PrimitiveType::ALL {
            assert_eq!(
                format!("{:?}", topology(guest)),
                crate::naming::vulkan_topology_spelling(&format!("{guest:?}")),
                "{guest:?}"
            );
        }
    }
}
