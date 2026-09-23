//! Where a resource's bytes live on this host, decided in one order from the
//! guest's declaration and this device's measured capabilities.
//!
//! # The order, and why it is an order
//!
//! 1. Apply storage-mode semantics.
//! 2. Take the representation the resource actually requires — format, tiling,
//!    stride, plane layout, ownership — as an input, because deriving it needs
//!    the descriptor and this decides placement.
//! 3. Ask whether the guest's own backing can serve that representation
//!    directly.
//! 4. If not, ask whether imported guest backing can at least be a GPU transfer
//!    endpoint.
//! 5. Otherwise use a host-visible staging representation.
//! 6. Choose the working representation's memory class from the topology.
//!
//! Written as a sequence because the capability cells *constrain* the available
//! routes rather than selecting one. A table keyed on `(topology, import)` that
//! returned a route directly would have to encode every combination of the
//! resource's own requirements too, and the first requirement it did not encode
//! would silently take somebody else's cell.
//!
//! # Private does not skip coherence, and that is not this module's opinion
//!
//! The plan describes private content as bypassing guest-CPU coherence. On
//! *this* wire that reading does not hold, and
//! [`reims_vgpu_core::storage_mode`] carries the evidence: backing is
//! allocated mode-blind, the guest CPU-touches private resources through the
//! same mapping, and what the mode gates is the modified-range
//! *announcement*. So [`Placement`] never uses [`StorageMode::Private`] to drop
//! a guest-visible representation; it uses it only where the API's own
//! transitions do.
//!
//! Recorded here rather than in a note somewhere, because this is the module
//! that would otherwise take the shortcut, and the resulting stale-page defect
//! has no counter — its only symptom is wrong content.
//!
//! # A route is a decision about performance, never about meaning
//!
//! Every route below produces the same guest-visible behaviour. That is the
//! constraint the whole crate is under: a guest stream executed on an
//! import-capable unified host and on a discrete host without import must mean
//! the same thing. So the routes differ in how many copies happen and where the
//! bytes sit, and never in what a read returns.

use crate::memory::{MemoryClass, MemoryTopology};
use reims_vgpu_core::storage_mode::StorageMode;

/// What this host offers, as capabilities rather than names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostCell {
    pub topology: MemoryTopology,
    /// Guest pages can be imported as device memory.
    ///
    /// Not "`VK_EXT_external_memory_host` is present": the extension publishes
    /// no feature structure, so the name says only that the entry points
    /// resolve. [`crate::census::Census::take`] qualifies it against the
    /// device's own answers --- the host-allocation handle type is importable,
    /// and the stated import alignment is one a pointer can meet --- because
    /// [`Route::DirectAlias`] is the one route with no copy left to fall back
    /// to if the import is then refused.
    pub host_pointer_import: bool,
}

/// Whether the guest's own backing can serve the representation the resource
/// requires.
///
/// An input rather than something derived here: answering it needs the format,
/// tiling, stride and plane layout, and this module decides placement. Carrying
/// *why* it cannot lets the census say which requirement is costing the copies
/// rather than reporting a bare count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectFit {
    /// Format, tiling, stride, plane layout and ownership all agree, so the
    /// guest's pages are the resource.
    Exact,
    /// A linear buffer's bytes are its bytes; nothing about a buffer can
    /// disagree with the guest's layout.
    LinearBuffer,
    /// The representation needs a tiling the guest's linear pages do not have.
    TilingDiffers,
    /// Stride, plane layout or extent do not line up with the guest's.
    LayoutDiffers,
    /// The resource is a render target or otherwise needs usage an imported
    /// allocation cannot carry here.
    UsageUnavailable,
}

impl DirectFit {
    #[must_use]
    pub const fn fits(self) -> bool {
        matches!(self, Self::Exact | Self::LinearBuffer)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::LinearBuffer => "linear_buffer",
            Self::TilingDiffers => "tiling_differs",
            Self::LayoutDiffers => "layout_differs",
            Self::UsageUnavailable => "usage_unavailable",
        }
    }
}

/// Where the resource's bytes end up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// The guest's pages, imported, are the resource. No copy ever happens and
    /// a guest write is visible without one.
    DirectAlias,
    /// The guest's pages are imported as a transfer endpoint, and a working
    /// representation is copied to and from them on the GPU.
    ImportedTransfer { working: MemoryClass },
    /// The guest's pages are copied through a host-visible staging
    /// representation, because they cannot be imported at all.
    HostStaging { working: MemoryClass },
}

impl Route {
    /// The working representation's memory class, when there is one.
    #[must_use]
    pub const fn working(self) -> Option<MemoryClass> {
        match self {
            Self::DirectAlias => None,
            Self::ImportedTransfer { working } | Self::HostStaging { working } => Some(working),
        }
    }

    /// Whether a guest write reaches the GPU without a copy.
    #[must_use]
    pub const fn copies_guest_writes(self) -> bool {
        !matches!(self, Self::DirectAlias)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DirectAlias => "direct_alias",
            Self::ImportedTransfer { .. } => "imported_transfer",
            Self::HostStaging { .. } => "host_staging",
        }
    }
}

/// The full decision, with the inputs that produced it.
///
/// The inputs travel with the answer because a route on its own cannot be
/// argued with: "host staging on a unified host" reads as a defect until the
/// `DirectFit` beside it says the representation needed a tiling the guest's
/// pages do not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub route: Route,
    pub mode: StorageMode,
    pub fit: DirectFit,
    pub cell: HostCell,
}

impl Placement {
    /// One line for the always-on report.
    #[must_use]
    pub fn census(&self) -> String {
        format!(
            "vk_placement route={} mode={} fit={} topology={} import={}",
            self.route.name(),
            self.mode.name(),
            self.fit.name(),
            self.cell.topology.slug(),
            u8::from(self.cell.host_pointer_import),
        )
    }
}

/// Decide where a resource's bytes live.
///
/// The mode is carried on the answer and does not select the route: every mode
/// this wire carries obliges the device to keep the guest's pages current, so
/// there is nothing here for it to change. It travels because a placement that
/// could not say which mode it was made for could not be argued with, and
/// because a future contract change would arrive as a route this function does
/// not currently draw.
#[must_use]
pub fn place(mode: StorageMode, fit: DirectFit, cell: HostCell) -> Placement {
    let route = route(fit, cell);
    Placement {
        route,
        mode,
        fit,
        cell,
    }
}

fn route(fit: DirectFit, cell: HostCell) -> Route {
    // 3. The guest's own backing serves the representation, and this host can
    //    import it. Nothing is copied, ever.
    if fit.fits() && cell.host_pointer_import {
        return Route::DirectAlias;
    }
    // 6. The working representation's class, chosen from the topology.
    //
    //    Unified prefers a placement the CPU can also reach: it is the same
    //    DRAM, so a host-visible working resource costs nothing in bandwidth
    //    and saves the copy back. Discrete wants device-local, where the GPU's
    //    reads are not crossing the bus.
    let working = match cell.topology {
        MemoryTopology::Unified => MemoryClass::DeviceLocalPreferred,
        MemoryTopology::Discrete => MemoryClass::DeviceLocal,
    };
    if cell.host_pointer_import {
        // 4. The representation does not fit the guest's pages, but they can
        //    still be one end of a GPU copy — which is a DMA the GPU performs
        //    rather than bytes the CPU moves.
        Route::ImportedTransfer { working }
    } else {
        // 5. No import at all, so the bytes cross through a host-visible
        //    staging representation the CPU writes and the GPU reads.
        Route::HostStaging { working }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIFIED_IMPORT: HostCell = HostCell {
        topology: MemoryTopology::Unified,
        host_pointer_import: true,
    };
    const UNIFIED_NO_IMPORT: HostCell = HostCell {
        topology: MemoryTopology::Unified,
        host_pointer_import: false,
    };
    const DISCRETE_IMPORT: HostCell = HostCell {
        topology: MemoryTopology::Discrete,
        host_pointer_import: true,
    };
    const DISCRETE_NO_IMPORT: HostCell = HostCell {
        topology: MemoryTopology::Discrete,
        host_pointer_import: false,
    };

    const CELLS: [HostCell; 4] = [
        UNIFIED_IMPORT,
        UNIFIED_NO_IMPORT,
        DISCRETE_IMPORT,
        DISCRETE_NO_IMPORT,
    ];
    const MODES: [StorageMode; 3] = [
        StorageMode::Shared,
        StorageMode::Managed,
        StorageMode::Private,
    ];
    const FITS: [DirectFit; 5] = [
        DirectFit::Exact,
        DirectFit::LinearBuffer,
        DirectFit::TilingDiffers,
        DirectFit::LayoutDiffers,
        DirectFit::UsageUnavailable,
    ];

    /// A resource whose representation is exactly its guest pages, on a host
    /// that can import them, is those pages. This is the cell the whole import
    /// mechanism exists for.
    #[test]
    fn an_exact_fit_with_import_is_never_copied() {
        for cell in [UNIFIED_IMPORT, DISCRETE_IMPORT] {
            for mode in MODES {
                let p = place(mode, DirectFit::Exact, cell);
                assert_eq!(p.route, Route::DirectAlias, "{}", p.census());
                assert!(!p.route.copies_guest_writes());
                assert_eq!(p.route.working(), None);
            }
        }
    }

    /// A linear buffer's bytes are its bytes: nothing about a buffer's layout
    /// can disagree with the guest's.
    #[test]
    fn a_linear_buffer_fits_wherever_import_exists() {
        assert_eq!(
            place(
                StorageMode::Shared,
                DirectFit::LinearBuffer,
                DISCRETE_IMPORT
            )
            .route,
            Route::DirectAlias
        );
    }

    /// Without import there is no direct alias on any topology, however well
    /// the representation fits — the pages simply cannot become device memory.
    #[test]
    fn without_import_an_exact_fit_still_copies() {
        for cell in [UNIFIED_NO_IMPORT, DISCRETE_NO_IMPORT] {
            let p = place(StorageMode::Shared, DirectFit::Exact, cell);
            assert!(
                matches!(p.route, Route::HostStaging { .. }),
                "{}",
                p.census()
            );
            assert!(p.route.copies_guest_writes());
        }
    }

    /// A representation the guest's pages cannot serve still uses them as a
    /// copy endpoint when import exists — a DMA the GPU performs rather than
    /// bytes the CPU moves.
    #[test]
    fn a_misfitting_representation_still_imports_as_a_transfer_endpoint() {
        for fit in [
            DirectFit::TilingDiffers,
            DirectFit::LayoutDiffers,
            DirectFit::UsageUnavailable,
        ] {
            assert_eq!(
                place(StorageMode::Shared, fit, DISCRETE_IMPORT).route,
                Route::ImportedTransfer {
                    working: MemoryClass::DeviceLocal
                },
                "{}",
                fit.name()
            );
            assert_eq!(
                place(StorageMode::Shared, fit, UNIFIED_IMPORT).route,
                Route::ImportedTransfer {
                    working: MemoryClass::DeviceLocalPreferred
                }
            );
        }
    }

    /// The topology chooses the working representation's class and nothing
    /// else. On unified it is the same DRAM either way, so a placement the CPU
    /// can also reach is free; on discrete the GPU's reads must not cross the
    /// bus.
    #[test]
    fn the_topology_chooses_the_working_class_and_not_the_route() {
        for import in [true, false] {
            let unified = place(
                StorageMode::Shared,
                DirectFit::TilingDiffers,
                HostCell {
                    topology: MemoryTopology::Unified,
                    host_pointer_import: import,
                },
            );
            let discrete = place(
                StorageMode::Shared,
                DirectFit::TilingDiffers,
                HostCell {
                    topology: MemoryTopology::Discrete,
                    host_pointer_import: import,
                },
            );
            assert_eq!(
                unified.route.name(),
                discrete.route.name(),
                "the topology did not change which route was taken"
            );
            assert_eq!(
                unified.route.working(),
                Some(MemoryClass::DeviceLocalPreferred)
            );
            assert_eq!(discrete.route.working(), Some(MemoryClass::DeviceLocal));
        }
    }

    /// The claim the whole crate is under: the same guest stream must mean the
    /// same thing on every cell. A storage mode that changed the route would be
    /// a mode that changed meaning, since a route is a performance decision.
    #[test]
    fn the_storage_mode_never_changes_the_route() {
        for cell in CELLS {
            for fit in FITS {
                let routes: Vec<Route> = MODES.iter().map(|m| place(*m, fit, cell).route).collect();
                assert!(
                    routes.windows(2).all(|w| w[0] == w[1]),
                    "storage mode changed the route for {} on {}/{}: {routes:?}",
                    fit.name(),
                    cell.topology.slug(),
                    u8::from(cell.host_pointer_import)
                );
            }
        }
    }

    /// Private is not a licence to stop keeping guest pages current on this
    /// wire, and the shortcut would be taken here if anywhere. The evidence is
    /// in the protocol vocabulary; this is the check that the shortcut was not
    /// taken.
    #[test]
    fn private_content_is_placed_exactly_as_shared_content_is() {
        for cell in CELLS {
            for fit in FITS {
                assert_eq!(
                    place(StorageMode::Private, fit, cell).route,
                    place(StorageMode::Shared, fit, cell).route,
                    "private took a route shared did not, which is the \
                     stale-page defect with no counter"
                );
            }
        }
    }

    /// Every cell of the matrix resolves to a route, and the four cells are not
    /// all the same route — a policy that answered identically everywhere would
    /// be one with no capability input at all.
    #[test]
    fn every_cell_resolves_and_the_cells_differ() {
        let mut seen: Vec<&str> = Vec::new();
        for cell in CELLS {
            for fit in FITS {
                let p = place(StorageMode::Managed, fit, cell);
                assert!(p.census().contains("vk_placement"));
                seen.push(p.route.name());
            }
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            3,
            "all three routes are reachable from the matrix"
        );
    }

    #[test]
    fn only_the_fitting_variants_fit() {
        assert!(DirectFit::Exact.fits());
        assert!(DirectFit::LinearBuffer.fits());
        assert!(!DirectFit::TilingDiffers.fits());
        assert!(!DirectFit::LayoutDiffers.fits());
        assert!(!DirectFit::UsageUnavailable.fits());
        let mut names: Vec<&str> = FITS.iter().map(|f| f.name()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }
}
