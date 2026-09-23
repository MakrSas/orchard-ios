//! `MTLPrimitiveType`, and the three classes its five values fall into.
//!
//! # A class is not a decoration
//!
//! A line list and a line strip are two arrangements of the same primitive:
//! the same rasterization rules, the same clip behaviour, a different reading
//! of the same vertices. Both APIs draw that line somewhere — Metal in
//! `MTLPrimitiveTopologyClass`, Vulkan in what a pipeline may change without
//! being rebuilt — so the grouping is a contract term rather than a
//! convenience, and it is named here where both rails can read it.
//!
//! # What is deliberately not here
//!
//! Whether a host can change topology without rebuilding a pipeline, and how
//! far within or across a class it may move. Those are device capabilities and
//! belong to the executor that queried one.

/// `MTLPrimitiveType`.
pub const MTL_PRIMITIVE_TYPE_POINT: u32 = 0;
pub const MTL_PRIMITIVE_TYPE_LINE: u32 = 1;
pub const MTL_PRIMITIVE_TYPE_LINE_STRIP: u32 = 2;
pub const MTL_PRIMITIVE_TYPE_TRIANGLE: u32 = 3;
pub const MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP: u32 = 4;

/// How a draw reads its vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimitiveType {
    Point,
    Line,
    LineStrip,
    Triangle,
    TriangleStrip,
}

/// The primitive a type rasterizes, without how its vertices are grouped.
///
/// Three, because the two strips share their class with their list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TopologyClass {
    Point,
    Line,
    Triangle,
}

impl TopologyClass {
    pub const ALL: [TopologyClass; 3] = [Self::Point, Self::Line, Self::Triangle];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Line => "line",
            Self::Triangle => "triangle",
        }
    }

    /// The one member of this class that is not a strip.
    ///
    /// Every class has exactly one, which is what makes this total and what
    /// makes it the class's canonical member: a layer that must name *some*
    /// primitive type standing for the whole class — a pipeline declaring a
    /// topology it may then be told to change within the class, say — names
    /// this one and cannot pick differently from one call to the next.
    #[must_use]
    pub const fn list(self) -> PrimitiveType {
        match self {
            Self::Point => PrimitiveType::Point,
            Self::Line => PrimitiveType::Line,
            Self::Triangle => PrimitiveType::Triangle,
        }
    }
}

impl PrimitiveType {
    pub const ALL: [PrimitiveType; 5] = [
        Self::Point,
        Self::Line,
        Self::LineStrip,
        Self::Triangle,
        Self::TriangleStrip,
    ];

    #[must_use]
    pub const fn parse(ordinal: u32) -> Option<Self> {
        Some(match ordinal {
            MTL_PRIMITIVE_TYPE_POINT => Self::Point,
            MTL_PRIMITIVE_TYPE_LINE => Self::Line,
            MTL_PRIMITIVE_TYPE_LINE_STRIP => Self::LineStrip,
            MTL_PRIMITIVE_TYPE_TRIANGLE => Self::Triangle,
            MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP => Self::TriangleStrip,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::Point => MTL_PRIMITIVE_TYPE_POINT,
            Self::Line => MTL_PRIMITIVE_TYPE_LINE,
            Self::LineStrip => MTL_PRIMITIVE_TYPE_LINE_STRIP,
            Self::Triangle => MTL_PRIMITIVE_TYPE_TRIANGLE,
            Self::TriangleStrip => MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP,
        }
    }

    #[must_use]
    pub const fn class(self) -> TopologyClass {
        match self {
            Self::Point => TopologyClass::Point,
            Self::Line | Self::LineStrip => TopologyClass::Line,
            Self::Triangle | Self::TriangleStrip => TopologyClass::Triangle,
        }
    }

    /// Whether consecutive primitives share vertices.
    ///
    /// The distinction a restart index would act on, and the reason its
    /// absence matters — see the executors.
    #[must_use]
    pub const fn is_strip(self) -> bool {
        matches!(self, Self::LineStrip | Self::TriangleStrip)
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Line => "line",
            Self::LineStrip => "line_strip",
            Self::Triangle => "triangle",
            Self::TriangleStrip => "triangle_strip",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    /// Every class's canonical member is in that class, is not a strip, and is
    /// the only member of the class that is not — so the choice is forced
    /// rather than picked.
    #[test]
    fn each_class_has_exactly_one_non_strip_member_and_that_is_its_list() {
        for class in TopologyClass::ALL {
            let list = class.list();
            assert_eq!(list.class(), class);
            assert!(!list.is_strip());
            let non_strips: alloc::vec::Vec<PrimitiveType> = PrimitiveType::ALL
                .into_iter()
                .filter(|p| p.class() == class && !p.is_strip())
                .collect();
            assert_eq!(non_strips.as_slice(), &[list]);
        }
    }

    #[test]
    fn the_set_is_closed_at_five_and_round_trips() {
        let ordinals: BTreeSet<u32> = PrimitiveType::ALL.iter().map(|p| p.ordinal()).collect();
        assert_eq!(ordinals, BTreeSet::from([0, 1, 2, 3, 4]));
        for p in PrimitiveType::ALL {
            assert_eq!(PrimitiveType::parse(p.ordinal()), Some(p));
        }
        assert_eq!(PrimitiveType::parse(5), None);
        let names: BTreeSet<&str> = PrimitiveType::ALL.iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), PrimitiveType::ALL.len());
    }

    /// Five types, three classes, and each strip in its list's class. This is
    /// the grouping an executor decides what a pipeline may change from.
    #[test]
    fn each_strip_shares_its_lists_class_and_the_classes_are_three() {
        assert_eq!(PrimitiveType::LineStrip.class(), TopologyClass::Line);
        assert_eq!(PrimitiveType::Line.class(), TopologyClass::Line);
        assert_eq!(
            PrimitiveType::TriangleStrip.class(),
            TopologyClass::Triangle
        );
        assert_eq!(PrimitiveType::Triangle.class(), TopologyClass::Triangle);
        assert_eq!(PrimitiveType::Point.class(), TopologyClass::Point);

        let classes: BTreeSet<TopologyClass> =
            PrimitiveType::ALL.iter().map(|p| p.class()).collect();
        assert_eq!(classes.len(), TopologyClass::ALL.len());
        // Every class is reachable from some primitive type, so no class is a
        // grouping nothing lands in.
        for class in TopologyClass::ALL {
            assert!(PrimitiveType::ALL.iter().any(|p| p.class() == class));
        }
    }

    #[test]
    fn exactly_the_two_strips_share_vertices() {
        let strips: BTreeSet<u32> = PrimitiveType::ALL
            .iter()
            .filter(|p| p.is_strip())
            .map(|p| p.ordinal())
            .collect();
        assert_eq!(strips, BTreeSet::from([2, 4]));
    }
}
