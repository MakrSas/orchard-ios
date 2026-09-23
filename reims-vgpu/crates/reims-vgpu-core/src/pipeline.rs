//! The pipeline lifetime, and the rule that a draw never compiles.
//!
//! # Why this is a state machine and not a cache
//!
//! A pipeline that is compiled lazily on first use is a pipeline whose
//! compilation cost lands on a draw — and a draw is the one place in this
//! architecture that may not block, wait on a host, or discover work. Making
//! the lifetime explicit moves the cost to where the guest actually asked for
//! it: the object-creation packet starts the work, and a transaction that wants
//! the pipeline holds a lease and is not ready until the lease resolves.
//!
//! So "not compiled yet" is a state a transaction can wait on rather than a
//! cache miss a draw has to handle, and "refused" is a state rather than an
//! error return that some call sites check and others do not.
//!
//! # Refused is terminal and says why
//!
//! A pipeline the device cannot build does not retry on the next draw. It stays
//! refused for the lifetime of the object, with the reason attached, so a guest
//! re-binding it every frame produces one refusal rather than one per frame —
//! and so that the reason survives to whoever reads the failure channel.
//!
//! # What this crate cannot see
//!
//! Nothing here names a shader, a module, a descriptor layout or a native
//! handle. The translation and compilation *happen* somewhere that does; this
//! owns when they may start, what a waiting transaction observes, and when the
//! result may be dropped.

use crate::access::AccessMode;
use crate::identity::{ResourceId, SessionGeneration};
use reims_vgpu_protocol::render::ShaderStage;
use std::collections::HashMap;

/// Where a pipeline is in its life.
///
/// The order is the lifetime's order, and [`Ord`] follows it, so "has it got at
/// least as far as X" is a comparison rather than a match. `Refused` and
/// `Retired` are both terminal and are deliberately not comparable-as-progress
/// with each other; a caller asking "is this usable" asks [`Self::is_ready`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineState {
    /// The guest has created the object; no work has started.
    Declared,
    /// The guest's shader form is being turned into the host's.
    Translating,
    /// The host is building the pipeline.
    Compiling,
    /// Usable.
    Ready,
    /// The device cannot build it, and will not try again.
    Refused,
    /// The guest deleted it, or its generation closed.
    Retired,
}

impl PipelineState {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Refused | Self::Retired)
    }

    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Translating => "translating",
            Self::Compiling => "compiling",
            Self::Ready => "ready",
            Self::Refused => "refused",
            Self::Retired => "retired",
        }
    }

    /// Whether `next` is a legal step from here.
    ///
    /// Written as a table rather than as a set of guards at each mutator,
    /// because a guard is a thing one caller can be written without.
    #[must_use]
    pub const fn may_become(self, next: PipelineState) -> bool {
        matches!(
            (self, next),
            (Self::Declared, Self::Translating)
                | (Self::Translating, Self::Compiling)
                | (Self::Compiling, Self::Ready)
                // Translation and compilation can each fail, and a pipeline can
                // be refused before either starts — a descriptor the model
                // cannot represent is refused at declaration.
                | (Self::Declared | Self::Translating | Self::Compiling, Self::Refused)
                // Deletion can arrive at any point that is not already
                // terminal. A guest deleting a pipeline mid-compile is
                // ordinary, and the compile finishing afterwards must not
                // resurrect it.
                | (
                    Self::Declared | Self::Translating | Self::Compiling | Self::Ready,
                    Self::Retired
                )
                // **`Ready` is not terminal, and a driven macos-15 desktop is
                // why.** These states say whether *the device* can execute the
                // pipeline now, not what the guest's object list says it is.
                // The guest rewrites the shader behind a live pipeline ref in
                // place — no delete, no new generation, so no new
                // [`ResourceId`] — and the executor's translation is keyed by
                // the shader's content, so it stops holding one. Four refs a
                // desktop, measured.
                //
                // With no step back, `Ready` meant "was translated once" and a
                // transaction binding that ref was released against a shader
                // the executor was still translating. The alternative — leaving
                // it `Ready` and letting the record refuse — is a lost draw
                // with the model still claiming the pipeline was usable, which
                // is a model that disagrees with the device.
                //
                // Only the executor may take this step, and only because it is
                // the layer that discovers it: this crate cannot read a shader.
                // It is not a reset — `Declared` is unreachable from here,
                // because the pipeline is still declared and always was.
                | (Self::Ready, Self::Translating)
        )
    }
}

/// What a compiled pipeline does with each slot it binds.
///
/// # An immutable fact, published by whoever compiled it
///
/// Nothing in this crate can read a shader. Which of an encoder's bound slots a
/// pipeline actually references, and in which direction, is discovered during
/// translation — by the executor, which is the layer that has the shader — and
/// it reaches the model as this, once, when the pipeline becomes ready. That is
/// the plan's rule about what advances semantic state: an immutable fact
/// returned by an executor, not a query the model makes.
///
/// # Why the alternative is expensive rather than wrong
///
/// Without one, [`crate::encoder`] gives every bound slot
/// [`AccessMode::Unknown`], which conflicts with everything. No edge is missed;
/// a great many are added. The point of publishing this is to buy those back,
/// and the point of `Unknown` being its own variant is that the census can say
/// how many are still being paid for.
///
/// # A slot past the end is unreferenced, not unknown
///
/// The tables are as long as the pipeline's own binding set. A bound slot
/// beyond them is one the shader does not name, so it contributes nothing —
/// falling back to `Unknown` there would make a guest with a long-tailed bind
/// table pay forever for slots no shader reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BindingUsage {
    buffers: Vec<Option<AccessMode>>,
    textures: Vec<Option<AccessMode>>,
}

impl BindingUsage {
    #[must_use]
    pub fn new(buffers: Vec<Option<AccessMode>>, textures: Vec<Option<AccessMode>>) -> Self {
        Self { buffers, textures }
    }

    /// What the pipeline does with buffer slot `slot`, or `None` if it does not
    /// reference it.
    #[must_use]
    pub fn buffer(&self, slot: u32) -> Option<AccessMode> {
        self.buffers.get(slot as usize).copied().flatten()
    }

    /// What the pipeline does with texture slot `slot`.
    #[must_use]
    pub fn texture(&self, slot: u32) -> Option<AccessMode> {
        self.textures.get(slot as usize).copied().flatten()
    }

    /// Whether the pipeline writes anything through its bindings.
    ///
    /// A pipeline that only reads cannot be the producer half of a hazard, so
    /// this is worth one question rather than a scan at every draw.
    #[must_use]
    pub fn writes_anything(&self) -> bool {
        self.buffers
            .iter()
            .chain(self.textures.iter())
            .flatten()
            .any(|m| m.writes())
    }
}

/// Why a pipeline will not be built.
///
/// A payload rather than a slug, because the reason has to survive to a failure
/// channel this crate cannot reach — and because a refusal without a reason is
/// how a guest ends up rendering nothing with a clean log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalReason {
    /// A descriptor field the model has no representation for.
    Undescribable(&'static str),
    /// The guest's shader form could not be translated.
    TranslationFailed(&'static str),
    /// The host refused to build it.
    CompilationFailed(&'static str),
}

impl RefusalReason {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Undescribable(_) => "pipeline_undescribable",
            Self::TranslationFailed(_) => "pipeline_translation_failed",
            Self::CompilationFailed(_) => "pipeline_compilation_failed",
        }
    }

    pub const fn detail(self) -> &'static str {
        match self {
            Self::Undescribable(d) | Self::TranslationFailed(d) | Self::CompilationFailed(d) => d,
        }
    }
}

/// One pipeline object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pipeline {
    pub id: ResourceId,
    /// The semantic lifetime it was declared in. A pipeline outliving its
    /// generation is not usable by work from a later one, however healthy the
    /// host object is.
    pub generation: SessionGeneration,
    pub state: PipelineState,
    pub refusal: Option<RefusalReason>,
}

/// What a transaction that wants a pipeline observes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lease {
    /// Usable now.
    Ready,
    /// Not yet. The transaction is not ready either, and nothing blocks.
    Pending,
    /// It will never be usable, with the reason.
    Refused(RefusalReason),
    /// There is no such pipeline in this generation, and which of the three
    /// ways that happened.
    Absent(AbsentBecause),
}

/// The three ways a pipeline this session holds no lease for got that way.
///
/// **They are different defects and the slug alone cannot tell them apart.**
/// A driven macos-26 boot refused 914 exec packets on `pipeline_absent` where a
/// macos-15 boot refused none, and the three candidates predict different fixes:
/// a name nothing declared is a lease list built from something other than the
/// walk, a stale generation is a reset that did not close its table, and a
/// retired one is the guest's own delete arriving before work that still binds
/// it — which under a model that parks packets is a delete resolved at the
/// wrong moment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsentBecause {
    /// No entry at all: nothing ever declared this id.
    Undeclared,
    /// An entry from a generation this session has left behind.
    OtherGeneration,
    /// The guest deleted it, or its generation closed.
    Retired,
}

impl AbsentBecause {
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Undeclared => "pipeline_absent_undeclared",
            Self::OtherGeneration => "pipeline_absent_other_generation",
            Self::Retired => "pipeline_absent_retired",
        }
    }
}

/// Why a transaction can never use a pipeline it binds.
///
/// Both variants are terminal for the work, and they are kept apart because
/// they are different defects: a refused pipeline is this device failing to
/// build what the guest asked for, and an absent one is work naming an object
/// this generation does not have — a use-after-delete, or a packet that
/// outlived a reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeaseRefusal {
    Refused {
        pipeline: ResourceId,
        reason: RefusalReason,
    },
    Absent {
        pipeline: ResourceId,
        because: AbsentBecause,
    },
}

impl LeaseRefusal {
    /// The name this reaches a failure channel under. A refused pipeline
    /// reports the build's own reason, because "the draw could not run" is not
    /// the fact anyone reading the log needs.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Refused { reason, .. } => reason.slug(),
            Self::Absent { because, .. } => because.slug(),
        }
    }
}

/// One pipeline's published usage, per shader stage.
///
/// A stage is [`Option`] and not a defaulted [`BindingUsage`], and the
/// difference is the whole safety argument. An empty `BindingUsage` says every
/// slot is past the end of the pipeline's binding set — that is, *unreferenced*,
/// contributing nothing to the footprint. "Nothing has been published for this
/// stage" says the opposite: nothing is known, so every bound slot participates
/// as [`AccessMode::Unknown`]. Collapsing the two would turn a pipeline nobody
/// has reflected into a draw with no memory footprint at all, which is every
/// hazard edge missed rather than a coarser answer.
///
/// A render pipeline publishes the two render stages and leaves compute unset;
/// a compute pipeline does the reverse. Asking a render pipeline for its
/// compute stage therefore answers "not published", which is the true answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PublishedUsage {
    vertex: Option<BindingUsage>,
    fragment: Option<BindingUsage>,
    compute: Option<BindingUsage>,
}

impl PublishedUsage {
    /// What a render executor knows about the two render stages.
    ///
    /// Each stage is an [`Option`] the caller states, because knowing one and
    /// not the other is a real position for an executor to be in — a vertex
    /// module that reflected and a fragment module that did not — and the
    /// difference between "published, references nothing" and "not published"
    /// is exactly the difference between no participation and an `Unknown` one.
    #[must_use]
    pub fn render(vertex: Option<BindingUsage>, fragment: Option<BindingUsage>) -> Self {
        Self {
            vertex,
            fragment,
            compute: None,
        }
    }

    /// What a compute executor knows, once it has compiled the kernel.
    #[must_use]
    pub fn compute(kernel: BindingUsage) -> Self {
        Self {
            vertex: None,
            fragment: None,
            compute: Some(kernel),
        }
    }

    /// The stage's usage, or `None` where nothing was published for it.
    ///
    /// `None` for the stage argument is the compute stage: a dispatch reads one
    /// set of tables and there is no [`ShaderStage`] value that names it.
    #[must_use]
    pub fn stage(&self, stage: Option<ShaderStage>) -> Option<&BindingUsage> {
        match stage {
            Some(ShaderStage::Vertex) => self.vertex.as_ref(),
            Some(ShaderStage::Fragment) => self.fragment.as_ref(),
            None => self.compute.as_ref(),
        }
    }
}

/// What an executor has published about the shaders behind a pipeline.
///
/// The read side of [`PipelineTable::publish_usage`], as a trait because the
/// walk that needs it must not depend on where the table lives — see
/// [`crate::walk::exec`]. Every implementation answers `None` for a pipeline it
/// knows nothing about, and `None` is the conservative answer: the encoder then
/// gives every bound slot [`AccessMode::Unknown`].
pub trait UsageSource: core::fmt::Debug {
    /// One stage of one pipeline, or `None` if nothing is published for it.
    fn binding_usage(
        &self,
        pipeline: ResourceId,
        stage: Option<ShaderStage>,
    ) -> Option<&BindingUsage>;
}

/// Nothing is published, for a caller with no executor answer to give.
///
/// A statement and not an opt-out: it produces exactly the conservative
/// footprint the model took before anything published, so a caller using it
/// pays for every edge and misses none.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoUsage;

impl UsageSource for NoUsage {
    fn binding_usage(
        &self,
        _pipeline: ResourceId,
        _stage: Option<ShaderStage>,
    ) -> Option<&BindingUsage> {
        None
    }
}

/// The pipeline objects of one session.
#[derive(Debug, Default)]
pub struct PipelineTable {
    pipelines: HashMap<ResourceId, Pipeline>,
    /// What an executor has published about each pipeline's shaders.
    ///
    /// Beside [`Self::pipelines`] rather than a field of [`Pipeline`], which is
    /// `Copy` and is passed around by value everywhere a lease is answered.
    /// Its lifetime is the pipeline entry's: every door that removes an entry
    /// or un-builds it removes this too, because a reflection is a fact about a
    /// build and a pipeline that has to be built again has not published
    /// anything about the build that will replace it.
    usage: HashMap<ResourceId, PublishedUsage>,
    census: Census,
}

impl UsageSource for PipelineTable {
    fn binding_usage(
        &self,
        pipeline: ResourceId,
        stage: Option<ShaderStage>,
    ) -> Option<&BindingUsage> {
        self.usage.get(&pipeline)?.stage(stage)
    }
}

/// What the table has seen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub declared: usize,
    pub ready: usize,
    pub refused: usize,
    pub retired: usize,
    /// Leases taken while a pipeline was still being built. The number that
    /// says whether starting compilation at declaration is early enough.
    pub leases_pending: usize,
    /// Leases taken on a pipeline that was already ready. The steady state.
    pub leases_ready: usize,
    /// Pipelines that left [`PipelineState::Ready`] because the executor
    /// stopped holding their translation.
    ///
    /// Counted rather than merely allowed: the step exists for a guest that
    /// rewrites a shader in place, and a count that climbs with the frame rate
    /// would mean something else is provoking it — a cycle between the
    /// withdrawal and whatever re-readies, which is a hang rather than four
    /// events a desktop.
    pub withdrawn: usize,
}

/// What the table is holding, by state, at one moment.
///
/// See [`PipelineTable::resting`] for why this is a separate question from
/// [`Census`], which counts events.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resting {
    pub declared: usize,
    pub translating: usize,
    pub compiling: usize,
    pub ready: usize,
    pub refused: usize,
    pub retired: usize,
}

impl Resting {
    /// The pipelines a transaction can be waiting on: everything a lease
    /// answers `Pending` for.
    ///
    /// One number rather than three, because "is anything parked on a build"
    /// is the question a live boot asks, and three columns are three things to
    /// forget to add up.
    #[must_use]
    pub const fn pending(self) -> usize {
        self.declared + self.translating + self.compiling
    }

    #[must_use]
    pub const fn total(self) -> usize {
        self.pending() + self.ready + self.refused + self.retired
    }
}

/// What a closed semantic generation left behind.
///
/// Two lists over the same removal, because two callers ask different
/// questions of it and answering only one of them strands the other — see
/// [`PipelineTable::generation_closed`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[must_use = "a closed generation's pipelines are host objects to destroy and waits to discharge"]
pub struct Closed {
    /// Removed pipelines a host object exists for, in id order. Destroyed
    /// rather than abandoned: a closed generation leaves the handles usable
    /// and merely unnameable.
    pub destroy: Vec<ResourceId>,
    /// Every removed pipeline, in id order — including the ones no host object
    /// was ever built for. These are the names nothing may reach again, which
    /// is what a parked transaction was waiting on.
    pub removed: Vec<ResourceId>,
}

impl PipelineTable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn census(&self) -> Census {
        self.census
    }

    /// How many pipelines the table is **holding** in each state right now.
    ///
    /// # Not the same question as [`Self::census`], and the difference is a hang
    ///
    /// The census counts *events*: how many declarations happened, how many
    /// builds finished. A pipeline that was declared and never advanced is
    /// counted once there and never again, so a table quietly accumulating
    /// pipelines nothing will ever build reads exactly like a healthy one —
    /// the two numbers only differ by a subtraction across counters that
    /// nobody performs while looking at a live boot.
    ///
    /// That accumulation is precisely the failure mode this lifetime has.
    /// `Declared`, `Translating` and `Compiling` are the three states a
    /// transaction *waits* on, so an occupancy at any of them that does not
    /// fall is work parked on a build nobody is running — and a rising one is
    /// a hang forming, visible before the guest stops drawing rather than
    /// after.
    ///
    /// `Refused` and `Retired` are terminal and their occupancy only grows;
    /// they are here so the six sum to the table's size and a reader can see
    /// that they do.
    #[must_use]
    pub fn resting(&self) -> Resting {
        let mut out = Resting::default();
        for p in self.pipelines.values() {
            let slot = match p.state {
                PipelineState::Declared => &mut out.declared,
                PipelineState::Translating => &mut out.translating,
                PipelineState::Compiling => &mut out.compiling,
                PipelineState::Ready => &mut out.ready,
                PipelineState::Refused => &mut out.refused,
                PipelineState::Retired => &mut out.retired,
            };
            *slot += 1;
        }
        out
    }

    /// Declare a pipeline. Returns whether it was new.
    ///
    /// Re-declaring an id that is live is not an update: the guest's object
    /// namespace produces a new generation for a reused slot, so two live
    /// declarations of one [`ResourceId`] would mean the namespace failed to do
    /// that, and silently replacing the first would hide it.
    pub fn declare(&mut self, id: ResourceId, generation: SessionGeneration) -> bool {
        if self.pipelines.contains_key(&id) {
            return false;
        }
        self.census.declared += 1;
        self.pipelines.insert(
            id,
            Pipeline {
                id,
                generation,
                state: PipelineState::Declared,
                refusal: None,
            },
        );
        true
    }

    #[must_use]
    pub fn get(&self, id: ResourceId) -> Option<&Pipeline> {
        self.pipelines.get(&id)
    }

    /// Whether an executor has already said what this pipeline's shaders do
    /// with each bound slot.
    ///
    /// The read side of [`Self::publish_usage`] as a *question*, for a rail
    /// whose pre-scan reaches the same pipeline once per packet and must build
    /// its tables once per build. Enumerating a shader's binding set allocates,
    /// and a rail-side "already done" set would be a second record of a fact
    /// this table already holds — one that a withdrawal or a retirement could
    /// leave disagreeing with it. Asking here cannot disagree.
    ///
    /// `false` for a pipeline this table does not hold, which is the answer
    /// that makes the caller try: an executor that publishes before the
    /// pipeline is declared is refused by [`Self::publish_usage`], and the next
    /// pass is what lands it.
    #[must_use]
    pub fn usage_published(&self, id: ResourceId) -> bool {
        self.usage.contains_key(&id)
    }

    /// Advance a pipeline. Returns whether the step was legal and taken.
    ///
    /// An illegal step is refused rather than applied, and a caller that has to
    /// care can ask. The common illegal step is real: a compile that finishes
    /// after the guest deleted the pipeline, which must not resurrect it.
    ///
    /// # `Refused` is not a step this door can take
    ///
    /// [`PipelineState::may_become`] says the transition is legal, and it is —
    /// but the state is not the whole fact. A refused pipeline carries the
    /// reason it was refused, because a refusal without one is how a guest
    /// ends up rendering nothing with a clean log, and this door has no reason
    /// to write. Taking the step here produced `state: Refused, refusal: None`
    /// — a state the lease answer `expect`s cannot exist, so the next guest draw
    /// binding that pipeline panicked in the semantic model, and the census
    /// read zero refusals meanwhile.
    ///
    /// So the door that can supply a reason is the only door: [`Self::refuse`].
    pub fn advance(&mut self, id: ResourceId, next: PipelineState) -> bool {
        if next == PipelineState::Refused {
            return false;
        }
        // `Ready -> Translating` is a legal step and this is not the door for
        // it, for the same reason `Refused` is not: it is a step only one
        // caller may take, and this door is walked by callers that must not.
        //
        // The Metal rail retains no pipeline state, so it walks
        // `Translating -> Compiling -> Ready` on **every draw** that binds a
        // pipeline. If those calls could take a ready pipeline backwards, every
        // draw would un-ready its own pipeline and every transaction parked on
        // it would wait again — a hot loop where the withdrawal exists for four
        // events a desktop. [`Self::withdraw`] is the door, and only the
        // executor discovering it no longer holds the translation may open it.
        if next == PipelineState::Translating
            && self.pipelines.get(&id).map(|p| p.state) == Some(PipelineState::Ready)
        {
            return false;
        }
        let Some(p) = self.pipelines.get_mut(&id) else {
            return false;
        };
        if !p.state.may_become(next) {
            return false;
        }
        p.state = next;
        match next {
            PipelineState::Ready => self.census.ready += 1,
            PipelineState::Retired => {
                self.census.retired += 1;
                // The name is gone, so the reflection under it is too. Left
                // behind it would be answered for a later pipeline that
                // reused the id, which is a narrower footprint derived from
                // another object's shader.
                self.usage.remove(&id);
            }
            _ => {}
        }
        true
    }

    /// Take a ready pipeline back to [`PipelineState::Translating`], because
    /// the executor no longer holds a translation for it.
    ///
    /// Its own door rather than an [`Self::advance`] step, and the reason is
    /// the same one [`Self::refuse`] has: the step is legal but only one caller
    /// may take it. `advance` is walked by a rail that retains no pipeline
    /// state and re-walks the build on every draw, and that caller taking this
    /// step would un-ready every pipeline it draws with.
    ///
    /// Returns whether anything moved. `false` for a pipeline that was already
    /// waiting, terminal, or absent — all of which are ordinary: the caller
    /// asks for every ref its pre-scan found pending, and almost all of those
    /// are a cold pipeline that has simply not finished yet.
    pub fn withdraw(&mut self, id: ResourceId) -> bool {
        let Some(p) = self.pipelines.get_mut(&id) else {
            return false;
        };
        if p.state != PipelineState::Ready {
            return false;
        }
        p.state = PipelineState::Translating;
        // The build this described is the one the executor just said it no
        // longer holds. What the rebuild publishes is the rebuild's.
        self.usage.remove(&id);
        self.census.withdrawn += 1;
        true
    }

    /// Refuse a pipeline, with the reason.
    pub fn refuse(&mut self, id: ResourceId, reason: RefusalReason) -> bool {
        let Some(p) = self.pipelines.get_mut(&id) else {
            return false;
        };
        if !p.state.may_become(PipelineState::Refused) {
            return false;
        }
        p.state = PipelineState::Refused;
        p.refusal = Some(reason);
        self.census.refused += 1;
        // A build that will not happen publishes nothing, and anything already
        // published described a build this refusal supersedes.
        self.usage.remove(&id);
        true
    }

    /// What a transaction wanting this pipeline in this generation observes.
    ///
    /// The generation is a parameter rather than read from the pipeline,
    /// because the question is whether *this* work may use it: a pipeline
    /// declared in a closed generation is absent to work from a later one even
    /// though the object is intact.
    pub fn lease(&mut self, id: ResourceId, generation: SessionGeneration) -> Lease {
        let lease = self.peek(id, generation);
        self.charge(lease);
        lease
    }

    /// What a lease would answer, without charging the census for it.
    ///
    /// The census counts leases *taken*, and a lease is taken only if the
    /// transaction that wanted it is admitted. [`Self::waits_for`] therefore
    /// asks this for every pipeline first and charges only once it knows it is
    /// returning `Ok` — otherwise a list whose third pipeline is refused
    /// charges two leases for a transaction that never ran, and the number that
    /// says whether compilation starts early enough grows with refusals.
    fn peek(&self, id: ResourceId, generation: SessionGeneration) -> Lease {
        let Some(p) = self.pipelines.get(&id) else {
            return Lease::Absent(AbsentBecause::Undeclared);
        };
        if p.generation != generation {
            return Lease::Absent(AbsentBecause::OtherGeneration);
        }
        if p.state == PipelineState::Retired {
            return Lease::Absent(AbsentBecause::Retired);
        }
        match p.state {
            PipelineState::Ready => Lease::Ready,
            PipelineState::Refused => {
                Lease::Refused(p.refusal.expect("a refused pipeline carries its reason"))
            }
            PipelineState::Declared | PipelineState::Translating | PipelineState::Compiling => {
                Lease::Pending
            }
            PipelineState::Retired => Lease::Absent(AbsentBecause::Retired),
        }
    }

    /// Charge the census for a lease that was actually taken.
    const fn charge(&mut self, lease: Lease) {
        match lease {
            Lease::Ready => self.census.leases_ready += 1,
            Lease::Pending => self.census.leases_pending += 1,
            Lease::Refused(_) | Lease::Absent(_) => {}
        }
    }

    /// Turn a transaction's pipeline leases into the waits it is admitted with.
    ///
    /// **The join between "a transaction leases pipelines" and "a transaction
    /// is held until they are built".** [`crate::exec::ExecWork::pipeline_leases`]
    /// says which pipelines the records bind and
    /// [`crate::session::SessionModel::admit`] takes a list of waits, and
    /// nothing turned one into the other — so the only ways to call `admit`
    /// were with no waits, which runs a draw against a pipeline that is still
    /// compiling, or with a list built somewhere else, which is a second
    /// opinion about what the records bind.
    ///
    /// A lease is taken for every pipeline, in order, which is what the
    /// census counts. Only the pending ones come back: a ready pipeline is
    /// nothing to wait for, and returning it would park the transaction on a
    /// completion that has already happened.
    ///
    /// # Errors
    ///
    /// [`LeaseRefusal`] at the first pipeline this work can never use. The
    /// transaction is refused rather than admitted with a wait that will never
    /// resolve — which is the same choice `admit` makes for every other
    /// unsatisfiable packet, and for the same reason: a completion word the
    /// guest waits on forever is worse than a refusal it is told about.
    ///
    /// Nothing is charged for a list that refuses. The walk decides in full
    /// first and only then charges, so a refusal at the third pipeline leaves
    /// the census exactly as it was — which is what
    /// `an_unusable_pipeline_stops_the_walk_and_nothing_is_counted` asserts.
    pub fn waits_for(
        &mut self,
        leases: &[ResourceId],
        generation: SessionGeneration,
    ) -> Result<Vec<ResourceId>, LeaseRefusal> {
        let mut taken = Vec::new();
        let mut waits = Vec::new();
        for &pipeline in leases {
            match self.peek(pipeline, generation) {
                Lease::Ready => taken.push(Lease::Ready),
                // **A pipeline the guest deleted is not a reason to refuse the
                // packet.** It is neither a wait — nothing will build it — nor a
                // failure of this device, and the transaction's other records
                // have nothing to do with it. On this interface a command buffer
                // recorded before the delete still binds the object, and the
                // guest is entitled to submit it: the host encoder retained the
                // pipeline when the record was written, and the guest's delete
                // ends the *name*.
                //
                // Refusing cost whole packets. A driven macos-26 boot refused
                // **9374 exec packets** with `pipeline_absent_retired` — the
                // guest deletes 145 pipelines a boot and keeps binding them —
                // while the macos-15 boot beside it deletes four and refuses
                // none. Every one of those packets carried records that had
                // nothing to do with the deleted pipeline.
                //
                // What the executor does with a bind it cannot satisfy is the
                // executor's, and it already answers: the rail drops its own
                // retained object on the same delete and reports the missing
                // bind per *draw*. One draw is what the guest loses, which is
                // what it lost before this table existed.
                //
                // The other two ways to be absent stay refusals, because
                // neither is the guest's doing. `Undeclared` means the lease
                // list and the declarations disagree — the caller declares every
                // lease before admitting — and `OtherGeneration` is work that
                // outlived a reset, whose host objects a destroyer has already
                // been handed.
                Lease::Absent(AbsentBecause::Retired) => {
                    taken.push(Lease::Absent(AbsentBecause::Retired))
                }
                Lease::Pending => {
                    taken.push(Lease::Pending);
                    waits.push(pipeline);
                }
                Lease::Refused(reason) => return Err(LeaseRefusal::Refused { pipeline, reason }),
                Lease::Absent(because) => {
                    return Err(LeaseRefusal::Absent { pipeline, because });
                }
            }
        }
        // Decided in full first: nothing is charged for a list that refuses.
        for lease in taken {
            self.charge(lease);
        }
        Ok(waits)
    }

    /// An executor finished compiling and says what its shaders do with each
    /// bound slot.
    ///
    /// **The one door the model's own conservatism is bought back through.**
    /// Without a publication [`crate::encoder`] gives every bound slot
    /// [`AccessMode::Unknown`], which conflicts with everything; the first boot
    /// that counted found 81 % of its accesses in that column. This is the fact
    /// that narrows them, and only the layer holding the shader can state it —
    /// which is why it arrives as an immutable value an executor pushes rather
    /// than as a question the model asks.
    ///
    /// Refused for a pipeline this table does not hold, and for one that has
    /// reached a terminal state: a reflection landing after the guest deleted
    /// the pipeline describes a build nothing may bind, and keeping it would
    /// leave an entry alive under a name that is gone. Returns whether it
    /// landed.
    ///
    /// Republication is allowed and replaces. A pipeline taken back to
    /// [`PipelineState::Translating`] by [`Self::withdraw`] or
    /// [`Self::device_lost`] has already had its usage dropped, so what lands
    /// after a rebuild is the rebuild's own answer and never the previous
    /// build's.
    pub fn publish_usage(&mut self, id: ResourceId, usage: PublishedUsage) -> bool {
        let Some(p) = self.pipelines.get(&id) else {
            return false;
        };
        if p.state.is_terminal() {
            return false;
        }
        self.usage.insert(id, usage);
        true
    }

    /// Retire a pipeline the guest deleted.
    pub fn retire(&mut self, id: ResourceId) -> bool {
        self.advance(id, PipelineState::Retired)
    }

    /// A semantic generation closed: its pipelines can never be named again.
    ///
    /// Returns the ones a host object exists for, in id order.
    ///
    /// # Why this is a door and not a consequence of the generation check
    ///
    /// [`Self::lease`] already answers `Absent` for a pipeline of another
    /// generation, so a closed generation's pipelines are unusable without
    /// this. What they are not is *gone*: the entries stay, so a guest that
    /// resets in a loop grows this table without bound, and — the reason it
    /// matters more — nothing hands their host objects to whoever destroys
    /// them. [`PipelineState::Retired`]'s own doc names two ways in, "the
    /// guest deleted it, or its generation closed", and only the first had a
    /// path.
    ///
    /// The objects are *destroyed*, not abandoned: a closed generation leaves
    /// the handles perfectly usable and merely unnameable, which is exactly
    /// [`crate::retire::Validity::SemanticallyClosed`]. That is the difference
    /// from [`Self::device_lost`], where the handles are what went.
    ///
    /// Only the states a host object can exist in are offered for
    /// destruction. `Declared` never reached the host, and `Refused` is a
    /// build that did not happen; both leave the table with the rest and
    /// neither is offered to a destroyer that has nothing to destroy.
    ///
    /// **Removal and destruction are different scopes, and both come back.**
    /// Every pipeline of the closed generation leaves the table, whatever its
    /// state, and a transaction parked on one is parked on a wait that nothing
    /// can now discharge — [`Self::advance`] answers `false` for a pipeline
    /// with no entry, so the compile that lands afterwards releases nobody.
    /// Returning only the destroyable ones scoped this answer to whether a
    /// host object existed, which is the right question for a destroyer and
    /// the wrong one for the waiters: a `Declared` pipeline has no host object
    /// and a draw can be parked on it just the same.
    #[must_use = "a pipeline nobody destroys is a host object nothing frees, and one nobody names is a wait nothing discharges"]
    pub fn generation_closed(&mut self, closed: SessionGeneration) -> Closed {
        let mut out = Closed::default();
        self.pipelines.retain(|id, p| {
            if p.generation != closed {
                return true;
            }
            if matches!(
                p.state,
                PipelineState::Translating | PipelineState::Compiling | PipelineState::Ready
            ) {
                self.census.retired += 1;
                out.destroy.push(*id);
            }
            out.removed.push(*id);
            false
        });
        for id in &out.removed {
            self.usage.remove(id);
        }
        out.destroy.sort_unstable();
        out.removed.sort_unstable();
        out
    }

    /// The host device incarnation ended: every build it performed is gone.
    ///
    /// Returns the pipelines whose build has to start again, in id order.
    ///
    /// # Why a build and not the object
    ///
    /// [`PipelineState::Ready`] is a statement that the *host* has built this
    /// pipeline, and a host build belongs to one device incarnation. The
    /// semantic object does not: a device loss is not a reset, so the guest
    /// still names what it named — see [`crate::retire`], whose whole subject
    /// is that these are two lifetimes and that answering with one produces
    /// "dead handles reachable under a live name". This is that name.
    ///
    /// It is worse than the usual shape of that failure because the lease is
    /// taken at *admission*. A transaction binding a pipeline still reading
    /// `Ready` is admitted with no wait, given an ordering position and a
    /// completion obligation, and only then does an executor find there is
    /// nothing to record with — a refusal after admission, which
    /// [`crate::session`] opens by saying costs the guest the channel.
    ///
    /// So the built ones go back to [`PipelineState::Declared`] and a
    /// transaction binding one waits, exactly as it would have waited the
    /// first time. `Refused` and `Retired` do not move: a pipeline this device
    /// could not describe is not one the next incarnation describes either,
    /// and an object the guest deleted is not resurrected by a new device.
    #[must_use = "a build nobody restarts is a transaction that waits forever"]
    pub fn device_lost(&mut self) -> Vec<ResourceId> {
        let mut rebuilding = Vec::new();
        for (id, p) in &mut self.pipelines {
            if p.state.is_terminal() {
                continue;
            }
            // Counted as a declaration, because that is what it now is: the
            // number says how many builds this session has asked for, and a
            // rebuild is one more of them.
            self.census.declared += 1;
            // Set rather than stepped through [`Self::advance`], and
            // deliberately: [`PipelineState::may_become`] describes the
            // lifetime's own forward order, and this is not a step in it. It
            // is the incarnation underneath the lifetime changing, which is
            // the fact `may_become` cannot express and should not be widened
            // to — a `Ready -> Declared` step admitted there would let an
            // ordinary caller un-build a live pipeline.
            p.state = PipelineState::Declared;
            // Every reflection here described a build performed by the device
            // incarnation that just ended. The shader is the guest's and does
            // not change, but the pipeline these were published against is
            // gone, and a rebuild republishes.
            self.usage.remove(id);
            rebuilding.push(*id);
        }
        rebuilding.sort_unstable();
        rebuilding
    }

    /// Drop retired pipelines' bookkeeping.
    ///
    /// Separate from [`Self::retire`] for the reason retirement and compaction
    /// are separate everywhere else here: one is a lifetime fact and the other
    /// is housekeeping, and doing the second inside the first charges it to the
    /// wrong event.
    pub fn compact(&mut self) {
        self.pipelines
            .retain(|_, p| p.state != PipelineState::Retired);
        // Belt and braces against a retired entry that reached this table by
        // some door other than `advance`: an entry here with no pipeline beside
        // it would be answered for whatever next takes the id.
        self.usage.retain(|id, _| self.pipelines.contains_key(id));
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }
}

#[cfg(test)]
mod resting_tests {
    use super::*;

    fn name(slot: u32) -> ResourceId {
        ResourceId {
            slot: crate::identity::ObjectListRef(slot),
            generation: crate::identity::SlotGeneration(1),
        }
    }

    /// Occupancy and the census answer different questions, and the difference
    /// is the shape of a hang.
    ///
    /// A pipeline declared and never advanced is one event in the census and
    /// stays one forever. A table accumulating those reads identically to a
    /// healthy one there — and every one of them is a transaction that will
    /// lease `Pending` and never be released. `resting` is where that is a
    /// number rather than a subtraction nobody performs.
    #[test]
    fn occupancy_shows_what_the_census_counts_once_and_never_again() {
        let mut table = PipelineTable::new();
        let generation = SessionGeneration::FIRST;

        for slot in 0..3 {
            assert!(table.declare(name(slot), generation));
        }
        assert_eq!(table.census().declared, 3);
        assert_eq!(table.resting().pending(), 3, "nothing has been built yet");

        // One completes its build. The census gains an event; the occupancy
        // *moves* — which is the half a count of events cannot express.
        assert!(table.advance(name(0), PipelineState::Translating));
        assert!(table.advance(name(0), PipelineState::Compiling));
        assert!(table.advance(name(0), PipelineState::Ready));
        let resting = table.resting();
        assert_eq!((resting.pending(), resting.ready), (2, 1));
        assert_eq!(
            table.census().declared,
            3,
            "the census still reports three declarations, because three happened"
        );

        // One is refused: terminal, so it leaves `pending` and never returns.
        assert!(table.refuse(name(1), RefusalReason::TranslationFailed("t")));
        let resting = table.resting();
        assert_eq!((resting.pending(), resting.refused), (1, 1));

        // And the last one is the reading that matters: a pipeline nothing is
        // building, indistinguishable in the census from the two that finished.
        assert_eq!(table.resting().declared, 1);
        assert_eq!(
            table.resting().total(),
            3,
            "the six columns account for every pipeline the table holds"
        );
    }

    /// A retired pipeline stays in the occupancy, because it stays in the
    /// table.
    ///
    /// It is terminal and no transaction waits on it, so it cannot be a hang —
    /// but leaving it out would make the columns stop summing to the table's
    /// size, and a reader who cannot check that the parts add up cannot trust
    /// the part they came for.
    #[test]
    fn a_retired_pipeline_is_still_something_the_table_holds() {
        let mut table = PipelineTable::new();
        assert!(table.declare(name(7), SessionGeneration::FIRST));
        assert!(table.retire(name(7)));
        let resting = table.resting();
        assert_eq!(
            (resting.pending(), resting.retired, resting.total()),
            (0, 1, 1)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ObjectListRef, SlotGeneration};

    fn id(slot: u32) -> ResourceId {
        ResourceId {
            slot: ObjectListRef(slot),
            generation: SlotGeneration::default(),
        }
    }
    const GEN: SessionGeneration = SessionGeneration::FIRST;

    #[test]
    fn the_happy_path_is_the_only_forward_path() {
        let mut t = PipelineTable::new();
        assert!(t.declare(id(1), GEN));
        assert_eq!(t.lease(id(1), GEN), Lease::Pending);
        assert!(t.advance(id(1), PipelineState::Translating));
        assert!(t.advance(id(1), PipelineState::Compiling));
        assert!(t.advance(id(1), PipelineState::Ready));
        assert_eq!(t.lease(id(1), GEN), Lease::Ready);
        assert_eq!(t.census().leases_pending, 1);
        assert_eq!(t.census().leases_ready, 1);
    }

    /// The memo a rail's pre-scan skips on is the same record the answer lands
    /// in, so it cannot outlive the answer.
    ///
    /// This is why the question is the table's and not the rail's. A pre-scan
    /// reaches the same pipeline once per packet and must enumerate its binding
    /// set once per *build*; a set kept beside the rail would go on saying
    /// "covered" after a withdrawal threw the published usage away, and the
    /// rebuilt shader's usage would never be published at all.
    ///
    /// Also the undeclared case, which is what makes the gate safe to try
    /// early: an executor holding a translation before the table holds the
    /// pipeline gets `false` from both doors, so nothing is recorded and the
    /// next pass publishes.
    #[test]
    fn the_published_usage_memo_is_dropped_by_everything_that_drops_the_usage() {
        let mut t = PipelineTable::new();
        let usage = || PublishedUsage::compute(BindingUsage::new(vec![], vec![]));

        assert!(
            !t.usage_published(id(1)),
            "nothing is published for a pipeline this table does not hold"
        );
        assert!(
            !t.publish_usage(id(1), usage()),
            "and publishing into one is refused rather than recorded"
        );
        assert!(!t.usage_published(id(1)));

        assert!(t.declare(id(1), GEN));
        assert!(t.advance(id(1), PipelineState::Translating));
        assert!(t.advance(id(1), PipelineState::Compiling));
        assert!(t.advance(id(1), PipelineState::Ready));
        assert!(t.publish_usage(id(1), usage()));
        assert!(t.usage_published(id(1)), "the pass that lands it");

        assert!(t.withdraw(id(1)));
        assert!(
            !t.usage_published(id(1)),
            "the rebuild's usage is not the previous build's, so the memo must              let the rail publish again"
        );

        assert!(t.advance(id(1), PipelineState::Compiling));
        assert!(t.advance(id(1), PipelineState::Ready));
        assert!(t.publish_usage(id(1), usage()));
        assert!(t.retire(id(1)));
        assert!(
            !t.usage_published(id(1)),
            "a retired pipeline holds no answer, and a rail that thought it did              would never publish for the name that replaces it"
        );
    }

    /// A pipeline the executor stopped holding a translation for goes back to
    /// waiting, and a transaction binding it waits again.
    ///
    /// This is the step a driven macos-15 desktop needs and the table did not
    /// have. The guest rewrites the shader behind a live pipeline ref in place,
    /// so no delete and no new generation arrives and the [`ResourceId`] is
    /// unchanged; the executor's translation is keyed by the shader's content
    /// and it stops holding one. `Ready` was terminal, so the ordering plane
    /// went on releasing transactions against a shader that was mid-translation.
    ///
    /// `Declared` stays unreachable from here: the pipeline is still declared,
    /// and a step back to it would say the guest had not created it.
    #[test]
    fn a_pipeline_whose_translation_the_executor_no_longer_holds_waits_again() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            assert!(t.advance(id(1), step));
        }
        assert_eq!(
            t.waits_for(&[id(1)], GEN),
            Ok(vec![]),
            "nothing to wait for"
        );

        assert!(
            !t.advance(id(1), PipelineState::Translating),
            "not through the door a rail walks on every draw"
        );
        assert!(
            t.withdraw(id(1)),
            "the executor withdraws what it no longer holds"
        );
        assert_eq!(t.census().withdrawn, 1);
        assert_eq!(
            t.waits_for(&[id(1)], GEN),
            Ok(vec![id(1)]),
            "and the next transaction binding it waits"
        );
        assert_eq!(t.resting().translating, 1);
        assert_eq!(t.resting().ready, 0);

        assert!(
            !t.advance(id(1), PipelineState::Declared),
            "it is still declared; there is no step back to saying it is not"
        );

        // And it comes back the same way it came the first time.
        assert!(t.advance(id(1), PipelineState::Compiling));
        assert!(t.advance(id(1), PipelineState::Ready));
        assert_eq!(t.waits_for(&[id(1)], GEN), Ok(vec![]));
        assert_eq!(
            t.census().ready,
            2,
            "the census counts events, so the second readying is its own"
        );
        assert_eq!(t.census().withdrawn, 1, "and one withdrawal, not two");

        // Asked for a pipeline that is not ready, the door moves nothing —
        // which is the common case, because the caller asks about every ref its
        // pre-scan found still building.
        t.declare(id(2), GEN);
        assert!(!t.withdraw(id(2)), "declared is not ready");
        assert!(!t.withdraw(id(999)), "and absent is not ready either");
        assert_eq!(t.census().withdrawn, 1);
    }

    /// The rule: a refusal that carries no reason is not a refusal this table
    /// can hold, so the door that has no reason to write cannot take the step.
    #[test]
    fn a_refusal_only_arrives_through_the_door_that_carries_its_reason() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        assert!(
            !t.advance(id(1), PipelineState::Refused),
            "the step `may_become` calls legal is not one this door can supply"
        );
        assert_eq!(
            t.lease(id(1), GEN),
            Lease::Pending,
            "and nothing moved: a draw binding it still waits"
        );
        assert_eq!(t.census().refused, 0);

        let reason = RefusalReason::CompilationFailed("no");
        assert!(t.refuse(id(1), reason));
        assert_eq!(t.lease(id(1), GEN), Lease::Refused(reason));
        assert_eq!(t.census().refused, 1);
    }

    /// Only the pipelines that are still being built come back as waits.
    ///
    /// A ready pipeline is nothing to wait for, and returning it would park the
    /// transaction on a completion that has already happened — a frame that
    /// never arrives with nothing to explain it.
    #[test]
    fn a_ready_pipeline_is_not_something_to_wait_for() {
        let mut t = PipelineTable::new();
        for slot in [1, 2] {
            t.declare(id(slot), GEN);
        }
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            t.advance(id(1), step);
        }
        assert_eq!(t.waits_for(&[id(1), id(2)], GEN), Ok(vec![id(2)]));
        // Every lease was taken, and the census says which kind each was.
        assert_eq!(t.census().leases_ready, 1);
        assert_eq!(t.census().leases_pending, 1);
        assert_eq!(
            t.waits_for(&[], GEN),
            Ok(Vec::new()),
            "nothing bound, nothing held"
        );
    }

    /// A pipeline that will never build refuses the work that binds it, with
    /// the build's own reason.
    ///
    /// Admitting it with a wait that cannot resolve is a completion word the
    /// guest waits on forever, which is worse than a refusal it is told about.
    /// The slug is the compilation's, because "the draw could not run" is not
    /// the fact anyone reading the failure channel needs.
    #[test]
    fn a_pipeline_that_will_never_build_refuses_the_work_that_binds_it() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.refuse(id(1), RefusalReason::TranslationFailed("no such stage"));
        let refusal = t
            .waits_for(&[id(1)], GEN)
            .expect_err("a refused pipeline is terminal");
        assert_eq!(
            refusal,
            LeaseRefusal::Refused {
                pipeline: id(1),
                reason: RefusalReason::TranslationFailed("no such stage"),
            }
        );
        assert_eq!(refusal.slug(), "pipeline_translation_failed");
    }

    /// Work naming a pipeline this generation does not have is refused, and is
    /// not the same failure as one that could not be built.
    ///
    /// A pipeline declared in a closed generation is intact and unusable, which
    /// is why the generation is asked about rather than read off the object.
    #[test]
    fn a_pipeline_from_another_generation_is_absent_rather_than_refused() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            t.advance(id(1), step);
        }
        assert_eq!(t.waits_for(&[id(1)], GEN), Ok(Vec::new()));
        assert_eq!(
            t.waits_for(&[id(1)], GEN.next()),
            Err(LeaseRefusal::Absent {
                pipeline: id(1),
                because: AbsentBecause::OtherGeneration,
            })
        );
        assert_eq!(
            t.waits_for(&[id(9)], GEN),
            Err(LeaseRefusal::Absent {
                pipeline: id(9),
                because: AbsentBecause::Undeclared,
            }),
            "and one that was never declared at all"
        );
    }

    /// **The first unusable pipeline ends the answer, and nothing is counted.**
    ///
    /// This used to charge the pipelines ahead of the refused one, on the
    /// reading that the walk had already leased them. It had not: a
    /// `waits_for` that refuses refuses the whole list, its caller refuses the
    /// packet, and no transaction ever holds any of them. `leases_pending` is
    /// the number that says whether starting compilation at declaration is
    /// early enough, and a lease charged for work that never ran cannot be part
    /// of that answer.
    #[test]
    fn an_unusable_pipeline_stops_the_walk_and_nothing_is_counted() {
        let mut t = PipelineTable::new();
        for slot in [1, 2, 3] {
            t.declare(id(slot), GEN);
        }
        t.refuse(id(2), RefusalReason::CompilationFailed("out of registers"));
        assert!(matches!(
            t.waits_for(&[id(1), id(2), id(3)], GEN),
            Err(LeaseRefusal::Refused { pipeline, .. }) if pipeline == id(2)
        ));
        assert_eq!(
            t.census().leases_pending,
            0,
            "no transaction held any of them"
        );
        // The same list without the refusal charges every one of them.
        assert_eq!(t.waits_for(&[id(1), id(3)], GEN), Ok(vec![id(1), id(3)]));
        assert_eq!(t.census().leases_pending, 2);
    }

    /// Skipping a step is not a shortcut. A pipeline that reached `Ready`
    /// without compiling is one whose host object nobody built.
    #[test]
    fn a_pipeline_cannot_skip_to_ready() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        assert!(!t.advance(id(1), PipelineState::Ready));
        assert!(!t.advance(id(1), PipelineState::Compiling));
        assert_eq!(t.get(id(1)).unwrap().state, PipelineState::Declared);
    }

    /// The step that actually happens: a compile finishing after the guest
    /// deleted the pipeline. It must not resurrect it.
    #[test]
    fn a_compile_that_lands_after_deletion_does_not_resurrect() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.advance(id(1), PipelineState::Translating);
        t.advance(id(1), PipelineState::Compiling);
        assert!(t.retire(id(1)));
        assert!(
            !t.advance(id(1), PipelineState::Ready),
            "the host finished building an object the guest no longer has"
        );
        assert_eq!(t.lease(id(1), GEN), Lease::Absent(AbsentBecause::Retired));
    }

    /// A refusal is terminal and carries its reason to whoever reads it, so a
    /// guest re-binding the pipeline every frame produces one refusal rather
    /// than one per frame.
    #[test]
    fn a_refusal_is_terminal_and_keeps_its_reason() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        assert!(t.refuse(id(1), RefusalReason::TranslationFailed("no_air")));
        assert_eq!(
            t.lease(id(1), GEN),
            Lease::Refused(RefusalReason::TranslationFailed("no_air"))
        );
        assert_eq!(
            t.lease(id(1), GEN),
            Lease::Refused(RefusalReason::TranslationFailed("no_air"))
        );
        assert!(
            !t.advance(id(1), PipelineState::Translating),
            "a refused pipeline does not retry"
        );
        assert_eq!(t.census().refused, 1);
        assert_eq!(
            t.census().leases_pending,
            0,
            "a refusal is not a pending lease; counting it as one would make \
             the number that argues for earlier compilation include work that \
             will never compile"
        );
    }

    /// A descriptor the model cannot represent is refused before any work
    /// starts, which is the whole reason `Declared -> Refused` is legal.
    #[test]
    fn a_pipeline_can_be_refused_before_anything_starts() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        assert!(t.refuse(id(1), RefusalReason::Undescribable("tessellation")));
        assert_eq!(t.get(id(1)).unwrap().state, PipelineState::Refused);
    }

    /// A pipeline from a closed generation is absent to later work even though
    /// the object is intact — the host handle may be perfectly healthy, and
    /// that is not the question.
    #[test]
    fn a_pipeline_from_a_closed_generation_is_absent_to_later_work() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.advance(id(1), PipelineState::Translating);
        t.advance(id(1), PipelineState::Compiling);
        t.advance(id(1), PipelineState::Ready);
        assert_eq!(t.lease(id(1), GEN), Lease::Ready);
        assert_eq!(
            t.lease(id(1), GEN.next()),
            Lease::Absent(AbsentBecause::OtherGeneration)
        );
    }

    /// A pipeline the guest deleted is not a wait and not a refusal.
    ///
    /// **The three ways to be absent are not one answer.** A guest delete ends
    /// the *name*, and a command buffer recorded before it still binds the
    /// object — the host encoder retained it when the record was written — so a
    /// transaction carrying that record is entitled to run, and the records
    /// beside it have nothing to do with the deleted pipeline. A driven macos-26
    /// boot refused 9374 exec packets on this before the distinction existed.
    ///
    /// The other two stay refusals and this asserts that too: `Undeclared` means
    /// the lease list disagrees with what was declared, and `OtherGeneration` is
    /// work that outlived a reset whose host objects are already gone.
    #[test]
    fn a_retired_pipeline_is_neither_a_wait_nor_a_refusal() {
        let mut t = PipelineTable::new();
        assert!(t.declare(id(1), GEN));
        t.advance(id(1), PipelineState::Translating);
        t.advance(id(1), PipelineState::Compiling);
        t.advance(id(1), PipelineState::Ready);
        assert!(t.retire(id(1)));

        assert_eq!(
            t.waits_for(&[id(1)], GEN),
            Ok(Vec::new()),
            "the guest ended the name; there is nothing left to wait for and \
             nothing here that makes the packet unrunnable"
        );

        assert_eq!(
            t.waits_for(&[id(9)], GEN),
            Err(LeaseRefusal::Absent {
                pipeline: id(9),
                because: AbsentBecause::Undeclared,
            }),
            "a lease nothing declared is the caller's two lists disagreeing"
        );
        assert_eq!(
            t.waits_for(&[id(1)], GEN.next()),
            Err(LeaseRefusal::Absent {
                pipeline: id(1),
                because: AbsentBecause::OtherGeneration,
            }),
            "and work that outlived a reset names host objects already handed away"
        );
    }

    /// Two live declarations of one id would mean the object namespace failed
    /// to produce a new generation for a reused slot. Replacing the first
    /// silently would hide that.
    #[test]
    fn redeclaring_a_live_id_is_refused_rather_than_applied() {
        let mut t = PipelineTable::new();
        assert!(t.declare(id(1), GEN));
        t.advance(id(1), PipelineState::Translating);
        assert!(!t.declare(id(1), GEN));
        assert_eq!(t.get(id(1)).unwrap().state, PipelineState::Translating);
        // A reused slot is a different id, and declares cleanly.
        let reused = ResourceId {
            slot: ObjectListRef(1),
            generation: SlotGeneration::default().next(),
        };
        assert!(t.declare(reused, GEN));
    }

    /// A host build belongs to the incarnation that performed it.
    #[test]
    fn a_device_loss_takes_the_builds_and_leaves_the_objects() {
        let gen = SessionGeneration::FIRST;
        let mut t = PipelineTable::new();
        // One at each state the loss can find a pipeline in.
        let declared = id(1);
        let translating = id(2);
        let compiling = id(3);
        let ready = id(4);
        let refused = id(5);
        let retired = id(6);
        for p in [declared, translating, compiling, ready, refused, retired] {
            assert!(t.declare(p, gen));
        }
        for (p, steps) in [(translating, 1), (compiling, 2), (ready, 3)] {
            for step in [
                PipelineState::Translating,
                PipelineState::Compiling,
                PipelineState::Ready,
            ]
            .into_iter()
            .take(steps)
            {
                assert!(t.advance(p, step));
            }
        }
        assert!(t.refuse(refused, RefusalReason::CompilationFailed("no")));
        assert!(t.retire(retired));
        let before = t.census();

        assert_eq!(
            t.device_lost(),
            vec![declared, translating, compiling, ready],
            "every build in flight or finished starts again, and the two \
             terminal states do not"
        );
        for p in [declared, translating, compiling, ready] {
            assert_eq!(
                t.get(p).expect("still an object").state,
                PipelineState::Declared
            );
            assert_eq!(t.lease(p, gen), Lease::Pending);
        }
        assert_eq!(
            t.lease(refused, gen),
            Lease::Refused(RefusalReason::CompilationFailed("no"))
        );
        assert_eq!(t.lease(retired, gen), Lease::Absent(AbsentBecause::Retired));
        assert_eq!(
            t.census().declared,
            before.declared + 4,
            "a rebuild is a build this session asked for"
        );

        // And the lifetime runs forward from there exactly as it did the first
        // time --- a set state is not a state the machine cannot leave.
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            assert!(t.advance(ready, step));
        }
        assert_eq!(t.lease(ready, gen), Lease::Ready);
        assert!(
            t.device_lost().contains(&ready),
            "and a second loss takes the second build"
        );
    }

    /// The incarnation change is not a step in the lifetime, and the lifetime's
    /// own order does not admit it.
    #[test]
    fn nothing_can_step_a_built_pipeline_back_to_declared() {
        assert!(!PipelineState::Ready.may_become(PipelineState::Declared));
        let gen = SessionGeneration::FIRST;
        let mut t = PipelineTable::new();
        let p = id(1);
        t.declare(p, gen);
        for step in [
            PipelineState::Translating,
            PipelineState::Compiling,
            PipelineState::Ready,
        ] {
            assert!(t.advance(p, step));
        }
        assert!(
            !t.advance(p, PipelineState::Declared),
            "un-building a live pipeline is not something a caller may ask for"
        );
        assert_eq!(t.lease(p, gen), Lease::Ready);
    }

    #[test]
    fn compaction_drops_only_retired_pipelines() {
        let mut t = PipelineTable::new();
        t.declare(id(1), GEN);
        t.declare(id(2), GEN);
        t.retire(id(1));
        t.compact();
        assert_eq!(t.len(), 1);
        assert!(t.get(id(1)).is_none());
        assert!(t.get(id(2)).is_some());
    }

    #[test]
    fn an_absent_pipeline_answers_absent_rather_than_pending() {
        let mut t = PipelineTable::new();
        assert_eq!(
            t.lease(id(9), GEN),
            Lease::Absent(AbsentBecause::Undeclared)
        );
        assert!(!t.advance(id(9), PipelineState::Translating));
        assert!(!t.refuse(id(9), RefusalReason::CompilationFailed("x")));
    }
}
