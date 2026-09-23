//! Reims vGPU protocol: the first layer allowed to say what a wire tag *means*.
//!
//! # Why this is a crate
//!
//! `reims-vgpu-wire` answers one question and refuses the next one. It says
//! which bytes a serializer record is made of, and it is deliberately unable to
//! say what the device owes the guest in return — a view that knew that would be
//! a view with a policy in it. Everything above wire has historically answered
//! the second question wherever it happened to be standing: a decode arm, an
//! executor match, a census route name, a comment. That is why the same
//! operation could be a no-op in one rail's reading and a dropped command in
//! another's, with nothing able to compare the two.
//!
//! This crate is the one place the second question is answered, and it depends
//! only on wire so the compiler can keep it that way: no device state, no
//! backend, no host OS, no allocation policy. `#![no_std]`.
//!
//! # The parts
//!
//! - [`bind`] — the argument-table sizes a plural bind is truncated at, and
//!   why they are capacity hints rather than bounds.
//! - [`closure`] — the refusal-closure ledger. For every decodable operation,
//!   exactly one outcome: implemented, contract-proven no-op on a named
//!   capability cell, contract-proven unsupported with its exact refusal, or
//!   unresolved and therefore blocking. "The current workload does not use it"
//!   and "the old backend drops it" are not outcomes and cannot be spelled here.
//! - [`packets`] — the same ledger for the FIFO packet classes, which are the
//!   other half of what a guest sends and which the manifest cannot enumerate.
//! - [`blit`] — which transfer a blit opcode names, which is its shape rather
//!   than its closure.
//! - [`compute`] — which compute-encoder record an opcode names, and the pass
//!   dispatch type that only reaches the wire through the descriptor.
//! - [`decode`] — records lifted out of bytes, with the guest's own refs still
//!   on them.
//! - [`extent`] — the guest API's three-dimensional extent, its mip-level
//!   dimensions, and the byte arithmetic of a tightly-packed image.
//! - [`fifo`] — the FIFO packet payload layouts: the resource-list, invalidate,
//!   synchronize and replace-physical records, the `EXEC_INDIRECT2` header and
//!   its per-resource table, and the display descriptor's timing entries.
//! - [`sync`] — which opcode is a fence, an event or a barrier, on which rail,
//!   and what a barrier's scope word names.
//! - [`segment`] — what a segment-type byte means: which encoder wrote it and
//!   which rail its records are read on, and how a command stream divides into
//!   the segments carrying them.
//! - [`resource_state`] — what the content-representation records ask for:
//!   which directive, at which granularity.
//! - [`info_reply`] — the key/value table an info query is answered with, and
//!   the three separate bounds on how much of it may be written.
//! - [`storage_mode`] — which storage mode a resource declares, and the one
//!   thing this wire's use of it does not license.
//! - [`present`] — which of the three present commands a packet is, and where
//!   its trailer keeps the target it names.
//! - [`render`] — which render-encoder record an opcode names, the eight draw
//!   shapes behind its fourteen draw opcodes, and the stage no wire field
//!   carries.
//! - [`pass_action`] — the load and store ordinals a render-pass attachment
//!   carries, and the closed sets behind them.
//! - [`residency`] — what a `useResource`/`useHeap` declaration says, split so
//!   that the half a per-draw binder owes nothing on cannot hide the half it
//!   does.
//! - [`pixel_format`] — the format ordinals, what each one is made of, and the
//!   conversions between a texel and the eight-bit colour the device carries.
//! - [`blend`] — a colour attachment's blend equations and write mask, and
//!   which of them a cleared `blendingEnabled` puts out of reach.
//! - [`depth_stencil`] — `MTLDepthStencilDescriptor`'s ordinals, and which
//!   of its two faces the record actually wrote.
//! - [`sampler`] — `MTLSamplerDescriptor`'s ordinals, and the combinations
//!   the guest API itself does not admit.
//! - [`vertex_format`] — `MTLVertexFormat` and the geometry every rail
//!   derives from one.
//! - [`topology`] — `MTLPrimitiveType` and the three classes its five
//!   values fall into.
//! - [`texture_shape`] — what a texture declaration is: its type ordinal, the
//!   dimensions that type uses, and the field pairs the guest API does not
//!   admit.
//! - [`mipmap`] — what a mipmap-generation request must satisfy before a
//!   backend sees it, which is arithmetic over a format code and a size.
//! - [`iosurface_pages`] — the mapper's descriptors and the page span a shared
//!   surface occupies.
//! - [`gva`] and [`gva_resolve`] — guest virtual addresses, and the guest
//!   page-table walk's statuses on the failure channel.
//! - [`draw`], [`dispatch`], [`vertex_step`], [`visibility`] — the closed
//!   ordinal rules: which primitive types are executable, what a dispatch
//!   record's grid resolves to, how a vertex step and its rate read together,
//!   and what a visibility mode names.
//! - [`checked`], [`endian`], [`fnv`] — the arithmetic underneath all of it.
//!
//! # Backend-neutral, and provably so
//!
//! What belongs *above* this crate is everything that knows how a host draws —
//! Vulkan handles, SPIR-V, memory placement, descriptors, queue families, image
//! layouts, host capability policy — and everything that knows the device is
//! attached to QEMU. That used to be a rule held by habit while the vocabulary
//! sat in a second crate beside this one. It is a fact here: `ash`, Metal, QEMU
//! and the device's own state are not in scope, so a check cannot reach one by
//! accident and a reviewer does not have to notice that it did.
//!
//! The one dependency that looks like a device dependency and is not is
//! `reims_vgpu_observe`, taken **without** its `std` feature: a check that
//! refuses has to be able to *name* its refusal, and the
//! [`Decline`](reims_vgpu_observe::Decline) vocabulary is that name. It carries
//! no policy, selects nothing, and does not bring the sink that logs it.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_op_in_unsafe_fn)]

// `extent::tight_pyramid_spans` returns one span per mip level, a count the
// caller does not know in advance. `alloc` is still `no_std`; what this crate
// must not reach is the host, not the heap.
extern crate alloc;

pub mod bind;
pub mod blend;
pub mod blit;
pub mod checked;
pub mod closure;
pub mod compute;
pub mod decode;
pub mod depth_stencil;
pub mod destroy;
pub mod dispatch;
pub mod draw;
pub mod endian;
pub mod extent;
pub mod fifo;
pub mod fnv;
pub mod gva;
pub mod gva_resolve;
pub mod info_reply;
pub mod iosurface_pages;
pub mod mipmap;
pub mod packets;
pub mod pass_action;
pub mod pixel_format;
pub mod present;
pub mod render;
pub mod residency;
pub mod resource_state;
pub mod sampler;
pub mod segment;
pub mod serializer_object;
pub mod storage_mode;
pub mod sync;
pub mod texture_shape;
pub mod topology;
pub mod vertex_format;
pub mod vertex_step;
pub mod visibility;
