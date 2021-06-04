//! light _Merkle Tree_ implementation.
//!
//! Merkle tree (MT) implemented as a full (power of 2) arity tree allocated as a vec
//! of statically sized hashes to give hashes more locality (although disk based backings
//! are supported, as a partial tree disk based backings).  MT is specialized
//! to the extent of arity, hashing algorithm and hash item. [`Hashable`] trait is
//! compatible to the `std::hash::Hasher` and supports custom hash algorithms.
//! Implementation does not depend on any external crypto libraries, and tries
//! to be as performant as possible (CPU support only; GPU hashing currently unsupported).
//!
//! This tree implementation uses encoding scheme as in _Certificate Transparency_
//! by default. Encoding scheme for leafs and nodes can be overridden though.
//! [RFC 6962](https://tools.ietf.org/html/rfc6962):
//!
//! ```text
//! MTH({d(0)}) = ALG(0x00 || d(0)).
//! For n > 1, let k be the largest power of two smaller than n (i.e.,
//! k < n <= 2k).  The Merkle tree Hash of an n-element list D[n] is then
//! defined recursively as
//! MTH(D[n]) = ALG(0x01 || MTH(D[0:k]) || MTH(D[k:n])),
//! ```
//!
//! Link: [](https://en.wikipedia.org/wiki/Merkle_tree)
//!
//! # Implementation choices
//!
//! Main idea is the whole code must obtain specialization at compile time with
//! minimum allocations calls, hashes must be of fixed size arrays known at
//! compile time, hash algorithm must be a trait and must not depend on any
//! external cryptographic libraries and the lib itself must somehow mimic std Rust api.
//!
//! Standard way in Rust is to hash objects with a `std::hash::Hasher`, and mainly
//! that is the reason behind the choice of the abstractions:
//!
//! `Object : Hashable<H> -> Hasher + Algorithm <- Merkle Tree`
//!
//! Custom [`merkle::hash::Hashable`] trait allows implementations differ
//! from [`std::collection`] related hashes, different implementations for
//! different hashing algorithms / schemas and conforms object-safety trait rules.
//!
//! [`Algorithm`] complements [`Hasher`] to be reusable and follows the idea
//! that the result hash is a mapping of the data stream.
//!
//! [`Algorithm.hash`] had to change its signature to be `&mut self` (`&self`) because
//! most of the cryptographic digest algorithms breaks current state on finalization
//! into unusable. `ring` libra tho contains interfaces incompatible to
//! `start-update-finish-reset` lifecycle. It requires either `cloning()` its state
//! on finalization, or `Cell`-ing via unsafe.
//!
//! Turning back to having [`Algorithm.write(&mut self, &[u8])`] instead of
//! `write(T)` allows to relax [`Algorithm`] trait [`Hasher`] constraint, even tho
//! works together well still.
//!
//! # Interface
//!
//! ```text
//! - build_tree (items) -> tree
//! - get_root -> hash
//! - gen_proof -> proof
//! - validate_proof (proof, leaf, root) -> bool
//! ```
//!
//! # Examples
//!
//! [`test_common.rs`]: custom hash example xor128, misc shared utils
//! [`test_xor128.rs`]: most comprehensive tests for library features
//! [`proof.rs`]: contains impl and tests for proofs across pow2 arity trees
//!

// missing_docs,
#![deny(
    unused_qualifications,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces
)]
#![cfg_attr(feature = "nightly", allow(unstable_features))]

#[macro_use]
extern crate anyhow;

/// Hash infrastructure for items in Merkle tree.
pub mod hash;

/// Common implementations for [`Hashable`].
mod hash_impl;

/// Store implementations.
pub mod store;

/// Merkle tree inclusion proof.
pub mod proof;

/// Merkle tree abstractions, implementation and algorithms.
pub mod merkle;

/// Re-usable Testing primitives
pub mod test_common;

/// Tests XOR128.
#[cfg(test)]
mod test_xor128;

/// Tests data.
#[cfg(test)]
mod test_item;

/// Tests SIP.
#[cfg(test)]
mod test_sip;

/// Tests for Merkle Hasher Customization.
#[cfg(test)]
mod test_cmh;

#[macro_use]
extern crate arrayref;
