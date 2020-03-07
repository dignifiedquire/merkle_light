use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use log::debug;
use rayon::prelude::*;
use typenum::marker_traits::Unsigned;
use typenum::U2;

use crate::hash::{Algorithm, Hashable};
use crate::proof::Proof;
use crate::store::{Store, StoreConfig, VecStore, BUILD_CHUNK_NODES};

// Number of batched nodes processed and stored together when
// populating from the data leaves.
pub const BUILD_DATA_BLOCK_SIZE: usize = 64 * BUILD_CHUNK_NODES;

// FIXME: Hand-picked constants, some proper benchmarks should be done
// to choose more appropriate values and document the decision.

/// Merkle Tree.
///
/// All leafs and nodes are stored in a linear array (vec).
///
/// A merkle tree is a tree in which every non-leaf node is the hash of its
/// child nodes. A diagram depicting how it works:
///
/// ```text
///         root = h1234 = h(h12 + h34)
///        /                           \
///  h12 = h(h1 + h2)            h34 = h(h3 + h4)
///   /            \              /            \
/// h1 = h(tx1)  h2 = h(tx2)    h3 = h(tx3)  h4 = h(tx4)
/// ```
///
/// In memory layout:
///
/// ```text
///     [h1 h2 h3 h4 h12 h34 root]
/// ```
///
/// Merkle root is always the last element in the array.
///
/// The number of inputs is not always a power of two which results in a
/// balanced tree structure as above.  In that case, parent nodes with no
/// children are also zero and parent nodes with only a single left node
/// are calculated by concatenating the left node with itself before hashing.
/// Since this function uses nodes that are pointers to the hashes, empty nodes
/// will be nil.
///
/// TODO: Ord
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct MerkleTree<T, A, K, U = U2>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
    U: Unsigned,
{
    data: K,
    leafs: usize,
    height: usize,

    // Cache with the `root` of the tree built from `data`. This allows to
    // not access the `Store` (e.g., access to disks in `DiskStore`).
    root: T,

    _u: PhantomData<U>,
    _a: PhantomData<A>,
    _t: PhantomData<T>,
}

/// Element stored in the merkle tree.
pub trait Element: Ord + Clone + AsRef<[u8]> + Sync + Send + Default + std::fmt::Debug {
    /// Returns the length of an element when serialized as a byte slice.
    fn byte_len() -> usize;

    /// Creates the element from its byte form. Panics if the slice is not appropriately sized.
    fn from_slice(bytes: &[u8]) -> Self;

    fn copy_to_slice(&self, bytes: &mut [u8]);
}

impl<T: Element, A: Algorithm<T>, K: Store<T>, U: Unsigned> MerkleTree<T, A, K, U> {
    /// Creates new merkle from a sequence of hashes.
    pub fn new<I: IntoIterator<Item = T>>(data: I) -> Result<MerkleTree<T, A, K, U>> {
        Self::try_from_iter(data.into_iter().map(Ok))
    }

    /// Creates new merkle from a sequence of hashes.
    pub fn new_with_config<I: IntoIterator<Item = T>>(
        data: I,
        config: StoreConfig,
    ) -> Result<MerkleTree<T, A, K, U>> {
        Self::try_from_iter_with_config(data.into_iter().map(Ok), config)
    }

    /// Creates new merkle tree from a list of hashable objects.
    pub fn from_data<O: Hashable<A>, I: IntoIterator<Item = O>>(
        data: I,
    ) -> Result<MerkleTree<T, A, K, U>> {
        let mut a = A::default();
        Self::try_from_iter(data.into_iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            Ok(a.hash())
        }))
    }

    /// Creates new merkle tree from a list of hashable objects.
    pub fn from_data_with_config<O: Hashable<A>, I: IntoIterator<Item = O>>(
        data: I,
        config: StoreConfig,
    ) -> Result<MerkleTree<T, A, K, U>> {
        let mut a = A::default();
        Self::try_from_iter_with_config(
            data.into_iter().map(|x| {
                a.reset();
                x.hash(&mut a);
                Ok(a.hash())
            }),
            config,
        )
    }

    /// Creates new merkle tree from an already allocated 'Store'
    /// (used with 'Store::new_from_disk').  The specified 'size' is
    /// the number of base data leafs in the MT.
    pub fn from_data_store(data: K, leafs: usize) -> Result<MerkleTree<T, A, K, U>> {
        let branches = U::to_usize();
        ensure!(next_pow2(leafs) == leafs, "leafs MUST be a power of 2");
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let tree_len = get_merkle_tree_len(leafs, branches);
        ensure!(tree_len == data.len(), "Inconsistent tree data");

        ensure!(
            is_merkle_tree_size_valid(leafs, branches),
            "MerkleTree size is invalid given the arity"
        );

        let height = get_merkle_tree_height(leafs, branches);
        let root = data.read_at(data.len() - 1)?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    /// Represent a fully constructed merkle tree from a provided slice.
    pub fn from_tree_slice(data: &[u8], leafs: usize) -> Result<MerkleTree<T, A, K, U>> {
        let branches = U::to_usize();
        let height = get_merkle_tree_height(leafs, branches);
        let tree_len = get_merkle_tree_len(leafs, branches);
        ensure!(
            tree_len == data.len() / T::byte_len(),
            "Inconsistent tree data"
        );

        ensure!(
            is_merkle_tree_size_valid(leafs, branches),
            "MerkleTree size is invalid given the arity"
        );

        let store = K::new_from_slice(tree_len, &data).context("failed to create data store")?;
        let root = store.read_at(data.len() - 1)?;

        Ok(MerkleTree {
            data: store,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    /// Represent a fully constructed merkle tree from a provided slice.
    pub fn from_tree_slice_with_config(
        data: &[u8],
        leafs: usize,
        config: StoreConfig,
    ) -> Result<MerkleTree<T, A, K, U>> {
        let branches = U::to_usize();
        let height = get_merkle_tree_height(leafs, branches);
        let tree_len = get_merkle_tree_len(leafs, branches);
        ensure!(
            tree_len == data.len() / T::byte_len(),
            "Inconsistent tree data"
        );

        ensure!(
            is_merkle_tree_size_valid(leafs, branches),
            "MerkleTree size is invalid given the arity"
        );

        let store = K::new_from_slice_with_config(tree_len, branches, &data, config)
            .context("failed to create data store")?;
        let root = store.read_at(data.len() - 1)?;

        Ok(MerkleTree {
            data: store,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    #[inline]
    fn build_partial_tree(
        mut data: VecStore<T>,
        leafs: usize,
        height: usize,
    ) -> Result<MerkleTree<T, A, VecStore<T>, U>> {
        let root = VecStore::build::<A, U>(&mut data, leafs, height, None)?;
        let branches = U::to_usize();

        let tree_len = get_merkle_tree_len(leafs, branches);
        ensure!(tree_len == Store::len(&data), "Inconsistent tree data");

        ensure!(
            is_merkle_tree_size_valid(leafs, branches),
            "MerkleTree size is invalid given the arity"
        );

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    /// Generate merkle tree inclusion proof for leaf `i`
    #[inline]
    pub fn gen_proof(&self, i: usize) -> Result<Proof<T, U>> {
        ensure!(
            i < self.leafs,
            "{} is out of bounds (max: {})",
            i,
            self.leafs
        ); // i in [0 .. self.leafs)

        let mut base = 0;
        let mut j = i;

        // level 1 width
        let mut width = self.leafs;
        let branches = U::to_usize();
        ensure!(width == next_pow2(width), "Must be a power of 2 tree");
        ensure!(
            branches == next_pow2(branches),
            "branches must be a power of 2"
        );
        let shift = log2_pow2(branches);

        let mut lemma: Vec<T> =
            Vec::with_capacity(get_merkle_proof_lemma_len(self.height, branches));
        let mut path: Vec<usize> = Vec::with_capacity(self.height - 1); // path - 1

        // item is first
        lemma.push(self.read_at(j)?);
        while base + 1 < self.len() {
            let hash_index = (j / branches) * branches;
            for k in hash_index..hash_index + branches {
                if k != j {
                    lemma.push(self.read_at(base + k)?)
                }
            }

            path.push(j % branches); // path_index

            base += width;
            width >>= shift; // width /= branches;
            j >>= shift; // j /= branches;
        }

        // root is final
        lemma.push(self.root());

        // Sanity check: if the `MerkleTree` lost its integrity and `data` doesn't match the
        // expected values for `leafs` and `height` this can get ugly.
        ensure!(
            lemma.len() == get_merkle_proof_lemma_len(self.height, branches),
            "Invalid proof lemma length"
        );
        ensure!(path.len() == self.height - 1, "Invalid proof path length");

        Proof::new(lemma, path)
    }

    /// Generate merkle tree inclusion proof for leaf `i` by first
    /// building a partial tree (returned) along with the proof.
    /// Return value is a Result tuple of the proof and the partial
    /// tree that was constructed.
    #[allow(clippy::type_complexity)]
    pub fn gen_proof_and_partial_tree(
        &self,
        i: usize,
        levels: usize,
    ) -> Result<(Proof<T, U>, MerkleTree<T, A, VecStore<T>, U>)> {
        ensure!(
            i < self.leafs,
            "{} is out of bounds (max: {})",
            i,
            self.leafs
        ); // i in [0 .. self.leafs]

        // For partial tree building, the data layer width must be a
        // power of 2.
        ensure!(
            self.leafs == next_pow2(self.leafs),
            "The size of the data layer must be a power of 2"
        );

        let branches = U::to_usize();
        let total_size = get_merkle_tree_len(self.leafs, branches);
        let cache_size = get_merkle_tree_cache_size(self.leafs, branches, levels);
        ensure!(
            cache_size < total_size,
            "Generate a partial proof with all data available?"
        );

        let cached_leafs = get_merkle_tree_leafs(cache_size, branches);
        ensure!(
            cached_leafs == next_pow2(cached_leafs),
            "The size of the cached leafs must be a power of 2"
        );

        let cache_height = get_merkle_tree_height(cached_leafs, branches);
        let partial_height = self.height - cache_height + 1;

        // Calculate the subset of the base layer data width that we
        // need in order to build the partial tree required to build
        // the proof (termed 'segment_width'), given the data
        // configuration specified by 'levels'.
        let segment_width = self.leafs / cached_leafs;
        let segment_start = (i / segment_width) * segment_width;
        let segment_end = segment_start + segment_width;

        debug!("leafs {}, branches {}, total size {}, total height {}, cache_size {}, cached levels above base {}, \
                partial_height {}, cached_leafs {}, segment_width {}, segment range {}-{} for {}",
               self.leafs, branches, total_size, self.height, cache_size, levels, partial_height,
               cached_leafs, segment_width, segment_start, segment_end, i);

        // Copy the proper segment of the base data into memory and
        // initialize a VecStore to back a new, smaller MT.
        let mut data_copy = vec![0; segment_width * T::byte_len()];
        self.data
            .read_range_into(segment_start, segment_end, &mut data_copy)?;
        let partial_store = VecStore::new_from_slice(segment_width, &data_copy)?;
        ensure!(
            Store::len(&partial_store) == segment_width,
            "Inconsistent store length"
        );

        // Before building the tree, resize the store where the tree
        // will be built to allow space for the newly constructed layers.
        data_copy.resize(
            get_merkle_tree_len(segment_width, branches) * T::byte_len(),
            0,
        );

        // Build the optimally small tree.
        let partial_tree: MerkleTree<T, A, VecStore<T>, U> =
            Self::build_partial_tree(partial_store, segment_width, partial_height)?;
        ensure!(
            partial_height == partial_tree.height(),
            "Inconsistent partial tree height"
        );

        // Generate entire proof with access to the base data, the
        // cached data, and the partial tree.
        let proof = self.gen_proof_with_partial_tree(i, levels, &partial_tree)?;

        debug!(
            "generated partial_tree of height {} and len {} with {} branches for proof at {}",
            partial_tree.height,
            partial_tree.len(),
            branches,
            i
        );

        Ok((proof, partial_tree))
    }

    /// Generate merkle tree inclusion proof for leaf `i` given a
    /// partial tree for lookups where data is otherwise unavailable.
    pub fn gen_proof_with_partial_tree(
        &self,
        i: usize,
        levels: usize,
        partial_tree: &MerkleTree<T, A, VecStore<T>, U>,
    ) -> Result<Proof<T, U>> {
        ensure!(
            i < self.leafs,
            "{} is out of bounds (max: {})",
            i,
            self.leafs
        ); // i in [0 .. self.leafs)

        // For partial tree building, the data layer width must be a
        // power of 2.
        let mut width = self.leafs;
        let branches = U::to_usize();
        ensure!(width == next_pow2(width), "Must be a power of 2 tree");
        ensure!(
            branches == next_pow2(branches),
            "branches must be a power of 2"
        );

        let data_width = width;
        let total_size = get_merkle_tree_len(data_width, branches);
        let cache_size = get_merkle_tree_cache_size(self.leafs, branches, levels);
        let cache_index_start = total_size - cache_size;
        let cached_leafs = get_merkle_tree_leafs(cache_size, branches);
        ensure!(
            cached_leafs == next_pow2(cached_leafs),
            "Cached leafs size must be a power of 2"
        );

        // Calculate the subset of the data layer width that we need
        // in order to build the partial tree required to build the
        // proof (termed 'segment_width').
        let mut segment_width = width / cached_leafs;
        let segment_start = (i / segment_width) * segment_width;

        // shift is the amount that we need to decrease the width by
        // the number of branches at each level up the main merkle
        // tree.
        let shift = log2_pow2(branches);

        // segment_shift is the amount that we need to offset the
        // partial tree offsets to keep them within the space of the
        // partial tree as we move up it.
        //
        // segment_shift is conceptually (segment_start >>
        // (current_height * shift)), which tracks an offset in the
        // main merkle tree that we apply to the partial tree.
        let mut segment_shift = segment_start;

        // 'j' is used to track the challenged nodes required for the
        // proof up the tree.
        let mut j = i;

        // 'base' is used to track the data index of the layer that
        // we're currently processing in the main merkle tree that's
        // represented by the store.
        let mut base = 0;

        // 'partial_base' is used to track the data index of the layer
        // that we're currently processing in the partial tree.
        let mut partial_base = 0;

        let mut lemma: Vec<T> =
            Vec::with_capacity(get_merkle_proof_lemma_len(self.height, branches));
        let mut path: Vec<usize> = Vec::with_capacity(self.height - 1); // path - 1

        lemma.push(self.read_at(j)?);
        while base + 1 < self.len() {
            let hash_index = (j / branches) * branches;
            for k in hash_index..hash_index + branches {
                if k != j {
                    let read_index = base + k;
                    lemma.push(
                        if read_index < data_width || read_index >= cache_index_start {
                            self.read_at(base + k)?
                        } else {
                            let read_index = partial_base + k - segment_shift;
                            partial_tree.read_at(read_index)?
                        },
                    );
                }
            }

            path.push(j % branches); // path_index

            base += width;
            width >>= shift; // width /= branches

            partial_base += segment_width;
            segment_width >>= shift; // segment_width /= branches

            segment_shift >>= shift; // segment_shift /= branches

            j >>= shift; // j /= branches;
        }

        // root is final
        lemma.push(self.root());

        // Sanity check: if the `MerkleTree` lost its integrity and `data` doesn't match the
        // expected values for `leafs` and `height` this can get ugly.
        ensure!(
            lemma.len() == get_merkle_proof_lemma_len(self.height, branches),
            "Invalid proof lemma length"
        );
        ensure!(path.len() == self.height - 1, "Invalid proof path length");

        Proof::new(lemma, path)
    }

    /// Returns merkle root
    #[inline]
    pub fn root(&self) -> T {
        self.root.clone()
    }

    /// Returns number of elements in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Truncates the data for later access via LevelCacheStore
    /// interface.
    #[inline]
    pub fn compact(&mut self, config: StoreConfig, store_version: u32) -> Result<bool> {
        let branches = U::to_usize();
        self.data.compact(branches, config, store_version)
    }

    #[inline]
    pub fn reinit(&mut self) -> Result<()> {
        self.data.reinit()
    }

    /// Removes the backing store for this merkle tree.
    #[inline]
    pub fn delete(&self, config: StoreConfig) -> Result<()> {
        K::delete(config)
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns height of the tree
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Returns original number of elements the tree was built upon.
    #[inline]
    pub fn leafs(&self) -> usize {
        self.leafs
    }

    /// Returns data reference
    #[inline]
    pub fn data(&self) -> &K {
        &self.data
    }

    /// Returns mutable data reference
    #[inline]
    pub fn data_mut(&mut self) -> &mut K {
        &mut self.data
    }

    /// Returns merkle leaf at index i
    #[inline]
    pub fn read_at(&self, i: usize) -> Result<T> {
        self.data.read_at(i)
    }

    pub fn read_range(&self, start: usize, end: usize) -> Result<Vec<T>> {
        ensure!(start < end, "start must be less than end");
        self.data.read_range(start..end)
    }

    pub fn read_range_into(&self, start: usize, end: usize, buf: &mut [u8]) -> Result<()> {
        ensure!(start < end, "start must be less than end");
        self.data.read_range_into(start, end, buf)
    }

    /// Reads into a pre-allocated slice (for optimization purposes).
    pub fn read_into(&self, pos: usize, buf: &mut [u8]) -> Result<()> {
        self.data.read_into(pos, buf)
    }

    /// Build the tree given a slice of all leafs, in bytes form.
    pub fn from_byte_slice_with_config(leafs: &[u8], config: StoreConfig) -> Result<Self> {
        ensure!(
            leafs.len() % T::byte_len() == 0,
            "{} ist not a multiple of {}",
            leafs.len(),
            T::byte_len()
        );

        let leafs_count = leafs.len() / T::byte_len();
        let branches = U::to_usize();
        ensure!(leafs_count > 1, "not enough leaves");
        ensure!(
            next_pow2(leafs_count) == leafs_count,
            "size MUST be a power of 2"
        );
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let size = get_merkle_tree_len(leafs_count, branches);
        let height = get_merkle_tree_height(leafs_count, branches);

        let mut data = K::new_from_slice_with_config(size, branches, leafs, config.clone())
            .context("failed to create data store")?;
        let root = K::build::<A, U>(&mut data, leafs_count, height, Some(config))?;

        Ok(MerkleTree {
            data,
            leafs: leafs_count,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    /// Build the tree given a slice of all leafs, in bytes form.
    pub fn from_byte_slice(leafs: &[u8]) -> Result<Self> {
        ensure!(
            leafs.len() % T::byte_len() == 0,
            "{} is not a multiple of {}",
            leafs.len(),
            T::byte_len()
        );

        let leafs_count = leafs.len() / T::byte_len();
        let branches = U::to_usize();
        ensure!(leafs_count > 1, "not enough leaves");
        ensure!(
            next_pow2(leafs_count) == leafs_count,
            "size MUST be a power of 2"
        );
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let size = get_merkle_tree_len(leafs_count, branches);
        let height = get_merkle_tree_height(leafs_count, branches);

        let mut data = K::new_from_slice(size, leafs).context("failed to create data store")?;

        let root = K::build::<A, U>(&mut data, leafs_count, height, None)?;

        Ok(MerkleTree {
            data,
            leafs: leafs_count,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }
}

pub trait FromIndexedParallelIterator<T, U>: Sized
where
    T: Send,
{
    fn from_par_iter<I>(par_iter: I) -> Result<Self>
    where
        U: Unsigned,
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator;

    fn from_par_iter_with_config<I>(par_iter: I, config: StoreConfig) -> Result<Self>
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator,
        U: Unsigned;
}

impl<T: Element, A: Algorithm<T>, K: Store<T>, U: Unsigned> FromIndexedParallelIterator<T, U>
    for MerkleTree<T, A, K, U>
{
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I>(into: I) -> Result<Self>
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator,
    {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let branches = U::to_usize();
        ensure!(leafs > 1, "not enough leaves");
        ensure!(next_pow2(leafs) == leafs, "size MUST be a power of 2");
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let size = get_merkle_tree_len(leafs, branches);
        let height = get_merkle_tree_height(leafs, branches);

        let mut data = K::new(size).expect("failed to create data store");

        populate_data_par::<T, A, K, U, _>(&mut data, iter)?;
        let root = K::build::<A, U>(&mut data, leafs, height, None)?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter_with_config<I>(into: I, config: StoreConfig) -> Result<Self>
    where
        U: Unsigned,
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator,
    {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let branches = U::to_usize();
        ensure!(leafs > 1, "not enough leaves");
        ensure!(next_pow2(leafs) == leafs, "size MUST be a power of 2");
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let size = get_merkle_tree_len(leafs, branches);
        let height = get_merkle_tree_height(leafs, branches);

        let mut data = K::new_with_config(size, branches, config.clone())
            .context("failed to create data store")?;

        // If the data store was loaded from disk, we know we have
        // access to the full merkle tree.
        if data.loaded_from_disk() {
            let root = data.last().context("failed to read root")?;

            return Ok(MerkleTree {
                data,
                leafs,
                height,
                root,
                _u: PhantomData,
                _a: PhantomData,
                _t: PhantomData,
            });
        }

        populate_data_par::<T, A, K, U, _>(&mut data, iter)?;
        let root = K::build::<A, U>(&mut data, leafs, height, Some(config))?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>, U: Unsigned> MerkleTree<T, A, K, U> {
    /// Attempts to calculate the root hash of a new merkle tree using hashable objects yielded by
    /// the provided iterator. This method returns the first error yielded by the iterator, if the
    /// iterator yielded an error.
    pub fn try_root_from_iter<I: IntoIterator<Item = Result<T>>>(into: I) -> Result<T> {
        let iter = into.into_iter();

        let (_, n) = iter.size_hint();
        let leafs = n.ok_or_else(|| anyhow!("could not get size hint from iterator"))?;
        let branches = U::to_usize();
        ensure!(leafs > 1, "not enough leaves");
        ensure!(next_pow2(leafs) == leafs, "size MUST be a power of 2");
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let height = get_merkle_tree_height(leafs, branches);

        // tmp_nodes is a two-dimensional vector, where the first dimension is the level of the
        // tree (`0` are the leaves). The second dimension is a fixed sized buffer that is the
        // same size as the branching factor, as it holds the temporary nodes for the current
        // level.
        let mut tmp_nodes = Vec::with_capacity(height);
        (0..height).for_each(|_| {
            // TODO vmx 2020-02-27: It should be possible to use GenericArray to create the per
            // level buffers
            let level_buffer: Vec<T> = Vec::with_capacity(branches);
            tmp_nodes.push(level_buffer)
        });

        for leaf in iter {
            // Leaf nodes need some processing
            let mut item = A::default().leaf(leaf?);

            let mut level = 0;
            loop {
                // There is still enough space within the current node
                if tmp_nodes[level].len() < branches - 1 {
                    tmp_nodes[level].push(item);
                    break;
                }
                // The node becomes full when adding the current item
                else {
                    // Add the current item, now the node is full
                    tmp_nodes[level].push(item);

                    // Hash the node to put the result into its parent
                    // NOTE vmx 2020-02-27: Last parameter isn't used, hence using `0`
                    item = A::default().multi_node(&tmp_nodes[level], 0);
                    // Clear the node
                    tmp_nodes[level].clear();
                    // Move on to the parent level
                    level += 1;
                }
            }
        }

        let root = tmp_nodes[height - 1]
            .pop()
            .context("failed to calculate root node")?;
        Ok(root)
    }

    /// Attempts to create a new merkle tree using hashable objects yielded by
    /// the provided iterator. This method returns the first error yielded by
    /// the iterator, if the iterator yielded an error.
    pub fn try_from_iter<I: IntoIterator<Item = Result<T>>>(into: I) -> Result<Self> {
        let iter = into.into_iter();

        let (_, n) = iter.size_hint();
        let leafs = n.ok_or_else(|| anyhow!("could not get size hint from iterator"))?;
        let branches = U::to_usize();
        ensure!(leafs > 1, "not enough leaves");
        ensure!(next_pow2(leafs) == leafs, "size MUST be a power of 2");
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let size = get_merkle_tree_len(leafs, branches);
        let height = get_merkle_tree_height(leafs, branches);

        let mut data = K::new(size).context("failed to create data store")?;
        populate_data::<T, A, K, U, I>(&mut data, iter).context("failed to populate data")?;
        let root = K::build::<A, U>(&mut data, leafs, height, None)?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    /// Attempts to create a new merkle tree using hashable objects yielded by
    /// the provided iterator and store config. This method returns the first
    /// error yielded by the iterator, if the iterator yielded an error.
    pub fn try_from_iter_with_config<I: IntoIterator<Item = Result<T>>>(
        into: I,
        config: StoreConfig,
    ) -> Result<Self> {
        let iter = into.into_iter();

        let (_, n) = iter.size_hint();
        let leafs = n.ok_or_else(|| anyhow!("could not get size hint from iterator"))?;
        let branches = U::to_usize();
        ensure!(leafs > 1, "not enough leaves");
        ensure!(next_pow2(leafs) == leafs, "size MUST be a power of 2");
        ensure!(
            next_pow2(branches) == branches,
            "branches MUST be a power of 2"
        );

        let size = get_merkle_tree_len(leafs, branches);
        let height = get_merkle_tree_height(leafs, branches);

        let mut data = K::new_with_config(size, branches, config.clone())
            .context("failed to create data store")?;

        // If the data store was loaded from disk, we know we have
        // access to the full merkle tree.
        if data.loaded_from_disk() {
            let root = data.last().context("failed to read root")?;

            return Ok(MerkleTree {
                data,
                leafs,
                height,
                root,
                _u: PhantomData,
                _a: PhantomData,
                _t: PhantomData,
            });
        }

        populate_data::<T, A, K, U, I>(&mut data, iter).expect("failed to populate data");
        let root = K::build::<A, U>(&mut data, leafs, height, Some(config))?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _u: PhantomData,
            _a: PhantomData,
            _t: PhantomData,
        })
    }
}

impl Element for [u8; 32] {
    fn byte_len() -> usize {
        32
    }

    fn from_slice(bytes: &[u8]) -> Self {
        if bytes.len() != 32 {
            panic!("invalid length {}, expected 32", bytes.len());
        }
        *array_ref!(bytes, 0, 32)
    }

    fn copy_to_slice(&self, bytes: &mut [u8]) {
        bytes.copy_from_slice(self);
    }
}

// Tree length calculation given the number of leafs in the tree and the branches.
pub fn get_merkle_tree_len(leafs: usize, branches: usize) -> usize {
    // Optimization
    if branches == 2 {
        assert!(leafs == next_pow2(leafs));
        return 2 * leafs - 1;
    }

    let mut len = leafs;
    let mut cur = leafs;
    let shift = log2_pow2(branches);
    while cur > 0 {
        cur >>= shift; // cur /= branches
        assert!(cur < leafs);
        len += cur;
    }

    len
}

// Tree length calculation given the number of leafs in the tree, the
// cached levels above the base, and the branches.
pub fn get_merkle_tree_cache_size(leafs: usize, branches: usize, levels: usize) -> usize {
    let shift = log2_pow2(branches);
    let len = get_merkle_tree_len(leafs, branches);
    let mut height = get_merkle_tree_height(leafs, branches);
    let stop_height = height - levels;

    let mut cache_size = len;
    let mut cur_leafs = leafs;

    while height > stop_height {
        cache_size -= cur_leafs;
        cur_leafs >>= shift; // cur /= branches
        height -= 1;
    }

    cache_size
}

pub fn is_merkle_tree_size_valid(leafs: usize, branches: usize) -> bool {
    let mut cur = leafs;
    let shift = log2_pow2(branches);
    while cur != 1 {
        cur >>= shift; // cur /= branches
        assert!(cur < leafs);
        if cur == 0 {
            return false;
        }
    }

    true
}

// Height calculation given the number of leafs in the tree and the branches.
pub fn get_merkle_tree_height(leafs: usize, branches: usize) -> usize {
    // Optimization
    if branches == 2 {
        (leafs * branches).trailing_zeros() as usize
    } else {
        (branches as f64 * leafs as f64).log(branches as f64) as usize
    }
}

// Given a tree of 'height' with the specified number of 'branches',
// calculate the length of hashes required for the proof.
pub fn get_merkle_proof_lemma_len(height: usize, branches: usize) -> usize {
    2 + ((branches - 1) * (height - 1))
}

// This method returns the number of 'leafs' given a merkle tree
// length of 'len', where leafs must be a power of 2, respecting the
// number of branches.
pub fn get_merkle_tree_leafs(len: usize, branches: usize) -> usize {
    // Optimization:
    if branches == 2 {
        return (len >> 1) + 1;
    }

    let mut leafs = 1;
    let mut cur = len;
    let shift = log2_pow2(branches);
    while cur != 1 {
        leafs <<= shift; // leafs *= branches
        cur -= leafs;
        assert!(cur < len);
    }

    leafs
}

/// returns next highest power of two from a given number if it is not
/// already a power of two.
pub fn next_pow2(n: usize) -> usize {
    n.next_power_of_two()
}

/// find power of 2 of a number which is power of 2
pub fn log2_pow2(n: usize) -> usize {
    n.trailing_zeros() as usize
}

pub fn populate_data<
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
    U: Unsigned,
    I: IntoIterator<Item = Result<T>>,
>(
    data: &mut K,
    iter: <I as std::iter::IntoIterator>::IntoIter,
) -> Result<()> {
    if !data.is_empty() {
        return Ok(());
    }

    let mut buf = Vec::with_capacity(BUILD_DATA_BLOCK_SIZE * T::byte_len());

    let mut a = A::default();
    for item in iter {
        // short circuit the tree-populating routine if the iterator yields an
        // error
        let item = item?;

        a.reset();
        buf.extend(a.leaf(item).as_ref());
        if buf.len() >= BUILD_DATA_BLOCK_SIZE * T::byte_len() {
            let data_len = data.len();
            // FIXME: Integrate into `len()` call into `copy_from_slice`
            // once we update to `stable` 1.36.
            data.copy_from_slice(&buf, data_len)?;
            buf.clear();
        }
    }
    let data_len = data.len();
    data.copy_from_slice(&buf, data_len)?;
    data.sync()?;

    Ok(())
}

fn populate_data_par<T, A, K, U, I>(data: &mut K, iter: I) -> Result<()>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
    U: Unsigned,
    I: ParallelIterator<Item = T> + IndexedParallelIterator,
{
    if !data.is_empty() {
        return Ok(());
    }

    let store = Arc::new(RwLock::new(data));

    iter.chunks(BUILD_DATA_BLOCK_SIZE)
        .enumerate()
        .try_for_each(|(index, chunk)| {
            let mut a = A::default();
            let mut buf = Vec::with_capacity(BUILD_DATA_BLOCK_SIZE * T::byte_len());

            for item in chunk {
                a.reset();
                buf.extend(a.leaf(item).as_ref());
            }
            store
                .write()
                .unwrap()
                .copy_from_slice(&buf[..], BUILD_DATA_BLOCK_SIZE * index)
        })?;

    store.write().unwrap().sync()?;
    Ok(())
}
