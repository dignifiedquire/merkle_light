use memmap::MmapOptions;
use std::fs::OpenOptions;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use log::debug;
use rayon::prelude::*;

use crate::hash::{Algorithm, Hashable};
use crate::proof::Proof;
use crate::store::{Store, StoreConfig, StoreType, VecStore};

/// Tree size (number of nodes) used as threshold to decide which build algorithm
/// to use. Small trees (below this value) use the old build algorithm, optimized
/// for speed rather than memory, allocating as much as needed to allow multiple
/// threads to work concurrently without interrupting each other. Large trees (above)
/// use the new build algorithm, optimized for memory rather than speed, allocating
/// as less as possible with multiple threads competing to get the write lock.
pub const SMALL_TREE_BUILD: usize = 1024;

// Number of nodes to process in parallel during the `build` stage.
pub const BUILD_CHUNK_NODES: usize = 1024 * 4;

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
/// children nodes. A diagram depicting how it works:
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
pub struct MerkleTree<T, A, K>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
{
    data: K,
    leafs: usize,
    height: usize,

    // Cache with the `root` of the tree built from `data`. This allows to
    // not access the `Store` (e.g., access to disks in `DiskStore`).
    root: T,

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

impl<T: Element, A: Algorithm<T>, K: Store<T>> MerkleTree<T, A, K> {
    /// Creates new merkle from a sequence of hashes.
    pub fn new<I: IntoIterator<Item = T>>(data: I) -> Result<MerkleTree<T, A, K>> {
        Self::try_from_iter(data.into_iter().map(Ok))
    }

    /// Creates new merkle from a sequence of hashes.
    pub fn new_with_config<I: IntoIterator<Item = T>>(
        data: I,
        config: StoreConfig,
    ) -> Result<MerkleTree<T, A, K>> {
        Self::try_from_iter_with_config(data.into_iter().map(Ok), config)
    }

    /// Creates new merkle tree from a list of hashable objects.
    pub fn from_data<O: Hashable<A>, I: IntoIterator<Item = O>>(
        data: I,
    ) -> Result<MerkleTree<T, A, K>> {
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
    ) -> Result<MerkleTree<T, A, K>> {
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
    pub fn from_data_store(data: K, size: usize) -> Result<MerkleTree<T, A, K>> {
        let pow = next_pow2(size);
        let height = log2_pow2(2 * pow);
        let root = data.last()?;

        Ok(MerkleTree {
            data,
            leafs: size,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    fn build(data: K, leafs: usize, height: usize) -> Result<Self> {
        ensure!(data.len() == leafs, "Inconsistent data");
        if leafs <= SMALL_TREE_BUILD {
            return Self::build_small_tree(data, leafs, height);
        }

        let data_lock = Arc::new(RwLock::new(data));

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `data`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        while width > 1 {
            if width & 1 == 1 {
                // Odd number of nodes, duplicate last.
                let mut active_store = data_lock.write().unwrap();
                let last_node = active_store.last()?;
                active_store.push(last_node)?;

                width += 1;
            }

            // Start reading at the beginning of the current level, and writing the next
            // level immediate after.  `level_node_index` keeps track of the current read
            // starts, and width is updated accordingly at each level so that we know where
            // to start writing.
            let (read_start, write_start) = if level == 0 {
                // Note that we previously asserted that data.len() == leafs.
                (0, data_lock.read().unwrap().len())
            } else {
                (level_node_index, level_node_index + width)
            };

            Self::process_layer(data_lock.clone(), width, level, read_start, write_start)?;

            level_node_index += width;
            level += 1;
            width >>= 1;

            data_lock.write().unwrap().sync()?;
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = {
            let data = data_lock.read().unwrap();
            data.last()?
        };

        Ok(MerkleTree {
            data: Arc::try_unwrap(data_lock).unwrap().into_inner().unwrap(),
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    fn process_layer(
        data_lock: Arc<RwLock<K>>,
        width: usize,
        level: usize,
        read_start: usize,
        write_start: usize,
    ) -> Result<()> {
        // Allocate `width` indexes during operation (which is a negligible memory bloat
        // compared to the 32-bytes size of the nodes stored in the `Store`s) and hash each
        // pair of nodes to write them to the next level in concurrent threads.
        // Process `BUILD_CHUNK_NODES` nodes in each thread at a time to reduce contention,
        // optimized for big sector sizes (small ones will just have one thread doing all
        // the work).
        debug_assert_eq!(BUILD_CHUNK_NODES % 2, 0);
        Vec::from_iter((read_start..read_start + width).step_by(BUILD_CHUNK_NODES))
            .par_iter()
            .try_for_each(|&chunk_index| -> Result<()> {
                let chunk_size = std::cmp::min(BUILD_CHUNK_NODES, read_start + width - chunk_index);

                let chunk_nodes = {
                    // Read everything taking the lock once.
                    data_lock
                        .read()
                        .unwrap()
                        .read_range(chunk_index..chunk_index + chunk_size)?
                };

                // We write the hashed nodes to the next level in the position that
                // would be "in the middle" of the previous pair (dividing by 2).
                let write_delta = (chunk_index - read_start) / 2;

                let nodes_size = (chunk_nodes.len() / 2) * T::byte_len();
                let hashed_nodes_as_bytes = chunk_nodes.chunks(2).fold(
                    Vec::with_capacity(nodes_size),
                    |mut acc, node_pair| {
                        let h =
                            A::default().node(node_pair[0].clone(), node_pair[1].clone(), level);
                        acc.extend_from_slice(h.as_ref());
                        acc
                    },
                );

                // Check that we correctly pre-allocated the space.
                debug_assert_eq!(hashed_nodes_as_bytes.len(), chunk_size / 2 * T::byte_len());

                // Write the data into the store.
                data_lock
                    .write()
                    .unwrap()
                    .copy_from_slice(&hashed_nodes_as_bytes, write_start + write_delta)?;

                Ok(())
            })
    }

    fn process_disk_layer(
        data: &mut K,
        file: &std::fs::File,
        width: usize,
        level: usize,
        read_start: usize,
        write_start: usize,
    ) -> Result<()> {
        let data_lock = Arc::new(RwLock::new(data));

        Vec::from_iter((read_start..read_start + width).step_by(BUILD_CHUNK_NODES))
            .par_iter()
            .try_for_each(move |&chunk_index| -> Result<()> {
                let chunk_size = std::cmp::min(BUILD_CHUNK_NODES, read_start + width - chunk_index);

                let chunk_nodes = {
                    // Read everything taking the lock once.
                    data_lock
                        .read()
                        .unwrap()
                        .read_range(chunk_index..chunk_index + chunk_size)?
                };

                // We write the hashed nodes to the next level in the position that
                // would be "in the middle" of the previous pair (dividing by 2).
                let write_delta = (chunk_index - read_start) / 2;

                let nodes_size = (chunk_nodes.len() / 2) * T::byte_len();
                let hashed_nodes_as_bytes = chunk_nodes.chunks(2).fold(
                    Vec::with_capacity(nodes_size),
                    |mut acc, node_pair| {
                        let h =
                            A::default().node(node_pair[0].clone(), node_pair[1].clone(), level);
                        acc.extend_from_slice(h.as_ref());
                        acc
                    },
                );

                // Check that we correctly pre-allocated the space.
                let chunk_size = std::cmp::min(BUILD_CHUNK_NODES, read_start + width - chunk_index);

                let hashed_nodes_as_bytes_len = hashed_nodes_as_bytes.len();
                debug_assert_eq!(hashed_nodes_as_bytes_len, chunk_size / 2 * T::byte_len());

                let map_write_start =
                    (write_start * T::byte_len() + write_delta * T::byte_len()) as u64;

                // Safety: this operation is safe becase it's a limited
                // writable region that is not overlapped with any
                // other mapping that could happen in parallel.
                let mut mmap = unsafe {
                    let mut mmap_options = MmapOptions::new();
                    mmap_options
                        .offset(map_write_start)
                        .len(hashed_nodes_as_bytes_len)
                        .map_mut(&file)?
                };

                mmap[0..hashed_nodes_as_bytes_len].copy_from_slice(&hashed_nodes_as_bytes);

                Ok(())
            })
    }

    fn build_from_disk_store(
        mut data: K,
        leafs: usize,
        height: usize,
        config: StoreConfig,
    ) -> Result<Self> {
        ensure!(data.len() == leafs, "Inconsistent data");

        // Open the backing file of the store for future mmap'd writes.
        let data_path = StoreConfig::data_path(&config.path, &config.id);
        let file = OpenOptions::new().read(true).write(true).open(data_path)?;

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `data`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        while width > 1 {
            // Start reading at the beginning of the current level, and writing the next
            // level immediate after.  `level_node_index` keeps track of the current read
            // starts, and width is updated accordingly at each level so that we know where
            // to start writing.
            let (read_start, write_start) = if level == 0 {
                // Note that we previously asserted that data.len() == leafs.
                (0, leafs)
            } else {
                (level_node_index, level_node_index + width)
            };

            Self::process_disk_layer(&mut data, &file, width, level, read_start, write_start)?;

            level_node_index += width;
            level += 1;
            width >>= 1;

            // When the layer is complete, update the store length
            // since we know the backing file was updated outside of
            // the store interface.
            data.set_len(Store::len(&data) + width);
        }

        // Ensure every element is accounted for.
        assert_eq!(Store::len(&data), get_merkle_tree_len(leafs));

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = data.last()?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    #[inline]
    fn build_small_tree(mut data: K, leafs: usize, height: usize) -> Result<Self> {
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;

        while width > 1 {
            if width & 1 == 1 {
                let last_node = data.read_at(Store::len(&data) - 1)?;
                data.push(last_node)?;

                width += 1;
            }

            // Same indexing logic as `build`.
            let (layer, write_start) = {
                let (read_start, write_start) = if level == 0 {
                    // Note that we previously asserted that data.len() == leafs.
                    (0, data.len())
                } else {
                    (level_node_index, level_node_index + width)
                };

                let layer: Vec<_> = data
                    .read_range(read_start..read_start + width)?
                    .par_chunks(2)
                    .map(|v| {
                        let lhs = v[0].to_owned();
                        let rhs = v[1].to_owned();
                        A::default().node(lhs, rhs, level)
                    })
                    .collect();

                (layer, write_start)
            };

            for (i, node) in layer.into_iter().enumerate() {
                data.write_at(node, write_start + i)?;
            }

            level_node_index += width;
            level += 1;
            width >>= 1;
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = data.last()?;

        Ok(MerkleTree {
            data,
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    #[inline]
    fn build_partial_tree(
        data: VecStore<T>,
        leafs: usize,
        height: usize,
    ) -> Result<MerkleTree<T, A, VecStore<T>>> {
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;

        let data_len = Store::len(&data);
        let data_lock = Arc::new(RwLock::new(data));

        while width > 1 {
            // For partial tree building, the data layers can never be
            // odd lengths.
            assert!(width % 2 == 0);

            // Same indexing logic as `build`.
            let (read_start, write_start) = if level == 0 {
                (0, data_len)
            } else {
                (level_node_index, level_node_index + width)
            };

            MerkleTree::<T, A, VecStore<T>>::process_layer(
                data_lock.clone(),
                width,
                level,
                read_start,
                write_start,
            )?;

            level_node_index += width;
            level += 1;
            width >>= 1;
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = data_lock.read().unwrap().last()?;

        Ok(MerkleTree {
            data: Arc::try_unwrap(data_lock).unwrap().into_inner().unwrap(),
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        })
    }

    /// Generate merkle tree inclusion proof for leaf `i`
    #[inline]
    pub fn gen_proof(&self, i: usize) -> Result<Proof<T>> {
        ensure!(
            i < self.leafs,
            "{} is out of bounds (max: {})",
            i,
            self.leafs
        ); // i in [0 .. self.leafs)

        let mut lemma: Vec<T> = Vec::with_capacity(self.height + 1); // path + root
        let mut path: Vec<bool> = Vec::with_capacity(self.height - 1); // path - 1

        let mut base = 0;
        let mut j = i;

        // level 1 width
        let mut width = self.leafs;
        if width & 1 == 1 {
            width += 1;
        }

        lemma.push(self.read_at(j)?);
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                self.read_at(base + j + 1)?
            } else {
                // j is right
                self.read_at(base + j - 1)?
            });
            path.push(j & 1 == 0);

            base += width;
            width >>= 1;
            if width & 1 == 1 {
                width += 1;
            }
            j >>= 1;
        }

        // root is final
        lemma.push(self.root());

        // Sanity check: if the `MerkleTree` lost its integrity and `data` doesn't match the
        // expected values for `leafs` and `height` this can get ugly.
        debug_assert!(lemma.len() == self.height + 1);
        debug_assert!(path.len() == self.height - 1);

        Ok(Proof::new(lemma, path))
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
    ) -> Result<(Proof<T>, MerkleTree<T, A, VecStore<T>>)> {
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

        let total_size = 2 * self.leafs - 1;
        let cache_size = total_size >> levels;
        if cache_size >= total_size {
            return Ok((self.gen_proof(i)?, Default::default()));
        }

        let cached_leafs = get_merkle_tree_leafs(cache_size);
        ensure!(
            cached_leafs == next_pow2(cached_leafs),
            "The size of the cached leafs must be a power of 2"
        );

        let cache_height = log2_pow2(cached_leafs);
        let partial_height = self.height - cache_height;

        // Calculate the subset of the base layer data width that we
        // need in order to build the partial tree required to build
        // the proof (termed 'segment_width'), given the data
        // configuration specified by 'levels'.
        let segment_width = self.leafs / cached_leafs;
        let segment_start = (i / segment_width) * segment_width;
        let segment_end = segment_start + segment_width;

        debug!("leafs {}, total size {}, total height {}, cache_size {}, cached levels above base {}, \
                partial_height {}, cached_leafs {}, segment_width {}, segment range {}-{} for {}",
               self.leafs, total_size, self.height, cache_size, levels, partial_height,
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
        data_copy.resize(((2 * segment_width) - 1) * T::byte_len(), 0);

        // Build the optimally small tree.
        let partial_tree: MerkleTree<T, A, VecStore<T>> =
            Self::build_partial_tree(partial_store, segment_width, partial_height)?;
        ensure!(
            partial_height == partial_tree.height(),
            "Inconsistent partial tree height"
        );

        // Generate entire proof with access to the base data, the
        // cached data, and the partial tree.
        let proof = self.gen_proof_with_partial_tree(i, levels, &partial_tree)?;

        debug!(
            "generated partial_tree of height {} and len {} for proof at {}",
            partial_tree.height,
            partial_tree.len(),
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
        partial_tree: &MerkleTree<T, A, VecStore<T>>,
    ) -> Result<Proof<T>> {
        ensure!(
            i < self.leafs,
            "{} is out of bounds (max: {})",
            i,
            self.leafs
        ); // i in [0 .. self.leafs)

        // For partial tree building, the data layer width must be a
        // power of 2.
        let mut width = self.leafs;
        ensure!(width == next_pow2(width), "Must be a power of 2 tree");

        let data_width = width;
        let total_size = 2 * data_width - 1;
        let cache_size = total_size >> levels;
        let cache_index_start = total_size - cache_size;
        let cached_leafs = get_merkle_tree_leafs(cache_size);
        ensure!(
            cached_leafs == next_pow2(cached_leafs),
            "Cached leafs size must be a power of 2"
        );

        // Calculate the subset of the data layer width that we need
        // in order to build the partial tree required to build the
        // proof (termed 'segment_width').
        let mut segment_width = width / cached_leafs;
        let segment_start = (i / segment_width) * segment_width;

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

        // 'current_height' tracks the layer count being processed,
        // starting from the bottom and increasing toward the root.
        let mut current_height = 0;

        let mut lemma: Vec<T> = Vec::with_capacity(self.height + 1); // path + root
        let mut path: Vec<bool> = Vec::with_capacity(self.height - 1); // path - 1

        lemma.push(self.read_at(j)?);
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                let left_index = base + j + 1;

                // Check if we can read from either the base data layer, or the cached region
                // (accessed the same way via the store interface).
                if left_index < data_width || left_index >= cache_index_start {
                    self.read_at(left_index)?
                } else {
                    // Otherwise, read from the partially built sub-tree with a properly
                    // adjusted index.
                    let partial_tree_index =
                        partial_base + j + 1 - (segment_start >> current_height);
                    partial_tree.read_at(partial_tree_index)?
                }
            } else {
                // j is right
                let right_index = base + j - 1;

                // Check if we can read from either the base data layer, or the cached region
                // (accessed the same way via the store interface).
                if right_index < data_width || right_index >= cache_index_start {
                    self.read_at(right_index)?
                } else {
                    // Otherwise, read from the partially built sub-tree with a properly
                    // adjusted index.
                    let partial_tree_index =
                        partial_base + j - 1 - (segment_start >> current_height);
                    partial_tree.read_at(partial_tree_index)?
                }
            });

            path.push(j & 1 == 0);

            base += width;
            width >>= 1;

            partial_base += segment_width;
            segment_width >>= 1;

            j >>= 1;
            current_height += 1;
        }

        // root is final
        lemma.push(self.root());

        // Sanity check: if the `MerkleTree` lost its integrity and `data` doesn't match the
        // expected values for `leafs` and `height` this can get ugly.
        debug_assert!(lemma.len() == self.height + 1);
        debug_assert!(path.len() == self.height - 1);

        Ok(Proof::new(lemma, path))
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
        self.data.compact(config, store_version)
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

    /// Returns merkle root
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
        ensure!(leafs_count > 1, "Must have at least 1 leaf");

        let pow = next_pow2(leafs_count);
        let data =
            K::new_from_slice_with_config(get_merkle_tree_len(leafs_count), leafs, config.clone())
                .context("failed to create data store")?;

        if data.store_type() == StoreType::Disk {
            Self::build_from_disk_store(data, leafs_count, log2_pow2(2 * pow), config)
        } else {
            Self::build(data, leafs_count, log2_pow2(2 * pow))
        }
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
        ensure!(leafs_count > 1, "Must have at least 1 leaf");

        let pow = next_pow2(leafs_count);
        let data = K::new_from_slice(get_merkle_tree_len(leafs_count), leafs)
            .context("failed to create data store")?;

        Self::build(data, leafs_count, log2_pow2(2 * pow))
    }
}

pub trait FromIndexedParallelIterator<T>: Sized
where
    T: Send,
{
    fn from_par_iter<I>(par_iter: I) -> Result<Self>
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator;

    fn from_par_iter_with_config<I>(par_iter: I, config: StoreConfig) -> Result<Self>
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator;
}

// NOTE: This use cannot accept a StoreConfig.
impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIndexedParallelIterator<T>
    for MerkleTree<T, A, K>
{
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I>(into: I) -> Result<Self>
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator,
    {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);

        let mut data = K::new(get_merkle_tree_len(leafs)).expect("failed to create data store");
        populate_data_par::<T, A, K, _>(&mut data, iter)?;

        Self::build(data, leafs, log2_pow2(2 * pow))
    }

    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter_with_config<I>(into: I, config: StoreConfig) -> Result<Self>
    where
        I: IntoParallelIterator<Item = T>,
        I::Iter: IndexedParallelIterator,
    {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);
        let height = log2_pow2(2 * pow);

        let mut data = K::new_with_config(get_merkle_tree_len(leafs), config.clone())
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
                _a: PhantomData,
                _t: PhantomData,
            });
        }

        populate_data_par::<T, A, K, _>(&mut data, iter)?;
        if data.store_type() == StoreType::Disk {
            Self::build_from_disk_store(data, leafs, height, config)
        } else {
            Self::build(data, leafs, height)
        }
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> MerkleTree<T, A, K> {
    /// Attempts to create a new merkle tree using hashable objects yielded by
    /// the provided iterator. This method returns the first error yielded by
    /// the iterator, if the iterator yielded an error.
    pub fn try_from_iter<I: IntoIterator<Item = Result<T>>>(into: I) -> Result<Self> {
        let iter = into.into_iter();

        let (_, n) = iter.size_hint();
        let leafs = n.ok_or_else(|| anyhow!("could not get size hint from iterator"))?;
        ensure!(leafs > 1, "not enough leaves");

        let pow = next_pow2(leafs);
        let mut data = K::new(get_merkle_tree_len(leafs)).context("failed to create data store")?;
        populate_data::<T, A, K, I>(&mut data, iter).context("failed to populate data")?;

        Self::build(data, leafs, log2_pow2(2 * pow))
    }

    /// Attempts to create a new merkle tree using hashable objects yielded by
    /// the provided iterator and store config. This method returns the first
    /// error yielded by the iterator, if the iterator yielded an error.
    pub fn try_from_iter_with_config<I: IntoIterator<Item = Result<T>>>(
        into: I,
        config: StoreConfig,
    ) -> Result<Self> {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);
        let height = log2_pow2(2 * pow);

        let mut data = K::new_with_config(get_merkle_tree_len(leafs), config.clone())
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
                _a: PhantomData,
                _t: PhantomData,
            });
        }

        populate_data::<T, A, K, I>(&mut data, iter).expect("failed to populate data");

        if data.store_type() == StoreType::Disk {
            Self::build_from_disk_store(data, leafs, height, config)
        } else {
            Self::build(data, leafs, height)
        }
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

// This method returns the actual merkle tree length, but requires
// that leafs is a power of 2.
pub fn get_merkle_tree_len(leafs: usize) -> usize {
    2 * next_pow2(leafs) - 1
}

// This method returns the minimal number of 'leafs' given a merkle
// tree length of 'len', where leafs must be a power of 2.
pub fn get_merkle_tree_leafs(len: usize) -> usize {
    (len >> 1) + 1
}

/// `next_pow2` returns next highest power of two from a given number if
/// it is not already a power of two.
///
/// [](http://locklessinc.com/articles/next_pow2/)
/// [](https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2/466242#466242)
pub fn next_pow2(mut n: usize) -> usize {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n + 1
}

/// find power of 2 of a number which is power of 2
pub fn log2_pow2(n: usize) -> usize {
    n.trailing_zeros() as usize
}

pub fn populate_data<
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
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

fn populate_data_par<T, A, K, I>(data: &mut K, iter: I) -> Result<()>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
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
