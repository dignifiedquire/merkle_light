use hash::{Algorithm, Hashable};
use memmap::MmapMut;
use memmap::MmapOptions;
use proof::Proof;
use rayon::prelude::*;
use std::fs::File;
use std::fs::OpenOptions;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::{self, Index};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tempfile::tempfile;

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
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MerkleTree<T, A, K>
where
    T: Element,
    A: Algorithm<T>,
    K: Store<T>,
{
    leaves: K,
    top_half: K,
    leafs: usize,
    height: usize,

    // Cache with the `root` of the tree built from `data`. This allows to
    // not access the `Store` when offloaded (`DiskMmapStore` case).
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

/// Backing store of the merkle tree.
pub trait Store<E: Element>:
    ops::Deref<Target = [E]> + std::fmt::Debug + Clone + Send + Sync
{
    /// Creates a new store which can store up to `size` elements.
    // FIXME: Return errors on failure instead of panicking
    //  (see https://github.com/filecoin-project/merkle_light/issues/19).
    fn new(size: usize) -> Self;

    fn new_from_slice(size: usize, data: &[u8]) -> Self;

    fn write_at(&mut self, el: E, i: usize);

    // Used to reduce lock contention and do the `E` to `u8`
    // *outside* the lock.
    // FIXME: `start` is the position of the elements even though
    // we're dealing with bytes here.
    // FIXME: Do we really want to expose writing `u8`?
    fn write_range(&mut self, buf: &[u8], start: usize);

    fn read_at(&self, i: usize) -> E;
    fn read_range(&self, r: ops::Range<usize>) -> Vec<E>;
    fn read_into(&self, pos: usize, buf: &mut [u8]);

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn push(&mut self, el: E);

    // Signal to offload the `data` from memory if possible (`DiskMmapStore`
    // case). When the `data` is read/written again it should be automatically
    // reloaded. This function is only a hint with an optional implementation
    // (its mechanism should be transparent to the user who doesn't need to
    // manually reload).
    // Returns `true` if it was able to comply.
    fn try_offload(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct VecStore<E: Element>(Vec<E>);

impl<E: Element> ops::Deref for VecStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<E: Element> Store<E> for VecStore<E> {
    fn new(size: usize) -> Self {
        VecStore(Vec::with_capacity(size))
    }

    fn write_at(&mut self, el: E, i: usize) {
        if self.0.len() <= i {
            self.0.resize(i + 1, E::default());
        }

        self.0[i] = el;
    }

    // NOTE: Performance regression. To conform with the current API we are
    // unnecessarily converting to and from `&[u8]` in the `VecStore` which
    // already stores `E` (in contrast with the `mmap` versions). We are
    // prioritizing performance for the `mmap` case which will be used in
    // production (`VecStore` is mainly for testing and backwards compatibility).
    fn write_range(&mut self, buf: &[u8], start: usize) {
        assert_eq!(buf.len() % E::byte_len(), 0);
        let num_elem = buf.len() / E::byte_len();

        if self.0.len() < start+num_elem {
            self.0.resize(start+num_elem, E::default());
        }

        self.0.splice(start..start+num_elem, buf
            .chunks_exact(E::byte_len())
            .map(E::from_slice));
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Self {
        let mut v: Vec<_> = data
            .chunks_exact(E::byte_len())
            .map(E::from_slice)
            .collect();
        let additional = size - v.len();
        v.reserve(additional);

        VecStore(v)
    }

    fn read_at(&self, i: usize) -> E {
        self.0[i].clone()
    }

    fn read_into(&self, i: usize, buf: &mut [u8]) {
        self.0[i].copy_to_slice(buf);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        self.0.index(r).to_vec()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn push(&mut self, el: E) {
        self.0.push(el);
    }

    fn try_offload(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MmapStore<E: Element> {
    store: MmapMut,
    len: usize,
    _e: PhantomData<E>,
}

impl<E: Element> ops::Deref for MmapStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

impl<E: Element> Store<E> for MmapStore<E> {
    #[allow(unsafe_code)]
    fn new(size: usize) -> Self {
        let byte_len = E::byte_len() * size;

        MmapStore {
            store: MmapOptions::new().len(byte_len).map_anon().unwrap(),
            len: 0,
            _e: Default::default(),
        }
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Self {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut res = Self::new(size);

        let end = data.len();
        res.store[..end].copy_from_slice(data);
        res.len = data.len() / E::byte_len();

        res
    }

    fn write_at(&mut self, el: E, i: usize) {
        let b = E::byte_len();
        self.store[i * b..(i + 1) * b].copy_from_slice(el.as_ref());
        self.len += 1;
    }

    fn write_range(&mut self, buf: &[u8], start: usize) {
        let b = E::byte_len();
        assert_eq!(buf.len() % b, 0);
        let r = std::ops::Range {
                            start: start * b,
                            end: start * b + buf.len(),
                        };
        self.store[r].copy_from_slice(buf);
        self.len += buf.len()/ b;
    }

    fn read_at(&self, i: usize) -> E {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store[start..end])
    }

    fn read_into(&self, i: usize, buf: &mut [u8]) {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        buf.copy_from_slice(&self.store[start..end]);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let b = E::byte_len();
        let start = r.start * b;
        let end = r.end * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store[start..end]
            .chunks(b)
            .map(E::from_slice)
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) {
        let l = self.len;
        assert!(
            (l + 1) * E::byte_len() <= self.store.len(),
            "not enough space"
        );

        self.write_at(el, l);
    }

    fn try_offload(&self) -> bool {
        false
    }
}

impl<E: Element> Clone for MmapStore<E> {
    fn clone(&self) -> MmapStore<E> {
        MmapStore::new_from_slice(
            self.store.len() / E::byte_len(),
            &self.store[..(self.len() * E::byte_len())],
        )
    }
}

/// File-mapping version of `MmapStore` with the added `new_with_path` method
/// that allows to set its path (otherwise a temporary file is used which is
/// cleaned up after we drop this structure).
#[derive(Debug)]
pub struct DiskMmapStore<E: Element> {
    // Implementing the `store` with `Arc`/`RwLock` to avoid adding lifetimes
    // parameters to the struct (which might have larger repercussions in the
    // definitions of the `MerkleTree` and its consumers. Also used for its
    // coordination mechanisms, but it's not clearly defined if `MerkleTree`
    // (and `Store`) should be thread-safe (so they might be removed later).
    store: Arc<RwLock<Option<MmapMut>>>,

    len: usize,
    _e: PhantomData<E>,
    file: File,
    // We need to save the `File` in case we're creating a `tempfile()`
    // otherwise it will get cleaned after we return from `new()`.

    // We cache the `store.len()` call to avoid accessing the store when
    // it's offloaded. Not to be confused with `len`, this saves the total
    // size of the `store` and the other one keeps track of used `E` slots
    // in the `DiskMmapStore`.
    store_size: usize,

    // We save the arguments of `new_with_path` to reconstruct it and reload
    // the `MmapMut` after offload has been called.
    path: PathBuf,
    size: Option<usize>,
}

impl<E: Element> ops::Deref for DiskMmapStore<E> {
    type Target = [E];

    fn deref(&self) -> &Self::Target {
        unimplemented!()
    }
}

impl<E: Element> Store<E> for DiskMmapStore<E> {
    #[allow(unsafe_code)]
    fn new(size: usize) -> Self {
        let byte_len = E::byte_len() * size;
        let file: File = tempfile().expect("couldn't create temp file");
        file.set_len(byte_len as u64)
            .unwrap_or_else(|_| panic!("couldn't set len of {}", byte_len));

        let mmap = unsafe { MmapMut::map_mut(&file).expect("couldn't create map_mut") };
        let mmap_size = mmap.len();
        DiskMmapStore {
            store: Arc::new(RwLock::new(Some(mmap))),
            len: 0,
            _e: Default::default(),
            file,
            store_size: mmap_size,
            path: PathBuf::new(),
            size: None,
        }
    }

    fn new_from_slice(size: usize, data: &[u8]) -> Self {
        assert_eq!(data.len() % E::byte_len(), 0);

        let mut res = Self::new(size);

        let end = data.len();
        res.store_copy_from_slice(0, end, data);
        res.len = data.len() / E::byte_len();

        res
    }

    fn write_at(&mut self, el: E, i: usize) {
        let b = E::byte_len();
        self.store_copy_from_slice(i * b, (i + 1) * b, el.as_ref());
        self.len += 1;
    }

    fn write_range(&mut self, buf: &[u8], start: usize) {
        let b = E::byte_len();
        self.store_copy_from_slice(start * b, start * b + buf.len(), buf);
        self.len += buf.len()/ b;
        // FIXME: is it safe to increment `len` here? we may not be necessarily 
        // writing to the end of the `Store`. Although this function is used at
        // the moment in `build` which guarantees we're writing *all* the
        // elements only *once* (so `len` will be correct at the end).
    }

    fn read_at(&self, i: usize) -> E {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store_read_range(start, end))
    }

    fn read_into(&self, i: usize, buf: &mut [u8]) {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_into(start, end, buf);
    }

    fn read_range(&self, r: ops::Range<usize>) -> Vec<E> {
        let b = E::byte_len();
        let start = r.start * b;
        let end = r.end * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        self.store_read_range(start, end)
            .chunks(b)
            .map(E::from_slice)
            .collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn push(&mut self, el: E) {
        let l = self.len;
        assert!(
            (l + 1) * E::byte_len() <= self.store_size(),
            format!(
                "not enough space, l: {}, E size {}, store len {}",
                l,
                E::byte_len(),
                self.store_size()
            )
        );

        self.write_at(el, l);
    }

    // Offload the `store` in the case it was constructed with `new_with_path`.
    // Temporary files with no path (created from `new`) can't be offloaded.
    fn try_offload(&self) -> bool {
        if self.path.as_os_str().is_empty() {
            // Temporary file.
            return false;
        }

        *self.store.write().unwrap() = None;

        true
    }
}

impl<E: Element> DiskMmapStore<E> {
    #[allow(unsafe_code)]
    // FIXME: Return errors on failure instead of panicking
    //  (see https://github.com/filecoin-project/merkle_light/issues/19).
    pub fn new_with_path(size: usize, path: &Path) -> Self {
        let byte_len = E::byte_len() * size;
        let file: File = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .expect("cannot create file");
        file.set_len(byte_len as u64).unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).expect("couldn't create map_mut") };
        let mmap_size = mmap.len();
        DiskMmapStore {
            store: Arc::new(RwLock::new(Some(mmap))),
            len: 0,
            _e: Default::default(),
            file,
            store_size: mmap_size,
            path: path.to_path_buf(),
            size: Some(size),
        }
    }

    pub fn store_size(&self) -> usize {
        self.store_size
    }

    pub fn store_read_range(&self, start: usize, end: usize) -> Vec<u8> {
        self.reload_store();
        // FIXME: Not actually thread safe, the `store` could have been offloaded
        //  after this call (but we're not striving for thread-safety at the moment).

        match *self.store.read().unwrap() {
            Some(ref mmap) => mmap[start..end].to_vec(),
            None => panic!("The store has not been reloaded"),
        }
    }

    pub fn store_read_into(&self, start: usize, end: usize, buf: &mut [u8]) {
        self.reload_store();
        // FIXME: Not actually thread safe, the `store` could have been offloaded
        //  after this call (but we're not striving for thread-safety at the moment).

        match *self.store.read().unwrap() {
            Some(ref mmap) => buf.copy_from_slice(&mmap[start..end]),
            None => panic!("The store has not been reloaded"),
        }
    }

    pub fn store_copy_from_slice(&self, start: usize, end: usize, slice: &[u8]) {
        self.reload_store();
        match *self.store.write().unwrap() {
            Some(ref mut mmap) => mmap[start..end].copy_from_slice(slice),
            None => panic!("The store has not been reloaded"),
        }
    }

    // Checks if the `store` is loaded and reloads it if necessary.
    // FIXME: Check how to compact this logic.
    fn reload_store(&self) {
        let need_to_reload_store = self.store.read().unwrap().is_none();

        if need_to_reload_store {
            let new_store: DiskMmapStore<E> = DiskMmapStore::new_with_path(
                self.size.expect("couldn't find size"),
                Path::new(&self.path),
            );
            //            self.store = Arc::new(RwLock::new();
            // FIXME: Extract part of the `MmapMut` creation logic to avoid
            //  recreating the entire `DiskMmapStore`.

            let mut store = self.store.write().unwrap();
            let new_store = Arc::try_unwrap(new_store.store)
                .unwrap()
                .into_inner()
                .unwrap();

            std::mem::replace(&mut *store, new_store);
        }
    }
}

// FIXME: Fake `Clone` implementation to accomodate the artificial call in
//  `from_data_with_store`, we won't actually duplicate the mmap memory,
//  just recreate the same object (as the original will be dropped).
impl<E: Element> Clone for DiskMmapStore<E> {
    fn clone(&self) -> DiskMmapStore<E> {
        unimplemented!("We can't clone a mmap with an already associated file");
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> MerkleTree<T, A, K> {
    /// Creates new merkle from a sequence of hashes.
    pub fn new<I: IntoIterator<Item = T>>(data: I) -> MerkleTree<T, A, K> {
        Self::from_iter(data)
    }

    /// Creates new merkle tree from a list of hashable objects.
    pub fn from_data<O: Hashable<A>, I: IntoIterator<Item = O>>(data: I) -> MerkleTree<T, A, K> {
        let mut a = A::default();
        Self::from_iter(data.into_iter().map(|x| {
            a.reset();
            x.hash(&mut a);
            a.hash()
        }))
    }

    /// Creates new merkle from an already allocated `Store` (used with
    /// `DiskMmapStore::new_with_path` to set its path before instantiating
    /// the MT, which would otherwise just call `DiskMmapStore::new`).
    // FIXME: Taken from `MerkleTree::from_iter` to avoid adding more complexity,
    //  it should receive a `parallel` flag to decide what to do.
    // FIXME: We're repeating too much code here, `from_iter` (and
    //  `from_par_iter`) should be extended to handled a pre-allocated `Store`.
    pub fn from_data_with_store<I: IntoIterator<Item = T>>(
        into: I,
        mut leaves: K,
        top_half: K,
    ) -> MerkleTree<T, A, K> {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);

        // leafs
        let mut a = A::default();
        for item in iter {
            a.reset();
            leaves.push(a.leaf(item));
        }

        Self::build(leaves, top_half, leafs, log2_pow2(2 * pow))
    }

    #[inline]
    pub fn try_offload_store(&self) -> bool {
        self.leaves.try_offload() && self.top_half.try_offload()
    }

    #[inline]
    fn build(leaves: K, top_half: K, leafs: usize, height: usize) -> Self {
        // This algorithms assumes that the underlying store has preallocated enough space.
        // TODO: add an assert here to ensure this is the case.

        let leaves_lock: Arc<RwLock<K>> = Arc::new(RwLock::new(leaves));
        let top_half_lock: Arc<RwLock<K>> = Arc::new(RwLock::new(top_half));

        // Process one `level` at a time of `width` nodes. Each level has half the nodes
        // as the previous one; the first level, completely stored in `leaves`, has `leafs`
        // nodes. We guarantee an even number of nodes per `level`, duplicating the last
        // node if necessary.
        // `level_node_index` keeps the "global" index of the first node of the current
        // level: the index we would have if the `leaves` and `top_half` were unified
        // in the same `Store`; it is later converted to the "local" index to access each
        // individual `Store` (according to which `level` we're processing at the moment).
        // We always write to the `top_half` (which contains all the levels but the first
        // one) of the tree and only read from the `leaves` in the first iteration
        // (at `level` 0).
        let mut level: usize = 0;
        let mut width = leafs;
        let mut level_node_index = 0;
        while width > 1 {
            if width & 1 == 1 {
                // Odd number of nodes, duplicate last.
                let mut active_store = if level == 0 {
                    leaves_lock.write().unwrap()
                } else {
                    top_half_lock.write().unwrap()
                };
                let last_node = active_store.read_at(active_store.len() - 1);
                active_store.push(last_node);

                width += 1;
            }

            // We read the `width` nodes of the current `level` from `read_store` and
            // write (half of it) in the `write_store` (which contains the next level).
            // Both `read_start` and `write_start` are "local" indexes with respect to
            // the `read_store` and `write_store` they are accessing.
            let (read_store_lock, write_store_lock, read_start, write_start) = if level == 0 {
                // The first level is in the `leaves`, which is all it contains so the
                // next level to write to will be in the `top_half`. Since we are "jumping"
                // from one `Store` to the other both read/write start indexes start at zero.
                (leaves_lock.clone(), top_half_lock.clone(), 0, 0)
            } else {
                // For all other levels we'll read/write from/to the `top_half` adjusting the
                // "global" index to access this `Store` (offsetting `leaves` length). All levels
                // are contiguous so we read/write `width` nodes apart.
                let read_start = level_node_index - leaves_lock.read().unwrap().len();

                (
                    top_half_lock.clone(),
                    top_half_lock.clone(),
                    read_start,
                    read_start + width,
                )
            };

            // Allocate `width` indexes during operation (which is a negligible memory bloat
            // compared to the 32-bytes size of the nodes stored in the `Store`s) and hash each
            // pair of nodes to write them to the next level in concurrent threads.
            // NOTE: This may lead to *considerable* lock contention (especially when reading
            // and writing to the same `Store`, `top_half`).
            // FIXME: Process more than 2 nodes at a time to reduce contention (changing the
            // "pair" terminology to the more general "chunk" and removing the hard-coded 2's).
            let chunk_size = 1024;
            debug_assert_eq!(chunk_size % 2, 0);
            Vec::from_iter((read_start..read_start + width).step_by(chunk_size))
                .par_chunks(1)
                .for_each(|chunk_index| {
                    let chunk_index = chunk_index[0];
                    let chunk_size = std::cmp::min(chunk_size, read_start + width - chunk_index);

                    let chunk_nodes = {
                        // Read everything taking the lock once.
                        let read_store = read_store_lock.read().unwrap();
                        read_store.read_range(std::ops::Range {
                            start: chunk_index,
                            end: chunk_index + chunk_size,
                        })
                    };

                    let hashed_nodes = chunk_nodes.chunks(2).map(|node_pair| {
                        A::default().node(node_pair[0].clone(), node_pair[1].clone(), level)
                        // FIXME: Change `node()` to receive references to avoid the `clone` here,
                        //  this might be an API-breaking change.
                    });

                    // We write the hashed nodes to the next level in the position that
                    // would be "in the middle" of the previous pair (dividing it by 2).
                    let write_delta = (chunk_index - read_start) / 2;

                    let mut tmp: Vec<u8> = Vec::with_capacity(hashed_nodes.len() * T::byte_len());
                    hashed_nodes.for_each(|node| { tmp.append(&mut node.as_ref().to_vec()) });
                    debug_assert_eq!(tmp.len(), chunk_size/2 * T::byte_len());
                    let hashed_nodes_as_bytes: &[u8] = tmp.as_slice();
                    // FIXME: Pre-allocate slice. Simplify this conversion.
                    
                    let mut write_store = write_store_lock.write().unwrap();
                    write_store.write_range(hashed_nodes_as_bytes, write_start + write_delta);
                });

            level_node_index += width;
            level += 1;
            width >>= 1;
        }

        assert_eq!(height, level + 1);
        // The root isn't part of the previous loop so `height` is
        // missing one level.

        let root = {
            let top_half = top_half_lock.read().unwrap();
            top_half.read_at(top_half.len() - 1)
        };

        MerkleTree {
            leaves: Arc::try_unwrap(leaves_lock).unwrap().into_inner().unwrap(),
            top_half: Arc::try_unwrap(top_half_lock)
                .unwrap()
                .into_inner()
                .unwrap(),
            leafs,
            height,
            root,
            _a: PhantomData,
            _t: PhantomData,
        }
    }

    /// Generate merkle tree inclusion proof for leaf `i`
    #[inline]
    pub fn gen_proof(&self, i: usize) -> Proof<T> {
        assert!(i < self.leafs); // i in [0 .. self.leafs)

        let mut lemma: Vec<T> = Vec::with_capacity(self.height + 1); // path + root
        let mut path: Vec<bool> = Vec::with_capacity(self.height - 1); // path - 1

        let mut base = 0;
        let mut j = i;

        // level 1 width
        let mut width = self.leafs;
        if width & 1 == 1 {
            width += 1;
        }

        lemma.push(self.read_at(j));
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                self.read_at(base + j + 1)
            } else {
                // j is right
                self.read_at(base + j - 1)
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
        self.leaves.len() + self.top_half.len()
    }

    /// Returns `true` if the vector contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty() && self.top_half.is_empty()
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

    /// Returns merkle root
    #[inline]
    pub fn read_at(&self, i: usize) -> T {
        if i < self.leaves.len() {
            self.leaves.read_at(i)
        } else {
            self.top_half.read_at(i - self.leaves.len())
        }
    }

    // With the leaves decoupled from the rest of the tree we need to split
    // the range if necessary. If the range is covered by a single `Store`
    // we just call its `read_range`, if not, we need to form a new `Vec`
    // to hold both parts.
    // FIXME: The second mechanism can be *very* expensive with big sectors,
    // should the consumer be aware of this to avoid memory bloats?
    pub fn read_range(&self, start: usize, end: usize) -> Vec<T> {
        if start > end {
            panic!("read_range: start > end ({} > {})", start, end);
            // FIXME: Do we need to check this? The implementations of
            // `Store` don't (does `Range` take care of it?).
        }

        let leaves_len = self.leaves.len();
        if end <= self.leaves.len() {
            self.leaves.read_range(std::ops::Range { start, end })
        } else if start >= self.leaves.len() {
            self.top_half.read_range(std::ops::Range {
                start: start - leaves_len,
                end: end - leaves_len,
            })
        } else {
            let mut joined = Vec::with_capacity(end - start);
            joined.append(&mut self.leaves.read_range(std::ops::Range {
                start,
                end: leaves_len,
            }));
            joined.append(&mut self.top_half.read_range(std::ops::Range {
                start: 0,
                end: end - leaves_len,
            }));
            joined
        }
    }

    /// Reads into a pre-allocated slice (for optimization purposes).
    pub fn read_into(&self, pos: usize, buf: &mut [u8]) {
        if pos < self.leaves.len() {
            self.leaves.read_into(pos, buf);
        } else {
            self.top_half.read_into(pos - self.leaves.len(), buf);
        }
    }

    /// Build the tree given a slice of all leafs, in bytes form.
    pub fn from_byte_slice(leafs: &[u8]) -> Self {
        assert_eq!(
            leafs.len() % T::byte_len(),
            0,
            "{} not a multiple of {}",
            leafs.len(),
            T::byte_len()
        );

        let leafs_count = leafs.len() / T::byte_len();
        let pow = next_pow2(leafs_count);

        let leaves = K::new_from_slice(pow, leafs);
        let top_half = K::new(pow);

        assert!(leafs_count > 1);
        Self::build(leaves, top_half, leafs_count, log2_pow2(2 * pow))
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromParallelIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I: IntoParallelIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);

        let mut leaves = K::new(pow);
        let top_half = K::new(pow);

        // leafs
        let vs = iter
            .map(|item| {
                let mut a = A::default();
                a.leaf(item)
            })
            .collect::<Vec<_>>();

        for v in vs.into_iter() {
            leaves.push(v);
        }

        assert!(leafs > 1);
        Self::build(leaves, top_half, leafs, log2_pow2(2 * pow))
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_iter<I: IntoIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);

        let mut leaves = K::new(pow);
        let top_half = K::new(pow);

        // leafs
        let mut a = A::default();
        for item in iter {
            a.reset();
            leaves.push(a.leaf(item));
        }

        Self::build(leaves, top_half, leafs, log2_pow2(2 * pow))
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
