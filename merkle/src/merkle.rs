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
use std::path::Path;
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
    data: K,
    leafs: usize,
    height: usize,
    _a: PhantomData<A>,
    _t: PhantomData<T>,
}

/// Element stored in the merkle tree.
pub trait Element: Ord + Clone + AsRef<[u8]> + Sync + Send + Default + std::fmt::Debug {
    /// Returns the length of an element when serialized as a byte slice.
    fn byte_len() -> usize;

    /// Creates the element from its byte form. Panics if the slice is not appropriately sized.
    fn from_slice(bytes: &[u8]) -> Self;
}

/// Backing store of the merkle tree.
pub trait Store<E: Element>: ops::Deref<Target = [E]> + std::fmt::Debug + Clone {
    /// Creates a new store which can store up to `size` elements.
    fn new(size: usize) -> Self;

    fn new_from_slice(size: usize, data: &[u8]) -> Self;

    fn write_at(&mut self, el: E, i: usize);

    fn read_at(&self, i: usize) -> E;
    fn read_range(&self, r: ops::Range<usize>) -> Vec<E>;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn push(&mut self, el: E);

    // Signal to offload the `data` from memory if possible (`DiskMmapStore`
    // case). When the `data` is read/written again it should be automatically
    // reloaded. This function is only a hint with an optional implementation
    // (its mechanism should be transparent to the user who doesn't need to
    // manually reload).
    // Returns `true` if it was able to comply.
    fn offload(&mut self) -> bool;
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

    fn offload(&mut self) -> bool {
        return false;
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

    fn read_at(&self, i: usize) -> E {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store[start..end])
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

    fn offload(&mut self) -> bool {
        return false;
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
    path: String,
    // FIXME: Using `String` instead of `&Path` to avoid using lifetimes
    //  or errors like "the size for values of type `[u8]` cannot be known
    //  at compilation time" (with `Option<Box<Path>>` or similar).
    size: Option<usize>,
    // FIXME: Needed to reconstruct the `new_with_path` call, especially
    //  since we are adjusting the size there. Ideally this value should
    //  be deduced from the MT settings and not stored here.
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
            .expect(&format!("couldn't set len of {}", byte_len));

        let mmap = unsafe { MmapMut::map_mut(&file).expect("couldn't create map_mut") };
        let mmap_size = mmap.len();
        DiskMmapStore {
            store: Arc::new(RwLock::new(Some(mmap))),
            len: 0,
            _e: Default::default(),
            file,
            store_size: mmap_size,
            path: "".to_string(),
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

    fn read_at(&self, i: usize) -> E {
        let b = E::byte_len();
        let start = i * b;
        let end = (i + 1) * b;
        let len = self.len * b;
        assert!(start < len, "start out of range {} >= {}", start, len);
        assert!(end <= len, "end out of range {} > {}", end, len);

        E::from_slice(&self.store_read_range(start, end))
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
    fn offload(&mut self) -> bool {
        if self.path.is_empty() {
            // Temporary file.
            return false;
        }

        *self.store.write().unwrap() = None;

        return true;
    }
}

impl<E: Element> DiskMmapStore<E> {
    pub fn new_with_path(size: usize, path: &Path) -> Self {
        // FIXME: CROSSING API BOUNDARIES HERE.
        // This is used by `rust-fil-proofs` code when it allocates a `Store`
        // to use it with `MerkleTree::from_data_with_store`, but the `Store`
        // created there has an assigned size of *only* the leaves (the data,
        // as it's seen there), not the *entire* tree, which is expected since
        // the consumer shouldn't need to know how big the final `Store` is
        // going to be. To *temporarily* accommodate this we *mangle* the
        // received `size` (in a similar way to `MerkleTree::from_iter`) to
        // expand it to the entire tree.
        let pow = next_pow2(size);
        let size = 2 * pow - 1;

        let byte_len = E::byte_len() * size;
        let file: File = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .unwrap();
        file.set_len(byte_len as u64).unwrap();

        let mmap = unsafe { MmapMut::map_mut(&file).expect("couldn't create map_mut") };
        let mmap_size = mmap.len();
        DiskMmapStore {
            store: Arc::new(RwLock::new(Some(mmap))),
            len: 0,
            _e: Default::default(),
            file,
            store_size: mmap_size,
            path: path.to_str().expect("couldn't find path").to_string(),
            size: Some(size),
        }
    }

    pub fn store_size(&self) -> usize {
        return self.store_size;
    }

    pub fn store_read_range(&self, start: usize, end: usize) -> Vec<u8> {
        self.reload_store();
        // FIXME: Not actually thread safe, the `store` could have been offloaded
        //  after this call (but we're not striving for thread-safety at the moment).

        match *self.store.read().unwrap() {
            Some(ref mmap) => return mmap[start..end].to_vec(),
            None => panic!("The store has not been reloaded"),
        }
        // FIXME: Remove clone, this is duplicating lots of data.
    }

    pub fn store_copy_from_slice(&self, start: usize, end: usize, slice: &[u8]) {
        self.reload_store();
        match *self.store.write().unwrap() {
            Some(ref mut mmap) => return mmap[start..end].copy_from_slice(slice),
            None => panic!("The store has not been reloaded"),
        }
    }

    // Checks if the `store` is loaded and reloads it if necessary.
    // FIXME: Check how to compact this logic.
    fn reload_store(&self) {
        let mut need_to_reload_store = false;
        {
            if self.store.read().unwrap().is_none() {
                need_to_reload_store = true;
            }
        }
        if need_to_reload_store {
            let mmap = self.store.write().unwrap();

            //        let new_store: DiskMmapStore<E> = DiskMmapStore::new_with_path(self.size.expect("couldn't find size"), Path::new(&self.path));
            //        self.store = Arc::new(RwLock::new(*new_store.store()));
            //        // FIXME: Extract part of the `MmapMut` creation logic to avoid
            //        //  recreating the entire `DiskMmapStore`.
        }
    }
}

// FIXME: Fake `Clone` implementation to accomodate the artificial call in
//  `from_data_with_store`, we won't actually duplicate the mmap memory,
//  just recreate the same object (as the original will be dropped).
impl<E: Element> Clone for DiskMmapStore<E> {
    fn clone(&self) -> DiskMmapStore<E> {
        if self.path.is_empty() {
            panic!("We don't allow cloning non-temporary files at the moment");
        }

        // FIXME: How to implement the clone? (the clone itself shouldn't
        //  actually be used).
        match *self.store.read().unwrap() {
            Some(ref mmap) => {
                return DiskMmapStore {
                    store: Arc::new(RwLock::new(Some(*self.store.read().unwrap()))),
                    len: self.len(),
                    _e: Default::default(),
                    file: self.file,
                    store_size: self.store_size,
                    path: self.path,
                    size: self.size,
                };
            }
            None => panic!("Trying to copy an unloaded store"),
        }
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
        data: &mut K,
    ) -> MerkleTree<T, A, K> {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);
        let size = 2 * pow - 1;

        let mut data = data.clone();
        // FIXME: We shouldn't clone here (which would reallocate), instead use
        //  a mutable reference, that will make necessary to change also the
        //  `MerkleTree::data` to a reference.

        // leafs
        let mut a = A::default();
        for item in iter {
            a.reset();
            data.push(a.leaf(item));
        }

        let mut mt: MerkleTree<T, A, K> = MerkleTree {
            data,
            leafs,
            height: log2_pow2(size + 1),
            _a: PhantomData,
            _t: PhantomData,
        };

        mt.build();
        mt
    }

    #[inline]
    pub fn offload_store(&mut self) -> bool {
        return self.data.offload();
    }

    #[inline]
    fn build(&mut self) {
        // This algorithms assumes that the underlying store has preallocated enough space.
        // TODO: add an assert here to ensure this is the case.

        let mut width = self.leafs;

        // build tree
        let mut i: usize = 0;
        let mut j: usize = width;
        let mut height: usize = 0;

        while width > 1 {
            // if there is odd num of elements, fill in to the even
            if width & 1 == 1 {
                let el = self.data.read_at(self.data.len() - 1);
                self.data.push(el);

                width += 1;
                j += 1;
            }

            // elements are in [i..j] and they are even
            let layer: Vec<_> = self
                .data
                .read_range(i..j)
                .par_chunks(2)
                .map(|v| {
                    let lhs = v[0].to_owned();
                    let rhs = v[1].to_owned();
                    A::default().node(lhs, rhs, height)
                })
                .collect();

            // TODO: avoid collecting into a vec and write the results direclty if possible.
            for el in layer.into_iter() {
                self.data.push(el);
            }

            width >>= 1;
            i = j;
            j = j + width;
            height += 1;
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

        lemma.push(self.data.read_at(j));
        while base + 1 < self.len() {
            lemma.push(if j & 1 == 0 {
                // j is left
                self.data.read_at(base + j + 1)
            } else {
                // j is right
                self.data.read_at(base + j - 1)
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
        self.data.read_at(self.data.len() - 1)
    }

    /// Returns number of elements in the tree.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
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

    /// Returns merkle root
    #[inline]
    pub fn read_at(&self, i: usize) -> T {
        self.data.read_at(i)
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
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
        let size = 2 * pow - 1;

        let data = K::new_from_slice(size, leafs);

        assert!(leafs_count > 1);
        let mut mt: MerkleTree<T, A, K> = MerkleTree {
            data,
            leafs: leafs_count,
            height: log2_pow2(size + 1),
            _a: PhantomData,
            _t: PhantomData,
        };

        mt.build();

        mt
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromParallelIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_par_iter<I: IntoParallelIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_par_iter();

        let leafs = iter.opt_len().expect("must be sized");
        let pow = next_pow2(leafs);
        let size = 2 * pow - 1;

        let mut data = K::new(size);

        // leafs
        let vs = iter
            .map(|item| {
                let mut a = A::default();
                a.leaf(item)
            })
            .collect::<Vec<_>>();

        for v in vs.into_iter() {
            data.push(v);
        }

        assert!(leafs > 1);
        let mut mt: MerkleTree<T, A, K> = MerkleTree {
            data,
            leafs,
            height: log2_pow2(size + 1),
            _a: PhantomData,
            _t: PhantomData,
        };

        mt.build();

        mt
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> FromIterator<T> for MerkleTree<T, A, K> {
    /// Creates new merkle tree from an iterator over hashable objects.
    fn from_iter<I: IntoIterator<Item = T>>(into: I) -> Self {
        let iter = into.into_iter();

        let leafs = iter.size_hint().1.unwrap();
        assert!(leafs > 1);

        let pow = next_pow2(leafs);
        let size = 2 * pow - 1;

        let mut data = K::new(size);

        // leafs
        let mut a = A::default();
        for item in iter {
            a.reset();
            data.push(a.leaf(item));
        }

        let mut mt: MerkleTree<T, A, K> = MerkleTree {
            data,
            leafs,
            height: log2_pow2(size + 1),
            _a: PhantomData,
            _t: PhantomData,
        };

        mt.build();
        mt
    }
}

impl<T: Element, A: Algorithm<T>, K: Store<T>> ops::Deref for MerkleTree<T, A, K> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.data.deref()
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
