# merkle

[![Build Status](https://travis-ci.com/filecoin-project/merkle_light.svg?branch=master&style=flat)](https://travis-ci.com/filecoin-project/merkle_light)
[![Issues](http://img.shields.io/github/issues/filecoin-project/merkle_light.svg?style=flat)](https://github.com/filecoin_project/merkle_light/issues)
![License](https://img.shields.io/badge/license-bsd3-brightgreen.svg?style=flat)

*merkle* is a lightweight Rust implementation of a [Merkle tree](https://en.wikipedia.org/wiki/Merkle_tree).

## Features

- external dependency agnostic
- `std::hash::Hasher` compatibility
- standard types hasher implementations
- `#[derive(Hashable)]` support for simple struct
- customizable merkle leaf/node hashing algorithm
- support for custom hash types (e.g. [u8; 16], [u64; 4], [u128; 2], struct)
- customizable hashing algorithm
- linear memory layout, no nodes on heap
- buildable from iterator, objects or hashes
- certificate transparency style merkle hashing support
- SPV support included (via proof type)
- supports power of 2 arity merkletrees (only)
- supports compound merkletrees (a tree of merkletrees)
- supports compound-compound merkletrees (a tree of compound merkletrees)


## Documentation

Documentation is [available](https://docs.rs/merkletree).

# Examples

The most relevant examples are located in the following files:

* `test_common.rs`: custom hash example xor128, misc shared utils
* `test_xor128.rs`: most comprehensive tests for library features
* `proof.rs`: contains impl and tests for proofs across pow2 arity trees

# Building and testing

```
# Run tests in release mode
cargo test --release --all

# Run ignored tests in release mode
cargo test --release --all -- --ignored
```



## Bug Reporting

Please report bugs either as pull requests or as issues in [the issue
tracker](https://github.com/filecoin-project/merkle_light). *merkle* has a
**full disclosure** vulnerability policy. **Please do NOT attempt to report
any security vulnerability in this code privately to anybody.**

## License

See [LICENSE](LICENSE).
