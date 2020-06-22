# λ ∈ MLIR | ![mu-squared](https://wikimedia.org/api/rest_v1/media/math/render/svg/68b31ff8082976980ee460521d36b8a30f9603a2)
- [Lioville function](https://en.wikipedia.org/wiki/Liouville_function)

### Good resources:

- [SPIR-V dialect spec](https://github.com/bollu/mlir/blob/master/orig_docs/Dialects/SPIR-V.md)
- [SPIR-V dialect base tablegen](https://github.com/bollu/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVBase.td)
- [SPIR-V dialect header](https://github.com/bollu/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVDialect.h)
- [SPIR-V dialect cpp](https://github.com/bollu/mlir/blob/master/lib/Dialect/SPIRV/SPIRVDialect.cpp)
- [SPIR-V types header](https://github.com/bollu/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVTypes.h)
- [SPIR-V types cpp](https://github.com/bollu/mlir/blob/master/lib/Dialect/SPIRV/SPIRVTypes.cpp)

### Round tripping

- What stops someone from defining a printer that is completely different
  from what the parser wants?

### GHC Core

- [`CoreSyn`](https://downloads.haskell.org/~ghc/8.8.3/docs/html/libraries/ghc-8.8.3/CoreSyn.html)

```hs
data Expr b
  = Var   Id
  | Lit   Literal
  | App   (Expr b) (Arg b)
  | Lam   b (Expr b)
  | Let   (Bind b) (Expr b)
  | Case  (Expr b) b Type [Alt b]       -- See #case_invariants#
  | Cast  (Expr b) Coercion
  | Tick  (Tickish Id) (Expr b)
  | Type  Type
  | Coercion Coercion
  deriving Data
```

### `hask-programs/`

To run these, first install the GHC plugin from `ghc-dump`. This installs
both the plugin that is loaded with `-fplugin GhcDump.Plugin` [The package `ghc-dump-core`]
as well as the utility tool for dumping these called `ghc-dump` [The package `ghc-dump-util`].

```
# install tool and compiler plugin 
$ cd ghc-dump && cabal install --lib all && cabal install all
```

Then run the makefile:

```
# run the makefile, generate the dump, pretty print the dump
$ cd hask-programs && make && ghc-dump --show fib.pass-0000.cbor
```

### TODO

- [ ] Parse `case x of { ... }` style input.
- [ ] Parse `let var = val in ...` input.
- [ ] Add `PrimInt` type.
- [ ] Get factorial up and running.
- [ ] Get a notion of explicit GC. 
