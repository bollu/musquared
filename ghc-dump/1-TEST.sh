#!/usr/bin/env bash
set -o xtrace
rm fib.pass-0000.core
rm fib.pass-0000.mlir
cabal build ghc-dump-util
cabal install ghc-dump-util --overwrite-policy=always
ghc-dump show  ~/work/mlir/musquared/hask-programs/dump/fib.pass-0000.cbor | tee fib.pass-0000.core
ghc-dump-mlir show ~/work/mlir/musquared/hask-programs/dump/fib.pass-0000.cbor  | tee fib.pass-0000.mlir
mlir-opt fib.pass-0000.mlir
