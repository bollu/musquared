#!/usr/bin/env bash
set -e
set -o xtrace
rm fib.pass-0000.core || true
rm fib.pass-0000.mlir || true
cabal build ghc-dump-util ghc-dump-mlir
# cabal install ghc-dump-util ghc-dump-mlir --overwrite-policy=always || true
cabal exec ghc-dump show  ~/work/mlir/musquared/hask-programs/dump/fib.pass-0000.cbor | tee fib.pass-0000.core
cabal exec ghc-dump-mlir show ~/work/mlir/musquared/hask-programs/dump/fib.pass-0000.cbor  | tee fib.pass-0000.mlir
mlir-opt fib.pass-0000.mlir -allow-unregistered-dialect
