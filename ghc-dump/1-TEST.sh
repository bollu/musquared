#!/usr/bin/env bash
cabal install ghc-dump-util --overwrite-policy=always && ghc-dump mlir ~/work/mlir/musquared/hask-programs/dump/fib.pass-0000.cbor 
