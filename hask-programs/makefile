.PHONY: fib.core shotest.core
fib.core: fib.hs
	- rm fib.hi rm.o fib
	ghc -fplugin GhcDump.Plugin -dumpdir=dump -O fib.hs
	- rm fib

showtest.core: showtest.hs
	- rm showtest.hi rm.o showtest
	ghc -fplugin GhcDump.Plugin -dumpdir=dump -O showtest.hs
	- rm showtest

XXXX.core: XXXX.hs
	- rm XXXX.hi rm.o XXXX
	ghc -fplugin GhcDump.Plugin -dumpdir=dump -O XXXX.hs
	- rm XXXX
