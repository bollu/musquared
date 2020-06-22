fib.core: fib.hs
	- rm fib.hi rm.o fib
	ghc -fplugin GhcDump.Plugin -dumpdir=dump -O fib.hs
	- rm fib
