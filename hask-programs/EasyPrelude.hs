{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE MagicHash #-}
-- https://downloads.haskell.org/~ghc/8.2.1/docs/html/libraries/base-4.10.0.0/src/GHC-Base.html
-- https://hackage.haskell.org/package/base-4.3.1.0/docs/src/GHC-Int.html
--  https://hackage.haskell.org/package/ghc-prim-0.6.1/docs/GHC-Prim.html
import GHC.Prim

data Int = IntConstructor Int#

data IO a = MkIO 


-- | wired in 
addInt :: Int -> Int -> Int
addInt (IntConstructor i) (IntConstructor j) = IntConstructor (i +# j)

-- | wired in 
printInt :: Int -> IO ()
printInt = undefined

-- | loop
undefined :: a
undefined = undefined

-- | wired in
printRawInt :: Int# -> IO ()
printRawInt = undefined
