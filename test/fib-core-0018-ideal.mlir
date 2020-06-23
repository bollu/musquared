{- Demand analysis: -}
module Main where

$trModule :: Addr#

{- Core Size{terms=1 types=0 cos=0 vbinds=0 jbinds=0} -}
$trModule = "main"#

$trModule :: TrName

{- Core Size{terms=2 types=0 cos=0 vbinds=0 jbinds=0} -}
$trModule =
  GHC.Types.TrNameS $trModule

$trModule :: Addr#

{- Core Size{terms=1 types=0 cos=0 vbinds=0 jbinds=0} -}
$trModule = "Main"#

$trModule :: TrName

{- Core Size{terms=2 types=0 cos=0 vbinds=0 jbinds=0} -}
$trModule =
  GHC.Types.TrNameS $trModule

$trModule :: Module

{- Core Size{terms=3 types=0 cos=0 vbinds=0 jbinds=0} -}
$trModule =
  GHC.Types.Module
    $trModule $trModule

rec {
$wfib :: Int# -> Int#

{- Core Size{terms=21 types=4 cos=0 vbinds=0 jbinds=0} -}
$wfib =
  λ ww →
    case ww of ds {
      DEFAULT →
        case $wfib ds of ww {
          DEFAULT →
            case $wfib (GHC.Prim.-# ds 1#)
            of ww {
              DEFAULT → GHC.Prim.+# ww ww
            }
        }
      0# → 0#
      1# → 1#
    }
}
main :: String

{- Core Size{terms=13 types=12 cos=0 vbinds=0 jbinds=0} -}
main =
  case $wfib 10# of ww {
    DEFAULT →
      case GHC.Show.$wshowSignedInt
             0# ww (GHC.Types.[] @Char)
      of ww4 {
        (#,#) ww5 ww6 →
          GHC.Types.: @Char ww5 ww6
      }
  }

main :: IO ()

{- Core Size{terms=4 types=0 cos=0 vbinds=0 jbinds=0} -}
main =
  GHC.IO.Handle.Text.hPutStr'
    GHC.IO.Handle.FD.stdout
    main
    GHC.Types.True

main :: State# RealWorld ->
        (#,#) (TupleRep ([] RuntimeRep)) LiftedRep (State# RealWorld) ()

{- Core Size{terms=2 types=1 cos=0 vbinds=0 jbinds=0} -}
main =
  GHC.TopHandler.runMainIO1
    @() Main.main

main :: IO ()

{- Core Size{terms=1 types=0 cos=3 vbinds=0 jbinds=0} -}
main = main

