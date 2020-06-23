// module Main where
// 
// $trModule :: Module
// 
// {- Core Size{terms=5 types=0 cos=0 vbinds=0 jbinds=0} -}
// $trModule =
//   GHC.Types.Module
//     (GHC.Types.TrNameS "main"#)
//     (GHC.Types.TrNameS "Main"#)
// 
// rec {
// fib :: Int -> Int
// 
// {- Core Size{terms=23 types=6 cos=0 vbinds=0 jbinds=0} -}
// fib =
//   λ i →
//     case i of wild {
//       I# ds →
//         case ds of ds {
//           DEFAULT →
//             GHC.Num.+
//               @Int
//               GHC.Num.$fNumInt
//               (Main.fib i)
//               (Main.fib
//                  (GHC.Num.-
//                     @Int
//                     GHC.Num.$fNumInt
//                     i
//                     (GHC.Types.I# 1#)))
//           0# → GHC.Types.I# 0#
//           1# → GHC.Types.I# 1#
//         }
//     }
// }
// main :: IO ()
// 
// {- Core Size{terms=6 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   System.IO.putStrLn
//     (GHC.Show.show
//        @Int
//        GHC.Show.$fShowInt
//        (Main.fib (GHC.Types.I# 10#)))
// 
main :: IO ()

{- Core Size{terms=2 types=1 cos=0 vbinds=0 jbinds=0} -}
main =
  GHC.TopHandler.runMainIO
    @() Main.main


module {
func @fib(i: core.Int) -> core.Int {
  return lean.case(%i : Ihash, Ihash, 
      "Ihash" -> [%ds : i64] {
         return lean.case(%ds : i64, I, 
                 0 -> [] { core.construct(I, Ihash, 0) }
                 1 -> [] { core.construct(I, Ihash, 1) })
      }
}

  func @main() -> !core.IO<none> {
     // typeclass: lookup a typeclass
     return core.putStrLn (core.typeclass(core.show, core.Int)((fib (core.construct(I, Ihash, constant 10)))))
  }
}


