// module Main where
// 
// $trModule :: Module
// 
// {- Core Size{terms=5 types=0 cos=0 vbinds=0 jbinds=0} -}
// $trModule =
//   APP(GHC.Types.Module
//     (APP(GHC.Types.TrNameS "main"#))
//     (APP(GHC.Types.TrNameS
//        "Main"#)))
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
//             APP(GHC.Num.+
//               @Int
//               GHC.Num.$fNumInt
//               (APP(Main.fib i))
//               (APP(Main.fib
//                  (APP(GHC.Num.-
//                     @Int
//                     GHC.Num.$fNumInt
//                     i
//                     (APP(GHC.Types.I# 1#)))))))
//           0# → APP(GHC.Types.I# 0#)
//           1# → APP(GHC.Types.I# 1#)
//         }
//     }
// }
// main :: IO ()
// 
// {- Core Size{terms=6 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(System.IO.putStrLn
//     (APP(GHC.Show.show
//        @Int
//        GHC.Show.$fShowInt
//        (APP(Main.fib
//           (APP(GHC.Types.I# 10#)))))))
// 
// main :: IO ()
// 
// {- Core Size{terms=2 types=1 cos=0 vbinds=0 jbinds=0} -}
// main =
//   APP(GHC.TopHandler.runMainIO
//     @() Main.main)main :: IO ()

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


