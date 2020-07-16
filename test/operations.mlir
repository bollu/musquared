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
//     @() Main.main)


module {
  func @arbitrary_mem_access(%I: memref<128xi32>, %M: memref<1024xf32>) {
    affine.for %i = 0 to 128 {
      // CHECK: %{{.*}} = affine.execute_region [{{.*}}] = ({{.*}}) : (memref<128xi32>, memref<1024xf32>) -> f32
      %ret = affine.for [%rI, %rM] = (%I, %M) : (memref<128xi32>, memref<1024xf32>) -> f32 {
        %idx = affine.load %rI[%i] : memref<128xi32>
        %index = index_cast %idx : i32 to index
        %v = affine.load %rM[%index]: memref<1024xf32>
        return %v : f32
      }
    }
    return
  }
}
