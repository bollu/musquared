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

module @lean_mod {
  // func @print_int(%arg: i32): !lean.IO<none>;  

  func @multiply_transpose(
                 %arg0: !lean.struct<tensor<*xf64>, tensor<*xf64>>, 
  						   %arg1: !lean.simple,
                 %arg2: !lean.IO<none>) {          
    %x = constant 10 : i64         
    %y = constant 20: i64
    %z = constant 30 : i32
    // %boxx = lean.box_int %x

    // %awesomeadd_stringy = "lean.awesome_add"(%x, %y) : (i64, i64) -> i64 
    // %awesomeadd_call = lean.awesome_add %x, %y : i64
    // lean.awesome_add %x, %y 
    %out = lean.awesome_add %x, %y
    %print_stringy = "lean.print_unboxed_int"(%x)  : (i64) -> !lean.IO<none> loc("example/file/path":1:1)
    %print_incorrect = "lean.print_unboxed_int"(%x)  : (i64) -> f64 loc("example/file/path":1:1)
    %foo = "lean.bar"(%x)  : (i64) -> f64 loc("example/file/path":1:1)
    %print_boxx = lean.case(%x : i64, i64, [%arg]  {
      %a = constant 20 : i64
      lean.return %arg : i64
    }, [%arg4] {
      %a = constant 30 : i64
      lean.return %arg4 : i64
    }) // , i64)
    // lean.case(%box  i64) 
    // lean.print_unboxed_int %x : i32 loc("example/file/path":1:1)
    return
  }

  
}
