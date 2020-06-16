module {
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
    %print_boxx = lean.case(%x : i64, f64, [%arg] {
      %a = constant 20 : i64
      lean.return %arg : i64
    }) // , i64)
    // lean.case(%box  i64) 
    // lean.print_unboxed_int %x : i32 loc("example/file/path":1:1)
    return
  }

  
}
