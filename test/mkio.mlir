module {
  // func @print_int(%arg: i32): !lean.IO<none>;  
  func @multiply_transpose(
                 %arg0: !lean.struct<tensor<*xf64>, tensor<*xf64>>, 
  						   %arg1: !lean.simple,
                 %arg2: !lean.IO<none>) {          
    %x = constant 10 : i32         
    %t_tensor = "toy.print_int"(%x)  : (i32) -> !lean.IO<none> loc("example/file/path":12:1)
    return
  }
}
