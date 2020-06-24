module {
  func @f() -> i64 {
    %y = call @g() : () -> i64
    return %y : i64
  }

  func @g() -> i64 {
    %z = call @f() : () -> i64
    return %z : i64
  }
  
}