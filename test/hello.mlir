module {
  func @main() -> !lean.IO<none> {
    %x = constant 10 : i64
    %z = lean.printUnboxedInt(%x)         
    return %z : !lean.IO<none>
  }

  
}


