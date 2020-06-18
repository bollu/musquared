module {
  func @main() {
    %x = constant 10 : i64
    %z = lean.printUnboxedInt(%x)         
    return 
  }

  
}


