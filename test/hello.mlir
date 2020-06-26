module {
  func @main() -> !lean.IO<none> {
    %x = constant 10 : i64
    %z = lean.printUnboxedInt(%x)         
    return %z : !lean.IO<none>
  }

  
  func @succeededDominanceFreeScope() -> () {
  lean.dominance_free_scope {
  // %1 is not dominated by its definition.
    %z = lean.printUnboxedInt(%x)         
    %x = constant 10 : i64
    %y = lean.printUnboxedInt(%x)
    "lean.terminator"() : () -> ()     
  }
  return
  }
}


