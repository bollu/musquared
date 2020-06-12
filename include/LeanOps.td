#ifndef LEAN_OPS
#define LEAN_OPS

// https://mlir.llvm.org/docs/OpDefinitions/

include "mlir/Interfaces/SideEffects.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// Lean dialect definition
//===----------------------------------------------------------------------===//

def Lean_Dialect : Dialect {
    let name = "lean";

    let description = [{
        A lean dialect in MLIR.
    }];

    let cppNamespace = "lean";
}

//===----------------------------------------------------------------------===//
// Lean type definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lean op definition
//===----------------------------------------------------------------------===//

// Base class for all Lean ops.
// class Lean_Op<string mnemonic, list<OpTrait> traits = []> :
//     Op<Lean_Dialect, mnemonic, traits> {

//     // For each Lean op, the following static functions need to be defined in
//     // LeanOps.cpp:
//     //
//     // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
//     //                                               OperationState &state);
//     // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
//     // * static LogicalResult verify(<op-c++-class-name> op)
//     let parser = [{ return ::parse$cppClass(parser, result); }];
//     let printer = [{ ::print(*this, p); }];
//     let verifier = [{ return ::verify(*this); }];
// }

class Lean_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Lean_Dialect, mnemonic, traits>;

// Provide a definition for the Toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def Lean_StructType :
    Type<CPred<"$_self.isa<StructType>()">, "Lean struct type">;

def Lean_SimpleType :
    Type<CPred<"$_self.isa<SimpleType>()">, "Lean simple type">;

def Lean_IOType :
    Type<CPred<"$_self.isa<IOType>()">, "Lean IO type">;

def Lean_BoxedI64Type :
    Type<CPred<"$_self.isa<BoxedI64Type>()">, "Lean Boxed I64 type">;

def PrintUnboxedIntOp : Lean_Op<"printUnboxedInt"> {

  let arguments = (ins I64:$input);
  let results = (outs Lean_IOType:$result);

  let verifier = [{ ::verify$cppClass(this); }];
}

def AwesomeAddOp : Lean_Op<"awesome_add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "awesomeadd" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I64:$lhs, I64:$rhs);
  let results = (outs I64:$r);

  // Allow building an AddOp with from the two input operands.
//   let builders = [
//     OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
//   ];

  // custom parser. WART: was unable to find this in toy???
  let parser = [{ return ::parse$cppClass(parser, result); }];
  // WART, LMAO: WTF, the _variable names_ are hard-coded, so we need
  // to name this `p`.
  let printer = [{ ::print$cppClass(this, p); }];

}

/*
/// SPV_I32Vec4
/// Consider a stripped definition of `toy.print` here.
/// TOY tutorial is woefully incomplete. It doesn't teach you enough of ODS
/// to be useful. Fuck this, go read SPIR-V definitions.
def PrintUnboxedIntOp : Lean_Op<"print_unboxed_int"> {
  let arguments = (ins I32:$input);
  let results = (outs 
                    LeanIO:$result
                );

  // Divert the printer and parser to static functions in our .cpp
  // file that correspond to 'print' and 'printPrintOp'. 'printer' and 'parser'
  // here correspond to an instance of a 'OpAsmParser' and 'OpAsmPrinter'. More
  // details on these classes is shown below.
  let printer = [{ return ::print(printer, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
*/
#endif // LEAN_OPS
