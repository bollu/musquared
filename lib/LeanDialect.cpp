#include "LeanDialect.h"
#include "LeanOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <string>

using namespace mlir;
using namespace mlir::lean;

//===----------------------------------------------------------------------===//
// Lean Dialect
//===----------------------------------------------------------------------===//

LeanDialect::LeanDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {


  // addInterfaces<ToyInlinerInterface>();
  addTypes<StructType, SimpleType, IOType, BoxedI64Type>();
  // addOperations<CppPrintUnboxedIntOp>();
  addOperations<
#define GET_OP_LIST
#include "LeanOps.cpp.inc"
#undef GET_OP_LIST
      >();


  // addTypes<SimpleType>();
  
  // Allow Lean operations to exist in their generic form
  allowUnknownOperations();
}

namespace mlir {
namespace lean {
namespace detail {
/// This class represents the internal storage of the Toy `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
}; // end struct StructTypeStorage

struct IOTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = mlir::Type;

  /// A constructor for the type storage instance.
  IOTypeStorage(mlir::Type elementType) : elementType(elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return mlir::hash_value((mlir::Type)key);
    // return llvm::hash_value(key.impl);

    // return llvm::getHashValue(key);
    // return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(mlir::Type elementType) {
    return elementType;
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static IOTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    // mlir::Type elementType = allocator.copyInto(key);

    // TODO: understand if I need to create a copy of `elementType`
    //       as well?
  
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<IOTypeStorage>()) IOTypeStorage(key);
  }

  /// The following field contains the element types of the struct.
  mlir::Type elementType;
}; // end struct IOTypeStorage

} // end namespace detail
} // end namespace lean
} // end namespace mlir

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {

  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first two parameters are the context to unique in and the
  // kind of the type. The parameters after the type kind are forwarded to the
  // storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();

  assert(ctx && "invalid ctx");
  StructType st = Base::get(ctx, LeanTypes::Struct, elementTypes);
  return st;
}


/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}


IOType IOType::get(mlir::Type elementType) {

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first two parameters are the context to unique in and the
  // kind of the type. The parameters after the type kind are forwarded to the
  // storage instance.
  mlir::MLIRContext *ctx = elementType.getContext();
  assert(ctx && "invalid ctx");
  IOType st = Base::get(ctx, LeanTypes::IO, elementType);
  return st;

}


mlir::Type IOType::getElementType() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementType;
}



/// Parse an instance of a type registered to the toy dialect.
mlir::Type LeanDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  llvm::errs () << "current location:\nvvvvv\n" << parser.getCurrentLocation().getPointer() << "\n^^^^^\n";

  
  if(!strcmp(parser.getCurrentLocation().getPointer(), "simple")) { 
      parser.parseKeyword("simple");
      SimpleType t = SimpleType::get(parser.getBuilder().getContext());
      llvm::errs() << "\tParsed a rawtype!: |" << t.getKind() << " ~= " << LeanTypes::Simple <<  "|\n";
      return t;
  }

  if (succeeded(parser.parseOptionalKeyword("BoxedI64"))) {  
    return BoxedI64Type::get(parser.getBuilder().getContext());
  }
  
  // else {
  //   llvm::errs() << "\tFailed at parsing a simple type\n";
  // }

Type t;
if (succeeded(parser.parseOptionalKeyword("IO"))) {  
    parser.parseLess(); parser.parseType(t); parser.parseGreater();
    return IOType::get(t);
}

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();


 
  // SimpleType st;
  // if (parser.parseType<SimpleType>(st)) {
  //   return st;
  // }

  // assert(false && "trying to parse");
  

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<mlir::TensorType>() &&
        !elementType.isa<StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}

/// Print an instance of a type registered to the toy dialect.
void LeanDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {

  if(StructType structType = type.dyn_cast<StructType>()) {
    // Print the struct type according to the parser format.
    printer << "struct<";
    mlir::interleaveComma(structType.getElementTypes(), printer);
    printer << '>';
  } else if(SimpleType simpleType = type.dyn_cast<SimpleType>()) {
    printer << "simple";
  } else if (IOType ioType = type.dyn_cast<IOType>()) {
    printer << "IO<" << ioType.getElementType() << ">";
  }
  else if (type.isa<BoxedI64Type>()) {
    printer << "BoxedI64";
  } 
  else {
    llvm::errs() << "unknown type:\n"; // |" << type << "|\n";
    llvm::errs() << "(DIALECT:" << type.getDialect().getNamespace() << 
        " | KIND: " << type.getKind() << ")\n";
    printer << "UNK: " << type << "\n";
    assert(false);
  }
}


// much magic, very wow :(
#define GET_OP_CLASSES
#include "LeanOps.cpp.inc"
#undef GET_OP_CLASSES 

// this needs to be here so it can see add
// void AwesomeAddOp::build(mlir::Builder *builder, mlir::OperationState &state,
//                   mlir::Value lhs, mlir::Value rhs) {
//   state.addTypes(builder->getIntegerType(64));  // IntegerType::get(builder->getF64Type()));
//   state.addOperands({lhs, rhs});
// }


namespace mlir {
namespace lean {
mlir::ParseResult parseAwesomeAddOp(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  // Parse the input operand, the attribute dictionary, and the type of the
  // input.
  mlir::OpAsmParser::OperandType inputOperand1;
  mlir::OpAsmParser::OperandType inputOperand2;
  if (parser.parseOperand(inputOperand1) ||
      parser.parseComma() ||
      parser.parseOperand(inputOperand2))
    return mlir::failure();

  // Resolve the input operand to the type we parsed in.
  if (parser.resolveOperand(inputOperand1, parser.getBuilder().getIntegerType(64), result.operands))
    return mlir::failure();

  if (parser.resolveOperand(inputOperand2, parser.getBuilder().getIntegerType(64), result.operands))
    return mlir::failure();

  llvm::outs () << "parsed AwesomeAdd: " << inputOperand1.name << " | " << inputOperand2.name << "\n";
  // assert(false);

  // Where is the result.addType API?
  result.addTypes({parser.getBuilder().getIntegerType(64) });

  return mlir::success();
}
void printAwesomeAddOp(AwesomeAddOp *op, mlir::OpAsmPrinter &p) {
  p << "lean.awesome_add " << op->getOperand(0) << ", " << op->getOperand(1);
}



LogicalResult verifyPrintUnboxedIntOp(PrintUnboxedIntOp *op) {
  // GG. When the fuck is verify() called?
  assert(false && "unable to verify unboxed int");
    if (IOType t = op->getResult().getType().dyn_cast<IOType>()) {
    return success();
  }
  return failure();
}


} // end namespace lean
} // end namespace mlir