#include "LeanDialect.h"
#include "LeanOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
// #include "mlir/Support/STLExtras.h"
// #include "llvm/ADT/None.h"
// #include "llvm/ADT/StringRef.h"
// #include "llvm/Support/ErrorHandling.h"
#include <algorithm>
// #include <bits/stdint-intn.h>
#include <cstddef>
#include <string>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"


using namespace mlir;
using namespace mlir::lean;

// static DialectRegistration<LeanDialect> StandardOps;


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

  addOperations<CaseOp>();
  addOperations<ReturnOp>();
  // addOperations>AltOp>();
  

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

    // Check that the type is either a T    ensorType or another StructType.
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
    // mlir::interleaveComma(structType.getElementTypes(), printer);
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


mlir::ParseResult parseGetIOTokenOp(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  result.addTypes({IOType::get(parser.getBuilder().getNoneType())});
  return mlir::success();
}

void printGetIOTokenOp(GetIOTokenOp *op, mlir::OpAsmPrinter &p) {
  p << "lean.get_IO_token";
}




// https://reviews.llvm.org/D72223
ParseResult CaseOp::parse(mlir::OpAsmParser &parser, OperationState &result) {

  mlir::OpAsmParser::OperandType scrutinee; 
  Type scrutineeTy;
  Type retty;
  SmallVector<Value, 1> resolvedScrutinee;


  llvm::outs() << "vvvPARSING CASEvvvv\n";

  // case(CASE-SCRUTINEE : SCURUTINEETY, RETTY, ALTS)
  // or
  // case (CASE-SCRUTINEE, RETTY, ALTS) 

  if (parser.parseLParen()) return failure();
  llvm::outs () << "\t-(\n";

  
  // CASE-SCRUTINEE
  if (parser.parseOperand(scrutinee)) return failure();

  // if (parser.parseRParen()) return failure();
  // return success();

  llvm::outs() << "\t-" << scrutinee.name << "\n";

  // if (parser.parseComma()) return failure();

  if(parser.parseColon()) return failure();

  // scrutineeTy
  if (parser.parseType(scrutineeTy)) return failure();
  llvm::outs() << "\t-" << scrutineeTy << "\n";
  result.addTypes({scrutineeTy});

  parser.resolveOperand(scrutinee, scrutineeTy, resolvedScrutinee);
  result.addOperands(resolvedScrutinee);
  
  if(parser.parseComma()) return failure();
  

  if(parser.parseType(retty)) return failure();
  // result.addTypes({retty});
  
  // )
  if(succeeded(parser.parseOptionalRParen())) {
    llvm::outs () << "\t-)\n";
    return success();
  } else if (parser.parseComma()) {
    llvm::outs() << __LINE__ << "\n";
    return failure();
  }

  llvm::outs() << __LINE__ << "\n";

  // ARG, 
  do {
    llvm::outs() << __LINE__ << "\n";
    SmallVector<mlir::OpAsmParser::OperandType, 4> regionArgs;
    parser.parseRegionArgumentList(regionArgs, mlir::OpAsmParser::Delimiter::OptionalSquare);
    llvm::outs() << __LINE__ << "\n";
    llvm::outs() << "\t- #region args:" << regionArgs.size() << "\n";
    if (regionArgs.size()) llvm::outs() << "\t-region arg[0]:  " << regionArgs[0].name << "\n";
    llvm::outs() << __LINE__ << "\n";
    Region *r = result.addRegion();
    
    // can I validate here? worth a shot.
   
    llvm::outs() << __LINE__ << "\n";
    
    // parser.parseRegion(*r, regionArgs, {scrutineeTy});
    parser.parseRegion(*r, regionArgs, {scrutineeTy});
    llvm::outs() << __LINE__ << "\n";
    llvm::outs() << __LINE__ << "\n";
    for(Block &BB : r->getBlocks()) {
      llvm::outs() << __LINE__ << "\n";
      assert(BB.getTerminator() != nullptr);
      if (ReturnOp retop = dyn_cast<ReturnOp>(BB.getTerminator())) {
        llvm::outs() << __LINE__ << "\n";
        // llvm::errs() << "-retop: "; retop.print(llvm::errs());
        // llvm::errs() << "\n-retop operand: "; retop.getOperand().print(llvm::errs());
        // llvm::errs() << "\n-retop operand type: "; retop.getOperand().getType(); 
        if(retop.getOperandType() != retty) {
          auto diagnostic = parser.emitError(parser.getCurrentLocation(), "mismatched alt type and return type");
          // diagnostic << "alt type ["; 
          return diagnostic;
        }
      }
  }
  llvm::outs() << __LINE__ << "\n";

  } while (succeeded(parser.parseOptionalComma()));

  llvm::outs() << __LINE__ << "\n";

  if(parser.parseRParen()) {
    llvm::outs() << __LINE__ << "\n";
    return failure();
  }
  
  llvm::outs() << __LINE__ << "\n";
  return success();

  // either we have:
  // 1. )
  // 2. ALT , ALTS
}

LogicalResult CaseOp::verify() {


  for(int i = 0; i < getNumAlts(); i++) {
    for(auto &BB : getAlt(i).getBlocks()) {
      assert(BB.getTerminator() != nullptr);
      if (ReturnOp retop = dyn_cast<ReturnOp>(BB.getTerminator())) {
        llvm::errs() << "-retop: "; retop.print(llvm::errs());
        llvm::errs() << "\n-retop operand: "; retop.getOperand().print(llvm::errs());
        llvm::errs() << "\n-retop operand type: "; retop.getOperand().getType(); 

        llvm::errs() << "\n-our return type: " << getReturnType() << "\n";
        assert(retop.getOperandType() == this->getReturnType());
      }
      // (*BB.getTerminator())
      // llvm::errs() << "BB:\n" << *BB.getTerminator() << "\n";

      // }
    }
  }

}

ParseResult ReturnOp::parse(mlir::OpAsmParser &parser, OperationState &result) {
  mlir::OpAsmParser::OperandType retoperand; 
  Type retty;
  if(parser.parseOperand(retoperand)) return failure();
  if (parser.parseColon()) return failure();
  if (parser.parseType(retty)) return failure();

  // how do I pass the type inference information?
  SmallVector<Value, 1> retval;
  parser.resolveOperand(retoperand, retty, retval);
  result.addOperands({retval });
  // result.addOperands(retval);
  return success();
}



LogicalResult verifyPrintUnboxedIntOp(PrintUnboxedIntOp *op) {
  // GG. When the fuck is verify() called?
  assert(false && "unable to verify unboxed int");
    if (IOType t = op->getResult().getType().dyn_cast<IOType>()) {
    return success();
  }
  return failure();
}

void printAwesomeAddOp(AwesomeAddOp *op, mlir::OpAsmPrinter &p) {
  p << "lean.awesome_add " << op->getOperand(0) << ", " << op->getOperand(1);
}


void printPrintUnboxedIntOp(PrintUnboxedIntOp *op, mlir::OpAsmPrinter &p) {
  p << "lean.printUnboxedInt (" << op->input() << ")";
}


mlir::ParseResult parsePrintUnboxedIntOp(mlir::OpAsmParser &parser,
                                      mlir::OperationState &result) {
  mlir::OpAsmParser::OperandType input;
  SmallVector<Value, 1> vs;
  if (parser.parseLParen() ||
      parser.parseOperand(input) ||
      parser.parseRParen() ||
      parser.resolveOperand(input, parser.getBuilder().getIntegerType(64), vs))
    return mlir::failure();
  
  result.addTypes({IOType::get(parser.getBuilder().getNoneType())});
  result.addOperands(vs);
  return success();

}

// REWRITING


void GetIOTokenOp::build(OpBuilder &b, OperationState &state) {
  return GetIOTokenOp::build(b, state, IOType::get(b.getNoneType()));
};


namespace {
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(lean::PrintUnboxedIntOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    {
      llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\nvvvvv";
      llvm::errs() << *op->getParentOp() << "\n";
      llvm::errs() << "^^^^^\n";
      GetIOTokenOp iotok = rewriter.create<GetIOTokenOp>(op->getLoc());
      rewriter.replaceOp(op, iotok.getResult());
      return success();

    }

    

    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    // assert(false && "died in rewrite");
    // return success();
    // (*op->operand_begin()).cast<mlir::Float
    // IntegerType i = (*op->operand_begin()).cast<mlir::IntegerType>();
    
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    // auto memRefType = (*op->operand_type_begin()).cast<>();
    // auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";


    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule, llvmDialect);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%d \0", 4), parentModule,
        llvmDialect);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule, llvmDialect);

    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";

    // Create a loop for each of the dimensions within the shape.
    /*
    SmallVector<Value, 4> loopIvs;
    
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
      auto upperBound = rewriter.create<ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }
    */
    auto printOp = cast<lean::PrintUnboxedIntOp>(op);

    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    Value input = printOp.input();
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    
    // Generate a call to printf for the current element of the loop.
    // auto elementLoad = rewriter.create<LoadOp>(loc, printOp.input(), loopIvs);
    CallOp call = rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                            ArrayRef<Value>({formatSpecifierCst, input }));
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    llvm::errs() << "loc: " << loc << " | op: " << *op << " | input: " << input << "\n";
    llvm::errs() << "call :"  << call << "\n";
    // rewriter.replaceOp(op, {call});
    // Notify the rewriter that this operation has been removed.
    
    GetIOTokenOp iotok = rewriter.create<GetIOTokenOp>(loc);
    // lmao, the fact that I need a cast<Value>(...) here is
    // definitely WTF++ for me. You get punished for not using auto
    // everywhere.
    // rewriter.replaceOpWithNewOp<GetIOTokenOp>(op);

    llvm::errs() << "iotok: " << iotok << " |iotokResult: " << iotok.getResult() << "\n";
    // rewriter.replaceOp(op, iotok.getOperation()->getOpResult(0));
    // rewriter.eraseOp(printOp);
    rewriter.replaceOp(op, iotok.getResult());

    // rewriter.replaceOpWithNewOp<GetIOTokenOp>(op);
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    llvm::errs() << "parent:vvvv\n"  << *call.getParentOp() << "\n^^^^^^\n";
    // return failure();
    return success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get("printf", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI8PtrTy,
                                                    /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get("printf", context);
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module,
                                       LLVM::LLVMDialect *llvmDialect) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), globalPtr,
        ArrayRef<Value>({cst0, cst0}));
  }
};
} // end anonymous namespace



namespace {
struct LeanToLLVMLoweringPass
    : public PassWrapper<LeanToLLVMLoweringPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
};
} // end anonymous namespace

void LeanToLLVMLoweringPass::runOnOperation() {
  
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  LLVMConversionTarget target(getContext());
  target.addLegalDialect<lean::LeanDialect, mlir::StandardOpsDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp, FuncOp>();
  // target.addLegalOp<FunctionOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  // LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  OwningRewritePatternList patterns;
  // populateAffineToStdConversionPatterns(patterns, &getContext());
  // populateLoopToStdConversionPatterns(patterns, &getContext());
  // populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  // patterns.insert<PrintOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  
  auto module = getOperation();
  // if (failed(applyFullConversion(module, target, patterns)))
  //   signalPassFailure();

  if (failed(applyPartialConversion(module, target, patterns))) {
    signalPassFailure();
  };
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LeanToLLVMLoweringPass>();
}


namespace {
struct LowerPrintPass
    : public PassWrapper<LowerPrintPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void LowerPrintPass::runOnFunction() {
  auto function = getFunction();


// assert(false )
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<LeanDialect, StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`.
  // target.addIllegalDialect<LeanDialect>();
  // target.addLegalOp<PrintUnboxedIntOp>();
  target.addIllegalOp<PrintUnboxedIntOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  OwningRewritePatternList patterns;
  patterns.insert<PrintOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns))) {
    llvm::errs() << __FUNCTION__ << ":" << __LINE__ << "\n";
    llvm::errs() << "fn\nvvvv\n";
    getFunction().dump() ;
    llvm::errs() << "\n^^^^^\n";
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createLowerPrintPass() {
  return std::make_unique<LowerPrintPass>();
}



} // end namespace lean
} // end namespace mlir
