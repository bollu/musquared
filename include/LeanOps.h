#ifndef MLIR_DIALECT_LEAN_LEANOPS_H
#define MLIR_DIALECT_LEAN_LEANOPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Interfaces/SideEffects.h"
#include <numeric>

namespace mlir {
namespace lean {

namespace {

/// Helper methods to convert between vectors and attributes
ArrayAttr convertSmallVectorToArrayAttr(ArrayRef<int64_t> vector,
                                        MLIRContext *context) {
  SmallVector<Attribute, 3> result;
  for (int64_t value : vector) {
    result.push_back(IntegerAttr::get(IntegerType::get(64, context), value));
  }
  return ArrayAttr::get(result, context);
}
SmallVector<int64_t, 3> convertArrayAttrToSmallVector(const ArrayAttr &array,
                                                      MLIRContext *context) {
  SmallVector<int64_t, 3> result;
  for (auto &attr : array) {
    result.push_back(attr.cast<IntegerAttr>().getValue().getSExtValue());
  }
  return result;
}
} // namespace

/// Retrieve the class declarations generated by TableGen
#define GET_OP_CLASSES
#include "LeanOps.h.inc"
} // namespace lean
} // namespace mlir

#endif // MLIR_DIALECT_LEAN_LEANOPS_H
