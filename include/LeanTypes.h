#ifndef MLIR_DIALECT_LEAN_LEANTYPES_H
#define MLIR_DIALECT_LEAN_LEANTYPES_H

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
//#include <bits/stdint-intn.h>
#include <cstdint>
#include <limits>

namespace mlir {
namespace lean {

namespace LeanTypes {
enum Kind {
  Field = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  View,
  LAST_USED_PRIVATE_EXPERIMENTAL_0_TYPE = View
};
}

//===----------------------------------------------------------------------===//
// FieldType
//===----------------------------------------------------------------------===//

struct FieldTypeStorage;
class FieldType : public Type::TypeBase<FieldType, Type, FieldTypeStorage> {
public:
  using Base::Base;

  static FieldType get(MLIRContext *context, Type elementType,
                       ArrayRef<int> dimensions);

  /// Used to implement LLVM-style casts
  static bool kindof(unsigned kind) { return kind == LeanTypes::Field; }

  /// Return the type of the field elements.
  Type getElementType();
  /// Return the allocated dimensions of the field.
  ArrayRef<int> getDimensions();
};

//===----------------------------------------------------------------------===//
// ViewType
//===----------------------------------------------------------------------===//

struct ViewTypeStorage;
class ViewType : public Type::TypeBase<ViewType, Type, ViewTypeStorage> {
public:
  using Base::Base;

  static ViewType get(MLIRContext *context, Type elementType,
                      ArrayRef<int> dimensions);

  /// Used to implement LLVM-style casts.
  static bool kindof(unsigned kind) { return kind == LeanTypes::View; }

  /// Return the type of the view elements.
  Type getElementType();
  /// Return the allocated dimensions of the view.
  ArrayRef<int> getDimensions();
};

} // namespace lean
} // namespace mlir

#endif // MLIR_DIALECT_LEAN_LEANTYPES_H
