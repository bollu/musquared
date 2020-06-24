add_mlir_dialect_library(MLIRLean
        LeanDialect.cpp
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/
        DEPENDS
        MLIRLeanOpsIncGen
	LINK_LIBS PUBLIC
	MLIRIR
	)


# set(LIBS
#   MLIREDSC
#   MLIRIR
#   MLIRStandardOps
#   MLIRSupport
#   MLIRTransformUtils
#   LLVMSupport)
# 
# add_llvm_library(MLIRLean
#   LeanDialect.cpp
#   # LowerToLLVM.cpp
#   # LeanOps.cpp
#   # LeanTypes.cpp
# 
#   ADDITIONAL_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}/include/Dialect/Lean
# )
# 
# # Make sure that the TableGen generated files are up-to-date
# add_dependencies(MLIRLean
# 	MLIRLeanOpsIncGen
#   ${LIBS})
# target_link_libraries(MLIRLean ${LIBS})
# 
# # specify header directories
# target_include_directories(MLIRLean PUBLIC "${PROJECT_SOURCE_DIR}/include")
# target_include_directories(MLIRLean PUBLIC "${PROJECT_BINARY_DIR}/include")

