module {
  func @arbitrary_mem_access(%I: memref<128xi32>, %M: memref<1024xf32>) {
    affine.for %i = 0 to 128 {
      // CHECK: %{{.*}} = affine.execute_region [{{.*}}] = ({{.*}}) : (memref<128xi32>, memref<1024xf32>) -> f32
      %ret = affine.for [%rI, %rM] = (%I, %M) : (memref<128xi32>, memref<1024xf32>) -> f32 {
        %idx = affine.load %rI[%i] : memref<128xi32>
        %index = index_cast %idx : i32 to index
        %v = affine.load %rM[%index]: memref<1024xf32>
        return %v : f32
      }
    }
    return
  }
}