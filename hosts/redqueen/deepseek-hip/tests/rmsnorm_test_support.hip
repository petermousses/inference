#include <hip/hip_runtime.h>

namespace {

__global__ void invalid_configuration_probe() {}

}  // namespace

// Test-only helper. Leaves the invalid launch status in HIP's per-thread
// last-error state so the RMSNorm wrapper can prove it won't enqueue on top.
extern "C" int redqueen_ds4_test_leave_pending_launch_error() {
  hipLaunchKernelGGL(invalid_configuration_probe, dim3(1), dim3(0), 0, nullptr);
  return static_cast<int>(hipPeekAtLastError());
}
