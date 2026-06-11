// Forward declaration callable from plain C++ translation units.
// Mirrors sm120/decode/sparse_decode.h for the prefill path.
#pragma once

#include "common/params.h"

namespace dsv4_kernel {
namespace sm120 {

void launch_dsv4_sparse_prefill(const SparseAttnPrefillParams &params);

}  // namespace sm120
}  // namespace dsv4_kernel
