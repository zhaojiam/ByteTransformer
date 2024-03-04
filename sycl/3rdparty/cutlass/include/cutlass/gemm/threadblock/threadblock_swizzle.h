/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Implements several possible threadblock-swizzling functions mapping blockIdx to 
      GEMM problems.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxX(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_local_id(2);
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxY(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_local_id(1);
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeThreadIdxZ(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_local_id(0);
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxX(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_group(2);
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxY(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_group(1);
}

/// Helper to rematerialize block Idx. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockIdxZ(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_group(0);
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimX(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_local_range(2);
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimY(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_local_range(1);
}

/// Helper to rematerialize block Dim. Reduces register liveness.
CUTLASS_DEVICE
int RematerializeBlockDimZ(const sycl::nd_item<3> &item_ct1) {
  return item_ct1.get_local_range(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
template <int N = 1>
struct GemmIdentityThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  GemmIdentityThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  /// *Gemm* problem size: gemm(M, N, K)
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv2d problem size: conv_operator(NPQK, NHWC, KRSC)
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv2dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* Conv3d problem size: conv_operator(NZPQK, NDHWC, KTRSC)
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    cutlass::conv::Operator conv_operator,
    cutlass::conv::Conv3dProblemSize const &problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    gemm::GemmCoord implicit_gemm_problem_size = 
    cutlass::conv::implicit_gemm_problem_size(conv_operator, problem_size);

    return get_tiled_shape(
      implicit_gemm_problem_size, tile_size, split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  sycl::range<3> get_grid_shape(GemmCoord tiled_shape) const {
    int tile = 1 << get_log_tile(tiled_shape);
    return sycl::range<3>(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile,
                          tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    auto n = tiled_shape.n();
    // Thresholds picked so that it doesn't cause too many no-op CTAs
    if (N >= 8 && n >= 6)
      return 3;
    else if (N >= 4 && n >= 3)
      return 2;
    else if (N >= 2 && n >= 2)
      return 1;
    else
      return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile, const sycl::nd_item<3> &item_ct1) const {
    int block_idx_x = RematerializeBlockIdxX(item_ct1);
    int block_idx_y = RematerializeBlockIdxY(item_ct1);
    int block_idx_z = RematerializeBlockIdxZ(item_ct1);

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, const sycl::nd_item<3> &item_ct1) const {

    int const kTile = N;
    int block_idx_x = RematerializeBlockIdxX(item_ct1);
    int block_idx_y = RematerializeBlockIdxY(item_ct1);

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ(item_ct1)};

    return GemmCoord{(block_idx_x / kTile), (block_idx_y * kTile) + (block_idx_x % kTile),
                     RematerializeBlockIdxZ(item_ct1)};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
struct GemmHorizontalThreadblockSwizzle {

  CUTLASS_HOST_DEVICE
  GemmHorizontalThreadblockSwizzle() { }

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  sycl::range<3> get_grid_shape(GemmCoord tiled_shape) const {
    return sycl::range<3>(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, const sycl::nd_item<3> &item_ct1) const {
    return GemmCoord{RematerializeBlockIdxY(item_ct1), RematerializeBlockIdxX(item_ct1),
                     RematerializeBlockIdxZ(item_ct1)};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMMs
struct GemmBatchedIdentityThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int batch_count) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      batch_count % (1 << 16));
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  sycl::range<3> get_grid_shape(GemmCoord tiled_shape) const {
    return sycl::range<3>(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, const sycl::nd_item<3> &item_ct1) const {
    return GemmCoord{RematerializeBlockIdxX(item_ct1), RematerializeBlockIdxY(item_ct1),
                     RematerializeBlockIdxZ(item_ct1)};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile, const sycl::nd_item<3> &item_ct1) const {
    int block_idx_x = RematerializeBlockIdxX(item_ct1);
    int block_idx_y = RematerializeBlockIdxY(item_ct1);
    int block_idx_z = RematerializeBlockIdxZ(item_ct1);

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Gets the batch index
  CUTLASS_DEVICE
  int get_batch_idx(const sycl::nd_item<3> &item_ct1) const {
    return RematerializeBlockIdxZ(item_ct1);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for split-K GEMMs
template <int N = 1>
struct GemmSplitKIdentityThreadblockSwizzle {

  int const kTile = N;

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    auto n = tiled_shape.n();
    // Thresholds picked so that it doesn't cause too many no-op CTAs
    if (N >= 8 && n >= 6)
      return 3;
    else if (N >= 4 && n >= 3)
      return 2;
    else if (N >= 2 && n >= 2)
      return 1;
    else
      return 0;
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  sycl::range<3> get_grid_shape(GemmCoord tiled_shape) const {
    int tile = 1 << get_log_tile(tiled_shape);
    return sycl::range<3>(tiled_shape.m() * tile, (tiled_shape.n() + tile - 1) / tile,
                          tiled_shape.k());
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile, const sycl::nd_item<3> &item_ct1) const {
    int block_idx_x = RematerializeBlockIdxX(item_ct1);
    int block_idx_y = RematerializeBlockIdxY(item_ct1);
    int block_idx_z = RematerializeBlockIdxZ(item_ct1);

    return GemmCoord{(block_idx_x >> log_tile),  //
                     (block_idx_y << log_tile) + ((block_idx_x) & ((1 << (log_tile)) - 1)),
                     block_idx_z};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, const sycl::nd_item<3> &item_ct1) const {

    int const kTile = N;
    int block_idx_x = RematerializeBlockIdxX(item_ct1);
    int block_idx_y = RematerializeBlockIdxY(item_ct1);

    if ((tiled_shape.m() < kTile) || (tiled_shape.n() < kTile))
      return GemmCoord{block_idx_x, block_idx_y, RematerializeBlockIdxZ(item_ct1)};

    return GemmCoord{(block_idx_x / kTile), (block_idx_y * kTile) + (block_idx_x % kTile),
                     RematerializeBlockIdxZ(item_ct1)};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for split-K GEMMs
struct GemmSplitKHorizontalThreadblockSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int partitions) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      partitions);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  sycl::range<3> get_grid_shape(GemmCoord tiled_shape) const {
    return sycl::range<3>(tiled_shape.n(), tiled_shape.m(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(int log_tile, const sycl::nd_item<3> &item_ct1) const {
    return GemmCoord{RematerializeBlockIdxY(item_ct1), RematerializeBlockIdxX(item_ct1),
                     RematerializeBlockIdxZ(item_ct1)};
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape, const sycl::nd_item<3> &item_ct1) const {
    return GemmCoord{RematerializeBlockIdxY(item_ct1), RematerializeBlockIdxX(item_ct1),
                     RematerializeBlockIdxZ(item_ct1)};
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for batched GEMVs
struct GemvBatchedStridedThreadblockDefaultSwizzle {

  /// Returns the shape of the problem in units of logical tiles
  CUTLASS_HOST_DEVICE
  BatchedGemmCoord get_tiled_shape(
    BatchedGemmCoord problem_size,
    BatchedGemmCoord tile_size) const {

    return BatchedGemmCoord(
      1, // M is always 1
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      (problem_size.k() + tile_size.k() - 1) / tile_size.k(),
      (problem_size.batch() + tile_size.batch() - 1) / tile_size.batch());
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  sycl::range<3> get_grid_shape(BatchedGemmCoord tiled_shape) const {
    return sycl::range<3>(tiled_shape.n(), tiled_shape.batch(), tiled_shape.k());
  }

  /// Calculates optimal swizzle width
  CUTLASS_HOST_DEVICE
  int get_log_tile(GemmCoord tiled_shape) const {
    return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  BatchedGemmCoord get_tile_offset(int log_tile, const sycl::nd_item<3> &item_ct1) const {
    return BatchedGemmCoord{
        0,  // M is always 1
        RematerializeBlockIdxX(item_ct1),
        RematerializeBlockIdxZ(item_ct1),
        RematerializeBlockIdxY(item_ct1),
    };
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  BatchedGemmCoord get_tile_offset(const sycl::nd_item<3> &item_ct1) const {
    return BatchedGemmCoord{
        0,  // M is always 1
        RematerializeBlockIdxX(item_ct1),
        RematerializeBlockIdxZ(item_ct1),
        RematerializeBlockIdxY(item_ct1),
    };
  }

  /// Gets the batch tile index
  CUTLASS_DEVICE
  int get_batch_tile_idx(const sycl::nd_item<3> &item_ct1) const {
    return RematerializeBlockIdxY(item_ct1);
  }

  /// Gets the absolute batch index
  CUTLASS_DEVICE
  int get_batch_idx(const sycl::nd_item<3> &item_ct1) const {
    return RematerializeBlockDimY(item_ct1) * RematerializeBlockIdxY(item_ct1) +
           RematerializeThreadIdxY(item_ct1);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

