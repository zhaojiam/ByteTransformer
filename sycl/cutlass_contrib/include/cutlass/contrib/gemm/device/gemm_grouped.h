/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This file is copied from NVIDIA/cutlass and modified.
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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/trace.h"
#include <cmath>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace contrib {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemm
template <typename GemmKernel_>
class GemmGrouped {
 public:
  using GemmKernel = GemmKernel_;

  /// Argument structure
  using Arguments = typename GemmKernel::Arguments;

 protected:
  /// Kernel parameters object
  typename GemmKernel::Params params_;

 public:
  /// Constructs the GEMM.
  GemmGrouped() {
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    // This kerenl does not utilize a workspace
    return size_t();
  }

  /// Computes the grid shape
  static sycl::range<3> get_grid_shape(Arguments const &args) {
    return sycl::range<3>(args.threadblock_count, 1, 1);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) try {
    CUTLASS_TRACE_HOST("GemmUniversalBase::maximum_active_blocks()");

    int max_active_blocks = -1;
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    CUTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");

    if (smem_size <= (48 << 10)) {
      /*
      DPCT1007:611: Migration of cudaOccupancyMaxActiveBlocksPerMultiprocessor is not supported.
      */
      dpct::err0 result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_active_blocks, Kernel<GemmKernel>, GemmKernel::kThreadCount, smem_size);

      if (result == 0) {
        CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
        return max_active_blocks;
      }
    } else {
      // Query assuming zero shared memory then compute occupancy limit based on SMEM
      /*
      DPCT1007:612: Migration of cudaOccupancyMaxActiveBlocksPerMultiprocessor is not supported.
      */
      dpct::err0 result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_active_blocks, Kernel<GemmKernel>, GemmKernel::kThreadCount, 0);

      if (smem_capacity < 0) {
        int device_idx = 0;
        result = DPCT_CHECK_ERROR(device_idx = dpct::dev_mgr::instance().current_device_id());

        dpct::device_info properties;
        result = DPCT_CHECK_ERROR(
            dpct::get_device_info(properties, dpct::dev_mgr::instance().get_device(device_idx)));

        /*
        DPCT1019:609: local_mem_size in SYCL is not a complete equivalent of
        sharedMemPerMultiprocessor in CUDA. You may need to adjust the code.
        */
        smem_capacity = static_cast<int>(properties.get_local_mem_size());
      }

      int occupancy = std::min(max_active_blocks, smem_capacity / smem_size);

      CUTLASS_TRACE_HOST("  occupancy: " << occupancy);

      return occupancy;
    }

    CUTLASS_TRACE_HOST("  returning internal error");

    return -1;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
              << std::endl;
    std::exit(1);
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr,
                    dpct::queue_ptr stream = &dpct::get_in_order_queue()) try {
    // CUTLASS_TRACE_HOST("GemmGrouped2D::initialize() - workspace "
    //                    << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Workspace
    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    // Initialize the Params structure
    params_ = typename GemmKernel::Params(args, workspace);

    // Specify shared memory capacity for kernel.
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      /*
      DPCT1027:613: The call to cudaFuncSetAttribute was replaced with 0 because SYCL currently
      does not support corresponding setting.
      */
      dpct::err0 result = 0;
    }

    return Status::kSuccess;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
              << std::endl;
    std::exit(1);
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    size_t workspace_bytes = get_workspace_size(args);

    if (workspace_bytes && !workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_.update(args, workspace);

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status run(dpct::queue_ptr stream = &dpct::get_in_order_queue()) {
    //
    // Configure grid and block dimensions
    //

    if (!params_.problem_visitor.problem_count) {
      return Status::kSuccess;
    }

    sycl::range<3> grid(1, 1, params_.threadblock_count);
    sycl::range<3> block(1, 1, GemmKernel::kThreadCount);

    /*
    DPCT1083:361: The size of local memory in the migrated code may be different from the original
    code. Check that the allocated memory size in the migrated code is correct.
    */
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

    //
    // Launch kernel
    //

    // Launch
    /*
    DPCT1049:360: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp64, sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(smem_size), cgh);

        auto params__ct0 = params_;

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              cutlass::Kernel<GemmKernel>(
                  params__ct0, item_ct1,
                  dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
      });
    }

    //
    // Query for errors
    //
    /*
    DPCT1010:610: SYCL uses exceptions to report errors and does not use the error codes. The call
    was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 result = 0;

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  Status operator()(dpct::queue_ptr stream = &dpct::get_in_order_queue()) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(Arguments const &args, void *workspace = nullptr,
                    dpct::queue_ptr stream = &dpct::get_in_order_queue()) {
    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace gemm
}  // namespace contrib
}  // namespace cutlass
