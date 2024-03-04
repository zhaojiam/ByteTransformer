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

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include "cutlass/cutlass.h"

/**
 * \file
 * \brief C++ interface to dump fragments and shared memory contents for
 * debugging.
 */

namespace cutlass {
namespace debug {

/******************************************************************************
 * Dump the fragments
 ******************************************************************************/

/// The first N threads dump the first M elements from their fragments with a
/// stride of S elements.  If N is not specified, dump the data of all the
/// threads.  If M is not specified, dump all the elements of the fragment.
template <typename Fragment>
CUTLASS_DEVICE void dump_fragment(Fragment const& frag, const sycl::nd_item<3> &item_ct1,
                                  const sycl::stream &stream_ct1, int N = 0, int M = 0,
                                  int S = 1) {
  int total_threads =
      item_ct1.get_local_range(2) * item_ct1.get_local_range(1) * item_ct1.get_local_range(0);
  int block_id = item_ct1.get_group(2) + item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                 item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
  int thread_id =
      (item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))) +
      (item_ct1.get_local_id(1) * item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);

  if (N < 0 || N > total_threads) {
    if (thread_id == 0 && block_id == 0)
      /*
      DPCT1015:372: Output needs adjustment.
      */
      stream_ct1 << "Thread number N = %d should between [1, %d].\n";

    /*
    DPCT1118:371: SYCL group functions and algorithms must be encountered in converged control
    flow. You may need to adjust the code.
    */
    /*
    DPCT1065:625: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    return;
  }

  int total_elements = frag.size();

  if (M < 0 || M > total_elements) {
    if (thread_id == 0 && block_id == 0)
      /*
      DPCT1015:374: Output needs adjustment.
      */
      stream_ct1 << "Element number M = %d should between [1, %d].\n";

    /*
    DPCT1118:373: SYCL group functions and algorithms must be encountered in converged control
    flow. You may need to adjust the code.
    */
    /*
    DPCT1065:626: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    return;
  }

  if (N == 0) N = total_threads;

  if (M == 0) M = total_elements;

  if (S < 1 || S > M) {
    if (thread_id == 0 && block_id == 0)
      /*
      DPCT1015:376: Output needs adjustment.
      */
      stream_ct1 << "Stride S = %d should between [1, %d].\n";

    /*
    DPCT1118:375: SYCL group functions and algorithms must be encountered in converged control
    flow. You may need to adjust the code.
    */
    /*
    DPCT1065:627: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    return;
  }

  if (thread_id == 0 && block_id == 0)
    stream_ct1 << "\n*******************Dumping the fragments*******************\n\n";

  CUTLASS_PRAGMA_NO_UNROLL
  for (int tid = 0; tid < N; ++tid) {
    if (tid == thread_id) {
      /*
      DPCT1015:378: Output needs adjustment.
      */
      stream_ct1 << "TB%d W%d T%d: ";
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < M; i += S) {
        /*
        DPCT1015:379: Output needs adjustment.
        */
        stream_ct1 << "%.0f ";
      }
      stream_ct1 << "\n";
    }

    /*
    DPCT1118:377: SYCL group functions and algorithms must be encountered in converged control
    flow. You may need to adjust the code.
    */
    /*
    DPCT1065:628: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();
  }

  if (thread_id == 0 && block_id == 0)
    stream_ct1 << "\n***********************************************************\n\n";

  /*
  DPCT1065:624: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  return;
}

/******************************************************************************
 * Dump the shared memory
 ******************************************************************************/

#define SHMEM_ROW_SIZE 128

/// Dump the shared memory contents.  ptr is the begin address, size specifies
/// the number of elements that need to be dumped, and S specifies the stride.
template <typename Element>
CUTLASS_DEVICE void dump_shmem(Element const* ptr, size_t size, const sycl::nd_item<3> &item_ct1,
                               const sycl::stream &stream_ct1, int S = 1) {
  int block_id = item_ct1.get_group(2) + item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                 item_ct1.get_group_range(2) * item_ct1.get_group_range(1) * item_ct1.get_group(0);
  int thread_id =
      (item_ct1.get_local_id(0) * (item_ct1.get_local_range(2) * item_ct1.get_local_range(1))) +
      (item_ct1.get_local_id(1) * item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);

  if (ptr == nullptr) {
    if (thread_id == 0 && block_id == 0) stream_ct1 << "ptr is null.\n";

    /*
    DPCT1118:380: SYCL group functions and algorithms must be encountered in converged control
    flow. You may need to adjust the code.
    */
    /*
    DPCT1065:631: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();
    return;
  }

  if (size < 1) {
    if (thread_id == 0 && block_id == 0)
      stream_ct1 << "Element size is less than 1\n";

    /*
    DPCT1118:381: SYCL group functions and algorithms must be encountered in converged control
    flow. You may need to adjust the code.
    */
    /*
    DPCT1065:632: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    return;
  }

  int row_elements = SHMEM_ROW_SIZE / sizeof(Element);

  if (S < 1 || S > row_elements) {
    if (thread_id == 0 && block_id == 0)
      /*
      DPCT1015:383: Output needs adjustment.
      */
      stream_ct1 << "Stride S = %d should between [1, %d].\n";

    /*
    DPCT1118:382: SYCL group functions and algorithms must be encountered in converged control
    flow. You may need to adjust the code.
    */
    /*
    DPCT1065:633: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there
    is no access to global memory.
    */
    item_ct1.barrier();

    return;
  }

  /*
  DPCT1065:629: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (thread_id == 0)
    /*
    DPCT1015:384: Output needs adjustment.
    */
    stream_ct1 << "\n********Dumping the shared memory of TB %d*******\n\n";

  if (thread_id == 0) {
    for (int i = 0; i < size; i += row_elements) {
      for (int j = 0; j < row_elements; j += S) {
        /*
        DPCT1015:385: Output needs adjustment.
        */
        stream_ct1 << "%.0f ";
      }

      stream_ct1 << "\n";
    }
  }

  if (thread_id == 0)
    stream_ct1 << "\n***********************************************************\n\n";

  /*
  DPCT1065:630: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  return;
}
}  // namespace debug
}  // namespace cutlass
