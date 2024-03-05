// Copyright 2023 Bytedance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common.h"

namespace bytetransformer {
#define FINAL_MASK 0xffffffff

__dpct_inline__ int getLaneId() {
  int laneId;
  /*
  DPCT1053:0: Migration of device assembly code is not supported.
  */
  asm("mov.s32 %0, %laneid;" : "=r"(laneId));
  return laneId;
}

template <typename T>
__inline__ T warpReduceSum(T val, const sycl::nd_item<3> &item_ct1) {
  for (int mask = 16; mask > 0; mask >>= 1)
    /*
    DPCT1096:587: The right-most dimension of the work-group used in the SYCL kernel that calls
    this function may be less than "32". The function "dpct::permute_sub_group_by_xor" may return
    an unexpected result on the CPU device. Modify the size of the work-group to ensure that the
    value of the right-most dimension is a multiple of "32".
    */
    val += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), val, mask);
  return val;
}

template <typename T>
__inline__ T blockReduceSum(T val, const sycl::nd_item<3> &item_ct1, T *shared) {
  int lane = item_ct1.get_local_id(2) & 0x1f;
  int wid = item_ct1.get_local_id(2) >> 5;

  val = warpReduceSum<T>(val, item_ct1);

  if (lane == 0)
    shared[wid] = val;
  /*
  DPCT1065:306: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  return wid == 0 ? warpReduceSum(item_ct1.get_local_id(2) < (item_ct1.get_local_range(2) >> 5)
                                      ? shared[lane]
                                      : (T)0.0f,
                                  item_ct1)
                  : 0.0f;
}

__inline__ sycl::half2 warpReduceSum(sycl::half2 val, const sycl::nd_item<3> &item_ct1) {
  sycl::half2 tmp_val;
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp_val = dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), val, mask);
    val = tmp_val + val;
  }
  return val;
}

__inline__ sycl::half __half2add(sycl::half2 val) {
  return val.x() + val.y();
}

__inline__ sycl::half blockReduceSum(sycl::half2 val, const sycl::nd_item<3> &item_ct1,
                                     sycl::half2 *shared) {
  int lane = item_ct1.get_local_id(2) & 0x1f;
  int wid = item_ct1.get_local_id(2) >> 5;

  val = warpReduceSum<sycl::half2>(val, item_ct1);

  if (lane == 0)
    shared[wid] = val;
  /*
  DPCT1065:307: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  return (sycl::half)(
      wid == 0 ? warpReduceSum(item_ct1.get_local_id(2) < (item_ct1.get_local_range(2) >> 5)
                                   ? (float)__half2add(shared[lane])
                                   : 0.0f,
                               item_ct1)
               : 0.0f);
}

template <typename T>
__inline__ T max_(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__inline__ T warpReduceMax(T val, const sycl::nd_item<3> &item_ct1) {
  for (int mask = 16; mask > 0; mask >>= 1)
    /*
    DPCT1096:586: The right-most dimension of the work-group used in the SYCL kernel that calls
    this function may be less than "32". The function "dpct::permute_sub_group_by_xor" may return
    an unexpected result on the CPU device. Modify the size of the work-group to ensure that the
    value of the right-most dimension is a multiple of "32".
    */
    val = max_(val, dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), val, mask));
  return val;
}

template <typename T>
__inline__ T blockReduceMax(T val, const sycl::nd_item<3> &item_ct1, T *shared) {
  int lane = item_ct1.get_local_id(2) & 0x1f;
  int wid = item_ct1.get_local_id(2) >> 5;

  val = warpReduceMax(val, item_ct1);

  if (lane == 0)
    shared[wid] = val;
  /*
  DPCT1065:308: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  return wid == 0 ? warpReduceMax(item_ct1.get_local_id(2) < (item_ct1.get_local_range(2) >> 5)
                                      ? shared[lane]
                                      : (T)-1e20f,
                                  item_ct1)
                  : (T)-1e20f;
}
}  // namespace bytetransformer
