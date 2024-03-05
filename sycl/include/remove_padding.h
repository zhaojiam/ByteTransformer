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
#include "reduce.h"

namespace bytetransformer {
// ************************** build_sequence_length_padding_offset **************************
template <typename T>
__inline__ T warpPrefixSum(int id, T count, const sycl::nd_item<3> &item_ct1) {
  for (int i = 1; i < 32; i <<= 1) {
    /*
    DPCT1121:302: Make sure that the "count" which is used in the SYCL group function/algorithm is
    initialized.
    */
    /*
    DPCT1096:590: The right-most dimension of the work-group used in the SYCL kernel that calls
    this function may be less than "32". The function "dpct::shift_sub_group_right" may return an
    unexpected result on the CPU device. Modify the size of the work-group to ensure that the value
    of the right-most dimension is a multiple of "32".
    */
    T val = dpct::shift_sub_group_right(item_ct1.get_sub_group(), count, i);
    if (id >= i)
      count += val;
  }
  return count;
}

template <typename T>
void parallel_prefix(const T *atten_mask, int *batch_idx, int *word_idx,
                                const int batch_size, const int max_seq_len,
                                const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  const int tid = item_ct1.get_local_id(2);
  const int warp_count = item_ct1.get_local_range(2) >> 5;
  int warp_id = tid >> 5;
  int warp_tid = tid & 0x1F;

  auto base = (int *)dpct_local;

  int *seq_len = base;
  int *seq_offset = base + batch_size;

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int count = 0;
    for (int i = warp_tid; i < (max_seq_len + 31) / 32 * 32; i += 32) {
      T mask = i < max_seq_len ? atten_mask[wid * max_seq_len * max_seq_len + i] : (T)0.0f;
      count += sycl::popcount(sycl::reduce_over_group(
          item_ct1.get_sub_group(),
          (0xFFFFFFFF & (0x1 << item_ct1.get_sub_group().get_local_linear_id())) && mask >= (T)0.5f
              ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
              : 0,
          sycl::ext::oneapi::plus<>()));
    }
    if (warp_tid == 0)
      seq_len[wid] = count;
  }

  /*
  DPCT1065:487: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (warp_id == 0) {
    int offset = 0, temp = 0;
    for (int i = warp_tid; i < ((batch_size + 31) / 32) * 32; i += 32) {
      offset = warp_tid == 0 ? temp : 0;
      int len = i < batch_size ? seq_len[i] : 0;
      temp = warpPrefixSum(warp_tid, offset + len, item_ct1);
      if (i < batch_size)
        seq_offset[i] = temp - len;

      /*
      DPCT1096:589: The right-most dimension of the work-group used in the SYCL kernel that calls
      this function may be less than "32". The function "dpct::select_from_sub_group" may return an
      unexpected result on the CPU device. Modify the size of the work-group to ensure that the
      value of the right-most dimension is a multiple of "32".
      */
      temp = dpct::select_from_sub_group(item_ct1.get_sub_group(), temp, 31);
    }
    if (warp_tid == 0)
      seq_offset[batch_size] = temp;
  }

  /*
  DPCT1065:488: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier(sycl::access::fence_space::local_space);

  for (int i = tid; i <= batch_size; i += item_ct1.get_local_range(2))
    batch_idx[i] = seq_offset[i];

  for (int wid = warp_id; wid < batch_size; wid += warp_count) {
    int offset = seq_offset[wid];
    for (int i = warp_tid; i < seq_len[wid]; i += 32)
      word_idx[offset + i] = wid * max_seq_len + i;
  }
}

template <typename T>
void build_sequence_length_padding_offset_kernelLauncher(const T *atten_mask, int *batch_idx,
                                                         int *word_idx, int *valid_word_num,
                                                         const int batch_size,
                                                         const int max_seq_len,
                                                         dpct::queue_ptr stream) {
  sycl::range<3> block(1, 1, batch_size * 32);  // one warp per sequence
  if (block[2] > 1024)
    block[2] = 1024;
  /*
  DPCT1049:216: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      /*
      DPCT1083:588: The size of local memory in the migrated code may be different from the
      original code. Check that the allocated memory size in the migrated code is correct.
      */
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
          sycl::range<1>((2 * batch_size + 1) * sizeof(int)), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(block, block),
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            parallel_prefix(atten_mask, batch_idx, word_idx, batch_size, max_seq_len, item_ct1,
                            dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
          });
    });
  }
  stream->memcpy(valid_word_num, batch_idx + batch_size, sizeof(int));
}

// *********************** compresse transformer input ***********************
template <typename T>
void compress_bert_input(const T *from_tensor, T *to_tensor, const int *batch_idx,
                                    const int *word_idx, int hidden_dim,
                                    const sycl::nd_item<3> &item_ct1) {
  /*
  DPCT1098:489: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:585: Migrated __ldg call is used in a macro/template definition and may not be valid for
  all macro/template uses. Adjust the code.
  */
  int offset = word_idx[item_ct1.get_group(2)];
  int dst_idx = item_ct1.get_group(2) * hidden_dim + item_ct1.get_local_id(2);
  int src_idx = offset * hidden_dim + item_ct1.get_local_id(2);
  ((sycl::float4 *)to_tensor)[dst_idx] = ((const sycl::float4 *)from_tensor)[src_idx];
}

template <typename T>
void compressBertInput_kernelLauncher(const T *from_tensor, T *to_tensor, int *batch_idx,
                                      int *word_idx, int valid_word_num, int batch_size,
                                      int hidden_dim, dpct::queue_ptr stream) {
  sycl::range<3> grid(1, 1, valid_word_num);
  sycl::range<3> block(1, 1, hidden_dim / 4);  // assert(hidden_dim / 4 <= 1024);
  /*
  DPCT1049:217: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
      compress_bert_input(from_tensor, to_tensor, batch_idx, word_idx, hidden_dim / 4, item_ct1);
    });
  }
}

// *********************** add bias input restore transformer output ***********************

template <typename T>
void add_bias_input(T *out, const T *input, const T *bias, int n, const sycl::nd_item<3> &item_ct1);

template <typename T>
void add_bias_input_restore_output(const T *out, const T *input, const T *bias, int n,
                                              T *out2, const int *batch_idx, const int *word_idx,
                                              const int seq_len, const sycl::nd_item<3> &item_ct1);

template <typename T>
void add_bias_half_input(T *out, const T *input, const T *bias, int n,
                         const sycl::nd_item<3> &item_ct1);

template <typename T>
void add_bias_half_input_restore_output(const T *out, const T *input, const T *bias,
                                                   int n, T *out2, const int *batch_idx,
                                                   const int *word_idx, const int seq_len,
                                                   const sycl::nd_item<3> &item_ct1);
}  // namespace bytetransformer
