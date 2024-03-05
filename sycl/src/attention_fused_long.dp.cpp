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
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "bytetransformer/include/attention.h"
#include "bytetransformer/include/reduce.h"
#include <cmath>

namespace bytetransformer {
#define SKEW_HALF 8  // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <const int max_seq_len, const int size_per_head, const int split_seq_len>
/*
DPCT1110:1: The total declared local variable size in device function wmma_attention_long_kernel
exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find
the total register size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/
void wmma_attention_long_kernel(const sycl::half2 *qkv, const sycl::half2 *qkv_bias,
                                const sycl::half *attention_mask, sycl::half *attention_output,
                                const int seq_len, const float scale,
                                const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  auto base = (sycl::half *)dpct_local;
  sycl::half(*s_kv)[size_per_head + SKEW_HALF] = (sycl::half(*)[size_per_head + SKEW_HALF]) base;
  sycl::half(*s_query)[size_per_head + SKEW_HALF] = (sycl::half(*)[size_per_head + SKEW_HALF])(
      base + (max_seq_len) * (size_per_head + SKEW_HALF));
  sycl::half(*s_logits)[max_seq_len + SKEW_HALF] = (sycl::half(*)[max_seq_len + SKEW_HALF])(
      base + (split_seq_len + max_seq_len) * (size_per_head + SKEW_HALF));

  const int warpNums = (split_seq_len / 16) * (max_seq_len / 16);  //(blockDim.x  >> 5);
  const int warpId = (item_ct1.get_local_id(2) >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = item_ct1.get_group_range(2) * (size_per_head / 2);
  const int thread_offset = item_ct1.get_group(2) * (size_per_head / 2) + warp_tid;

  const int batch_seq_offset = item_ct1.get_group(0) * seq_len;
  const int block_seq_len =
      sycl::min(split_seq_len, (int)(seq_len - (int)item_ct1.get_group(1) * split_seq_len));
  const int batch_seq_block_offset = batch_seq_offset + item_ct1.get_group(1) * split_seq_len;

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading Query
  /*
  DPCT1098:309: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 q_bias = qkv_bias[thread_offset];
  for (int seq_id = warpId; seq_id < block_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_block_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:310: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = qkv[pos] + q_bias;
  }

  // loading Key
  /*
  DPCT1098:311: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 k_bias = qkv_bias[thread_offset + half_hidden_dim];
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:312: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = qkv[pos + half_hidden_dim] + k_bias;
  }
  /*
  DPCT1065:315: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:2: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:3: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:4: Migration of nvcuda::wmma::row_major type is not supported.
    */
    wmma::fragment < wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> _N,
        WMMA_K,
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::col_major> MMA_N,
        WMMA_K,
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, sycl::half> int
            warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

#pragma unroll
    for (int k = 0; k < 4; k++) {
      /*
      DPCT1007:23: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      /*
      DPCT1007:24: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      /*
      DPCT1007:11: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    /*
    DPCT1007:22: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:316: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  // softmax
  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < seq_len) {
        float mask =
            /*
            DPCT1098:319: The '*' expression is used instead of the __ldg call. These two
            expressions do not provide the exact same functionality. Check the generated code for
            potential precision and/or performance issues.
            */
            (float)attention_mask[(batch_seq_block_offset + from_id) * seq_len + to_id[i]];
        mask = (1.0f - mask) * (-10000.0f);
        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + mask;
      }
      max_val = sycl::max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val, item_ct1);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = sycl::native::exp(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val, item_ct1) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] = (sycl::half)logits[i] / sum_val;
  }

  // loading Value
  /*
  DPCT1098:313: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 v_bias = qkv_bias[thread_offset + half_hidden_dim * 2];
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    ((sycl::half2 *)(s_kv[seq_id]))[warp_tid] =
        /*
        DPCT1098:314: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        qkv[pos + half_hidden_dim * 2] + v_bias;
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:317: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:12: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:13: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:14: Migration of nvcuda::wmma::row_major type is not supported.
    */
    wmma::fragment < wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> WMMA_N,
        WMMA_K,
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> MMA_N,
        WMMA_K,
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, sycl::half> t int
            warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      /*
      DPCT1007:26: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:27: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:21: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:25: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:318: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    int pos = (batch_seq_block_offset + from_id) * half_hidden_dim + thread_offset;
    ((sycl::half2 *)(attention_output))[pos] = ((sycl::half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head, const int split_seq_len>
/*
DPCT1110:28: The total declared local variable size in device function
wmma_attention_long_rm_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the code, or use smaller
sub-group size to avoid high register pressure.
*/
void wmma_attention_long_rm_kernel(const sycl::half2 *qkv, const sycl::half2 *qkv_bias,
                                   const sycl::half *attention_mask, sycl::half *attention_output,
                                   const float scale, const int *batch_idx,
                                   const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  auto base = (sycl::half *)dpct_local;
  sycl::half(*s_kv)[size_per_head + SKEW_HALF] = (sycl::half(*)[size_per_head + SKEW_HALF]) base;
  sycl::half(*s_query)[size_per_head + SKEW_HALF] = (sycl::half(*)[size_per_head + SKEW_HALF])(
      base + (max_seq_len) * (size_per_head + SKEW_HALF));
  sycl::half(*s_logits)[max_seq_len + SKEW_HALF] = (sycl::half(*)[max_seq_len + SKEW_HALF])(
      base + (split_seq_len + max_seq_len) * (size_per_head + SKEW_HALF));

  const int warpNums = (split_seq_len / 16) * (max_seq_len / 16);  //(blockDim.x  >> 5);
  const int warpId = (item_ct1.get_local_id(2) >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = item_ct1.get_group_range(2) * (size_per_head / 2);
  const int thread_offset = item_ct1.get_group(2) * (size_per_head / 2) + warp_tid;

  // const int batch_seq_offset = blockIdx.z * seq_len;
  /*
  DPCT1098:320: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_offset = batch_idx[item_ct1.get_group(0)];
  /*
  DPCT1098:321: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[item_ct1.get_group(0) + 1] - batch_seq_offset;
  const int block_seq_len =
      sycl::min(split_seq_len, (int)(batch_seq_len - (int)item_ct1.get_group(1) * split_seq_len));
  if (block_seq_len <= 0)
    return;
  const int batch_seq_len_pad = ((batch_seq_len + 15) >> 4) << 4;
  const int batch_seq_block_offset = batch_seq_offset + item_ct1.get_group(1) * split_seq_len;

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;

  // loading Query
  /*
  DPCT1098:322: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 q_bias = qkv_bias[thread_offset];
  for (int seq_id = warpId; seq_id < block_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_block_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:323: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = qkv[pos] + q_bias;
  }

  // loading Key
  /*
  DPCT1098:324: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 k_bias = qkv_bias[thread_offset + half_hidden_dim];
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:325: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = qkv[pos + half_hidden_dim] + k_bias;
  }
  /*
  DPCT1065:328: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:29: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:30: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:31: Migration of nvcuda::wmma::row_major type is not supported.
    */
    wmma::fragment < wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> _N,
        WMMA_K,
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::col_major> MMA_N,
        WMMA_K,
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, sycl::half> int
            warp_from_offset = (warpId / to_size) << 4;
    const int warp_to_offset = (warpId % to_size) << 4;

    if (warp_to_offset < batch_seq_len_pad) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        /*
        DPCT1007:50: Migration of nvcuda::wmma::load_matrix_sync is not supported.
        */
        wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                               size_per_head + SKEW_HALF);
        /*
        DPCT1007:51: Migration of nvcuda::wmma::load_matrix_sync is not supported.
        */
        wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K,
                               size_per_head + SKEW_HALF);
        /*
        DPCT1007:38: Migration of nvcuda::wmma::mma_sync is not supported.
        */
        wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
      }
      /*
      DPCT1007:49: Migration of nvcuda::wmma::store_matrix_sync is not supported.
      */
      wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                              max_seq_len + SKEW_HALF, wmma::mem_row_major);
    }
  }
  /*
  DPCT1065:329: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  // softmax
  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    float max_val = -1e20f;

    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = to_id[i] < batch_seq_len ? (float)(s_logits[from_id][to_id[i]]) * scale : -1e20f;
      max_val = sycl::max(max_val, logits[i]);
    }
    max_val = warpReduceMax(max_val, item_ct1);

    float sum_val = 0.0f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      logits[i] = sycl::native::exp(logits[i] - max_val);
      sum_val += logits[i];
    }
    sum_val = warpReduceSum(sum_val, item_ct1) + 1e-6f;

#pragma unroll
    for (int i = 0; i < n; i++)
      if (to_id[i] < batch_seq_len_pad)
        s_logits[from_id][to_id[i]] = (sycl::half)logits[i] / sum_val;
  }

  // loading Value
  /*
  DPCT1098:326: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 v_bias = qkv_bias[thread_offset + half_hidden_dim * 2];
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    ((sycl::half2 *)(s_kv[seq_id]))[warp_tid] =
        /*
        DPCT1098:327: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        qkv[pos + half_hidden_dim * 2] + v_bias;
  }

  // K dim clear 0
  for (int seq_id = batch_seq_len + warpId; seq_id < batch_seq_len_pad; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:330: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:39: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:40: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:41: Migration of nvcuda::wmma::row_major type is not supported.
    */
    wmma::fragment < wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> WMMA_N,
        WMMA_K,
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> MMA_N,
        WMMA_K,
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, sycl::half> t int
            warp_from_offset = (warpId >> 2) << 4;
    const int warp_to_offset = (warpId & 0x3) * WMMA_K;

#pragma unroll
    for (int k = 0; k < (batch_seq_len_pad / 16); k++) {
      /*
      DPCT1007:53: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:54: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:48: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:52: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:331: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    int pos = (batch_seq_block_offset + from_id) * half_hidden_dim + thread_offset;
    ((sycl::half2 *)(attention_output))[pos] = ((sycl::half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

// __shared__ __half     s_kv  [max_seq_len][size_per_head + SKEW_HALF];
// __shared__ __half  s_query[split_seq_len][size_per_head + SKEW_HALF];
// __shared__ __half s_logits[split_seq_len][max_seq_len   + SKEW_HALF];

/*
DPCT1049:55: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1026:304: The call to cudaFuncSetAttribute was removed because SYCL currently does not support
corresponding setting.
*/
#define WMMA_ATTENTION_LONG(SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN)                                    \
  shared_memory_size =                                                                            \
      ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * \
      2;                                                                                          \
  if (shared_memory_size > 48 * 1024)                                                             \
    ;                                                                                             \
  grid.x = head_num_, grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN, grid.z = batch_size,        \
  block.x = 32 * (SPLIT_LEN / 16 * split_count);                                                  \
  {                                                                                               \
    dpct::has_capability_or_fail(infer_param.stream->get_device(), {sycl::aspect::fp16});         \
                                                                                                  \
    infer_param.stream->submit([&](sycl::handler &cgh) {                                          \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_memory_size),     \
                                                          cgh);                                   \
                                                                                                  \
      const sycl::half2 *qkv_ptr_ct0 = qkv_ptr;                                                   \
      const sycl::half2 *qkv_bias_ptr_ct1 = qkv_bias_ptr;                                         \
      const sycl::half *atten_mask_ct2 = (sycl::half *)atten_mask;                                \
      sycl::half *attention_output_ct3 = (sycl::half *)attention_output;                          \
      const int seq_len_ct4 = seq_len;                                                            \
      const float scale_ct5 = scale;                                                              \
                                                                                                  \
      cgh.parallel_for(                                                                           \
          sycl::nd_range<3>(grid * block, block),                                                 \
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {                     \
            wmma_attention_long_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>(                        \
                qkv_ptr_ct0, qkv_bias_ptr_ct1, atten_mask_ct2, attention_output_ct3, seq_len_ct4, \
                scale_ct5, item_ct1,                                                              \
                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());           \
          });                                                                                     \
    });                                                                                           \
  }

/*
DPCT1049:56: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1026:305: The call to cudaFuncSetAttribute was removed because SYCL currently does not support
corresponding setting.
*/
#define WMMA_ATTENTION_LONG_RM(SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN)                                 \
  shared_memory_size =                                                                            \
      ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * \
      2;                                                                                          \
  if (shared_memory_size > 48 * 1024)                                                             \
    ;                                                                                             \
  grid.x = head_num_, grid.y = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN, grid.z = batch_size,        \
  block.x = 32 * (SPLIT_LEN / 16 * split_count);                                                  \
  {                                                                                               \
    dpct::has_capability_or_fail(infer_param.stream->get_device(), {sycl::aspect::fp16});         \
                                                                                                  \
    infer_param.stream->submit([&](sycl::handler &cgh) {                                          \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_memory_size),     \
                                                          cgh);                                   \
                                                                                                  \
      const sycl::half2 *qkv_ptr_ct0 = qkv_ptr;                                                   \
      const sycl::half2 *qkv_bias_ptr_ct1 = qkv_bias_ptr;                                         \
      const sycl::half *atten_mask_ct2 = (sycl::half *)atten_mask;                                \
      sycl::half *attention_output_ct3 = (sycl::half *)attention_output;                          \
      const float scale_ct4 = scale;                                                              \
      const int *et_param_batch_idx_ct5 = et_param.batch_idx;                                     \
                                                                                                  \
      cgh.parallel_for(                                                                           \
          sycl::nd_range<3>(grid * block, block),                                                 \
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {                     \
            wmma_attention_long_rm_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>(                     \
                qkv_ptr_ct0, qkv_bias_ptr_ct1, atten_mask_ct2, attention_output_ct3, scale_ct4,   \
                et_param_batch_idx_ct5, item_ct1,                                                 \
                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());           \
          });                                                                                     \
    });                                                                                           \
  }

template <OperationType OpType>
void Attention<OpType>::fused_long_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;

  if (OpType == OperationType::HALF) {
    const sycl::half2 *qkv_ptr = (const sycl::half2 *)infer_param.qkv;
    const sycl::half2 *qkv_bias_ptr = (const sycl::half2 *)param_.attr_bias_QKV;

    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

    sycl::range<3> grid(1, 1, 1), block(1, 1, 1);
    int shared_memory_size = 0;
    const int split_count = (seq_len + 15) / 16;
    switch (split_count) {
      case 6:
        WMMA_ATTENTION_LONG(96, 64, 48);
        break;  //  80 < seq_len <=  96
      case 7:
        WMMA_ATTENTION_LONG(112, 64, 64);
        break;  //  96 < seq_len <= 112
      case 8:
        WMMA_ATTENTION_LONG(128, 64, 64);
        break;  // 112 < seq_len <= 128
      case 9:
        WMMA_ATTENTION_LONG(144, 64, 48);
        break;  // 128 < seq_len <= 144
      case 10:
        WMMA_ATTENTION_LONG(160, 64, 48);
        break;  // 144 < seq_len <= 160
      case 11:
        WMMA_ATTENTION_LONG(176, 64, 32);
        break;  // 160 < seq_len <= 176
      case 12:
        WMMA_ATTENTION_LONG(192, 64, 32);
        break;  // 176 < seq_len <= 192
      case 13:
        WMMA_ATTENTION_LONG(208, 64, 32);
        break;  // 192 < seq_len <= 208
      case 14:
        WMMA_ATTENTION_LONG(224, 64, 32);
        break;  // 208 < seq_len <= 224
      case 15:
        WMMA_ATTENTION_LONG(240, 64, 32);
        break;  // 224 < seq_len <= 240
      case 16:
        WMMA_ATTENTION_LONG(256, 64, 32);
        break;  // 240 < seq_len <= 256
      case 17:
        WMMA_ATTENTION_LONG(272, 64, 16);
        break;  // 256 < seq_len <= 272
      case 18:
        WMMA_ATTENTION_LONG(288, 64, 16);
        break;  // 272 < seq_len <= 288
      case 19:
        WMMA_ATTENTION_LONG(304, 64, 16);
        break;  // 288 < seq_len <= 304
      case 20:
        WMMA_ATTENTION_LONG(320, 64, 16);
        break;  // 304 < seq_len <= 320
      case 21:
        WMMA_ATTENTION_LONG(336, 64, 16);
        break;  // 320 < seq_len <= 336
      case 22:
        WMMA_ATTENTION_LONG(352, 64, 16);
        break;  // 336 < seq_len <= 352
    }
  }
}

template <OperationType OpType>
void Attention<OpType>::fused_long_rm_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  ET_Param et_param = infer_param.et_param;

  if (OpType == OperationType::HALF) {
    const sycl::half2 *qkv_ptr = (const sycl::half2 *)infer_param.qkv;
    const sycl::half2 *qkv_bias_ptr = (const sycl::half2 *)param_.attr_bias_QKV;
    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

    sycl::range<3> grid(1, 1, 1), block(1, 1, 1);
    int shared_memory_size = 0;
    const int split_count = (seq_len + 15) / 16;
    switch (split_count) {
      case 6:
        WMMA_ATTENTION_LONG_RM(96, 64, 48);
        break;  //  80 < seq_len <=  96
      case 7:
        WMMA_ATTENTION_LONG_RM(112, 64, 64);
        break;  //  96 < seq_len <= 112
      case 8:
        WMMA_ATTENTION_LONG_RM(128, 64, 64);
        break;  // 112 < seq_len <= 128
      case 9:
        WMMA_ATTENTION_LONG_RM(144, 64, 48);
        break;  // 128 < seq_len <= 144
      case 10:
        WMMA_ATTENTION_LONG_RM(160, 64, 48);
        break;  // 144 < seq_len <= 160
      case 11:
        WMMA_ATTENTION_LONG_RM(176, 64, 32);
        break;  // 160 < seq_len <= 176
      case 12:
        WMMA_ATTENTION_LONG_RM(192, 64, 32);
        break;  // 176 < seq_len <= 192
      case 13:
        WMMA_ATTENTION_LONG_RM(208, 64, 32);
        break;  // 192 < seq_len <= 208
      case 14:
        WMMA_ATTENTION_LONG_RM(224, 64, 32);
        break;  // 208 < seq_len <= 224
      case 15:
        WMMA_ATTENTION_LONG_RM(240, 64, 32);
        break;  // 224 < seq_len <= 240
      case 16:
        WMMA_ATTENTION_LONG_RM(256, 64, 32);
        break;  // 240 < seq_len <= 256
      case 17:
        WMMA_ATTENTION_LONG_RM(272, 64, 16);
        break;  // 256 < seq_len <= 272
      case 18:
        WMMA_ATTENTION_LONG_RM(288, 64, 16);
        break;  // 272 < seq_len <= 288
      case 19:
        WMMA_ATTENTION_LONG_RM(304, 64, 16);
        break;  // 288 < seq_len <= 304
      case 20:
        WMMA_ATTENTION_LONG_RM(320, 64, 16);
        break;  // 304 < seq_len <= 320
      case 21:
        WMMA_ATTENTION_LONG_RM(336, 64, 16);
        break;  // 320 < seq_len <= 336
      case 22:
        WMMA_ATTENTION_LONG_RM(352, 64, 16);
        break;  // 336 < seq_len <= 352
    }
  }
}

template void Attention<OperationType::FP32>::fused_long_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_long_infer(AttentionInferParam infer_param);
template void Attention<OperationType::FP32>::fused_long_rm_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_long_rm_infer(AttentionInferParam infer_param);
}  // namespace bytetransformer
