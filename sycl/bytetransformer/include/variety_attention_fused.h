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

#include "reduce.h"
#include <cmath>

namespace bytetransformer {
#define SKEW_HALF 8  // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <const int max_seq_len, const int size_per_head>
/*
DPCT1110:243: The total declared local variable size in device function
variety_wmma_attention_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the code, or use smaller
sub-group size to avoid high register pressure.
*/

// __launch_bounds__(512,4)//THREADS_PER_BLOCK
void variety_wmma_attention_kernel(const sycl::half2 *query, const sycl::half2 *key,
                                   const sycl::half2 *value, const sycl::half *attention_mask,
                                   const sycl::half *qk_buf, const sycl::half *attention_bias,
                                   sycl::half *attention_output, const int seq_len,
                                   const float scale, const sycl::nd_item<3> &item_ct1,
                                   sycl::local_accessor<sycl::half, 2> s_kv,
                                   sycl::local_accessor<sycl::half, 2> s_query,
                                   sycl::local_accessor<sycl::half, 2> s_logits) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  const int warpNums = (item_ct1.get_local_range(2) >> 5);
  const int warpId = (item_ct1.get_local_id(2) >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = item_ct1.get_group_range(2) * (size_per_head / 2);
  const int thread_offset = item_ct1.get_group(2) * (size_per_head / 2) + warp_tid;
  const int batch_seq_offset = item_ct1.get_group(1) * seq_len;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  const int gemm_offset =
      (item_ct1.get_group(2) * item_ct1.get_group_range(1) + item_ct1.get_group(1)) * seq_len;

  // loading Query & Key
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_id) * (size_per_head / 2) + warp_tid;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:573: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = query[pos];
    /*
    DPCT1098:574: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = key[pos];
  }
  /*
  DPCT1065:576: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:244: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:245: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:246: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:265: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      /*
      DPCT1007:266: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      /*
      DPCT1007:253: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    /*
    DPCT1007:264: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:577: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  // softmax
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

    float max_val = -1e20f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < seq_len) {
        float qk = qk_buf ? (float)(qk_buf[(gemm_offset + from_id) * seq_len + to_id[i]]) : 0.0f;
        float bias =
            attention_bias
                ? (float)(attention_bias[(item_ct1.get_group(2) * seq_len + from_id) * seq_len +
                                         to_id[i]])
                : 0.0f;

        float mask =
            /*
            DPCT1098:580: The '*' expression is used instead of the __ldg call. These two
            expressions do not provide the exact same functionality. Check the generated code for
            potential precision and/or performance issues.
            */
            (float)attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]];
        mask = (1.0f - mask) * (-10000.0f);
        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + qk + bias + mask;
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

    // loading Value
    int pos = (gemm_offset + from_id) * (size_per_head / 2) + warp_tid;
    /*
    DPCT1098:575: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    ((sycl::half2 *)(s_kv[from_id]))[warp_tid] = value[pos];
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:578: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:254: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:255: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:256: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:268: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:269: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:263: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:267: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:579: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + thread_offset;
    ((sycl::half2 *)(attention_output))[pos] = ((sycl::half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head>
/*
DPCT1110:270: The total declared local variable size in device function
variety_wmma_attention_rm_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust the code, or use
smaller sub-group size to avoid high register pressure.
*/

// __launch_bounds__(512,4)//THREADS_PER_BLOCK
void variety_wmma_attention_rm_kernel(
    const sycl::half2 *query, const sycl::half2 *key, const sycl::half2 *value,
    const sycl::half *attention_mask, const sycl::half *qk_buf, const sycl::half *attention_bias,
    sycl::half *attention_output, const int seq_len, const float scale, const int *batch_idx,
    const sycl::nd_item<3> &item_ct1, sycl::local_accessor<sycl::half, 2> s_kv,
    sycl::local_accessor<sycl::half, 2> s_query, sycl::local_accessor<sycl::half, 2> s_logits) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  const int warpNums = (item_ct1.get_local_range(2) >> 5);
  const int warpId = (item_ct1.get_local_id(2) >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = item_ct1.get_group_range(2) * (size_per_head / 2);
  const int thread_offset = item_ct1.get_group(2) * (size_per_head / 2) + warp_tid;
  /*
  DPCT1098:581: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[item_ct1.get_group(1)];
  /*
  DPCT1098:582: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[item_ct1.get_group(1) + 1] - batch_offset;
  const int from_size = (batch_seq_len + 15) >> 4;
  const int to_size = (batch_seq_len + 15) >> 4;

  const int gemm_offset =
      (item_ct1.get_group(2) * item_ct1.get_group_range(1) + item_ct1.get_group(1)) * seq_len;

  // loading Query & Key
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_id) * (size_per_head / 2) + warp_tid;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:583: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = query[pos];
    /*
    DPCT1098:584: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = key[pos];
  }
  /*
  DPCT1065:586: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:271: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:272: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:273: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:292: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      /*
      DPCT1007:293: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      /*
      DPCT1007:280: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    /*
    DPCT1007:291: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:587: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  // softmax
  for (int from_id = warpId; from_id < batch_seq_len; from_id += warpNums) {
    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

    float max_val = -1e20f;
#pragma unroll
    for (int i = 0; i < n; i++) {
      to_id[i] = warp_tid + (i << 5);
      logits[i] = -1e20f;

      if (to_id[i] < batch_seq_len) {
        float qk = qk_buf ? (float)(qk_buf[(gemm_offset + from_id) * seq_len + to_id[i]]) : 0.0f;
        float bias =
            attention_bias
                ? (float)(attention_bias[(item_ct1.get_group(2) * seq_len + from_id) * seq_len +
                                         to_id[i]])
                : 0.0f;

        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + qk + bias;
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

    // loading Value
    int pos = (gemm_offset + from_id) * (size_per_head / 2) + warp_tid;
    /*
    DPCT1098:585: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    ((sycl::half2 *)(s_kv[from_id]))[warp_tid] = value[pos];
  }

  // K dim clear 0
  for (int seq_id = batch_seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:588: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:281: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:282: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:283: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:295: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:296: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:290: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:294: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:589: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  for (int from_id = warpId; from_id < batch_seq_len; from_id += warpNums) {
    int pos = (batch_offset + from_id) * half_hidden_dim + thread_offset;
    ((sycl::half2 *)(attention_output))[pos] = ((sycl::half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head, const int split_seq_len>
/*
DPCT1110:297: The total declared local variable size in device function
variety_wmma_attention_long_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust the code, or use
smaller sub-group size to avoid high register pressure.
*/
void variety_wmma_attention_long_kernel(const sycl::half2 *query, const sycl::half2 *key,
                                        const sycl::half2 *value, const sycl::half *attention_mask,
                                        const sycl::half *qk_buf, const sycl::half *attention_bias,
                                        sycl::half *attention_output, const int seq_len,
                                        const float scale, const sycl::nd_item<3> &item_ct1,
                                        uint8_t *dpct_local) {
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
  const int seq_block_offset = item_ct1.get_group(1) * split_seq_len;
  const int block_seq_len = sycl::min(split_seq_len, seq_len - seq_block_offset);

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;

  const int gemm_offset =
      (item_ct1.get_group(2) * item_ct1.get_group_range(0) + item_ct1.get_group(0)) * seq_len;

  // loading Query
  for (int seq_id = warpId; seq_id < block_seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_block_offset + seq_id) * (size_per_head / 2) + warp_tid;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:590: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = query[pos];
  }

  // loading Key
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_id) * (size_per_head / 2) + warp_tid;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:591: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = key[pos];
  }
  /*
  DPCT1065:593: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:298: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:299: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:300: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:319: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      /*
      DPCT1007:320: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      /*
      DPCT1007:307: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    /*
    DPCT1007:318: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:594: Consider replacing sycl::nd_item::barrier() with
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
        const int split_seq_offset = seq_block_offset + from_id;
        float qk =
            qk_buf ? (float)(qk_buf[(gemm_offset + split_seq_offset) * seq_len + to_id[i]]) : 0.0f;
        float bias =
            attention_bias
                ? (float)(attention_bias[(item_ct1.get_group(2) * seq_len + split_seq_offset) *
                                             seq_len +
                                         to_id[i]])
                : 0.0f;

        /*
        DPCT1098:597: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        float mask =
            (float)attention_mask[(batch_seq_offset + split_seq_offset) * seq_len + to_id[i]];
        mask = (1.0f - mask) * (-10000.0f);
        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + qk + bias + mask;
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
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_id) * (size_per_head / 2) + warp_tid;
    /*
    DPCT1098:592: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    ((sycl::half2 *)(s_kv[seq_id]))[warp_tid] = value[pos];
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:595: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:308: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:309: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:310: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:322: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:323: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:317: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:321: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:596: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + seq_block_offset + from_id) * half_hidden_dim + thread_offset;
    ((sycl::half2 *)(attention_output))[pos] = ((sycl::half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head, const int split_seq_len>
/*
DPCT1110:324: The total declared local variable size in device function
variety_wmma_attention_long_rm_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and adjust the code, or
use smaller sub-group size to avoid high register pressure.
*/
void variety_wmma_attention_long_rm_kernel(
    const sycl::half2 *query, const sycl::half2 *key, const sycl::half2 *value,
    const sycl::half *attention_mask, const sycl::half *qk_buf, const sycl::half *attention_bias,
    sycl::half *attention_output, const int seq_len, const float scale, const int *batch_idx,
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
  DPCT1098:598: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_offset = batch_idx[item_ct1.get_group(0)];
  /*
  DPCT1098:599: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[item_ct1.get_group(0) + 1] - batch_seq_offset;
  const int seq_block_offset = item_ct1.get_group(1) * split_seq_len;
  const int block_seq_len = sycl::min(split_seq_len, batch_seq_len - seq_block_offset);
  if (block_seq_len <= 0)
    return;
  const int batch_seq_len_pad = ((batch_seq_len + 15) >> 4) << 4;

  const int from_size = split_seq_len / 16;
  const int to_size = max_seq_len / 16;

  const int gemm_offset =
      (item_ct1.get_group(2) * item_ct1.get_group_range(0) + item_ct1.get_group(0)) * seq_len;

  // loading Query
  for (int seq_id = warpId; seq_id < block_seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_block_offset + seq_id) * (size_per_head / 2) + warp_tid;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:600: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = query[pos];
  }

  // loading Key
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_id) * (size_per_head / 2) + warp_tid;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:601: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = key[pos];
  }
  /*
  DPCT1065:603: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:325: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:326: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:327: Migration of nvcuda::wmma::row_major type is not supported.
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
        DPCT1007:346: Migration of nvcuda::wmma::load_matrix_sync is not supported.
        */
        wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                               size_per_head + SKEW_HALF);
        /*
        DPCT1007:347: Migration of nvcuda::wmma::load_matrix_sync is not supported.
        */
        wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K,
                               size_per_head + SKEW_HALF);
        /*
        DPCT1007:334: Migration of nvcuda::wmma::mma_sync is not supported.
        */
        wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
      }
      /*
      DPCT1007:345: Migration of nvcuda::wmma::store_matrix_sync is not supported.
      */
      wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                              max_seq_len + SKEW_HALF, wmma::mem_row_major);
    }
  }
  /*
  DPCT1065:604: Consider replacing sycl::nd_item::barrier() with
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

      if (to_id[i] < batch_seq_len) {
        const int split_seq_offset = seq_block_offset + from_id;
        float qk =
            qk_buf ? (float)(qk_buf[(gemm_offset + split_seq_offset) * seq_len + to_id[i]]) : 0.0f;
        float bias =
            attention_bias
                ? (float)(attention_bias[(item_ct1.get_group(2) * seq_len + split_seq_offset) *
                                             seq_len +
                                         to_id[i]])
                : 0.0f;

        logits[i] = (float)(s_logits[from_id][to_id[i]]) * scale + qk + bias;
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
      if (to_id[i] < batch_seq_len_pad)
        s_logits[from_id][to_id[i]] = (sycl::half)logits[i] / sum_val;
  }

  // loading Value
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (gemm_offset + seq_id) * (size_per_head / 2) + warp_tid;
    /*
    DPCT1098:602: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    ((sycl::half2 *)(s_kv[seq_id]))[warp_tid] = value[pos];
  }

  // K dim clear 0
  for (int seq_id = batch_seq_len + warpId; seq_id < batch_seq_len_pad; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:605: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:335: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:336: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:337: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:349: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:350: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:344: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:348: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:606: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  for (int from_id = warpId; from_id < block_seq_len; from_id += warpNums) {
    int pos = (batch_seq_offset + seq_block_offset + from_id) * half_hidden_dim + thread_offset;
    ((sycl::half2 *)(attention_output))[pos] = ((sycl::half2 *)(s_query[from_id]))[warp_tid];
  }
#endif
}

/*
DPCT1049:351: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define X_WMMA_ATTENTION(SEQ_LEN, SIZE_PER_HEAD)                                               \
  {                                                                                            \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});                  \
                                                                                               \
    stream->submit([&](sycl::handler &cgh) {                                                   \
      sycl::local_accessor<sycl::half, 2> s_kv_acc_ct1(                                        \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                            \
      sycl::local_accessor<sycl::half, 2> s_query_acc_ct1(                                     \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                            \
      sycl::local_accessor<sycl::half, 2> s_logits_acc_ct1(                                    \
          sycl::range<2>(SEQ_LEN, SEQ_LEN + SKEW_HALF), cgh);                                  \
                                                                                               \
      const sycl::half2 *query_ct0 = query;                                                    \
      const sycl::half2 *key_ct1 = key;                                                        \
      const sycl::half2 *value_ct2 = value;                                                    \
      const sycl::half *atten_mask_ct3 = atten_mask;                                           \
      const sycl::half *qk_buf_ct4 = qk_buf;                                                   \
      const sycl::half *attention_bias_ct5 = attention_bias;                                   \
      sycl::half *attention_output_ct6 = attention_output;                                     \
      const int seq_len_ct7 = seq_len;                                                         \
      const float scale_ct8 = scale;                                                           \
                                                                                               \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                 \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {     \
                         variety_wmma_attention_kernel<SEQ_LEN, SIZE_PER_HEAD>(                \
                             query_ct0, key_ct1, value_ct2, atten_mask_ct3, qk_buf_ct4,        \
                             attention_bias_ct5, attention_output_ct6, seq_len_ct7, scale_ct8, \
                             item_ct1, s_kv_acc_ct1, s_query_acc_ct1, s_logits_acc_ct1);       \
                       });                                                                     \
    });                                                                                        \
  }

/*
DPCT1049:352: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define X_WMMA_ATTENTION_RM(SEQ_LEN, SIZE_PER_HEAD)                                            \
  {                                                                                            \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});                  \
                                                                                               \
    stream->submit([&](sycl::handler &cgh) {                                                   \
      sycl::local_accessor<sycl::half, 2> s_kv_acc_ct1(                                        \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                            \
      sycl::local_accessor<sycl::half, 2> s_query_acc_ct1(                                     \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                            \
      sycl::local_accessor<sycl::half, 2> s_logits_acc_ct1(                                    \
          sycl::range<2>(SEQ_LEN, SEQ_LEN + SKEW_HALF), cgh);                                  \
                                                                                               \
      const sycl::half2 *query_ct0 = query;                                                    \
      const sycl::half2 *key_ct1 = key;                                                        \
      const sycl::half2 *value_ct2 = value;                                                    \
      const sycl::half *atten_mask_ct3 = atten_mask;                                           \
      const sycl::half *qk_buf_ct4 = qk_buf;                                                   \
      const sycl::half *attention_bias_ct5 = attention_bias;                                   \
      sycl::half *attention_output_ct6 = attention_output;                                     \
      const int seq_len_ct7 = seq_len;                                                         \
      const float scale_ct8 = scale;                                                           \
      const int *batch_idx_ct9 = batch_idx;                                                    \
                                                                                               \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                 \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {     \
                         variety_wmma_attention_rm_kernel<SEQ_LEN, SIZE_PER_HEAD>(             \
                             query_ct0, key_ct1, value_ct2, atten_mask_ct3, qk_buf_ct4,        \
                             attention_bias_ct5, attention_output_ct6, seq_len_ct7, scale_ct8, \
                             batch_idx_ct9, item_ct1, s_kv_acc_ct1, s_query_acc_ct1,           \
                             s_logits_acc_ct1);                                                \
                       });                                                                     \
    });                                                                                        \
  }

/*
DPCT1049:353: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1026:607: The call to cudaFuncSetAttribute was removed because SYCL currently does not support
corresponding setting.
*/
#define X_WMMA_ATTENTION_LONG(SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN)                                  \
  shared_memory_size =                                                                            \
      ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * \
      2;                                                                                          \
  if (shared_memory_size > 48 * 1024)                                                             \
    ;                                                                                             \
  grid[2] = head_num, grid[1] = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN, grid[0] = batch_size,      \
  block[2] = 32 * (SPLIT_LEN / 16 * split_count);                                                 \
  {                                                                                               \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});                     \
                                                                                                  \
    stream->submit([&](sycl::handler &cgh) {                                                      \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_memory_size),     \
                                                          cgh);                                   \
                                                                                                  \
      const sycl::half2 *query_ct0 = query;                                                       \
      const sycl::half2 *key_ct1 = key;                                                           \
      const sycl::half2 *value_ct2 = value;                                                       \
      const sycl::half *atten_mask_ct3 = atten_mask;                                              \
      const sycl::half *qk_buf_ct4 = qk_buf;                                                      \
      const sycl::half *attention_bias_ct5 = attention_bias;                                      \
      sycl::half *attention_output_ct6 = attention_output;                                        \
      const int seq_len_ct7 = seq_len;                                                            \
      const float scale_ct8 = scale;                                                              \
                                                                                                  \
      cgh.parallel_for(                                                                           \
          sycl::nd_range<3>(grid * block, block),                                                 \
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {                     \
            variety_wmma_attention_long_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>(                \
                query_ct0, key_ct1, value_ct2, atten_mask_ct3, qk_buf_ct4, attention_bias_ct5,    \
                attention_output_ct6, seq_len_ct7, scale_ct8, item_ct1,                           \
                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());           \
          });                                                                                     \
    });                                                                                           \
  }

/*
DPCT1049:354: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1026:608: The call to cudaFuncSetAttribute was removed because SYCL currently does not support
corresponding setting.
*/
#define X_WMMA_ATTENTION_LONG_RM(SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN)                               \
  shared_memory_size =                                                                            \
      ((SEQ_LEN + SPLIT_LEN) * (SIZE_PER_HEAD + SKEW_HALF) + SPLIT_LEN * (SEQ_LEN + SKEW_HALF)) * \
      2;                                                                                          \
  if (shared_memory_size > 48 * 1024)                                                             \
    ;                                                                                             \
  grid[2] = head_num, grid[1] = (SEQ_LEN + SPLIT_LEN - 1) / SPLIT_LEN, grid[0] = batch_size,      \
  block[2] = 32 * (SPLIT_LEN / 16 * split_count);                                                 \
  {                                                                                               \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});                     \
                                                                                                  \
    stream->submit([&](sycl::handler &cgh) {                                                      \
      sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shared_memory_size),     \
                                                          cgh);                                   \
                                                                                                  \
      const sycl::half2 *query_ct0 = query;                                                       \
      const sycl::half2 *key_ct1 = key;                                                           \
      const sycl::half2 *value_ct2 = value;                                                       \
      const sycl::half *atten_mask_ct3 = atten_mask;                                              \
      const sycl::half *qk_buf_ct4 = qk_buf;                                                      \
      const sycl::half *attention_bias_ct5 = attention_bias;                                      \
      sycl::half *attention_output_ct6 = attention_output;                                        \
      const int seq_len_ct7 = seq_len;                                                            \
      const float scale_ct8 = scale;                                                              \
      const int *batch_idx_ct9 = batch_idx;                                                       \
                                                                                                  \
      cgh.parallel_for(                                                                           \
          sycl::nd_range<3>(grid * block, block),                                                 \
          [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {                     \
            variety_wmma_attention_long_rm_kernel<SEQ_LEN, SIZE_PER_HEAD, SPLIT_LEN>(             \
                query_ct0, key_ct1, value_ct2, atten_mask_ct3, qk_buf_ct4, attention_bias_ct5,    \
                attention_output_ct6, seq_len_ct7, scale_ct8, batch_idx_ct9, item_ct1,            \
                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());           \
          });                                                                                     \
    });                                                                                           \
  }

void fused_infer(const sycl::half2 *query, const sycl::half2 *key, const sycl::half2 *value,
                 const sycl::half *atten_mask, const sycl::half *qk_buf,
                 const sycl::half *attention_bias, sycl::half *attention_output,
                 const int head_num, const int batch_size, const int seq_len, const float scale,
                 dpct::queue_ptr stream) {
  const int size_per_head_ = 64;

  sycl::range<3> grid(1, batch_size, head_num), block(1, 1, 1);
  block[2] = 32 * ((seq_len + 15) / 16) * std::max(((seq_len + 15) / 16), size_per_head_ / 16);

  if (seq_len <= 16)
    X_WMMA_ATTENTION(16, 64);
  else if (seq_len <= 32)
    X_WMMA_ATTENTION(32, 64);
  else if (seq_len <= 48)
    X_WMMA_ATTENTION(48, 64);
  else if (seq_len <= 64)
    X_WMMA_ATTENTION(64, 64);
  else if (seq_len <= 80)
    X_WMMA_ATTENTION(80, 64);
}

void fused_rm_infer(const sycl::half2 *query, const sycl::half2 *key, const sycl::half2 *value,
                    const sycl::half *atten_mask, const sycl::half *qk_buf,
                    const sycl::half *attention_bias, sycl::half *attention_output,
                    const int head_num, const int batch_size, const int seq_len, const float scale,
                    dpct::queue_ptr stream, const int *batch_idx) {
  const int size_per_head_ = 64;
  sycl::range<3> grid(1, batch_size, head_num), block(1, 1, 1);
  block[2] = 32 * ((seq_len + 15) / 16) * std::max(((seq_len + 15) / 16), size_per_head_ / 16);

  if (seq_len <= 16)
    X_WMMA_ATTENTION_RM(16, 64);
  else if (seq_len <= 32)
    X_WMMA_ATTENTION_RM(32, 64);
  else if (seq_len <= 48)
    X_WMMA_ATTENTION_RM(48, 64);
  else if (seq_len <= 64)
    X_WMMA_ATTENTION_RM(64, 64);
  else if (seq_len <= 80)
    X_WMMA_ATTENTION_RM(80, 64);
}

void fused_long_infer(const sycl::half2 *query, const sycl::half2 *key, const sycl::half2 *value,
                      const sycl::half *atten_mask, const sycl::half *qk_buf,
                      const sycl::half *attention_bias, sycl::half *attention_output,
                      const int head_num, const int batch_size, const int seq_len,
                      const float scale, dpct::queue_ptr stream) {
  sycl::range<3> grid(1, 1, 1), block(1, 1, 1);
  int shared_memory_size = 0;
  const int split_count = (seq_len + 15) / 16;
  switch (split_count) {
    case 6:
      X_WMMA_ATTENTION_LONG(96, 64, 48);
      break;  //  80 < seq_len <=  96
    case 7:
      X_WMMA_ATTENTION_LONG(112, 64, 64);
      break;  //  96 < seq_len <= 112
    case 8:
      X_WMMA_ATTENTION_LONG(128, 64, 64);
      break;  // 112 < seq_len <= 128
    case 9:
      X_WMMA_ATTENTION_LONG(144, 64, 48);
      break;  // 128 < seq_len <= 144
    case 10:
      X_WMMA_ATTENTION_LONG(160, 64, 48);
      break;  // 144 < seq_len <= 160
    case 11:
      X_WMMA_ATTENTION_LONG(176, 64, 32);
      break;  // 160 < seq_len <= 176
    case 12:
      X_WMMA_ATTENTION_LONG(192, 64, 32);
      break;  // 176 < seq_len <= 192
    case 13:
      X_WMMA_ATTENTION_LONG(208, 64, 32);
      break;  // 192 < seq_len <= 208
    case 14:
      X_WMMA_ATTENTION_LONG(224, 64, 32);
      break;  // 208 < seq_len <= 224
    case 15:
      X_WMMA_ATTENTION_LONG(240, 64, 32);
      break;  // 224 < seq_len <= 240
    case 16:
      X_WMMA_ATTENTION_LONG(256, 64, 32);
      break;  // 240 < seq_len <= 256
    case 17:
      X_WMMA_ATTENTION_LONG(272, 64, 16);
      break;  // 256 < seq_len <= 272
    case 18:
      X_WMMA_ATTENTION_LONG(288, 64, 16);
      break;  // 272 < seq_len <= 288
    case 19:
      X_WMMA_ATTENTION_LONG(304, 64, 16);
      break;  // 288 < seq_len <= 304
    case 20:
      X_WMMA_ATTENTION_LONG(320, 64, 16);
      break;  // 304 < seq_len <= 320
    case 21:
      X_WMMA_ATTENTION_LONG(336, 64, 16);
      break;  // 320 < seq_len <= 336
    case 22:
      X_WMMA_ATTENTION_LONG(352, 64, 16);
      break;  // 336 < seq_len <= 352
  }
}

void fused_long_rm_infer(const sycl::half2 *query, const sycl::half2 *key,
                         const sycl::half2 *value, const sycl::half *atten_mask,
                         const sycl::half *qk_buf, const sycl::half *attention_bias,
                         sycl::half *attention_output, const int head_num, const int batch_size,
                         const int seq_len, const float scale, dpct::queue_ptr stream,
                         const int *batch_idx) {
  sycl::range<3> grid(1, 1, 1), block(1, 1, 1);
  int shared_memory_size = 0;
  const int split_count = (seq_len + 15) / 16;
  switch (split_count) {
    case 6:
      X_WMMA_ATTENTION_LONG_RM(96, 64, 48);
      break;  //  80 < seq_len <=  96
    case 7:
      X_WMMA_ATTENTION_LONG_RM(112, 64, 64);
      break;  //  96 < seq_len <= 112
    case 8:
      X_WMMA_ATTENTION_LONG_RM(128, 64, 64);
      break;  // 112 < seq_len <= 128
    case 9:
      X_WMMA_ATTENTION_LONG_RM(144, 64, 48);
      break;  // 128 < seq_len <= 144
    case 10:
      X_WMMA_ATTENTION_LONG_RM(160, 64, 48);
      break;  // 144 < seq_len <= 160
    case 11:
      X_WMMA_ATTENTION_LONG_RM(176, 64, 32);
      break;  // 160 < seq_len <= 176
    case 12:
      X_WMMA_ATTENTION_LONG_RM(192, 64, 32);
      break;  // 176 < seq_len <= 192
    case 13:
      X_WMMA_ATTENTION_LONG_RM(208, 64, 32);
      break;  // 192 < seq_len <= 208
    case 14:
      X_WMMA_ATTENTION_LONG_RM(224, 64, 32);
      break;  // 208 < seq_len <= 224
    case 15:
      X_WMMA_ATTENTION_LONG_RM(240, 64, 32);
      break;  // 224 < seq_len <= 240
    case 16:
      X_WMMA_ATTENTION_LONG_RM(256, 64, 32);
      break;  // 240 < seq_len <= 256
    case 17:
      X_WMMA_ATTENTION_LONG_RM(272, 64, 16);
      break;  // 256 < seq_len <= 272
    case 18:
      X_WMMA_ATTENTION_LONG_RM(288, 64, 16);
      break;  // 272 < seq_len <= 288
    case 19:
      X_WMMA_ATTENTION_LONG_RM(304, 64, 16);
      break;  // 288 < seq_len <= 304
    case 20:
      X_WMMA_ATTENTION_LONG_RM(320, 64, 16);
      break;  // 304 < seq_len <= 320
    case 21:
      X_WMMA_ATTENTION_LONG_RM(336, 64, 16);
      break;  // 320 < seq_len <= 336
    case 22:
      X_WMMA_ATTENTION_LONG_RM(352, 64, 16);
      break;  // 336 < seq_len <= 352
  }
}

template <typename T>
void variety_attention_fused_infer(const T *q, const T *k, const T *v, const T *atten_mask,
                                   const T *qk_buf, const T *attention_bias, T *attention_output,
                                   const int head_num, const int batch_size, const int seq_len,
                                   const int size_per_head, const float scale,
                                   dpct::queue_ptr stream, const int *batch_idx) {
  // query/key/value  [head_num, batch_size, seq_len, size_per_head]
  // atten_mask       [batch_size, seq_len, seq_len]
  // qk_buf           [head_num, batch_size, seq_len, seq_len]
  // attention_bias   [head_num, seq_len, seq_len]
  // attention_output [batch_size, seq_len, hidden_dim]

  // assert size_per_head == 64
  // assert seq_len <= 352

  const sycl::half2 *query = (const sycl::half2 *)q;
  const sycl::half2 *key = (const sycl::half2 *)k;
  const sycl::half2 *value = (const sycl::half2 *)v;

  if (seq_len <= 80) {
    if (batch_idx == NULL)
      fused_infer(query, key, value, atten_mask, qk_buf, attention_bias, attention_output,
                  head_num, batch_size, seq_len, scale, stream);
    else
      fused_rm_infer(query, key, value, atten_mask, qk_buf, attention_bias, attention_output,
                     head_num, batch_size, seq_len, scale, stream, batch_idx);
  } else {
    if (batch_idx == NULL)
      fused_long_infer(query, key, value, atten_mask, qk_buf, attention_bias, attention_output,
                       head_num, batch_size, seq_len, scale, stream);
    else
      fused_long_rm_infer(query, key, value, atten_mask, qk_buf, attention_bias, attention_output,
                          head_num, batch_size, seq_len, scale, stream, batch_idx);
  }
}
}  // namespace bytetransformer
