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
#include "bytetransformer/include/common.h"
#include <cmath>

namespace bytetransformer {
#define SKEW_HALF 8  // offset for avoding bank conflict
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <const int max_seq_len, const int size_per_head>
/*
DPCT1110:416: The total declared local variable size in device function wmma_attention_kernel_16
exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find
the total register size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/

// __launch_bounds__(512,4)//THREADS_PER_BLOCK
void wmma_attention_kernel_16(const sycl::half2 *qkv, const sycl::half2 *qkv_bias,
                              const sycl::half *attention_mask, sycl::half *attention_output,
                              const int seq_len, const float scale,
                              const sycl::nd_item<3> &item_ct1,
                              sycl::local_accessor<sycl::half, 2> s_kv,
                              sycl::local_accessor<sycl::half, 2> s_query,
                              sycl::local_accessor<sycl::half, 2> s_logits) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  const int warpNums = (item_ct1.get_local_range(2) >> 5);
  const int warpId = (item_ct1.get_local_id(2) >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = item_ct1.get_group_range(2) * (size_per_head / 2);
  const int batch_seq_offset = item_ct1.get_group(1) * seq_len;
  const int from_size = max_seq_len / 16;
  const int to_size = max_seq_len / 16;

  const int quart_warpId = item_ct1.get_local_id(2) >> 3;
  const int quart_warp_tid = item_ct1.get_local_id(2) & 0x7;
  const int quart_thread_offset = item_ct1.get_group(2) * (size_per_head / 2) + quart_warp_tid;

  // loading Query & Key
  /*
  DPCT1098:765: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 q_bias = qkv_bias[quart_thread_offset];
  /*
  DPCT1098:766: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 k_bias = qkv_bias[quart_thread_offset + half_hidden_dim];
  for (int seq_id = quart_warpId; seq_id < seq_len; seq_id += warpNums * 4) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + quart_thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (quart_warp_tid << 1);
    /*
    DPCT1098:767: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = qkv[pos] + q_bias;
    /*
    DPCT1098:768: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = qkv[pos + half_hidden_dim] + k_bias;
  }

  /*
  DPCT1065:771: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:417: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:418: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:419: Migration of nvcuda::wmma::row_major type is not supported.
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
    for (int k = 0; k < 1; k++) {
      /*
      DPCT1007:438: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      /*
      DPCT1007:439: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      /*
      DPCT1007:426: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    /*
    DPCT1007:437: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }

  /*
  DPCT1065:772: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  // softmax
  for (int from_id = warpId; from_id < seq_len; from_id += warpNums) {
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
            DPCT1098:775: The '*' expression is used instead of the __ldg call. These two
            expressions do not provide the exact same functionality. Check the generated code for
            potential precision and/or performance issues.
            */
            (float)attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]];
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
  DPCT1098:769: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 v_bias = qkv_bias[quart_thread_offset + half_hidden_dim * 2];
  for (int seq_id = quart_warpId; seq_id < seq_len; seq_id += warpNums * 4) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + quart_thread_offset;
    ((sycl::half2 *)(s_kv[seq_id]))[quart_warp_tid] =
        /*
        DPCT1098:770: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        qkv[pos + half_hidden_dim * 2] + v_bias;
  }

  // K dim clear 0
  for (int seq_id = seq_len + quart_warpId; seq_id < max_seq_len; seq_id += warpNums * 4)
    ((float *)(s_kv[seq_id]))[quart_warp_tid] = 0.0f;
  /*
  DPCT1065:773: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < from_size) {
    /*
    DPCT1082:427: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:428: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:429: Migration of nvcuda::wmma::row_major type is not supported.
    */
    wmma::fragment < wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> WMMA_N,
        WMMA_K,
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, sycl::half, wmma::row_major> MMA_N,
        WMMA_K,
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, sycl::half> t int
            warp_from_offset = (warpId) << 4;
    const int warp_to_offset = 0;

#pragma unroll
    for (int k = 0; k < to_size; k++) {
      /*
      DPCT1007:441: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:442: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:436: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:440: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:774: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  for (int from_id = quart_warpId; from_id < seq_len; from_id += warpNums * 4) {
    int pos = (batch_seq_offset + from_id) * half_hidden_dim + quart_thread_offset;
    ((sycl::half2 *)(attention_output))[pos] = ((sycl::half2 *)(s_query[from_id]))[quart_warp_tid];
  }
#endif
}

template <const int max_seq_len, const int size_per_head>
/*
DPCT1110:443: The total declared local variable size in device function wmma_attention_kernel
exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find
the total register size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/

// __launch_bounds__(512,4)//THREADS_PER_BLOCK
void wmma_attention_kernel(const sycl::half2 *qkv, const sycl::half2 *qkv_bias,
                           const sycl::half *attention_mask, sycl::half *attention_output,
                           const int seq_len, const float scale, const sycl::nd_item<3> &item_ct1,
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

  // loading Query & Key
  /*
  DPCT1098:776: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 q_bias = qkv_bias[thread_offset];
  /*
  DPCT1098:777: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 k_bias = qkv_bias[thread_offset + half_hidden_dim];
  for (int seq_id = warpId; seq_id < seq_len; seq_id += warpNums) {
    int pos = (batch_seq_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:778: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = qkv[pos] + q_bias;
    /*
    DPCT1098:779: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = qkv[pos + half_hidden_dim] + k_bias;
  }
  /*
  DPCT1065:782: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:444: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:445: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:446: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:465: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      /*
      DPCT1007:466: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      /*
      DPCT1007:453: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    /*
    DPCT1007:464: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:783: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  // softmax
  /*
  DPCT1098:780: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 v_bias = qkv_bias[thread_offset + half_hidden_dim * 2];
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
        float mask =
            /*
            DPCT1098:786: The '*' expression is used instead of the __ldg call. These two
            expressions do not provide the exact same functionality. Check the generated code for
            potential precision and/or performance issues.
            */
            (float)attention_mask[(batch_seq_offset + from_id) * seq_len + to_id[i]];
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

    // loading Value
    int pos = (batch_seq_offset + from_id) * (half_hidden_dim * 3) + thread_offset;
    ((sycl::half2 *)(s_kv[from_id]))[warp_tid] =
        /*
        DPCT1098:781: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        qkv[pos + half_hidden_dim * 2] + v_bias;
  }

  // K dim clear 0
  for (int seq_id = seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:784: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:454: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:455: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:456: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:468: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:469: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:463: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:467: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:785: Consider replacing sycl::nd_item::barrier() with
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
DPCT1110:470: The total declared local variable size in device function wmma_attention_rm_kernel
exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find
the total register size available and adjust the code, or use smaller sub-group size to avoid high
register pressure.
*/

// __launch_bounds__(512,4)//THREADS_PER_BLOCK
void wmma_attention_rm_kernel(const sycl::half2 *qkv, const sycl::half2 *qkv_bias,
                              const sycl::half *attention_mask, sycl::half *attention_output,
                              const float scale, const int *batch_idx,
                              const sycl::nd_item<3> &item_ct1,
                              sycl::local_accessor<sycl::half, 2> s_kv,
                              sycl::local_accessor<sycl::half, 2> s_query,
                              sycl::local_accessor<sycl::half, 2> s_logits) {
#if DPCT_COMPATIBILITY_TEMP >= 700

  const int warpNums = (item_ct1.get_local_range(2) >> 5);
  const int warpId = (item_ct1.get_local_id(2) >> 5);
  const int warp_tid = getLaneId();
  const int half_hidden_dim = item_ct1.get_group_range(2) * (size_per_head / 2);
  const int thread_offset = item_ct1.get_group(2) * (size_per_head / 2) + warp_tid;
  /*
  DPCT1098:787: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[item_ct1.get_group(1)];
  /*
  DPCT1098:788: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[item_ct1.get_group(1) + 1] - batch_offset;
  const int from_size = (batch_seq_len + 15) >> 4;
  const int to_size = (batch_seq_len + 15) >> 4;

  // loading Query & Key
  /*
  DPCT1098:789: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 q_bias = qkv_bias[thread_offset];
  /*
  DPCT1098:790: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 k_bias = qkv_bias[thread_offset + half_hidden_dim];
  for (int seq_id = warpId; seq_id < batch_seq_len; seq_id += warpNums) {
    int pos = (batch_offset + seq_id) * (half_hidden_dim * 3) + thread_offset;
    int offset = seq_id * (size_per_head + SKEW_HALF) + (warp_tid << 1);
    /*
    DPCT1098:791: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_query + offset) = qkv[pos] + q_bias;
    /*
    DPCT1098:792: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    *(sycl::half2 *)(*s_kv + offset) = qkv[pos + half_hidden_dim] + k_bias;
  }
  /*
  DPCT1065:795: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (warpId < from_size * to_size) {
    /*
    DPCT1082:471: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:472: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:473: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:492: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Q_mat, s_query[warp_from_offset] + k * WMMA_K,
                             size_per_head + SKEW_HALF);
      /*
      DPCT1007:493: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(K_mat, s_kv[warp_to_offset] + k * WMMA_K, size_per_head + SKEW_HALF);
      /*
      DPCT1007:480: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QK_mat, Q_mat, K_mat, QK_mat);
    }
    /*
    DPCT1007:491: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_logits[warp_from_offset] + warp_to_offset, QK_mat,
                            max_seq_len + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:796: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  // softmax
  /*
  DPCT1098:793: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 v_bias = qkv_bias[thread_offset + half_hidden_dim * 2];
  for (int from_id = warpId; from_id < batch_seq_len; from_id += warpNums) {
    const int n = (max_seq_len + 31) / 32;
    float logits[n];
    int to_id[n];

    float max_val = -1e20f;
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
      if (to_id[i] < max_seq_len)
        s_logits[from_id][to_id[i]] = (sycl::half)logits[i] / sum_val;

    // loading Value
    int pos = (batch_offset + from_id) * (half_hidden_dim * 3) + thread_offset;
    ((sycl::half2 *)(s_kv[from_id]))[warp_tid] =
        /*
        DPCT1098:794: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        qkv[pos + half_hidden_dim * 2] + v_bias;
  }

  // K dim clear 0
  for (int seq_id = batch_seq_len + warpId; seq_id < max_seq_len; seq_id += warpNums)
    ((float *)(s_kv[seq_id]))[warp_tid] = 0.0f;
  /*
  DPCT1065:797: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  //* V
  if (warpId < (from_size << 2)) {
    /*
    DPCT1082:481: Migration of nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half,
    wmma::row_major> type is not supported.
    */
    /*
    DPCT1082:482: Migration of nvcuda::wmma::matrix_a type is not supported.
    */
    /*
    DPCT1082:483: Migration of nvcuda::wmma::row_major type is not supported.
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
      DPCT1007:495: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(Logits_mat, s_logits[warp_from_offset] + k * WMMA_K,
                             max_seq_len + SKEW_HALF);
      /*
      DPCT1007:496: Migration of nvcuda::wmma::load_matrix_sync is not supported.
      */
      wmma::load_matrix_sync(V_mat, s_kv[k * WMMA_K] + warp_to_offset, size_per_head + SKEW_HALF);
      /*
      DPCT1007:490: Migration of nvcuda::wmma::mma_sync is not supported.
      */
      wmma::mma_sync(QKV_mat, Logits_mat, V_mat, QKV_mat);
    }
    /*
    DPCT1007:494: Migration of nvcuda::wmma::store_matrix_sync is not supported.
    */
    wmma::store_matrix_sync(s_query[warp_from_offset] + warp_to_offset, QKV_mat,
                            size_per_head + SKEW_HALF, wmma::mem_row_major);
  }
  /*
  DPCT1065:798: Consider replacing sycl::nd_item::barrier() with
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

/*
DPCT1049:498: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define WMMA_ATTENTION_16(SEQ_LEN, SIZE_PER_HEAD)                                                 \
  {                                                                                               \
    dpct::has_capability_or_fail(infer_param.stream->get_device(), {sycl::aspect::fp16});         \
                                                                                                  \
    infer_param.stream->submit([&](sycl::handler &cgh) {                                          \
      sycl::local_accessor<sycl::half, 2> s_kv_acc_ct1(                                           \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                               \
      sycl::local_accessor<sycl::half, 2> s_query_acc_ct1(                                        \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                               \
      sycl::local_accessor<sycl::half, 2> s_logits_acc_ct1(                                       \
          sycl::range<2>(SEQ_LEN, SEQ_LEN + SKEW_HALF), cgh);                                     \
                                                                                                  \
      const sycl::half2 *qkv_ptr_ct0 = qkv_ptr;                                                   \
      const sycl::half2 *qkv_bias_ptr_ct1 = qkv_bias_ptr;                                         \
      const sycl::half *atten_mask_ct2 = (sycl::half *)atten_mask;                                \
      sycl::half *attention_output_ct3 = (sycl::half *)attention_output;                          \
      const int seq_len_ct4 = seq_len;                                                            \
      const float scale_ct5 = scale;                                                              \
                                                                                                  \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                    \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {        \
                         wmma_attention_kernel_16<SEQ_LEN, SIZE_PER_HEAD>(                        \
                             qkv_ptr_ct0, qkv_bias_ptr_ct1, atten_mask_ct2, attention_output_ct3, \
                             seq_len_ct4, scale_ct5, item_ct1, s_kv_acc_ct1, s_query_acc_ct1,     \
                             s_logits_acc_ct1);                                                   \
                       });                                                                        \
    });                                                                                           \
  }

/*
DPCT1049:497: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define WMMA_ATTENTION(SEQ_LEN, SIZE_PER_HEAD)                                                    \
  {                                                                                               \
    dpct::has_capability_or_fail(infer_param.stream->get_device(), {sycl::aspect::fp16});         \
                                                                                                  \
    infer_param.stream->submit([&](sycl::handler &cgh) {                                          \
      sycl::local_accessor<sycl::half, 2> s_kv_acc_ct1(                                           \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                               \
      sycl::local_accessor<sycl::half, 2> s_query_acc_ct1(                                        \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                               \
      sycl::local_accessor<sycl::half, 2> s_logits_acc_ct1(                                       \
          sycl::range<2>(SEQ_LEN, SEQ_LEN + SKEW_HALF), cgh);                                     \
                                                                                                  \
      const sycl::half2 *qkv_ptr_ct0 = qkv_ptr;                                                   \
      const sycl::half2 *qkv_bias_ptr_ct1 = qkv_bias_ptr;                                         \
      const sycl::half *atten_mask_ct2 = (sycl::half *)atten_mask;                                \
      sycl::half *attention_output_ct3 = (sycl::half *)attention_output;                          \
      const int seq_len_ct4 = seq_len;                                                            \
      const float scale_ct5 = scale;                                                              \
                                                                                                  \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                    \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {        \
                         wmma_attention_kernel<SEQ_LEN, SIZE_PER_HEAD>(                           \
                             qkv_ptr_ct0, qkv_bias_ptr_ct1, atten_mask_ct2, attention_output_ct3, \
                             seq_len_ct4, scale_ct5, item_ct1, s_kv_acc_ct1, s_query_acc_ct1,     \
                             s_logits_acc_ct1);                                                   \
                       });                                                                        \
    });                                                                                           \
  }

/*
DPCT1049:499: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define WMMA_ATTENTION_RM(SEQ_LEN, SIZE_PER_HEAD)                                                 \
  {                                                                                               \
    dpct::has_capability_or_fail(infer_param.stream->get_device(), {sycl::aspect::fp16});         \
                                                                                                  \
    infer_param.stream->submit([&](sycl::handler &cgh) {                                          \
      sycl::local_accessor<sycl::half, 2> s_kv_acc_ct1(                                           \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                               \
      sycl::local_accessor<sycl::half, 2> s_query_acc_ct1(                                        \
          sycl::range<2>(SEQ_LEN, SIZE_PER_HEAD + SKEW_HALF), cgh);                               \
      sycl::local_accessor<sycl::half, 2> s_logits_acc_ct1(                                       \
          sycl::range<2>(SEQ_LEN, SEQ_LEN + SKEW_HALF), cgh);                                     \
                                                                                                  \
      const sycl::half2 *qkv_ptr_ct0 = qkv_ptr;                                                   \
      const sycl::half2 *qkv_bias_ptr_ct1 = qkv_bias_ptr;                                         \
      const sycl::half *atten_mask_ct2 = (sycl::half *)atten_mask;                                \
      sycl::half *attention_output_ct3 = (sycl::half *)attention_output;                          \
      const float scale_ct4 = scale;                                                              \
      const int *et_param_batch_idx_ct5 = et_param.batch_idx;                                     \
                                                                                                  \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                    \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {        \
                         wmma_attention_rm_kernel<SEQ_LEN, SIZE_PER_HEAD>(                        \
                             qkv_ptr_ct0, qkv_bias_ptr_ct1, atten_mask_ct2, attention_output_ct3, \
                             scale_ct4, et_param_batch_idx_ct5, item_ct1, s_kv_acc_ct1,           \
                             s_query_acc_ct1, s_logits_acc_ct1);                                  \
                       });                                                                        \
    });                                                                                           \
  }

template <OperationType OpType>
void Attention<OpType>::fused_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;

  sycl::range<3> grid(1, batch_size, head_num_), block(1, 1, 1);

  if (OpType == OperationType::HALF) {
    const sycl::half2 *qkv_ptr = (const sycl::half2 *)infer_param.qkv;
    const sycl::half2 *qkv_bias_ptr = (const sycl::half2 *)param_.attr_bias_QKV;
    float scale = (1.0f / sqrt(size_per_head_ * 1.0f)) / param_.tao;

    block[2] = 32 * ((seq_len + 15) / 16) * std::max(((seq_len + 15) / 16), size_per_head_ / 16);
    if (size_per_head_ == 64) {
      if (seq_len <= 16)
        WMMA_ATTENTION(16, 64);
      else if (seq_len <= 32)
        WMMA_ATTENTION(32, 64);
      else if (seq_len <= 48)
        WMMA_ATTENTION(48, 64);
      else if (seq_len <= 64)
        WMMA_ATTENTION(64, 64);
      else if (seq_len <= 80)
        WMMA_ATTENTION(80, 64);
    } else if (size_per_head_ == 16) {
      if (seq_len <= 48)
        WMMA_ATTENTION_16(48, 16);
    }
  }
}

template <OperationType OpType>
void Attention<OpType>::fused_rm_infer(AttentionInferParam infer_param) {
  const DataType_ *atten_mask = infer_param.atten_mask;
  DataType_ *attention_output = infer_param.attention_output;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  ET_Param et_param = infer_param.et_param;

  sycl::range<3> grid(1, batch_size, head_num_), block(1, 1, 1);

  if (OpType == OperationType::HALF) {
    const sycl::half2 *qkv_ptr = (const sycl::half2 *)infer_param.qkv;
    const sycl::half2 *qkv_bias_ptr = (const sycl::half2 *)param_.attr_bias_QKV;
    float scale = 1.0f / sqrt(size_per_head_ * 1.0f) / param_.tao;

    block[2] = 32 * ((seq_len + 15) / 16) * std::max(((seq_len + 15) / 16), size_per_head_ / 16);
    if (size_per_head_ == 64) {
      if (seq_len <= 16)
        WMMA_ATTENTION_RM(16, 64);
      else if (seq_len <= 32)
        WMMA_ATTENTION_RM(32, 64);
      else if (seq_len <= 48)
        WMMA_ATTENTION_RM(48, 64);
      else if (seq_len <= 64)
        WMMA_ATTENTION_RM(64, 64);
      else if (seq_len <= 80)
        WMMA_ATTENTION_RM(80, 64);
    }
  }
}

template void Attention<OperationType::FP32>::fused_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_infer(AttentionInferParam infer_param);
template void Attention<OperationType::FP32>::fused_rm_infer(AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::fused_rm_infer(AttentionInferParam infer_param);
}  // namespace bytetransformer
