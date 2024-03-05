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
__inline__ void layernorm(float local_out, const void *gamma, const void *beta,
                                     float *out_ptr, int n, float *s_,
                                     const sycl::nd_item<3> &item_ct1, float *shared) {
  float sum = blockReduceSum<float>(local_out, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[0] = sum / n;
  /*
  DPCT1065:392: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  local_out -= s_[0];
  float variance = blockReduceSum<float>(local_out * local_out, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[1] = sycl::rsqrt(variance / n + 1e-6f);
  /*
  DPCT1065:393: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  /*
  DPCT1098:395: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:578: Migrated __ldg call is used in a macro/template definition and may not be valid for
  all macro/template uses. Adjust the code.
  */
  *out_ptr = local_out * s_[1] * ((float *)gamma)[item_ct1.get_local_id(2)] +
             /*
             DPCT1098:394: The '*' expression is used instead of the __ldg call. These two
             expressions do not provide the exact same functionality. Check the generated code for
             potential precision and/or performance issues.
             */
             /*
             DPCT1064:577: Migrated __ldg call is used in a macro/template definition and may not
             be valid for all macro/template uses. Adjust the code.
             */
             ((float *)beta)[item_ct1.get_local_id(2)];
}

__inline__ void layernorm(sycl::half2 local_out, const void *gamma, const void *beta,
                          sycl::half2 *out_ptr, int n, float *s_, bool use_fp32,
                          const sycl::nd_item<3> &item_ct1, float *shared) {
  sycl::float2 local_out_fp2 = local_out.convert<float, sycl::rounding_mode::automatic>();
  float t_sum = local_out_fp2.x() + local_out_fp2.y();
  float sum = blockReduceSum<float>(t_sum, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[0] = sum / n;
  /*
  DPCT1065:396: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  local_out_fp2.x() -= s_[0];
  local_out_fp2.y() -= s_[0];
  float variance = 0.0f;
  if (item_ct1.get_local_id(2) < n / 2)
    variance = local_out_fp2.x() * local_out_fp2.x() + local_out_fp2.y() * local_out_fp2.y();
  variance = blockReduceSum<float>(variance, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[1] = sycl::rsqrt(variance / n + 1e-6f);
  /*
  DPCT1065:397: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  if (item_ct1.get_local_id(2) < n / 2) {
    sycl::float2 gamma_val, beta_val;
    if (use_fp32) {
      /*
      DPCT1098:398: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:579: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      gamma_val = ((const sycl::float2 *)gamma)[item_ct1.get_local_id(2)];
      /*
      DPCT1098:399: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:580: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      beta_val = ((const sycl::float2 *)beta)[item_ct1.get_local_id(2)];
    } else {
      /*
      DPCT1098:400: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:401: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      /*
      DPCT1064:581: Migrated __half22float2 call is used in a macro/template definition and may not
      be valid for all macro/template uses. Adjust the code.
      */
      gamma_val = ((const sycl::half2 *)gamma)[item_ct1.get_local_id(2)]
                      .convert<float, sycl::rounding_mode::automatic>();
      /*
      DPCT1098:402: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:403: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      /*
      DPCT1064:582: Migrated __half22float2 call is used in a macro/template definition and may not
      be valid for all macro/template uses. Adjust the code.
      */
      beta_val = ((const sycl::half2 *)beta)[item_ct1.get_local_id(2)]
                     .convert<float, sycl::rounding_mode::automatic>();
    }

    local_out_fp2.x() = local_out_fp2.x() * s_[1] * gamma_val.x() + beta_val.x();
    local_out_fp2.y() = local_out_fp2.y() * s_[1] * gamma_val.y() + beta_val.y();
    *out_ptr = local_out_fp2.convert<sycl::half, sycl::rounding_mode::rte>();
  }
}

template <typename T>
void input_layernorm(T *out, const T *input, const void *gamma, const void *beta, int n,
                                bool use_fp32, const sycl::nd_item<3> &item_ct1, float *s_,
                                float *shared);

template <typename T>
void input_compress_layernorm(T *out, const T *input, const void *gamma,
                                         const void *beta, int n, bool use_fp32, T *out2,
                                         const int *batch_idx, const int *word_idx,
                                         const sycl::nd_item<3> &item_ct1, float *shared, float *s_);

template <const int ite>
__inline__ void layernorm_v2(float *local_out, float sum, const void *gamma,
                                        const void *beta, float *out_ptr, int n, float *s_,
                                        const sycl::nd_item<3> &item_ct1, float *shared) {
  float mean = blockReduceSum<float>(sum, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[0] = mean / n;
  /*
  DPCT1065:404: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out[i] -= s_[0];
    var += local_out[i] * local_out[i];
  }

  float variance = blockReduceSum<float>(var, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[1] = sycl::rsqrt(variance / n + 1e-6f);
  /*
  DPCT1065:405: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    out_ptr[col_id] =
        /*
        DPCT1098:406: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        local_out[i] * s_[1] * ((float *)gamma)[col_id] + ((float *)beta)[col_id];
  }
}

template <const int ite>
__inline__ void layernorm_v2(sycl::float2 *local_out_fp2, float sum, const void *gamma,
                             const void *beta, sycl::half2 *out_ptr, int n, float *s_,
                             bool use_fp32, const sycl::nd_item<3> &item_ct1, float *shared) {
  float mean = blockReduceSum<float>(sum, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[0] = mean / n;
  /*
  DPCT1065:407: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x() -= s_[0];
    local_out_fp2[i].y() -= s_[0];
    variance +=
        local_out_fp2[i].x() * local_out_fp2[i].x() + local_out_fp2[i].y() * local_out_fp2[i].y();
  }

  variance = blockReduceSum<float>(variance, item_ct1, shared);
  if (item_ct1.get_local_id(2) == 0)
    s_[1] = sycl::rsqrt(variance / n + 1e-6f);
  /*
  DPCT1065:408: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better performance if there is
  no access to global memory.
  */
  item_ct1.barrier();

  sycl::float2 gamma_val[ite], beta_val[ite];
  if (use_fp32) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      /*
      DPCT1098:409: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      gamma_val[i] = ((const sycl::float2 *)gamma)[col_id];
      /*
      DPCT1098:410: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      beta_val[i] = ((const sycl::float2 *)beta)[col_id];
    }
  } else {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      /*
      DPCT1098:411: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      gamma_val[i] =
          ((const sycl::half2 *)gamma)[col_id].convert<float, sycl::rounding_mode::automatic>();
      /*
      DPCT1098:412: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      beta_val[i] =
          ((const sycl::half2 *)beta)[col_id].convert<float, sycl::rounding_mode::automatic>();
    }
  }

#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x() = local_out_fp2[i].x() * s_[1] * gamma_val[i].x() + beta_val[i].x();
    local_out_fp2[i].y() = local_out_fp2[i].y() * s_[1] * gamma_val[i].y() + beta_val[i].y();
    out_ptr[i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2)] =
        local_out_fp2[i].convert<sycl::half, sycl::rounding_mode::rte>();
  }
}

template <const int ite>
void input_layernorm_v2(float *out, const float *input, const void *gamma,
                                   const void *beta, int n, bool use_fp32,
                                   const sycl::nd_item<3> &item_ct1, float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n;
  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int id = offset + (i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2));
    /*
    DPCT1098:413: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = input[id];
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
void input_layernorm_v2(sycl::half *out, const sycl::half *input, const void *gamma,
                        const void *beta, int n, bool use_fp32, const sycl::nd_item<3> &item_ct1,
                        float *shared, float *s_) {
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  int offset = item_ct1.get_group(2) * n / 2;

  sycl::float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int id = offset + (i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2));
    /*
    DPCT1098:414: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out_fp2[i] = input_ptr[id].convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <const int ite>
void input_compress_layernorm_v2(float *out, const float *input, const void *gamma,
                                            const void *beta, int n, bool use_fp32, float *out2,
                                            const int *batch_idx, const int *word_idx,
                                            const sycl::nd_item<3> &item_ct1, float *shared,
                                            float *s_) {
  /*
  DPCT1098:415: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:583: Migrated __ldg call is used in a macro/template definition and may not be valid for
  all macro/template uses. Adjust the code.
  */
  int from_offset = word_idx[item_ct1.get_group(2)] * n;
  int offset = item_ct1.get_group(2) * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = from_offset + col_id;
    /*
    DPCT1098:416: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = input[id];
    out[offset + col_id] = local_out[i];
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out2 + offset, n, s_);
}

template <const int ite>
void input_compress_layernorm_v2(sycl::half *out, const sycl::half *input, const void *gamma,
                                 const void *beta, int n, bool use_fp32, sycl::half *out2,
                                 const int *batch_idx, const int *word_idx,
                                 const sycl::nd_item<3> &item_ct1, float *shared, float *s_) {
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  /*
  DPCT1098:417: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:584: Migrated __ldg call is used in a macro/template definition and may not be valid for
  all macro/template uses. Adjust the code.
  */
  int from_offset = word_idx[item_ct1.get_group(2)] * n / 2;
  int offset = item_ct1.get_group(2) * n / 2;

  sycl::float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    /*
    DPCT1098:418: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    sycl::half2 temp = input_ptr[from_offset + col_id];
    ((sycl::half2 *)out)[offset + col_id] = temp;
    local_out_fp2[i] = temp.convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out2) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <typename T>
void add_bias_input_layernorm(T *out, const T *input, const T *bias, const void *gamma,
                                         const void *beta, int n, bool use_fp32,
                                         const sycl::nd_item<3> &item_ct1, float *shared, float *s_);

template <typename T>
void add_bias_input_layernorm_restore_output(const T *out, const T *input,
                                                        const T *bias, const void *gamma,
                                                        const void *beta, int n, bool use_fp32,
                                                        T *out2, const int *batch_idx,
                                                        const int *word_idx, const int seq_len,
                                                        const sycl::nd_item<3> &item_ct1,
                                                        float *shared, float *s_);

template <const int ite>
void add_bias_input_layernorm_v2(float *out, const float *input, const float *bias,
                                            const void *gamma, const void *beta, int n,
                                            bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                            float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    /*
    DPCT1098:419: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = (float)(out[id] + input[id] + bias[col_id]);
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
void add_bias_input_layernorm_v2(sycl::half *out, const sycl::half *input, const sycl::half *bias,
                                 const void *gamma, const void *beta, int n, bool use_fp32,
                                 const sycl::nd_item<3> &item_ct1, float *shared, float *s_) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

  int offset = item_ct1.get_group(2) * n / 2;

  float sum = 0.0f;
  sycl::float2 local_out_fp2[ite];
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    local_out_fp2[i] = (out_ptr[id] + input_ptr[id] + bias_ptr[col_id])
                           .convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x() + local_out_fp2[i].y();
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <const int ite>
void add_bias_input_layernorm_restore_output_v2(const float *out, const float *input,
                                                           const float *bias, const void *gamma,
                                                           const void *beta, int n, bool use_fp32,
                                                           float *out2, const int *batch_idx,
                                                           const int *word_idx,
                                                           const int seq_len,
                                                           const sycl::nd_item<3> &item_ct1,
                                                           float *shared, float *s_) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:421: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:422: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int from_offset = (batch_offset + seq_id) * n;
  int offset = item_ct1.get_group(2) * n;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      out2[offset + col_id] = 0.0f;
    }
    return;
  }

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = from_offset + col_id;
    /*
    DPCT1098:423: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = (float)(out[id] + input[id] + bias[col_id]);
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out2 + offset, n, s_);
}

template <const int ite>
void add_bias_input_layernorm_restore_output_v2(const sycl::half *out, const sycl::half *input,
                                                const sycl::half *bias, const void *gamma,
                                                const void *beta, int n, bool use_fp32,
                                                sycl::half *out2, const int *batch_idx,
                                                const int *word_idx, const int seq_len,
                                                const sycl::nd_item<3> &item_ct1, float *shared,
                                                float *s_) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:424: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:425: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int from_offset = (batch_offset + seq_id) * n / 2;
  int offset = item_ct1.get_group(2) * n / 2;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      ((float *)out2)[offset + col_id] = 0.0f;
    }
    return;
  }

  sycl::float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = from_offset + col_id;
    local_out_fp2[i] = (out_ptr[id] + input_ptr[id] + bias_ptr[col_id])
                           .convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x() + local_out_fp2[i].y();
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out2) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <typename T>
void input_layernorm_kernel_launcher(T *output, const T *input, const void *gamma,
                                     const void *beta, int m, int n, int hidden_dim,
                                     dpct::queue_ptr stream, bool use_fp32) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);

  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:193: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           input_layernorm_v2<2>(
                               output, input, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:194: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           input_layernorm_v2<4>(
                               output, input, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else
  /*
  DPCT1049:192: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);
      sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                         input_layernorm(
                             output, input, gamma, beta, n, use_fp32, item_ct1,
                             s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                             shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
  }
}

template <typename T>
void input_compress_layernorm_kernel_launcher(T *output, const T *input, const void *gamma,
                                              const void *beta, int m, int n, int hidden_dim,
                                              dpct::queue_ptr stream, bool use_fp32, T *output2,
                                              int *batch_idx, int *word_idx) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:196: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           input_compress_layernorm_v2<2>(
                               output2, input, gamma, beta, n, use_fp32, output, batch_idx,
                               word_idx, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:197: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           input_compress_layernorm_v2<4>(
                               output2, input, gamma, beta, n, use_fp32, output, batch_idx,
                               word_idx, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else
  /*
  DPCT1049:195: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
      sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                         input_compress_layernorm(
                             output2, input, gamma, beta, n, use_fp32, output, batch_idx, word_idx,
                             item_ct1,
                             shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                             s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
  }
}

template <typename T>
void add_bias_input_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                              const void *gamma, const void *beta, int m, int n,
                                              int hidden_dim, dpct::queue_ptr stream,
                                              bool use_fp32) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:198: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_input_layernorm_v2<2>(
                               output, input, bias, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:199: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_input_layernorm_v2<4>(
                               output, input, bias, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else {
    if (block[2] < 32)
      block[2] = 32;
    /*
    DPCT1049:200: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_input_layernorm(
                               output, input, bias, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  }
}

template <typename T>
void add_bias_input_layernorm_restore_output_kernel_launcher(
    T *output, const T *input, const T *bias, const void *gamma, const void *beta, int m, int n,
    int hidden_dim, dpct::queue_ptr stream, bool use_fp32, T *output2, int *batch_idx,
    int *word_idx, const int seq_len) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:202: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_input_layernorm_restore_output_v2<2>(
                               output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx,
                               word_idx, seq_len, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:203: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_input_layernorm_restore_output_v2<4>(
                               output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx,
                               word_idx, seq_len, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else
  /*
  DPCT1049:201: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
      sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                         add_bias_input_layernorm_restore_output(
                             output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx,
                             word_idx, seq_len, item_ct1,
                             shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                             s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
  }
}

// ************** for pre_norm: out + bias + input -> out2, layernorm(out2) ->
// out ****************
template <typename T>
void add_bias_input_out_layernorm(T *out, const T *input, const T *bias, T *out2,
                                             const void *gamma, const void *beta, int n,
                                             bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                             float *shared, float *s_);

template <const int ite>
void add_bias_input_out_layernorm_v2(float *out, const float *input, const float *bias,
                                                float *out2, const void *gamma, const void *beta,
                                                int n, bool use_fp32,
                                                const sycl::nd_item<3> &item_ct1, float *shared,
                                                float *s_) {
  int offset = item_ct1.get_group(2) * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    /*
    DPCT1098:427: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = out[id] + input[id] + bias[col_id];
    out2[id] = local_out[i];
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
void add_bias_input_out_layernorm_v2(sycl::half *out, const sycl::half *input,
                                     const sycl::half *bias, sycl::half *out2, const void *gamma,
                                     const void *beta, int n, bool use_fp32,
                                     const sycl::nd_item<3> &item_ct1, float *shared, float *s_) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;
  sycl::half2 *out2_ptr = (sycl::half2 *)out2;

  int offset = item_ct1.get_group(2) * n / 2;

  sycl::float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    /*
    DPCT1098:428: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    sycl::half2 temp = out_ptr[id] + input_ptr[id] + bias_ptr[col_id];
    out2_ptr[id] = temp;
    local_out_fp2[i] = temp.convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <typename T>
void add_bias_input_out_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                                  T *output2, const void *gamma, const void *beta,
                                                  int m, int n, int hidden_dim,
                                                  dpct::queue_ptr stream, bool use_fp32) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:205: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_input_out_layernorm_v2<2>(
                               output, input, bias, output2, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:206: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_input_out_layernorm_v2<4>(
                               output, input, bias, output2, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else
  /*
  DPCT1049:204: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
      sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                         add_bias_input_out_layernorm(
                             output, input, bias, output2, gamma, beta, n, use_fp32, item_ct1,
                             shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                             s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
  }
}

// ************** for conformer: (out + bias) * 0.5 + input -> out2,
// layernorm(out2) -> out ****************
template <typename T>
void add_bias_half_input_layernorm(T *out, const T *input, const T *bias,
                                              const void *gamma, const void *beta, int n,
                                              bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                              float *shared, float *s_);

template <const int ite>
void add_bias_half_input_layernorm_v2(float *out, const float *input, const float *bias,
                                                 const void *gamma, const void *beta, int n,
                                                 bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                                 float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    /*
    DPCT1098:429: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = (out[id] + bias[col_id]) * 0.5f + input[id];
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
void add_bias_half_input_layernorm_v2(sycl::half *out, const sycl::half *input,
                                      const sycl::half *bias, const void *gamma, const void *beta,
                                      int n, bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                      float *shared, float *s_) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;
  int offset = item_ct1.get_group(2) * n / 2;

  sycl::float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    local_out_fp2[i] = (out_ptr[id] + bias_ptr[col_id] * sycl::half2(0.5f, 0.5f) + input_ptr[id])
                           .convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <typename T>
void add_bias_half_input_layernorm_restore_output(
    const T *out, const T *input, const T *bias, const void *gamma, const void *beta, int n,
    bool use_fp32, T *out2, const int *batch_idx, const int *word_idx, const int seq_len,
    const sycl::nd_item<3> &item_ct1, float *shared, float *s_);

template <const int ite>
void add_bias_half_input_layernorm_restore_output_v2(
    const float *out, const float *input, const float *bias, const void *gamma, const void *beta,
    int n, bool use_fp32, float *out2, const int *batch_idx, const int *word_idx,
    const int seq_len, const sycl::nd_item<3> &item_ct1, float *shared, float *s_) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:432: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:433: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int from_offset = (batch_offset + seq_id) * n;
  int offset = item_ct1.get_group(2) * n;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      out2[offset + col_id] = 0.0f;
    }
    return;
  }

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = from_offset + col_id;
    /*
    DPCT1098:434: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = (out[id] + bias[col_id]) * 0.5f + input[id];
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out2 + offset, n, s_);
}

template <const int ite>
void add_bias_half_input_layernorm_restore_output_v2(
    const sycl::half *out, const sycl::half *input, const sycl::half *bias, const void *gamma,
    const void *beta, int n, bool use_fp32, sycl::half *out2, const int *batch_idx,
    const int *word_idx, const int seq_len, const sycl::nd_item<3> &item_ct1, float *shared,
    float *s_) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:435: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:436: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int from_offset = (batch_offset + seq_id) * n / 2;
  int offset = item_ct1.get_group(2) * n / 2;
  if (seq_id >= batch_seq_len) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
      ((float *)out2)[offset + col_id] = 0.0f;
    }
    return;
  }

  sycl::float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = from_offset + col_id;
    local_out_fp2[i] = (out_ptr[id] + bias_ptr[col_id] * sycl::half2(0.5f, 0.5f) + input_ptr[id])
                           .convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out2) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <typename T>
void add_bias_half_input_out_layernorm(T *out, const T *input, const T *bias, T *out2,
                                                  const void *gamma, const void *beta, int n,
                                                  bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                                  float *shared, float *s_);

template <const int ite>
void add_bias_half_input_out_layernorm_v2(float *out, const float *input,
                                                     const float *bias, float *out2,
                                                     const void *gamma, const void *beta, int n,
                                                     bool use_fp32,
                                                     const sycl::nd_item<3> &item_ct1,
                                                     float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    /*
    DPCT1098:439: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out[i] = (out[id] + bias[col_id]) * 0.5f + input[id];
    out2[id] = local_out[i];
    sum += local_out[i];
  }

  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
void add_bias_half_input_out_layernorm_v2(sycl::half *out, const sycl::half *input,
                                          const sycl::half *bias, sycl::half *out2,
                                          const void *gamma, const void *beta, int n,
                                          bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                          float *shared, float *s_) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  sycl::half2 *out2_ptr = (sycl::half2 *)out2;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;
  int offset = item_ct1.get_group(2) * n / 2;

  sycl::float2 local_out_fp2[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    int id = offset + col_id;
    sycl::half2 temp =
        /*
        DPCT1098:440: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        out_ptr[id] + bias_ptr[col_id] * sycl::half2(0.5f, 0.5f) + input_ptr[id];
    out2_ptr[id] = temp;
    local_out_fp2[i] = temp.convert<float, sycl::rounding_mode::automatic>();
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((sycl::half2 *)out) + offset, n, s_,
                    use_fp32, item_ct1, shared);
}

template <typename T>
void add_bias_half_input_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                                   const void *gamma, const void *beta, int m,
                                                   int n, int hidden_dim, dpct::queue_ptr stream,
                                                   bool use_fp32) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:208: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_half_input_layernorm_v2<2>(
                               output, input, bias, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:209: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_half_input_layernorm_v2<4>(
                               output, input, bias, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else
  /*
  DPCT1049:207: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
      sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                         add_bias_half_input_layernorm(
                             output, input, bias, gamma, beta, n, use_fp32, item_ct1,
                             shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                             s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
  }
}

template <typename T>
void add_bias_half_input_layernorm_restore_output_kernel_launcher(
    T *output, const T *input, const T *bias, const void *gamma, const void *beta, int m, int n,
    int hidden_dim, dpct::queue_ptr stream, bool use_fp32, T *output2, int *batch_idx,
    int *word_idx, const int seq_len) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:211: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_half_input_layernorm_restore_output_v2<2>(
                               output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx,
                               word_idx, seq_len, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:212: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_half_input_layernorm_restore_output_v2<4>(
                               output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx,
                               word_idx, seq_len, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else
  /*
  DPCT1049:210: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
      sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                         add_bias_half_input_layernorm_restore_output(
                             output, input, bias, gamma, beta, n, use_fp32, output2, batch_idx,
                             word_idx, seq_len, item_ct1,
                             shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                             s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
  }
}

template <typename T>
void add_bias_half_input_out_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                                       T *output2, const void *gamma,
                                                       const void *beta, int m, int n,
                                                       int hidden_dim, dpct::queue_ptr stream,
                                                       bool use_fp32) {
  sycl::range<3> grid(1, 1, m), block(1, 1, hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      /*
      DPCT1049:214: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 2),
                                           sycl::range<3>(1, 1, block[2] / 2)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_half_input_out_layernorm_v2<2>(
                               output, input, bias, output2, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    } else
    /*
    DPCT1049:215: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
        sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, block[2] / 4),
                                           sycl::range<3>(1, 1, block[2] / 4)),
                         [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                           add_bias_half_input_out_layernorm_v2<4>(
                               output, input, bias, output2, gamma, beta, n, use_fp32, item_ct1,
                               shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                               s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
      });
    }
  } else
  /*
  DPCT1049:213: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> shared_acc_ct1(sycl::range<1>(32), cgh);
      sycl::local_accessor<float, 1> s__acc_ct1(sycl::range<1>(2), cgh);

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                         add_bias_half_input_out_layernorm(
                             output, input, bias, output2, gamma, beta, n, use_fp32, item_ct1,
                             shared_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get(),
                             s__acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
  }
}
}  // namespace bytetransformer
