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
#include "bytetransformer/include/layernorm.h"

namespace bytetransformer {
template <>
void input_layernorm<float>(float *out, const float *input, const void *gamma,
                                       const void *beta, int n, bool use_fp32,
                                       const sycl::nd_item<3> &item_ct1, float *s_, float *shared) {
  int offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);

  /*
  DPCT1098:691: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  float local_out = input[offset];

    // s_mean & s_variance
  layernorm(local_out, gamma, beta, out + offset, n, s_, item_ct1, shared);
}

template <>
void input_layernorm<sycl::half>(sycl::half *out, const sycl::half *input, const void *gamma,
                                 const void *beta, int n, bool use_fp32,
                                 const sycl::nd_item<3> &item_ct1, float *s_, float *shared) {
  int offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);

  sycl::half2 local_out((sycl::half)0.0f, (sycl::half)0.0f);
  if (item_ct1.get_local_id(2) < n / 2)
    /*
    DPCT1098:692: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out = ((const sycl::half2 *)input)[offset];

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out) + offset, n, s_, use_fp32, item_ct1,
            shared);
}

template <>
void input_compress_layernorm<float>(float *out, const float *input, const void *gamma,
                                                const void *beta, int n, bool use_fp32,
                                                float *out2, const int *batch_idx,
                                                const int *word_idx,
                                                const sycl::nd_item<3> &item_ct1, float *shared,
                                                float *s_) {
  /*
  DPCT1098:693: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  int src_offset = word_idx[item_ct1.get_group(2)] * n + item_ct1.get_local_id(2);
  int dst_offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);

  /*
  DPCT1098:694: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  float local_out = input[src_offset];
  out[dst_offset] = local_out;

  layernorm(local_out, gamma, beta, out2 + dst_offset, n, s_, item_ct1, shared);
}

template <>
void input_compress_layernorm<sycl::half>(sycl::half *out, const sycl::half *input,
                                          const void *gamma, const void *beta, int n,
                                          bool use_fp32, sycl::half *out2, const int *batch_idx,
                                          const int *word_idx, const sycl::nd_item<3> &item_ct1,
                                          float *shared, float *s_) {
  /*
  DPCT1098:695: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  int src_offset = word_idx[item_ct1.get_group(2)] * n / 2 + item_ct1.get_local_id(2);
  int dst_offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);

  /*
  DPCT1098:696: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::half2 local_out = ((const sycl::half2 *)input)[src_offset];
  ((sycl::half2 *)out)[dst_offset] = local_out;

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out2) + dst_offset, n, s_, use_fp32, item_ct1,
            shared);
}

template <>
void add_bias_input_layernorm<float>(float *out, const float *input, const float *bias,
                                                const void *gamma, const void *beta, int n,
                                                bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                                float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);

  /*
  DPCT1098:697: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  float local_out = (float)(out[offset] + input[offset] + bias[item_ct1.get_local_id(2)]);

  layernorm(local_out, gamma, beta, out + offset, n, s_, item_ct1, shared);
}

template <>
void add_bias_input_layernorm<sycl::half>(sycl::half *out, const sycl::half *input,
                                          const sycl::half *bias, const void *gamma,
                                          const void *beta, int n, bool use_fp32,
                                          const sycl::nd_item<3> &item_ct1, float *shared,
                                          float *s_) {
  int offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);

  sycl::half2 local_out((sycl::half)0.0f, (sycl::half)0.0f);
  if (item_ct1.get_local_id(2) < n / 2)
    /*
    DPCT1098:698: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    local_out = ((sycl::half2 *)out)[offset] + ((const sycl::half2 *)input)[offset] +
                ((const sycl::half2 *)bias)[item_ct1.get_local_id(2)];

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out) + offset, n, s_, use_fp32, item_ct1,
            shared);
}

template <>
void add_bias_input_layernorm_restore_output<float>(
    const float *out, const float *input, const float *bias, const void *gamma, const void *beta,
    int n, bool use_fp32, float *out2, const int *batch_idx, const int *word_idx,
    const int seq_len, const sycl::nd_item<3> &item_ct1, float *shared, float *s_) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:701: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:702: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int src_offset = (batch_offset + seq_id) * n + item_ct1.get_local_id(2);
  int dst_offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);
  if (seq_id >= batch_seq_len) {
    out2[dst_offset] = 0.0f;
    return;
  }

  float local_out =
      /*
      DPCT1098:703: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      (float)(out[src_offset] + input[src_offset] + bias[item_ct1.get_local_id(2)]);

  layernorm(local_out, gamma, beta, out2 + dst_offset, n, s_, item_ct1, shared);
}

template <>
void add_bias_input_layernorm_restore_output<sycl::half>(
    const sycl::half *out, const sycl::half *input, const sycl::half *bias, const void *gamma,
    const void *beta, int n, bool use_fp32, sycl::half *out2, const int *batch_idx,
    const int *word_idx, const int seq_len, const sycl::nd_item<3> &item_ct1, float *shared,
    float *s_) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:704: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:705: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int src_offset = (batch_offset + seq_id) * n / 2 + item_ct1.get_local_id(2);
  int dst_offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);
  if (seq_id >= batch_seq_len) {
    ((float *)out2)[dst_offset] = 0.0f;
    return;
  }

  sycl::half2 local_out = ((sycl::half2 *)out)[src_offset] +
                          ((const sycl::half2 *)input)[src_offset] +
                          ((const sycl::half2 *)bias)[item_ct1.get_local_id(2)];

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out2) + dst_offset, n, s_, use_fp32, item_ct1,
            shared);
}

// ************** for pre_norm: out + bias + input -> out2, layernorm(out2) -> out ****************
template <>
void add_bias_input_out_layernorm<float>(float *out, const float *input,
                                                    const float *bias, float *out2,
                                                    const void *gamma, const void *beta, int n,
                                                    bool use_fp32,
                                                    const sycl::nd_item<3> &item_ct1, float *shared,
                                                    float *s_) {
  int offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);

  /*
  DPCT1098:709: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  float local_out = out[offset] + input[offset] + bias[item_ct1.get_local_id(2)];
  out2[offset] = local_out;

  layernorm(local_out, gamma, beta, out + offset, n, s_, item_ct1, shared);
}

template <>
void add_bias_input_out_layernorm<sycl::half>(sycl::half *out, const sycl::half *input,
                                              const sycl::half *bias, sycl::half *out2,
                                              const void *gamma, const void *beta, int n,
                                              bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                              float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);

  sycl::half2 local_out =
      /*
      DPCT1098:710: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      ((sycl::half2 *)out)[offset] + ((const sycl::half2 *)input)[offset] +
      ((const sycl::half2 *)bias)[item_ct1.get_local_id(2)];
  ((sycl::half2 *)out2)[offset] = local_out;

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out) + offset, n, s_, use_fp32, item_ct1,
            shared);
}

// ************** for conformer: (out + bias) * 0.5 + input -> out2, layernorm(out2) -> out
// ****************
template <>
void add_bias_half_input_layernorm<float>(float *out, const float *input,
                                                     const float *bias, const void *gamma,
                                                     const void *beta, int n, bool use_fp32,
                                                     const sycl::nd_item<3> &item_ct1,
                                                     float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);

  /*
  DPCT1098:713: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  float local_out = (out[offset] + bias[item_ct1.get_local_id(2)]) * 0.5f + input[offset];

  layernorm(local_out, gamma, beta, out + offset, n, s_, item_ct1, shared);
}

template <>
void add_bias_half_input_layernorm<sycl::half>(sycl::half *out, const sycl::half *input,
                                               const sycl::half *bias, const void *gamma,
                                               const void *beta, int n, bool use_fp32,
                                               const sycl::nd_item<3> &item_ct1, float *shared,
                                               float *s_) {
  int offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);

  sycl::half2 local_out =
      /*
      DPCT1098:714: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:716: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      /*
      DPCT1064:717: Migrated __hadd2 call is used in a macro/template definition and may not be
      valid for all macro/template uses. Adjust the code.
      */
      /*
      DPCT1064:718: Migrated __hmul2 call is used in a macro/template definition and may not be
      valid for all macro/template uses. Adjust the code.
      */
      ((sycl::half2 *)out)[offset] +
      ((const sycl::half2 *)bias)[item_ct1.get_local_id(2)] * sycl::half2(0.5f, 0.5f) +
      ((const sycl::half2 *)input)[offset];

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out) + offset, n, s_, use_fp32, item_ct1,
            shared);
}

template <>
void add_bias_half_input_layernorm_restore_output<float>(
    const float *out, const float *input, const float *bias, const void *gamma, const void *beta,
    int n, bool use_fp32, float *out2, const int *batch_idx, const int *word_idx,
    const int seq_len, const sycl::nd_item<3> &item_ct1, float *shared, float *s_) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:719: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:720: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int src_offset = (batch_offset + seq_id) * n + item_ct1.get_local_id(2);
  int dst_offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);
  if (seq_id >= batch_seq_len) {
    out2[dst_offset] = 0.0f;
    return;
  }

  float local_out =
      /*
      DPCT1098:721: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      (out[src_offset] + bias[item_ct1.get_local_id(2)]) * 0.5f + input[src_offset];

  layernorm(local_out, gamma, beta, out2 + dst_offset, n, s_, item_ct1, shared);
}

template <>
void add_bias_half_input_layernorm_restore_output<sycl::half>(
    const sycl::half *out, const sycl::half *input, const sycl::half *bias, const void *gamma,
    const void *beta, int n, bool use_fp32, sycl::half *out2, const int *batch_idx,
    const int *word_idx, const int seq_len, const sycl::nd_item<3> &item_ct1, float *shared,
    float *s_) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:722: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:723: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int src_offset = (batch_offset + seq_id) * n / 2 + item_ct1.get_local_id(2);
  int dst_offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);
  if (seq_id >= batch_seq_len) {
    ((float *)out2)[dst_offset] = 0.0f;
    return;
  }

  /*
  DPCT1098:724: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:728: Migrated __hadd2 call is used in a macro/template definition and may not be valid
  for all macro/template uses. Adjust the code.
  */
  /*
  DPCT1064:729: Migrated __hmul2 call is used in a macro/template definition and may not be valid
  for all macro/template uses. Adjust the code.
  */
  sycl::half2 local_out =
      ((sycl::half2 *)out)[src_offset] +
      ((const sycl::half2 *)bias)[item_ct1.get_local_id(2)] * sycl::half2(0.5f, 0.5f) +
      ((const sycl::half2 *)input)[src_offset];

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out2) + dst_offset, n, s_, use_fp32, item_ct1,
            shared);
}

template <>
void add_bias_half_input_out_layernorm<float>(float *out, const float *input,
                                                         const float *bias, float *out2,
                                                         const void *gamma, const void *beta,
                                                         int n, bool use_fp32,
                                                         const sycl::nd_item<3> &item_ct1,
                                                         float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n + item_ct1.get_local_id(2);

  /*
  DPCT1098:730: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  float local_out = (out[offset] + bias[item_ct1.get_local_id(2)]) * 0.5f + input[offset];
  out2[offset] = local_out;

  layernorm(local_out, gamma, beta, out + offset, n, s_, item_ct1, shared);
}

template <>
void add_bias_half_input_out_layernorm<sycl::half>(sycl::half *out, const sycl::half *input,
                                                   const sycl::half *bias, sycl::half *out2,
                                                   const void *gamma, const void *beta, int n,
                                                   bool use_fp32, const sycl::nd_item<3> &item_ct1,
                                                   float *shared, float *s_) {
  int offset = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);

  sycl::half2 local_out =
      /*
      DPCT1098:731: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:733: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      /*
      DPCT1064:734: Migrated __hadd2 call is used in a macro/template definition and may not be
      valid for all macro/template uses. Adjust the code.
      */
      /*
      DPCT1064:735: Migrated __hmul2 call is used in a macro/template definition and may not be
      valid for all macro/template uses. Adjust the code.
      */
      ((sycl::half2 *)out)[offset] +
      ((const sycl::half2 *)bias)[item_ct1.get_local_id(2)] * sycl::half2(0.5f, 0.5f) +
      ((const sycl::half2 *)input)[offset];
  ((sycl::half2 *)out2)[offset] = local_out;

  layernorm(local_out, gamma, beta, ((sycl::half2 *)out) + offset, n, s_, use_fp32, item_ct1,
            shared);
}
}  // namespace bytetransformer
