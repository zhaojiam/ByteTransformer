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
#include "bytetransformer/include/remove_padding.h"

namespace bytetransformer {
template <>
void add_bias_input<float>(float *out, const float *input, const float *bias, int n,
                           const sycl::nd_item<3> &item_ct1) {
  int offset = item_ct1.get_group(2) * n;
  for (int i = item_ct1.get_local_id(2); i < n; i += item_ct1.get_local_range(2)) {
    int index = offset + i;
    /*
    DPCT1098:739: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    out[index] = out[index] + input[index] + bias[i];
  }
}

template <>
void add_bias_input<sycl::half>(sycl::half *out, const sycl::half *input, const sycl::half *bias,
                                int n, const sycl::nd_item<3> &item_ct1) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

  int id = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);
  out_ptr[id] =
      /*
      DPCT1098:740: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:741: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      out_ptr[id] + input_ptr[id] + bias_ptr[item_ct1.get_local_id(2)];
}

template <>
void add_bias_input_restore_output<float>(const float *out, const float *input,
                                                     const float *bias, int n, float *out2,
                                                     const int *batch_idx, const int *word_idx,
                                                     const int seq_len,
                                                     const sycl::nd_item<3> &item_ct1) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:742: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:743: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int src_offset = (batch_offset + seq_id) * n;
  int dst_offset = item_ct1.get_group(2) * n;
  if (seq_id >= batch_seq_len) {
    for (int i = item_ct1.get_local_id(2); i < n; i += item_ct1.get_local_range(2))
      out2[dst_offset + i] = 0.0f;
    return;
  }

  for (int i = item_ct1.get_local_id(2); i < n; i += item_ct1.get_local_range(2)) {
    int index = src_offset + i;
    /*
    DPCT1098:744: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    out2[dst_offset + i] = out[index] + input[index] + bias[i];
  }
}

template <>
void add_bias_input_restore_output<sycl::half>(const sycl::half *out, const sycl::half *input,
                                               const sycl::half *bias, int n, sycl::half *out2,
                                               const int *batch_idx, const int *word_idx,
                                               const int seq_len,
                                               const sycl::nd_item<3> &item_ct1) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:745: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:746: The '*' expression is used instead of the __ldg call. These two expressions do not
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

  ((sycl::half2 *)out2)[dst_offset] =
      out_ptr[src_offset] + input_ptr[src_offset] + bias_ptr[item_ct1.get_local_id(2)];
}

template <>
void add_bias_half_input<float>(float *out, const float *input, const float *bias,
                                           int n, const sycl::nd_item<3> &item_ct1) {
  int offset = item_ct1.get_group(2) * n;
  for (int i = item_ct1.get_local_id(2); i < n; i += item_ct1.get_local_range(2)) {
    int index = offset + i;
    /*
    DPCT1098:749: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    out[index] = input[index] + (out[index] + bias[i]) * 0.5f;
  }
}

template <>
void add_bias_half_input<sycl::half>(sycl::half *out, const sycl::half *input,
                                     const sycl::half *bias, int n,
                                     const sycl::nd_item<3> &item_ct1) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

  int id = item_ct1.get_group(2) * n / 2 + item_ct1.get_local_id(2);
  out_ptr[id] =
      /*
      DPCT1098:750: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      input_ptr[id] + out_ptr[id] + bias_ptr[item_ct1.get_local_id(2)] * sycl::half2(0.5f, 0.5f);
}

template <>
void add_bias_half_input_restore_output<float>(const float *out, const float *input,
                                                          const float *bias, int n, float *out2,
                                                          const int *batch_idx,
                                                          const int *word_idx, const int seq_len,
                                                          const sycl::nd_item<3> &item_ct1) {
  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:755: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:756: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[batch_id + 1] - batch_offset;
  int src_offset = (batch_offset + seq_id) * n;
  int dst_offset = item_ct1.get_group(2) * n;
  if (seq_id >= batch_seq_len) {
    for (int i = item_ct1.get_local_id(2); i < n; i += item_ct1.get_local_range(2))
      out2[dst_offset + i] = 0.0f;
    return;
  }

  for (int i = item_ct1.get_local_id(2); i < n; i += item_ct1.get_local_range(2)) {
    int index = src_offset + i;
    /*
    DPCT1098:757: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    out2[dst_offset + i] = input[index] + (out[index] + bias[i]) * 0.5f;
  }
}

template <>
void add_bias_half_input_restore_output<sycl::half>(const sycl::half *out, const sycl::half *input,
                                                    const sycl::half *bias, int n,
                                                    sycl::half *out2, const int *batch_idx,
                                                    const int *word_idx, const int seq_len,
                                                    const sycl::nd_item<3> &item_ct1) {
  sycl::half2 *out_ptr = (sycl::half2 *)out;
  const sycl::half2 *input_ptr = (const sycl::half2 *)input;
  const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

  const int batch_id = item_ct1.get_group(2) / seq_len;
  const int seq_id = item_ct1.get_group(2) % seq_len;
  /*
  DPCT1098:758: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[batch_id];
  /*
  DPCT1098:759: The '*' expression is used instead of the __ldg call. These two expressions do not
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

  ((sycl::half2 *)out2)[dst_offset] =
      /*
      DPCT1098:760: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      input_ptr[src_offset] + out_ptr[src_offset] +
      bias_ptr[item_ct1.get_local_id(2)] * sycl::half2(0.5f, 0.5f);
}
}  // namespace bytetransformer
