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
#include "bytetransformer/include/attention_nofused_utils.h"

namespace bytetransformer {
SYCL_EXTERNAL template <>
void add_QKV_bias<float>(const float *QKV, const float *bias_QKV, float *q_buf, float *k_buf,
                         float *v_buf, const int batch_size, const int seq_len, const int head_num,
                         const int half_size_per_head, const bool is_roformer,
                         const sycl::nd_item<3> &item_ct1) {
  int batch_id = item_ct1.get_group(1);
  int seq_id = item_ct1.get_group(2);
  int head_id = item_ct1.get_local_id(2) / half_size_per_head;
  int id = item_ct1.get_local_id(2) % half_size_per_head;
  int src_id = (item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2)) *
                   (item_ct1.get_local_range(2) * 3) +
               item_ct1.get_local_id(2);
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;

  /*
  DPCT1098:803: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  sycl::float2 q_value = ((sycl::float2 *)QKV)[src_id],
               q_bias = ((sycl::float2 *)bias_QKV)[item_ct1.get_local_id(2)];
  sycl::float2 k_value = ((sycl::float2 *)QKV)[src_id + item_ct1.get_local_range(2)],
               /*
               DPCT1098:804: The '*' expression is used instead of the __ldg call. These two
               expressions do not provide the exact same functionality. Check the generated code
               for potential precision and/or performance issues.
               */
      k_bias = ((sycl::float2 *)bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2)];
  sycl::float2 v_value = ((sycl::float2 *)QKV)[src_id + item_ct1.get_local_range(2) * 2],
               /*
               DPCT1098:805: The '*' expression is used instead of the __ldg call. These two
               expressions do not provide the exact same functionality. Check the generated code
               for potential precision and/or performance issues.
               */
      v_bias =
          ((sycl::float2 *)bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2) * 2];
  q_value.x() += q_bias.x(), q_value.y() += q_bias.y();
  k_value.x() += k_bias.x(), k_value.y() += k_bias.y();
  v_value.x() += v_bias.x(), v_value.y() += v_bias.y();

  if (is_roformer) {
    sycl::float2 ro_q = sycl::float2(-q_value.y(), q_value.x());
    sycl::float2 ro_k = sycl::float2(-k_value.y(), k_value.x());
    float position_enc = seq_id / dpct::pow(10000.0f, id / half_size_per_head);
    float sin_pos = sycl::sin(position_enc);
    float cos_pos = sycl::cos(position_enc);
    q_value.x() = q_value.x() * cos_pos + ro_q.x() * sin_pos,
    q_value.y() = q_value.y() * cos_pos + ro_q.y() * sin_pos;
    k_value.x() = k_value.x() * cos_pos + ro_k.x() * sin_pos,
    k_value.y() = k_value.y() * cos_pos + ro_k.y() * sin_pos;
  }

  ((sycl::float2 *)q_buf)[trt_id] = q_value;
  ((sycl::float2 *)k_buf)[trt_id] = k_value;
  ((sycl::float2 *)v_buf)[trt_id] = v_value;
}

SYCL_EXTERNAL template <>
void add_QKV_bias<sycl::half>(const sycl::half *QKV, const sycl::half *bias_QKV, sycl::half *q_buf,
                              sycl::half *k_buf, sycl::half *v_buf, const int batch_size,
                              const int seq_len, const int head_num, const int half_size_per_head,
                              const bool is_roformer, const sycl::nd_item<3> &item_ct1) {
  int batch_id = item_ct1.get_group(1);
  int seq_id = item_ct1.get_group(2);
  int head_id = item_ct1.get_local_id(2) / half_size_per_head;
  int id = item_ct1.get_local_id(2) % half_size_per_head;
  int src_id = (item_ct1.get_group(1) * item_ct1.get_group_range(2) + item_ct1.get_group(2)) *
                   (item_ct1.get_local_range(2) * 3) +
               item_ct1.get_local_id(2);
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
  sycl::half2 q_value =
      /*
      DPCT1098:806: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      /*
      DPCT1064:807: Migrated __ldg call is used in a macro/template definition and may not be valid
      for all macro/template uses. Adjust the code.
      */
      ((const sycl::half2 *)QKV)[src_id] +
      ((const sycl::half2 *)bias_QKV)[item_ct1.get_local_id(2)];
  sycl::half2 k_value =
      ((const sycl::half2 *)QKV)[src_id + item_ct1.get_local_range(2)] +
      ((const sycl::half2 *)bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2)];
  sycl::half2 v_value =
      ((const sycl::half2 *)QKV)[src_id + item_ct1.get_local_range(2) * 2] +
      ((const sycl::half2 *)bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2) * 2];

  if (is_roformer) {
    sycl::half2 ro_q = sycl::half2(-q_value.y(), q_value.x());
    sycl::half2 ro_k = sycl::half2(-k_value.y(), k_value.x());
    float position_enc = seq_id / dpct::pow(10000.0f, id / half_size_per_head);
    sycl::half2 sin_pos =
        sycl::float2(sycl::sin(position_enc)).convert<sycl::half, sycl::rounding_mode::rte>();
    sycl::half2 cos_pos =
        sycl::float2(sycl::cos(position_enc)).convert<sycl::half, sycl::rounding_mode::rte>();
    q_value = q_value * cos_pos + ro_q * sin_pos;
    k_value = k_value * cos_pos + ro_k * sin_pos;
  }

  ((sycl::half2 *)q_buf)[trt_id] = q_value;
  ((sycl::half2 *)k_buf)[trt_id] = k_value;
  ((sycl::half2 *)v_buf)[trt_id] = v_value;
}

SYCL_EXTERNAL template <>
void add_QKV_bias_padding<float>(const float *QKV, const float *bias_QKV, float *q_buf,
                                 float *k_buf, float *v_buf, const int batch_size,
                                 const int seq_len, const int head_num,
                                 const int half_size_per_head, const bool is_roformer,
                                 const int *batch_idx, const int *word_idx,
                                 const sycl::nd_item<3> &item_ct1) {
  const int batch_id = item_ct1.get_group(1);
  const int seq_id = item_ct1.get_group(2);
  int head_id = item_ct1.get_local_id(2) / half_size_per_head;
  int id = item_ct1.get_local_id(2) % half_size_per_head;
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
  /*
  DPCT1098:812: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[item_ct1.get_group(1)];
  /*
  DPCT1098:813: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[item_ct1.get_group(1) + 1] - batch_offset;
  if (seq_id < batch_seq_len) {
    int src_id =
        (batch_offset + seq_id) * (item_ct1.get_local_range(2) * 3) + item_ct1.get_local_id(2);
    /*
    DPCT1098:814: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    sycl::float2 q_value = ((sycl::float2 *)QKV)[src_id],
                 q_bias = ((sycl::float2 *)bias_QKV)[item_ct1.get_local_id(2)];
    sycl::float2 k_value = ((sycl::float2 *)QKV)[src_id + item_ct1.get_local_range(2)],
                 /*
                 DPCT1098:815: The '*' expression is used instead of the __ldg call. These two
                 expressions do not provide the exact same functionality. Check the generated code
                 for potential precision and/or performance issues.
                 */
        k_bias =
            ((sycl::float2 *)bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2)];
    sycl::float2 v_value = ((sycl::float2 *)QKV)[src_id + item_ct1.get_local_range(2) * 2],
                 /*
                 DPCT1098:816: The '*' expression is used instead of the __ldg call. These two
                 expressions do not provide the exact same functionality. Check the generated code
                 for potential precision and/or performance issues.
                 */
        v_bias =
            ((sycl::float2 *)bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2) * 2];
    q_value.x() += q_bias.x(), q_value.y() += q_bias.y();
    k_value.x() += k_bias.x(), k_value.y() += k_bias.y();
    v_value.x() += v_bias.x(), v_value.y() += v_bias.y();

    if (is_roformer) {
      sycl::float2 ro_q = sycl::float2(-q_value.y(), q_value.x());
      sycl::float2 ro_k = sycl::float2(-k_value.y(), k_value.x());
      float position_enc = seq_id / dpct::pow(10000.0f, id / half_size_per_head);
      float sin_pos = sycl::sin(position_enc);
      float cos_pos = sycl::cos(position_enc);
      q_value.x() = q_value.x() * cos_pos + ro_q.x() * sin_pos,
      q_value.y() = q_value.y() * cos_pos + ro_q.y() * sin_pos;
      k_value.x() = k_value.x() * cos_pos + ro_k.x() * sin_pos,
      k_value.y() = k_value.y() * cos_pos + ro_k.y() * sin_pos;
    }

    ((sycl::float2 *)q_buf)[trt_id] = q_value;
    ((sycl::float2 *)k_buf)[trt_id] = k_value;
    ((sycl::float2 *)v_buf)[trt_id] = v_value;
  } else {
    sycl::float2 zero = sycl::float2(0.0f, 0.0f);
    ((sycl::float2 *)q_buf)[trt_id] = zero;
    ((sycl::float2 *)k_buf)[trt_id] = zero;
    ((sycl::float2 *)v_buf)[trt_id] = zero;
  }
}

SYCL_EXTERNAL template <>
void add_QKV_bias_padding<sycl::half>(const sycl::half *QKV, const sycl::half *bias_QKV,
                                      sycl::half *q_buf, sycl::half *k_buf, sycl::half *v_buf,
                                      const int batch_size, const int seq_len, const int head_num,
                                      const int half_size_per_head, const bool is_roformer,
                                      const int *batch_idx, const int *word_idx,
                                      const sycl::nd_item<3> &item_ct1) {
  const int batch_id = item_ct1.get_group(1);
  const int seq_id = item_ct1.get_group(2);
  int head_id = item_ct1.get_local_id(2) / half_size_per_head;
  int id = item_ct1.get_local_id(2) % half_size_per_head;
  int trt_id = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half_size_per_head + id;
  /*
  DPCT1098:817: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_offset = batch_idx[item_ct1.get_group(1)];
  /*
  DPCT1098:818: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  const int batch_seq_len = batch_idx[item_ct1.get_group(1) + 1] - batch_offset;
  if (seq_id < batch_seq_len) {
    int src_id =
        (batch_offset + seq_id) * (item_ct1.get_local_range(2) * 3) + item_ct1.get_local_id(2);
    sycl::half2 q_value =
        /*
        DPCT1098:819: The '*' expression is used instead of the __ldg call. These two expressions
        do not provide the exact same functionality. Check the generated code for potential
        precision and/or performance issues.
        */
        /*
        DPCT1064:820: Migrated __ldg call is used in a macro/template definition and may not be
        valid for all macro/template uses. Adjust the code.
        */
        ((const sycl::half2 *)QKV)[src_id] +
        ((const sycl::half2 *)bias_QKV)[item_ct1.get_local_id(2)];
    sycl::half2 k_value =
        ((const sycl::half2 *)QKV)[src_id + item_ct1.get_local_range(2)] +
        ((const sycl::half2 *)bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2)];
    sycl::half2 v_value =
        ((const sycl::half2 *)QKV)[src_id + item_ct1.get_local_range(2) * 2] +
        ((const sycl::half2 *)
             bias_QKV)[item_ct1.get_local_id(2) + item_ct1.get_local_range(2) * 2];

    if (is_roformer) {
      sycl::half2 ro_q = sycl::half2(-q_value.y(), q_value.x());
      sycl::half2 ro_k = sycl::half2(-k_value.y(), k_value.x());
      float position_enc = seq_id / dpct::pow(10000.0f, id / half_size_per_head);
      sycl::half2 sin_pos =
          sycl::float2(sycl::sin(position_enc)).convert<sycl::half, sycl::rounding_mode::rte>();
      sycl::half2 cos_pos =
          sycl::float2(sycl::cos(position_enc)).convert<sycl::half, sycl::rounding_mode::rte>();
      q_value = q_value * cos_pos + ro_q * sin_pos;
      k_value = k_value * cos_pos + ro_k * sin_pos;
    }

    ((sycl::half2 *)q_buf)[trt_id] = q_value;
    ((sycl::half2 *)k_buf)[trt_id] = k_value;
    ((sycl::half2 *)v_buf)[trt_id] = v_value;
  } else {
    ((float *)q_buf)[trt_id] = 0.0f;
    ((float *)k_buf)[trt_id] = 0.0f;
    ((float *)v_buf)[trt_id] = 0.0f;
  }
}

SYCL_EXTERNAL template <>
void transpose<float>(const float *src, float *dst, const int batch_size, const int seq_len,
                      const int head_num, const int size_per_head,
                      const sycl::nd_item<3> &item_ct1) {
  int batch_id = item_ct1.get_group(2) / seq_len;
  int seq_id = item_ct1.get_group(2) % seq_len;
  int head_id = item_ct1.get_local_id(1);
  int src_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head +
                   item_ct1.get_local_id(2);
  int dst_offset =
      (item_ct1.get_group(2) * head_num + head_id) * size_per_head + item_ct1.get_local_id(2);
  dst[dst_offset] = src[src_offset];
}

SYCL_EXTERNAL template <>
void transpose<sycl::half>(const sycl::half *src, sycl::half *dst, const int batch_size,
                           const int seq_len, const int head_num, const int size_per_head,
                           const sycl::nd_item<3> &item_ct1) {
  int batch_id = item_ct1.get_group(2) / seq_len;
  int seq_id = item_ct1.get_group(2) % seq_len;
  int head_id = item_ct1.get_local_id(1);
  int src_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head +
                   item_ct1.get_local_id(2);
  int dst_offset =
      (item_ct1.get_group(2) * head_num + head_id) * size_per_head + item_ct1.get_local_id(2);
  ((sycl::half2 *)dst)[dst_offset] = ((const sycl::half2 *)src)[src_offset];
}

SYCL_EXTERNAL template <>
void transpose_rm_padding<float>(const float *src, float *dst, const int batch_size,
                                 const int seq_len, const int head_num, const int size_per_head,
                                 const int *batch_idx, const int *word_idx,
                                 const sycl::nd_item<3> &item_ct1) {
  int offset = word_idx[item_ct1.get_group(2)];
  int batch_id = offset / seq_len;  // batch_idx[blockIdx.x]
  int seq_id = offset % seq_len;    // word_idx[blockIdx.x]
  int head_id = item_ct1.get_local_id(1);
  int src_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head +
                   item_ct1.get_local_id(2);
  int dst_offset =
      (item_ct1.get_group(2) * head_num + head_id) * size_per_head + item_ct1.get_local_id(2);
  dst[dst_offset] = src[src_offset];
}

SYCL_EXTERNAL template <>
void transpose_rm_padding<sycl::half>(const sycl::half *src, sycl::half *dst, const int batch_size,
                                      const int seq_len, const int head_num,
                                      const int size_per_head, const int *batch_idx,
                                      const int *word_idx, const sycl::nd_item<3> &item_ct1) {
  int offset = word_idx[item_ct1.get_group(2)];
  int batch_id = offset / seq_len;  // batch_idx[blockIdx.x]
  int seq_id = offset % seq_len;    // word_idx[blockIdx.x]
  int head_id = item_ct1.get_local_id(1);
  int src_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * size_per_head +
                   item_ct1.get_local_id(2);
  int dst_offset =
      (item_ct1.get_group(2) * head_num + head_id) * size_per_head + item_ct1.get_local_id(2);
  ((sycl::half2 *)dst)[dst_offset] = ((const sycl::half2 *)src)[src_offset];
}
}  // namespace bytetransformer
