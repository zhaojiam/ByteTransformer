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
#include <cmath>

namespace bytetransformer {
template <typename T>
void softmax_kernel_warp(T *qk_buf, const T *atten_bias, const T *atten_mask,
                                    const int batch_size, const int head_num, const int seq_len,
                                    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  int word_id = item_ct1.get_group(2);
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = item_ct1.get_local_id(2);
  int head_id = item_ct1.get_local_id(1);
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * seq_len;

  auto shmem = (float *)dpct_local;
  float *s_row_qk = (float *)shmem + head_id * seq_len;

  float max_v = -1e20f;
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    float qk = (float)qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk += (float)atten_bias[((head_id * seq_len + seq_id) * seq_len) + col_id];
    float mask_val =
        (1.0f - (float)atten_mask[((batch_id * seq_len + seq_id) * seq_len) + col_id]) * -10000.0f;
    float tmp = qk + mask_val;
    s_row_qk[col_id] = tmp;
    max_v = tmp > max_v ? tmp : max_v;
  }
  max_v = warpReduceMax<float>(max_v, item_ct1);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    float qk = sycl::native::exp(s_row_qk[col_id] - max_v);
    s_row_qk[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum<float>(exp_sum, item_ct1);

  exp_sum = 1.0f / (exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0))
    qk_buf[qk_offset + col_id] = (T)(s_row_qk[col_id] * exp_sum);
}

template <typename T>
void softmax_kernel_warp_half2(sycl::half2 *qk_buf, const sycl::half2 *atten_bias,
                               const sycl::half2 *atten_mask, const int batch_size,
                               const int head_num, const int seq_len,
                               const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  int word_id = item_ct1.get_group(2);
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = item_ct1.get_local_id(2);
  int head_id = item_ct1.get_local_id(1);
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  auto shmem = (float *)dpct_local;
  float *s_qk_buf = (float *)shmem + head_id * seq_len;

  float max_val = -1e20f;
  for (int col_id = warp_tid; col_id < half2_seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    sycl::half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk = qk + atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id];
    sycl::half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    float mask_val_x = (1.0f - (float)mask_val.x()) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y()) * -10000.0f;
    float tmp_x = (float)qk.x() + mask_val_x, tmp_y = (float)qk.y() + mask_val_y;
    s_qk_buf[col_id * 2] = tmp_x, s_qk_buf[col_id * 2 + 1] = tmp_y;
    max_val = sycl::fmax(max_val, sycl::fmax(tmp_x, tmp_y));
  }
  max_val = warpReduceMax(max_val, item_ct1);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    float qk = sycl::native::exp(s_qk_buf[col_id] - max_val);
    s_qk_buf[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum(exp_sum, item_ct1);

  exp_sum = 1.0f / (exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < half2_seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0))
    qk_buf[qk_offset + col_id] = sycl::half2((sycl::half)(s_qk_buf[col_id * 2] * exp_sum),
                                             (sycl::half)(s_qk_buf[col_id * 2 + 1] * exp_sum));
}

template <typename T>
void softmax_kernel_warp_et(T *qk_buf, const T *atten_bias, const T *atten_mask,
                                       const int batch_size, const int head_num, const int seq_len,
                                       int *batch_idx, int *word_idx,
                                       const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local) {
  /*
  DPCT1098:559: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:560: Migrated __ldg call is used in a macro/template definition and may not be valid for
  all macro/template uses. Adjust the code.
  */
  int word_id = word_idx[item_ct1.get_group(2)];
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = item_ct1.get_local_id(2);
  int head_id = item_ct1.get_local_id(1);
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * seq_len;

  auto shmem = (float *)dpct_local;
  float *s_row_qk = (float *)shmem + head_id * seq_len;

  float max_v = -1e20f;
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    float qk = (float)qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk += (float)atten_bias[((head_id * seq_len + seq_id) * seq_len) + col_id];
    float mask_val =
        (1.0f - (float)atten_mask[((batch_id * seq_len + seq_id) * seq_len) + col_id]) * -10000.0f;
    float tmp = qk + mask_val;
    s_row_qk[col_id] = tmp;
    max_v = tmp > max_v ? tmp : max_v;
  }
  max_v = warpReduceMax(max_v, item_ct1);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    float qk = sycl::native::exp(s_row_qk[col_id] - max_v);
    s_row_qk[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum(exp_sum, item_ct1);

  exp_sum = 1.0f / (exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0))
    qk_buf[qk_offset + col_id] = (T)(s_row_qk[col_id] * exp_sum);
}

template <typename T>
void softmax_kernel_warp_half2_et(sycl::half2 *qk_buf, const sycl::half2 *atten_bias,
                                  const sycl::half2 *atten_mask, const int batch_size,
                                  const int head_num, const int seq_len, int *batch_idx,
                                  int *word_idx, const sycl::nd_item<3> &item_ct1,
                                  uint8_t *dpct_local) {
  /*
  DPCT1098:561: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:562: Migrated __ldg call is used in a macro/template definition and may not be valid for
  all macro/template uses. Adjust the code.
  */
  int word_id = word_idx[item_ct1.get_group(2)];
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = item_ct1.get_local_id(2);
  int head_id = item_ct1.get_local_id(1);
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  auto shmem = (float *)dpct_local;
  float *s_qk_buf = (float *)shmem + head_id * seq_len;

  float max_val = -1e20f;
  for (int col_id = warp_tid; col_id < half2_seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    sycl::half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk = qk + atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id];
    sycl::half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    float mask_val_x = (1.0f - (float)mask_val.x()) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y()) * -10000.0f;
    float tmp_x = (float)qk.x() + mask_val_x, tmp_y = (float)qk.y() + mask_val_y;
    s_qk_buf[col_id * 2] = tmp_x, s_qk_buf[col_id * 2 + 1] = tmp_y;
    max_val = sycl::fmax(max_val, sycl::fmax(tmp_x, tmp_y));
  }
  max_val = warpReduceMax(max_val, item_ct1);

  float exp_sum = 0.0f;
  for (int col_id = warp_tid; col_id < seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0)) {
    float qk = sycl::native::exp(s_qk_buf[col_id] - max_val);
    s_qk_buf[col_id] = qk;
    exp_sum += qk;
  }
  exp_sum = warpReduceSum(exp_sum, item_ct1);

  exp_sum = 1.0f / (exp_sum + 1e-6f);
  for (int col_id = warp_tid; col_id < half2_seq_len;
       col_id += item_ct1.get_sub_group().get_local_range().get(0))
    qk_buf[qk_offset + col_id] = sycl::half2((sycl::half)(s_qk_buf[col_id * 2] * exp_sum),
                                             (sycl::half)(s_qk_buf[col_id * 2 + 1] * exp_sum));
}

template <typename T, const int count, const bool need_padding>
/*
DPCT1110:231: The total declared local variable size in device function
softmax_kernel_warp_half2_register exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust the code, or use
smaller sub-group size to avoid high register pressure.
*/
void softmax_kernel_warp_half2_register(sycl::half2 *qk_buf, const sycl::half2 *atten_bias,
                                        const sycl::half2 *atten_mask, const int batch_size,
                                        const int head_num, const int seq_len,
                                        const sycl::nd_item<3> &item_ct1) {
  int word_id = item_ct1.get_group(2);
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = item_ct1.get_local_id(2);
  int head_id = item_ct1.get_local_id(1);
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  float s_qk_buf[count];
  if (need_padding)
    s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;

  float max_val = -1e20f;
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + item_ct1.get_sub_group().get_local_range().get(0) * i;
    if (need_padding && col_id >= half2_seq_len)
      break;

    sycl::half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk =
          /*
          DPCT1098:563: The '*' expression is used instead of the __ldg call. These two expressions
          do not provide the exact same functionality. Check the generated code for potential
          precision and/or performance issues.
          */
          qk + atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id];
    /*
    DPCT1098:564: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    sycl::half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    float mask_val_x = (1.0f - (float)mask_val.x()) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y()) * -10000.0f;
    s_qk_buf[i * 2] = (float)qk.x() + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y() + mask_val_y;
  }

  for (int i = 0; i < count; i++)
    max_val = sycl::fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val, item_ct1);

  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = sycl::native::exp(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  exp_sum = warpReduceSum(exp_sum, item_ct1);

  exp_sum = 1.0f / (exp_sum + 1e-6f);
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + item_ct1.get_sub_group().get_local_range().get(0) * i;
    if (need_padding && col_id >= half2_seq_len)
      return;
    qk_buf[qk_offset + col_id] = sycl::half2((sycl::half)(s_qk_buf[i * 2] * exp_sum),
                                             (sycl::half)(s_qk_buf[i * 2 + 1] * exp_sum));
  }
}

template <typename T, const int count, const bool need_padding>
/*
DPCT1110:232: The total declared local variable size in device function
softmax_kernel_warp_half2_register_et exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and adjust the code, or
use smaller sub-group size to avoid high register pressure.
*/
void softmax_kernel_warp_half2_register_et(sycl::half2 *qk_buf, const sycl::half2 *atten_bias,
                                           const sycl::half2 *atten_mask, const int batch_size,
                                           const int head_num, const int seq_len, int *batch_idx,
                                           int *word_idx, const sycl::nd_item<3> &item_ct1) {
  /*
  DPCT1098:565: The '*' expression is used instead of the __ldg call. These two expressions do not
  provide the exact same functionality. Check the generated code for potential precision and/or
  performance issues.
  */
  /*
  DPCT1064:568: Migrated __ldg call is used in a macro/template definition and may not be valid for
  all macro/template uses. Adjust the code.
  */
  int word_id = word_idx[item_ct1.get_group(2)];
  int batch_id = word_id / seq_len;
  int seq_id = word_id % seq_len;
  int warp_tid = item_ct1.get_local_id(2);
  int head_id = item_ct1.get_local_id(1);
  int half2_seq_len = seq_len / 2;
  int qk_offset = ((head_id * batch_size + batch_id) * seq_len + seq_id) * half2_seq_len;

  float s_qk_buf[count];
  if (need_padding)
    s_qk_buf[count - 2] = -10000.0f, s_qk_buf[count - 1] = -10000.0f;

  float max_val = -1e20f;
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + item_ct1.get_sub_group().get_local_range().get(0) * i;
    if (need_padding && col_id >= half2_seq_len)
      break;

    sycl::half2 qk = qk_buf[qk_offset + col_id];
    if (atten_bias)
      qk =
          /*
          DPCT1098:566: The '*' expression is used instead of the __ldg call. These two expressions
          do not provide the exact same functionality. Check the generated code for potential
          precision and/or performance issues.
          */
          qk + atten_bias[((head_id * seq_len + seq_id) * half2_seq_len) + col_id];
    /*
    DPCT1098:567: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    sycl::half2 mask_val = atten_mask[((batch_id * seq_len + seq_id) * half2_seq_len) + col_id];
    float mask_val_x = (1.0f - (float)mask_val.x()) * -10000.0f,
          mask_val_y = (1.0f - (float)mask_val.y()) * -10000.0f;
    s_qk_buf[i * 2] = (float)qk.x() + mask_val_x, s_qk_buf[i * 2 + 1] = (float)qk.y() + mask_val_y;
  }

  for (int i = 0; i < count; i++)
    max_val = sycl::fmax(max_val, s_qk_buf[i]);
  max_val = warpReduceMax(max_val, item_ct1);

  float exp_sum = 0.0f;
  for (int i = 0; i < count; i++) {
    s_qk_buf[i] = sycl::native::exp(s_qk_buf[i] - max_val);
    exp_sum += s_qk_buf[i];
  }
  exp_sum = warpReduceSum(exp_sum, item_ct1);

  exp_sum = 1.0f / (exp_sum + 1e-6f);
  for (int i = 0; i < count / 2; i++) {
    int col_id = warp_tid + item_ct1.get_sub_group().get_local_range().get(0) * i;
    if (need_padding && col_id >= half2_seq_len)
      return;
    qk_buf[qk_offset + col_id] = sycl::half2((sycl::half)(s_qk_buf[i * 2] * exp_sum),
                                             (sycl::half)(s_qk_buf[i * 2 + 1] * exp_sum));
  }
}

/*
DPCT1049:233: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1049:234: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define SOFTMAX_HALF2_REG(REG_COUNT)                                                        \
  if (seq_len % 64 == 0)                                                                    \
  {                                                                                         \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});               \
                                                                                            \
    stream->submit([&](sycl::handler &cgh) {                                                \
      sycl::half2 *qk_buf_ct0 = (sycl::half2 *)qk_buf;                                      \
      const sycl::half2 *atten_bias_ct1 = (sycl::half2 *)atten_bias;                        \
      const sycl::half2 *atten_mask_ct2 = (sycl::half2 *)atten_mask;                        \
      const int batch_size_ct3 = batch_size;                                                \
      const int head_num_ct4 = head_num;                                                    \
      const int seq_len_ct5 = seq_len;                                                      \
                                                                                            \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                              \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {  \
                         softmax_kernel_warp_half2_register<sycl::half2, REG_COUNT, false>( \
                             qk_buf_ct0, atten_bias_ct1, atten_mask_ct2, batch_size_ct3,    \
                             head_num_ct4, seq_len_ct5, item_ct1);                          \
                       });                                                                  \
    });                                                                                     \
  } else {                                                                                  \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});               \
                                                                                            \
    stream->submit([&](sycl::handler &cgh) {                                                \
      sycl::half2 *qk_buf_ct0 = (sycl::half2 *)qk_buf;                                      \
      const sycl::half2 *atten_bias_ct1 = (sycl::half2 *)atten_bias;                        \
      const sycl::half2 *atten_mask_ct2 = (sycl::half2 *)atten_mask;                        \
      const int batch_size_ct3 = batch_size;                                                \
      const int head_num_ct4 = head_num;                                                    \
      const int seq_len_ct5 = seq_len;                                                      \
                                                                                            \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                              \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {  \
                         softmax_kernel_warp_half2_register<sycl::half2, REG_COUNT, true>(  \
                             qk_buf_ct0, atten_bias_ct1, atten_mask_ct2, batch_size_ct3,    \
                             head_num_ct4, seq_len_ct5, item_ct1);                          \
                       });                                                                  \
    });                                                                                     \
  }

/*
DPCT1049:238: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
/*
DPCT1049:239: The work-group size passed to the SYCL kernel may exceed the limit. To get the device
limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
*/
#define SOFTMAX_HALF2_REG_RM(REG_COUNT)                                                         \
  if (seq_len % 64 == 0)                                                                        \
  {                                                                                             \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});                   \
                                                                                                \
    stream->submit([&](sycl::handler &cgh) {                                                    \
      sycl::half2 *qk_buf_ct0 = (sycl::half2 *)qk_buf;                                          \
      const sycl::half2 *atten_bias_ct1 = (sycl::half2 *)atten_bias;                            \
      const sycl::half2 *atten_mask_ct2 = (sycl::half2 *)atten_mask;                            \
      const int batch_size_ct3 = batch_size;                                                    \
      const int head_num_ct4 = head_num;                                                        \
      const int seq_len_ct5 = seq_len;                                                          \
      int *batch_idx_ct6 = batch_idx;                                                           \
      int *word_idx_ct7 = word_idx;                                                             \
                                                                                                \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                  \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {      \
                         softmax_kernel_warp_half2_register_et<sycl::half2, REG_COUNT, false>(  \
                             qk_buf_ct0, atten_bias_ct1, atten_mask_ct2, batch_size_ct3,        \
                             head_num_ct4, seq_len_ct5, batch_idx_ct6, word_idx_ct7, item_ct1); \
                       });                                                                      \
    });                                                                                         \
  } else {                                                                                      \
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});                   \
                                                                                                \
    stream->submit([&](sycl::handler &cgh) {                                                    \
      sycl::half2 *qk_buf_ct0 = (sycl::half2 *)qk_buf;                                          \
      const sycl::half2 *atten_bias_ct1 = (sycl::half2 *)atten_bias;                            \
      const sycl::half2 *atten_mask_ct2 = (sycl::half2 *)atten_mask;                            \
      const int batch_size_ct3 = batch_size;                                                    \
      const int head_num_ct4 = head_num;                                                        \
      const int seq_len_ct5 = seq_len;                                                          \
      int *batch_idx_ct6 = batch_idx;                                                           \
      int *word_idx_ct7 = word_idx;                                                             \
                                                                                                \
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),                                  \
                       [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {      \
                         softmax_kernel_warp_half2_register_et<sycl::half2, REG_COUNT, true>(   \
                             qk_buf_ct0, atten_bias_ct1, atten_mask_ct2, batch_size_ct3,        \
                             head_num_ct4, seq_len_ct5, batch_idx_ct6, word_idx_ct7, item_ct1); \
                       });                                                                      \
    });                                                                                         \
  }

template <OperationType OpType, typename T>
void softmax_kernelLauncher(T *qk_buf, const T *atten_bias, const T *atten_mask,
                            const int batch_size, const int seq_len, const int head_num,
                            dpct::queue_ptr stream) {
  sycl::range<3> grid(1, 1, batch_size * seq_len), block(1, head_num, 32);

  /*
  DPCT1083:236: The size of local memory in the migrated code may be different from the original
  code. Check that the allocated memory size in the migrated code is correct.
  */
  const int shmem_size = head_num * seq_len * sizeof(float);
  if (shmem_size > 64 * 1024)
    printf("Not Enough Shared Memory for Softmax\n");

  if ((seq_len & 0x1) == 0 && OpType == OperationType::HALF) {
    if (seq_len <= 1024) {
      switch ((seq_len + 63) / 64) {
        case 1:
          SOFTMAX_HALF2_REG(1 * 2);
          break;
        case 2:
          SOFTMAX_HALF2_REG(2 * 2);
          break;
        case 3:
          SOFTMAX_HALF2_REG(3 * 2);
          break;
        case 4:
          SOFTMAX_HALF2_REG(4 * 2);
          break;
        case 5:
          SOFTMAX_HALF2_REG(5 * 2);
          break;
        case 6:
          SOFTMAX_HALF2_REG(6 * 2);
          break;
        case 7:
          SOFTMAX_HALF2_REG(7 * 2);
          break;
        case 8:
          SOFTMAX_HALF2_REG(8 * 2);
          break;
        case 9:
          SOFTMAX_HALF2_REG(9 * 2);
          break;
        case 10:
          SOFTMAX_HALF2_REG(10 * 2);
          break;
        case 11:
          SOFTMAX_HALF2_REG(11 * 2);
          break;
        case 12:
          SOFTMAX_HALF2_REG(12 * 2);
          break;
        case 13:
          SOFTMAX_HALF2_REG(13 * 2);
          break;
        case 14:
          SOFTMAX_HALF2_REG(14 * 2);
          break;
        case 15:
          SOFTMAX_HALF2_REG(15 * 2);
          break;
        case 16:
          SOFTMAX_HALF2_REG(16 * 2);
          break;
      }
    } else {
      if (shmem_size > 48 * 1024)
        /*
        DPCT1026:569: The call to cudaFuncSetAttribute was removed because SYCL currently does not
        support corresponding setting.
        */
        ;
      /*
      DPCT1049:235: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
      {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shmem_size), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(grid * block, block),
              [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_kernel_warp_half2<sycl::half2>(
                    (sycl::half2 *)qk_buf, (sycl::half2 *)atten_bias, (sycl::half2 *)atten_mask,
                    batch_size, head_num, seq_len, item_ct1,
                    dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
              });
        });
      }
    }
  } else {
    if (shmem_size > 48 * 1024)
      /*
      DPCT1026:570: The call to cudaFuncSetAttribute was removed because SYCL currently does not
      support corresponding setting.
      */
      ;
    /*
    DPCT1049:237: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shmem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              softmax_kernel_warp<T>(
                  qk_buf, atten_bias, atten_mask, batch_size, head_num, seq_len, item_ct1,
                  dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
      });
    }
  }
}

template <OperationType OpType, typename T>
void softmax_et_kernelLauncher(T *qk_buf, const T *atten_bias, const T *atten_mask,
                               const int batch_size, const int seq_len, const int head_num,
                               dpct::queue_ptr stream, int *batch_idx, int *word_idx,
                               int valid_word_num) {
  sycl::range<3> grid(1, 1, valid_word_num), block(1, head_num, 32);

  /*
  DPCT1083:241: The size of local memory in the migrated code may be different from the original
  code. Check that the allocated memory size in the migrated code is correct.
  */
  const int shmem_size = head_num * seq_len * sizeof(float);
  if (shmem_size > 64 * 1024)
    printf("Not Enough Shared Memory for Softmax\n");

  if ((seq_len & 0x1) == 0 && OpType == OperationType::HALF) {
    if (seq_len <= 1024) {
      switch ((seq_len + 63) / 64) {
        case 1:
          SOFTMAX_HALF2_REG_RM(1 * 2);
          break;
        case 2:
          SOFTMAX_HALF2_REG_RM(2 * 2);
          break;
        case 3:
          SOFTMAX_HALF2_REG_RM(3 * 2);
          break;
        case 4:
          SOFTMAX_HALF2_REG_RM(4 * 2);
          break;
        case 5:
          SOFTMAX_HALF2_REG_RM(5 * 2);
          break;
        case 6:
          SOFTMAX_HALF2_REG_RM(6 * 2);
          break;
        case 7:
          SOFTMAX_HALF2_REG_RM(7 * 2);
          break;
        case 8:
          SOFTMAX_HALF2_REG_RM(8 * 2);
          break;
        case 9:
          SOFTMAX_HALF2_REG_RM(9 * 2);
          break;
        case 10:
          SOFTMAX_HALF2_REG_RM(10 * 2);
          break;
        case 11:
          SOFTMAX_HALF2_REG_RM(11 * 2);
          break;
        case 12:
          SOFTMAX_HALF2_REG_RM(12 * 2);
          break;
        case 13:
          SOFTMAX_HALF2_REG_RM(13 * 2);
          break;
        case 14:
          SOFTMAX_HALF2_REG_RM(14 * 2);
          break;
        case 15:
          SOFTMAX_HALF2_REG_RM(15 * 2);
          break;
        case 16:
          SOFTMAX_HALF2_REG_RM(16 * 2);
          break;
      }
    } else {
      if (shmem_size > 48 * 1024)
        /*
        DPCT1026:571: The call to cudaFuncSetAttribute was removed because SYCL currently does not
        support corresponding setting.
        */
        ;
      /*
      DPCT1049:240: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
      {
        dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
          sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shmem_size), cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(grid * block, block),
              [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                softmax_kernel_warp_half2_et<sycl::half2>(
                    (sycl::half2 *)qk_buf, (sycl::half2 *)atten_bias, (sycl::half2 *)atten_mask,
                    batch_size, head_num, seq_len, batch_idx, word_idx, item_ct1,
                    dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
              });
        });
      }
    }
  } else {
    if (shmem_size > 48 * 1024)
      /*
      DPCT1026:572: The call to cudaFuncSetAttribute was removed because SYCL currently does not
      support corresponding setting.
      */
      ;
    /*
    DPCT1049:242: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(shmem_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
              softmax_kernel_warp_et<T>(
                  qk_buf, atten_bias, atten_mask, batch_size, head_num, seq_len, batch_idx,
                  word_idx, item_ct1,
                  dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
            });
      });
    }
  }
}
}  // namespace bytetransformer
