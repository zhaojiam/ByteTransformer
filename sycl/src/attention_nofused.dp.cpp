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
#include "bytetransformer/include/attention_nofused_utils.h"
#include "bytetransformer/include/gemm.h"
#include "bytetransformer/include/softmax.h"
#include "bytetransformer/include/variety_attention_fused.h"
#include <cmath>

namespace bytetransformer {
template <OperationType OpType>
void Attention<OpType>::nofused_infer(AttentionInferParam infer_param) {
  void* buf = infer_param.buf;
  const int batch_size = infer_param.batch_size;
  const int seq_len = infer_param.seq_len;
  dpct::queue_ptr stream = infer_param.stream;
  ET_Param et_param = infer_param.et_param;

  int input_tensor_size = batch_size * head_num_ * seq_len * size_per_head_;
  int qk_buf_size = ((batch_size * head_num_ * seq_len * seq_len + 15) >> 4)
                    << 4;

  DataType_* query = (DataType_*)buf + 0 * input_tensor_size;
  DataType_* key = (DataType_*)buf + 1 * input_tensor_size;
  DataType_* value = (DataType_*)buf + 2 * input_tensor_size;
  DataType_* qk_buf = (DataType_*)buf + 3 * input_tensor_size;
  DataType_* transpose_dst = qk_buf + qk_buf_size;

  int size_per_head_half = (OpType == OperationType::HALF)
                               ? size_per_head_ / 2
                               : size_per_head_;  // Be careful.

  // [batch_size, seq_len, hidden_dim] -> [head_num, batch_size, seq_len,
  // size_per_head]
  sycl::range<3> grid(1, 1, 1), block(1, 1, 1);
  grid[2] = seq_len, grid[1] = batch_size;
  block[2] = head_num_ * (size_per_head_ / 2);  // Process two adjacent values for float/half
  const bool is_roformer = false;
  if (is_remove_padding_)
    /*
    DPCT1049:184: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      auto infer_param_qkv_ct0 = infer_param.qkv;
      auto param__attr_bias_QKV_ct1 = param_.attr_bias_QKV;
      auto head_num__ct7 = head_num_;
      auto size_per_head__ct8 = size_per_head_ / 2;

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
        add_QKV_bias_padding(infer_param_qkv_ct0, param__attr_bias_QKV_ct1, query, key, value,
                             batch_size, seq_len, head_num__ct7, size_per_head__ct8, is_roformer,
                             et_param.batch_idx, et_param.word_idx, item_ct1);
      });
    });
  } else
  /*
  DPCT1049:185: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->submit([&](sycl::handler &cgh) {
      auto infer_param_qkv_ct0 = infer_param.qkv;
      auto param__attr_bias_QKV_ct1 = param_.attr_bias_QKV;
      auto head_num__ct7 = head_num_;
      auto size_per_head__ct8 = size_per_head_ / 2;

      cgh.parallel_for(sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
        add_QKV_bias(infer_param_qkv_ct0, param__attr_bias_QKV_ct1, query, key, value, batch_size,
                     seq_len, head_num__ct7, size_per_head__ct8, is_roformer, item_ct1);
      });
    });
  }
  grid[1] = 1;

  DataType_ alpha =
                (DataType_)(1.0f / sqrtf(size_per_head_ * 1.0f) / param_.tao),
            beta = (DataType_)0.0f;
  bool add_qk_buf = false;

  if (transformer_variety_fuse_flag_)
    variety_attention_fused_infer(
        (const sycl::half *)query, (const sycl::half *)key, (const sycl::half *)value,
        (const sycl::half *)infer_param.atten_mask, add_qk_buf ? (const sycl::half *)qk_buf : NULL,
        (const sycl::half *)infer_param.attention_bias, (sycl::half *)infer_param.attention_output,
        head_num_, batch_size, seq_len, size_per_head_, (float)alpha, infer_param.stream,
        is_remove_padding_ ? et_param.batch_idx : NULL);
  else {
    cublas_Gemm_Strided_Batched(query, key, qk_buf, seq_len, size_per_head_, seq_len,
                                head_num_ * batch_size, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::trans, alpha, beta,
                                infer_param.cublas_handle, stream, param_.cublas_Algo[0]);

    if (is_remove_padding_)
      softmax_et_kernelLauncher<OpType, DataType_>(
          qk_buf, infer_param.attention_bias, infer_param.atten_mask,
          batch_size, seq_len, head_num_, stream, et_param.batch_idx,
          et_param.word_idx, et_param.valid_word_num);
    else
      softmax_kernelLauncher<OpType, DataType_>(
          qk_buf, infer_param.attention_bias, infer_param.atten_mask,
          batch_size, seq_len, head_num_, stream);

    alpha = (DataType_)1.0f, beta = (DataType_)0.0f;
    cublas_Gemm_Strided_Batched(qk_buf, value, transpose_dst, seq_len, seq_len, size_per_head_,
                                head_num_ * batch_size, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::nontrans, alpha, beta,
                                infer_param.cublas_handle, stream, param_.cublas_Algo[1]);

    block[2] = size_per_head_half, block[1] = head_num_;
    if (is_remove_padding_)
      /*
      DPCT1049:186: The work-group size passed to the SYCL kernel may exceed the limit. To get the
      device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
      */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        auto infer_param_attention_output_ct1 = infer_param.attention_output;
        auto head_num__ct4 = head_num_;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, et_param.valid_word_num) * block, block),
            [=](sycl::nd_item<3> item_ct1) {
              transpose_rm_padding(transpose_dst, infer_param_attention_output_ct1, batch_size,
                                   seq_len, head_num__ct4, size_per_head_half, et_param.batch_idx,
                                   et_param.word_idx, item_ct1);
            });
      });
    } else
    /*
    DPCT1049:187: The work-group size passed to the SYCL kernel may exceed the limit. To get the
    device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    {
      dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

      stream->submit([&](sycl::handler &cgh) {
        auto infer_param_attention_output_ct1 = infer_param.attention_output;
        auto head_num__ct4 = head_num_;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, batch_size * seq_len) * block, block),
            [=](sycl::nd_item<3> item_ct1) {
              transpose(transpose_dst, infer_param_attention_output_ct1, batch_size, seq_len,
                        head_num__ct4, size_per_head_half, item_ct1);
            });
      });
    }
  }
}

template void Attention<OperationType::FP32>::nofused_infer(
    AttentionInferParam infer_param);
template void Attention<OperationType::HALF>::nofused_infer(
    AttentionInferParam infer_param);
}  // namespace bytetransformer
