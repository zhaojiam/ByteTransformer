#define DPCT_COMPAT_RT_MAJOR_VERSION 12
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
#include "bytetransformer/include/gemm_bias_act.h"

namespace bytetransformer {
template <>
void add_bias_gelu<float>(float *output, const float *bias, const int M, const int N,
                          const sycl::nd_item<3> &item_ct1) {
  int row_offset = item_ct1.get_group(2) * N;
  for (int tid = item_ct1.get_local_id(2); tid < N; tid += item_ct1.get_local_range(2)) {
    /*
    DPCT1098:332: The '*' expression is used instead of the __ldg call. These two expressions do
    not provide the exact same functionality. Check the generated code for potential precision
    and/or performance issues.
    */
    float out = output[row_offset + tid] + bias[tid];
    output[row_offset + tid] = gelu(out);
  }
}

template <>
void add_bias_gelu<sycl::half>(sycl::half *output, const sycl::half *bias, const int M,
                               const int N, const sycl::nd_item<3> &item_ct1) {
  if (N % 4 != 0) {
    sycl::half2 *output_ptr = (sycl::half2 *)output;
    const sycl::half2 *bias_ptr = (const sycl::half2 *)bias;

    int row_offset = item_ct1.get_group(2) * N / 2;
    for (int tid = item_ct1.get_local_id(2); tid < N / 2; tid += item_ct1.get_local_range(2)) {
      /*
      DPCT1098:333: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      sycl::half2 out = output_ptr[row_offset + tid] + bias_ptr[tid];
      output_ptr[row_offset + tid] = gelu(out);
    }
  } else {
    sycl::float2 *output_ptr = (sycl::float2 *)output;
    const sycl::float2 *bias_ptr = (const sycl::float2 *)(bias);
    int row_offset = item_ct1.get_group(2) * N / 4;
    for (int tid = item_ct1.get_local_id(2); tid < N / 4; tid += item_ct1.get_local_range(2)) {
      half4 val, bias_val;
      val.x = output_ptr[row_offset + tid];
      /*
      DPCT1098:334: The '*' expression is used instead of the __ldg call. These two expressions do
      not provide the exact same functionality. Check the generated code for potential precision
      and/or performance issues.
      */
      bias_val.x = bias_ptr[tid];

      val.h[0] = gelu(val.h[0] + bias_val.h[0]);
      val.h[1] = gelu(val.h[1] + bias_val.h[1]);

      output_ptr[row_offset + tid] = val.x;
    }
  }
}

void cublas_gemm_bias_gelu(const sycl::half *A, const sycl::half *B, sycl::half *C,
                           const sycl::half *bias, int m, int k, int n, dpct::queue_ptr stream,
                           dpct::queue_ptr cublas_handle, int cublasAlgo) {
  dense_layer_kernel_launcher(A, B, C, m, k, n, cublas_handle, stream, cublasAlgo);
  /*
  DPCT1049:58: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, m) * sycl::range<3>(1, 1, n / 8),
                                           sycl::range<3>(1, 1, n / 8)),
                         [=](sycl::nd_item<3> item_ct1) {
                           add_bias_gelu(C, bias, m, n, item_ct1);
                         });
  }
}

template <>
void gemm_bias_gelu<float>(const float *A_, const float *B_, float *C_, const float *bias_, int m_,
                           int k_, int n_, dpct::queue_ptr stream, dpct::queue_ptr cublas_handle,
                           int cublasAlgo, int arch) {
  dense_layer_kernel_launcher(A_, B_, C_, m_, k_, n_, cublas_handle, stream, cublasAlgo);
  /*
  DPCT1049:59: The work-group size passed to the SYCL kernel may exceed the limit. To get the
  device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, m_) * sycl::range<3>(1, 1, n_ / 4),
                                           sycl::range<3>(1, 1, n_ / 4)),
                         [=](sycl::nd_item<3> item_ct1) {
                           add_bias_gelu(C_, bias_, m_, n_, item_ct1);
                         });
  }
}

template <>
void gemm_bias_gelu<sycl::half>(const sycl::half *A_, const sycl::half *B_, sycl::half *C_,
                                const sycl::half *bias_, int m_, int k_, int n_,
                                dpct::queue_ptr stream, dpct::queue_ptr cublas_handle,
                                int cublasAlgo, int arch) {
  if (m_ < 8)
    cublas_gemm_bias_gelu(A_, B_, C_, bias_, m_, k_, n_, stream, cublas_handle, cublasAlgo);
  else {
#if (DPCT_COMPAT_RT_MAJOR_VERSION >= 11)
    const ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0f);
    const ElementComputeEpilogue beta = ElementComputeEpilogue(1.0f);
    const int split_k_slices_ = 1;
    void *cutlass_workspace_ = nullptr;
    cutlass::gemm::GemmCoord problem_size(m_, n_, k_);

    if (arch == 70 && (m_ > 896 && m_ <= 7936)) {
      using SmArch = cutlass::arch::Sm70;
#define _inst_m 8
#define _inst_n 8
#define _inst_k 4
      constexpr int NumStages = 2;
      GEMM_TYPE(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k,
                NumStages)
      GEMM_BIAS_GELU(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n,
                     _inst_k)
      gemmBiasGelu_op;
      GEMM_INIT(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k)
#undef _inst_m
#undef _inst_n
#undef _inst_k
      CUTLASS_CHECK(gemmBiasGelu_op.initialize(args, cutlass_workspace_));
      CUTLASS_CHECK(gemmBiasGelu_op(stream));
    } else if (arch == 75 && (m_ > 192 && m_ <= 3456)) {
      using SmArch = cutlass::arch::Sm75;
#define _inst_m 16
#define _inst_n 8
#define _inst_k 8
      constexpr int NumStages = 2;
      GEMM_TYPE(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k,
                NumStages)
      GEMM_BIAS_GELU(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n,
                     _inst_k)
      gemmBiasGelu_op;
      GEMM_INIT(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k)
#undef _inst_m
#undef _inst_n
#undef _inst_k
      CUTLASS_CHECK(gemmBiasGelu_op.initialize(args, cutlass_workspace_));
      CUTLASS_CHECK(gemmBiasGelu_op(stream));
    } else if (arch == 80 && (m_ >= 384 && m_ <= 16384))  // < 19742
    {
      using SmArch = cutlass::arch::Sm80;
#define _inst_m 16
#define _inst_n 8
#define _inst_k 16
      constexpr int NumStages = 3;
      GEMM_TYPE(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k,
                NumStages)
      GEMM_BIAS_GELU(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n,
                     _inst_k)
      gemmBiasGelu_op;
      GEMM_INIT(_block_m, _block_n, _block_k, _warp_m, _warp_n, _warp_k, _inst_m, _inst_n, _inst_k)
#undef _inst_m
#undef _inst_n
#undef _inst_k
      CUTLASS_CHECK(gemmBiasGelu_op.initialize(args, cutlass_workspace_));
      CUTLASS_CHECK(gemmBiasGelu_op(stream));
    } else
      cublas_gemm_bias_gelu(A_, B_, C_, bias_, m_, k_, n_, stream, cublas_handle, cublasAlgo);
#else
    cublas_gemm_bias_gelu(A_, B_, C_, bias_, m_, k_, n_, stream, cublas_handle, cublasAlgo);
#endif
  }
}

template <>
void gemm_bias_relu<float>(const float *A_, const float *B_, float *C_, const float *bias_, int m_,
                           int k_, int n_, dpct::queue_ptr stream, dpct::queue_ptr cublas_handle,
                           int cublasAlgo, int arch) {
  dense_layer_kernel_launcher(A_, B_, C_, m_, k_, n_, cublas_handle, stream, cublasAlgo);
  // add_bias_relu<<<m_, n_ / 4, 0, stream>>>(C_, bias_, m_, n_);
}

template <>
void gemm_bias_relu<sycl::half>(const sycl::half *A_, const sycl::half *B_, sycl::half *C_,
                                const sycl::half *bias_, int m_, int k_, int n_,
                                dpct::queue_ptr stream, dpct::queue_ptr cublas_handle,
                                int cublasAlgo, int arch) {
}
}  // namespace bytetransformer
