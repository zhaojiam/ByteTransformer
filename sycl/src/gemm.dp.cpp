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
#include "bytetransformer/include/gemm.h"
#include <dpct/lib_common_utils.hpp>

namespace bytetransformer {
void dense_layer_kernel_launcher(const float *in, const float *weight, float *out, const int M,
                                 const int K, const int N, dpct::queue_ptr cublas_handle,
                                 dpct::queue_ptr stream, int cublasAlgo) {
  const float alpha = 1.0f, beta = 0.0f;
  check_cuda_error(DPCT_CHECK_ERROR(dpct::gemm(
      *cublas_handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, M, K,
      &alpha, weight, dpct::library_data_t::real_float, N, in, dpct::library_data_t::real_float, K,
      &beta, out, dpct::library_data_t::real_float, N, dpct::library_data_t::real_float)));
}

void dense_layer_kernel_launcher(const sycl::half *in, const sycl::half *weight, sycl::half *out,
                                 const int M, const int K, const int N,
                                 dpct::queue_ptr cublas_handle, dpct::queue_ptr stream,
                                 int cublasAlgo) {
  const sycl::half alpha = (sycl::half)1.0f, beta = (sycl::half)0.0f;
  check_cuda_error(DPCT_CHECK_ERROR(dpct::gemm(
      *cublas_handle, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, N, M, K,
      &alpha, weight, dpct::library_data_t::real_half, N, in, dpct::library_data_t::real_half, K,
      &beta, out, dpct::library_data_t::real_half, N, dpct::library_data_t::real_half)));
}

void cublas_Gemm_Strided_Batched(const float *A, const float *B, float *out, const int M,
                                 const int K, const int N, const int batch_count,
                                 oneapi::mkl::transpose trans_A, oneapi::mkl::transpose trans_B,
                                 float alpha, float beta, dpct::queue_ptr cublas_handle,
                                 dpct::queue_ptr stream, int cublasAlgo) {
  const int lda = (trans_A == oneapi::mkl::transpose::nontrans) ? K : M;
  const int ldb = (trans_B == oneapi::mkl::transpose::nontrans) ? N : K;

  check_cuda_error(DPCT_CHECK_ERROR(dpct::gemm_batch(
      *cublas_handle, trans_B, trans_A, N, M, K, &alpha, B, dpct::library_data_t::real_float, ldb,
      K * N, A, dpct::library_data_t::real_float, lda, M * K, &beta, out,
      dpct::library_data_t::real_float, N, M * N, batch_count, dpct::library_data_t::real_float)));
}

void cublas_Gemm_Strided_Batched(const sycl::half *A, const sycl::half *B, sycl::half *out,
                                 const int M, const int K, const int N, const int batch_count,
                                 oneapi::mkl::transpose trans_A, oneapi::mkl::transpose trans_B,
                                 sycl::half alpha, sycl::half beta, dpct::queue_ptr cublas_handle,
                                 dpct::queue_ptr stream, int cublasAlgo) {
  const int lda = (trans_A == oneapi::mkl::transpose::nontrans) ? K : M;
  const int ldb = (trans_B == oneapi::mkl::transpose::nontrans) ? N : K;

  check_cuda_error(DPCT_CHECK_ERROR(dpct::gemm_batch(
      *cublas_handle, trans_B, trans_A, N, M, K, &alpha, B, dpct::library_data_t::real_half, ldb,
      K * N, A, dpct::library_data_t::real_half, lda, M * K, &beta, out,
      dpct::library_data_t::real_half, N, M * N, batch_count, dpct::library_data_t::real_half)));
}
}  // namespace bytetransformer
