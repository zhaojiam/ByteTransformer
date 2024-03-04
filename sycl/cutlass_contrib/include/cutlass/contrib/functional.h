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
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/complex.h"
#include "cutlass/quaternion.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"

namespace cutlass {
namespace contrib {

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add {
  CUTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    return C(a) * C(b) + c;
  }
};

/// Fused multiply-add
template <int N>
struct multiply_add<Array<half_t, N>, Array<half_t, N>, Array<half_t, N>> {
  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &a, Array<half_t, N> const &b,
                              Array<half_t, N> const &c) const {
    Array<half_t, N> result;
#if defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 530)

    sycl::half2 *result_ptr = reinterpret_cast<sycl::half2 *>(&result);
    sycl::half2 const *a_ptr = reinterpret_cast<sycl::half2 const *>(&a);
    sycl::half2 const *b_ptr = reinterpret_cast<sycl::half2 const *>(&b);
    sycl::half2 const *c_ptr = reinterpret_cast<sycl::half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = sycl::fma(a_ptr[i], b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {
      sycl::half const *a_residual_ptr = reinterpret_cast<sycl::half const *>(&a);
      sycl::half const *b_residual_ptr = reinterpret_cast<sycl::half const *>(&b);
      sycl::half const *c_residual_ptr = reinterpret_cast<sycl::half const *>(&c);

      sycl::half d_residual =
          sycl::fma(a_residual_ptr[N - 1], b_residual_ptr[N - 1], c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c[i]);
    }
#endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(half_t const &a, Array<half_t, N> const &b,
                              Array<half_t, N> const &c) const {
    Array<half_t, N> result;
#if defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 530)

    sycl::half2 *result_ptr = reinterpret_cast<sycl::half2 *>(&result);
    sycl::half2 a_pair = sycl::half2(reinterpret_cast<sycl::half const &>(a));
    sycl::half2 const *b_ptr = reinterpret_cast<sycl::half2 const *>(&b);
    sycl::half2 const *c_ptr = reinterpret_cast<sycl::half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = sycl::fma(a_pair, b_ptr[i], c_ptr[i]);
    }

    if (N % 2) {
      sycl::half const *b_residual_ptr = reinterpret_cast<sycl::half const *>(&b);
      sycl::half const *c_residual_ptr = reinterpret_cast<sycl::half const *>(&c);
      sycl::half d_residual = sycl::fma(reinterpret_cast<sycl::half const &>(a),
                                        b_residual_ptr[N - 1], c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    ::cutlass::multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a, b[i], c[i]);
    }
#endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &a, half_t const &b,
                              Array<half_t, N> const &c) const {
    Array<half_t, N> result;
#if defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 530)

    sycl::half2 *result_ptr = reinterpret_cast<sycl::half2 *>(&result);
    sycl::half2 const *a_ptr = reinterpret_cast<sycl::half2 const *>(&a);
    sycl::half2 b_pair = sycl::half2(reinterpret_cast<sycl::half const &>(b));
    sycl::half2 const *c_ptr = reinterpret_cast<sycl::half2 const *>(&c);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = sycl::fma(a_ptr[i], b_pair, c_ptr[i]);
    }

    if (N % 2) {
      sycl::half const *a_residual_ptr = reinterpret_cast<sycl::half const *>(&a);
      sycl::half const *c_residual_ptr = reinterpret_cast<sycl::half const *>(&c);

      sycl::half d_residual = sycl::fma(
          a_residual_ptr[N - 1], reinterpret_cast<sycl::half const &>(b), c_residual_ptr[N - 1]);

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c[i]);
    }
#endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &a, Array<half_t, N> const &b,
                              half_t const &c) const {
    Array<half_t, N> result;
#if defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 530)

    sycl::half2 *result_ptr = reinterpret_cast<sycl::half2 *>(&result);
    sycl::half2 const *a_ptr = reinterpret_cast<sycl::half2 const *>(&a);
    sycl::half2 const *b_ptr = reinterpret_cast<sycl::half2 const *>(&b);
    sycl::half2 c_pair = sycl::half2(reinterpret_cast<sycl::half const &>(c));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = sycl::fma(a_ptr[i], b_ptr[i], c_pair);
    }

    if (N % 2) {
      sycl::half const *a_residual_ptr = reinterpret_cast<sycl::half const *>(&a);
      sycl::half const *b_residual_ptr = reinterpret_cast<sycl::half const *>(&b);

      sycl::half d_residual = sycl::fma(a_residual_ptr[N - 1], b_residual_ptr[N - 1],
                                        reinterpret_cast<sycl::half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b[i], c);
    }
#endif

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<half_t, N> const &a, half_t const &b, half_t const &c) const {
    Array<half_t, N> result;
#if defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 530)

    sycl::half2 *result_ptr = reinterpret_cast<sycl::half2 *>(&result);
    sycl::half2 const *a_ptr = reinterpret_cast<sycl::half2 const *>(&a);
    sycl::half2 b_pair = sycl::half2(reinterpret_cast<sycl::half const &>(b));
    sycl::half2 c_pair = sycl::half2(reinterpret_cast<sycl::half const &>(c));

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = sycl::fma(a_ptr[i], b_pair, c_pair);
    }

    if (N % 2) {
      sycl::half const *a_residual_ptr = reinterpret_cast<sycl::half const *>(&a);

      sycl::half d_residual =
          sycl::fma(a_residual_ptr[N - 1], reinterpret_cast<sycl::half const &>(b),
                    reinterpret_cast<sycl::half const &>(c));

      result[N - 1] = reinterpret_cast<half_t const &>(d_residual);
    }

#else

    multiply_add<half_t> op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = op(a[i], b, c);
    }
#endif

    return result;
  }
};
}  // namespace contrib
}  // namespace cutlass
