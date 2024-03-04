/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Matrix multiply
*/

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include "cutlass/arch/mma.h"

#include "cutlass/layout/matrix.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<2,1,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 1, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 1> const &b,
    Array<half_t, 2> const &c
  ) {
#if (defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 600))

    sycl::half2 const &A = reinterpret_cast<sycl::half2 const &>(a);
    sycl::half2 B = sycl::half2(reinterpret_cast<sycl::half const &>(b));
    sycl::half2 const &C = reinterpret_cast<sycl::half2 const &>(c);

    sycl::half2 D = sycl::fma(A, B, C);

    d = reinterpret_cast<Array<half_t, 2> &>(D);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[i] * b[0] + c[i];
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <typename LayoutA, typename LayoutB>
struct Mma<
  gemm::GemmShape<1,2,1>,
  1,
  half_t,
  LayoutA,
  half_t,
  LayoutB,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<1, 2, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 2> &d,
    Array<half_t, 1> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 2> const &c
  ) {
#if (defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 600))

    sycl::half2 const &A = sycl::half2(reinterpret_cast<sycl::half const &>(a));
    sycl::half2 B = reinterpret_cast<sycl::half2 const &>(b);
    sycl::half2 const &C = reinterpret_cast<sycl::half2 const &>(c);

    sycl::half2 D = sycl::fma(A, B, C);

    d = reinterpret_cast<Array<half_t, 2> &>(D);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      d[i] = a[0] * b[i] + c[i];
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma <
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {
#if (defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 600))

    sycl::half2 const &A = reinterpret_cast<sycl::half2 const &>(a);
    sycl::half2 Blo = sycl::half2((reinterpret_cast<sycl::half2 const &>(b))[0]);
    sycl::half2 Bhi = sycl::half2((reinterpret_cast<sycl::half2 const &>(b))[1]);

    sycl::half2 const *C = reinterpret_cast<sycl::half2 const *>(&c);

    sycl::half2 Dlo = sycl::fma(A, Blo, C[0]);
    sycl::half2 Dhi = sycl::fma(A, Bhi, C[1]);

    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> const &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> const &>(Dhi);

#else
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 2; ++i) {
        d[i + 2 * j] = a[i] * b[j] + c[i + 2 * j];
      }
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation
template <>
struct Mma<
  gemm::GemmShape<2, 2, 1>,
  1,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  half_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<2, 2, 1>;
  using Operator = OpMultiplyAdd;
  
  CUTLASS_HOST_DEVICE
  void operator()(
    Array<half_t, 4> &d,
    Array<half_t, 2> const &a,
    Array<half_t, 2> const &b,
    Array<half_t, 4> const &c
  ) {
#if (defined(DPCT_COMPATIBILITY_TEMP) && (DPCT_COMPATIBILITY_TEMP >= 600))

    sycl::half2 Alo = sycl::half2((reinterpret_cast<sycl::half2 const &>(a))[0]);
    sycl::half2 Ahi = sycl::half2((reinterpret_cast<sycl::half2 const &>(a))[1]);
    sycl::half2 const &B = reinterpret_cast<sycl::half2 const &>(b);

    sycl::half2 const *C = reinterpret_cast<sycl::half2 const *>(&c);

    sycl::half2 Dlo = sycl::fma(Alo, B, C[0]);
    sycl::half2 Dhi = sycl::fma(Ahi, B, C[0]);

    Array<half_t, 2> * D = reinterpret_cast<Array<half_t, 2> *>(&d);

    D[0] = reinterpret_cast<Array<half_t, 2> &>(Dlo);
    D[1] = reinterpret_cast<Array<half_t, 2> &>(Dhi);
#else
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < 2; ++j) {
        d[i * 2 + j] = a[i] * b[j] + c[i * 2 + j];
      }
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}
}

