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
#include <dpct/blas_utils.hpp>

#include <iostream>
#include <stdexcept>
#include <dpct/lib_common_utils.hpp>

namespace bytetransformer {
enum class OperationType { FP32, HALF };

template <OperationType OpType>
class Traits;

template <>
class Traits<OperationType::FP32> {
 public:
  typedef float DataType;
  // cuBLAS parameters
  static dpct::library_data_t const computeType = dpct::library_data_t::real_float;
  static dpct::library_data_t const AType = dpct::library_data_t::real_float;
  static dpct::library_data_t const BType = dpct::library_data_t::real_float;
  static dpct::library_data_t const CType = dpct::library_data_t::real_float;
  static const int algo = -1;
};

template <>
class Traits<OperationType::HALF> {
 public:
  typedef sycl::half DataType;
  // cuBLAS parameters
  static dpct::library_data_t const computeType = dpct::library_data_t::real_half;
  static dpct::library_data_t const AType = dpct::library_data_t::real_half;
  static dpct::library_data_t const BType = dpct::library_data_t::real_half;
  static dpct::library_data_t const CType = dpct::library_data_t::real_half;
  static const int algo = 99;
};

typedef struct dpct_type_707531 {
  int *batch_idx;
  int *word_idx;
  int valid_word_num;
} ET_Param;

enum ModelType { Bert };

enum ActType { Relu, Sigmoid, SoftPlus, No };

template <ActType act, typename T>
__inline__ T act_fun(T val) {
  if (act == ActType::Relu)
    return (val <= (T)0.0f) ? (T)0.0f : val;
  else if (act == ActType::SoftPlus)
    /*
    DPCT1064:799: Migrated __expf call is used in a macro/template definition and may not be valid
    for all macro/template uses. Adjust the code.
    */
    return sycl::log(sycl::native::exp((float)val) + 1.0f);
  else if (act == ActType::Sigmoid)
    /*
    DPCT1064:800: Migrated __expf call is used in a macro/template definition and may not be valid
    for all macro/template uses. Adjust the code.
    */
    return 1.0f / (1.0f + sycl::native::exp(-1.0f * (float)val));
  else
    return val;
}

typedef union half4 {
  sycl::float2 x{};
  sycl::half2 h[2];
} half4;

#define PRINT_FUNC_NAME_()                                          \
  do {                                                              \
    std::cout << "[BT][CALL] " << __FUNCTION__ << " " << std::endl; \
  } while (0)

static const char *_cudaGetErrorEnum(dpct::err0 error) {
  /*
  DPCT1009:503: SYCL uses exceptions to report errors and does not use the error codes. The
  original code was commented out and a warning string was inserted. You need to rewrite this code.
  */
  return "cudaGetErrorString is not supported" /*cudaGetErrorString(error)*/;
}

static const char *_cudaGetErrorEnum(int error) {
  switch (error) {
    case 0:
      return "CUBLAS_STATUS_SUCCESS";

    case 1:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case 3:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case 7:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case 8:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case 11:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case 13:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case 14:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case 15:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case 16:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  /*
  DPCT1000:802: Error handling if-stmt was detected but could not be rewritten.
  */
  if (result)
    /*
    DPCT1001:801: The statement could not be removed.
    */
    throw std::runtime_error(std::string("[BT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)

#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    ::cutlass::Status error = status;                                                            \
    if (error != ::cutlass::Status::kSuccess) {                                                  \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

}  // namespace bytetransformer
