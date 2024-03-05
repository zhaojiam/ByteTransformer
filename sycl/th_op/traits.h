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

namespace bytetransformer {
namespace torch_ext {

template <typename T>
class THTraits;

template <>
class THTraits<float> {
 public:
  static const OperationType OpType = OperationType::FP32;
};

template <>
class THTraits<sycl::half> {
 public:
  static const OperationType OpType = OperationType::HALF;
};
}  // namespace torch_ext
}
