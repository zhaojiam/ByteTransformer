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

namespace bytetransformer {
template <typename T>
SYCL_EXTERNAL void add_QKV_bias(const T *QKV, const T *bias_QKV, T *q_buf, T *k_buf, T *v_buf,
                                const int batch_size, const int seq_len, const int head_num,
                                const int half_size_per_head, const bool is_roformer,
                                const sycl::nd_item<3> &item_ct1);

template <typename T>
SYCL_EXTERNAL void add_QKV_bias_padding(const T *QKV, const T *bias_QKV, T *q_buf, T *k_buf,
                                        T *v_buf, const int batch_size, const int seq_len,
                                        const int head_num, const int half_size_per_head,
                                        const bool is_roformer, const int *batch_idx,
                                        const int *word_idx, const sycl::nd_item<3> &item_ct1);

template <typename T>
SYCL_EXTERNAL void transpose(const T *src, T *dst, const int batch_size, const int seq_len,
                             const int head_num, const int size_per_head,
                             const sycl::nd_item<3> &item_ct1);

template <typename T>
SYCL_EXTERNAL void transpose_rm_padding(const T *src, T *dst, const int batch_size,
                                        const int seq_len, const int head_num,
                                        const int size_per_head, const int *batch_idx,
                                        const int *word_idx, const sycl::nd_item<3> &item_ct1);
}  // namespace bytetransformer
