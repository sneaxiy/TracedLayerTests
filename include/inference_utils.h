// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// #include "paddle/include/paddle_inference_api.h" // NOLINT
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono> //NOLINT
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "paddle/include/paddle_inference_api.h" // NOLINT

#define PADDLE_CHECK(__val__)                                                  \
  do {                                                                         \
    if (!(__val__)) {                                                          \
      auto __err_msg__ = "Something wrong at " + std::string(#__val__);        \
      __err_msg__ += (__FILE__ + std::string(":") + std::to_string(__LINE__)); \
      throw std::runtime_error(__err_msg__.c_str());                           \
    }                                                                          \
  } while (0)

using paddle::PaddlePredictor;
using paddle::AnalysisConfig;

template <typename T>
std::unique_ptr<T[]> Random(size_t length, T low = 0, T high = 1,
                            int seed = 0) {
  constexpr bool kIsFloat = std::is_floating_point<T>::value;
  using IntegerType = typename std::conditional<kIsFloat, int, T>::type;
  using FloatType = typename std::conditional<!kIsFloat, float, T>::type;

  std::unique_ptr<T[]> result(new T[length]);
  auto *p = result.get();
  std::default_random_engine engine(seed);
  if (kIsFloat) {
    std::uniform_real_distribution<FloatType> dist(
        static_cast<FloatType>(low), static_cast<FloatType>(high));
    for (size_t i = 0; i < length; ++i) {
      p[i] = static_cast<T>(dist(engine));
    }
  } else {
    std::uniform_int_distribution<IntegerType> dist(
        static_cast<IntegerType>(low), static_cast<IntegerType>(high));
    for (size_t i = 0; i < length; ++i) {
      p[i] = static_cast<T>(dist(engine));
    }
  }
  return result;
}

template <typename T> T Numel(const std::vector<T> &shapes) {
  return std::accumulate(shapes.begin(), shapes.end(), static_cast<T>(1),
                         std::multiplies<T>());
}

inline double LoopRun(const std::unique_ptr<PaddlePredictor> &predictor,
                      size_t loop_num) {
  auto time_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < loop_num; ++i) {
    PADDLE_CHECK(const_cast<PaddlePredictor *>(predictor.get())->ZeroCopyRun());
  }
  auto time_end = std::chrono::high_resolution_clock::now();
  auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(
                       time_end - time_start)
                       .count();
  return static_cast<double>(time_cost) / loop_num;
}

template <typename T> void Print(std::ostream &out, const T *p, size_t length) {
  if (length == 0) {
    out << "[]";
    return;
  }

  out << "[" << p[0];

  for (size_t i = 1; i < length; ++i) {
    out << ", " << p[i];
  }

  out << "]";
}

template <typename T>
bool Compare(const T *p1, const T *p2, size_t length, double rtol,
             double atol) {
  double max_diff = 0;
  for (size_t i = 0; i < length; ++i) {
    auto value1 = static_cast<double>(p1[i]);
    auto value2 = static_cast<double>(p2[i]);
    auto diff = std::abs(value1 - value2) - (atol + rtol * std::abs(value2));
    max_diff = std::max(diff, max_diff);
  }

  if (max_diff > 0) {
    std::cerr << "Max diff is " << max_diff << std::endl;
    Print(std::cerr, p1, length);
    std::cerr << std::endl;
    Print(std::cerr, p2, length);
    std::cerr << std::endl;
  }

  return max_diff == 0;
}

inline std::string GetEnv(const std::string &env_name) {
  const char *env = std::getenv(env_name.c_str());
  PADDLE_CHECK(env != nullptr);
  return std::string(env);
}
