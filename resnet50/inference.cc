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

#include "inference_utils.h" // NOLINT

void BuildConfig(AnalysisConfig *config, const std::string &model_dirname) {
  config->SetModel(model_dirname);
  config->EnableUseGpu(2048, 0);
  config->SwitchUseFeedFetchOps(false);
}

std::vector<float> TestMain(const std::string &dirname, double *t = nullptr) {
  AnalysisConfig config;
  BuildConfig(&config, dirname);

  auto predictor = paddle::CreatePaddlePredictor(config);

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);

  std::vector<int> x_shape = {1, 3, 224, 224};

  auto x_t = predictor->GetInputTensor(input_names[0]);
  x_t->Reshape(x_shape);

  auto random_x = Random<float>(Numel(x_shape), -1, 1, 0);
  x_t->copy_from_cpu(random_x.get());

  LoopRun(predictor, 2000);

  int loop_num = 5000;
  auto time_cost = LoopRun(predictor, loop_num);
  if (t) {
    *t = time_cost;
  }

  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  out_data.resize(Numel(output_shape));
  output_t->copy_to_cpu(out_data.data());

  return out_data;
}

int main() {
  double time1, time2;

  auto out1 = TestMain(GetEnv("FLAGS_dygraph_dirname"), &time1);
  auto out2 = TestMain(GetEnv("FLAGS_static_graph_dirname"), &time2);

  PADDLE_CHECK(out1.size() == out2.size());

  auto time_diff_ratio = time1 / time2 - 1;

  std::cout << "Time cost (dygraph) = " << time1 << " ms" << std::endl;
  std::cout << "Time cost (static graph) = " << time2 << " ms" << std::endl;
  std::cout << "Time diff " << time_diff_ratio * 100 << "%" << std::endl;

  PADDLE_CHECK(time_diff_ratio <= 0.03);
  PADDLE_CHECK(Compare(out1.data(), out2.data(), out1.size(), 0, 0));

  return 0;
}
