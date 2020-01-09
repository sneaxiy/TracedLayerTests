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

  int batch_size = 1;
  int num_layers = 2;
  int num_steps = 20;
  int hidden_size = 200;

  int vocab_size = 10000;

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);

  std::vector<int> x_shape = {batch_size, num_steps};
  std::vector<int> y_shape = {batch_size * num_steps, 1};
  std::vector<int> init_hidden_shape = {num_layers, batch_size, hidden_size};
  std::vector<int> init_cell_shape = {num_layers, batch_size, hidden_size};

  auto x_t = predictor->GetInputTensor(input_names[0]);
  x_t->Reshape(x_shape);

  auto y_t = predictor->GetInputTensor(input_names[1]);
  y_t->Reshape(y_shape);

  auto init_hidden_t = predictor->GetInputTensor(input_names[2]);
  init_hidden_t->Reshape(init_hidden_shape);

  auto init_cell_t = predictor->GetInputTensor(input_names[3]);
  init_cell_t->Reshape(init_cell_shape);

  auto random_x = Random<int64_t>(Numel(x_shape), 0, vocab_size - 1, 0);
  x_t->copy_from_cpu(random_x.get());

  auto random_y = Random<int64_t>(Numel(y_shape), 0, vocab_size - 1, 1);
  y_t->copy_from_cpu(random_y.get());

  auto random_init_hidden = Random<float>(Numel(init_hidden_shape), 0, 1, 2);
  init_hidden_t->copy_from_cpu(random_init_hidden.get());

  auto random_init_cell = Random<float>(Numel(init_cell_shape), 0, 1, 3);
  init_cell_t->copy_from_cpu(random_init_cell.get());

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

  std::cout << "Loss (dygraph) = " << out1[0] << std::endl;
  std::cout << "Loss (static graph) = " << out2[0] << std::endl;

  PADDLE_CHECK(time_diff_ratio <= 0.03);
  PADDLE_CHECK(Compare(out1.data(), out2.data(), out1.size(), 0.05, 0));

  return 0;
}
