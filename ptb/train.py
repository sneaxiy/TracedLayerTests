# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import TracedLayer
import numpy as np
import random
import reader
import six
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ptb model")
    parser.add_argument(
        '--use_dygraph',
        type=int,
        default=1,
        help='Whether to use dygraph to train')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./simple-examples/data',
        help='Data path of training data')
    return parser.parse_args()


def get_place():
    if fluid.is_compiled_with_cuda():
        return fluid.CUDAPlace(0)
    else:
        return fluid.CPUPlace()


class SimpleLSTMRNN(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=2,
                 init_scale=0.1,
                 dropout=None):
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._num_steps = num_steps
        self._create_parameter()

    def _create_parameter(self):
        self.weight_1_arr = []
        self.weight_2_arr = []
        self.bias_arr = []
        self.mask_array = []

        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        cell_array = []
        hidden_array = []

        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        res = []
        for index in range(self._num_steps):
            _input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            _input = fluid.layers.reshape(
                _input, shape=[-1, self._hidden_size])
            for k in range(self._num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                nn = fluid.layers.concat([_input, pre_hidden], 1)
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                hidden_array[k] = m
                cell_array[k] = c
                _input = m

                if self._dropout is not None and self._dropout > 0.0:
                    _input = fluid.layers.dropout(
                        _input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
            res.append(
                fluid.layers.reshape(
                    _input, shape=[1, -1, self._hidden_size]))
        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        return real_res, last_hidden, last_cell


class PtbModel(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 is_sparse=False,
                 dropout=None):
        super(PtbModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        self.embedding = Embedding(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=is_sparse,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label, init_hidden, init_cell):
        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        x_emb = self.embedding(input)
        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.drop_out,
                dropout_implementation='upscale_in_train')
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)
        rnn_out = fluid.layers.reshape(
            rnn_out, shape=[-1, self.num_steps, self.hidden_size])
        projection = fluid.layers.matmul(rnn_out, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection,
                                                  self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss, last_hidden, last_cell


def train_ptb_model(train_data, use_dygraph=True, model_type='small'):
    vocab_size = 10000

    if model_type == "small":
        num_layers = 2
        batch_size = 20
        hidden_size = 200
        num_steps = 20
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 4
        max_epoch = 13
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
    elif model_type == "medium":
        num_layers = 2
        batch_size = 20
        hidden_size = 650
        num_steps = 35
        init_scale = 0.05
        max_grad_norm = 5.0
        epoch_start_decay = 6
        max_epoch = 39
        dropout = 0.5
        lr_decay = 0.8
        base_learning_rate = 1.0
    elif model_type == "large":
        num_layers = 2
        batch_size = 20
        hidden_size = 1500
        num_steps = 35
        init_scale = 0.04
        max_grad_norm = 10.0
        epoch_start_decay = 14
        max_epoch = 55
        dropout = 0.65
        lr_decay = 1.0 / 1.15
        base_learning_rate = 1.0
    else:
        print("model type not support")
        return

    batch_len = len(train_data) // batch_size
    total_batch_size = (batch_len - 1) // num_steps
    log_interval = total_batch_size // 20

    bd = []
    lr_arr = [1.0]
    for i in range(1, max_epoch):
        bd.append(total_batch_size * i)
        new_lr = base_learning_rate * (lr_decay
                                       **max(i + 1 - epoch_start_decay, 0.0))
        lr_arr.append(new_lr)

    def build_model():
        fluid.default_startup_program().random_seed = 1
        fluid.default_main_program().random_seed = 1
        np.random.seed(1)
        random.seed(1)

        return PtbModel(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_steps=num_steps,
            init_scale=init_scale,
            dropout=dropout)

    def generate_unique_ids():
        r = random.sample(six.moves.range(vocab_size), batch_size * num_steps)
        return np.array(r).reshape([batch_size, num_steps]).astype('int64')

    def gradient_clip():
        if use_dygraph:
            #return fluid.dygraph_grad_clip.GradClipByValue(min_value=-0.01, max_value=0.01)
            return fluid.dygraph_grad_clip.GradClipByGlobalNorm(max_grad_norm)
        else:
            #return fluid.clip.GradientClipByValue(min=-0.01, max=0.01)
            return fluid.clip.GradientClipByGlobalNorm(clip_norm=max_grad_norm)

    def train_static_graph():
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            ptb_model = build_model()

            x = fluid.data(shape=[-1, num_steps], name='x', dtype='int64')
            y = fluid.data(shape=[-1, 1], name='y', dtype='int64')
            init_hidden = fluid.data(
                shape=[num_layers, -1, hidden_size],
                dtype='float32',
                name='init_hidden')
            init_cell = fluid.data(
                shape=[num_layers, -1, hidden_size],
                dtype='float32',
                name='init_cell')

            loss, last_hidden, last_cell = ptb_model(x, y, init_hidden,
                                                     init_cell)

            fluid.clip.set_gradient_clip(gradient_clip())

            sgd = SGDOptimizer(learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr_arr))
            sgd.minimize(loss)

            place = get_place()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            main_program = fluid.default_main_program()

            for epoch_id in range(max_epoch):
                total_loss = 0.0
                iters = 0.0

                init_hidden_data = np.zeros(
                    [num_layers, batch_size, hidden_size], dtype='float32')
                init_cell_data = np.zeros(
                    [num_layers, batch_size, hidden_size], dtype='float32')

                train_data_iter = reader.get_data_iter(train_data, batch_size,
                                                       num_steps)

                for batch_id, batch in enumerate(train_data_iter):
                    x_data, y_data = batch
                    x_data = x_data.reshape((-1, num_steps)).astype('int64')
                    y_data = y_data.reshape((-1, 1)).astype('int64')

                    # x_data = generate_unique_ids()

                    feed = {
                        x.name: x_data,
                        y.name: y_data,
                        init_hidden.name: init_hidden_data,
                        init_cell.name: init_cell_data,
                    }

                    out_loss, init_hidden_data, init_cell_data = \
                        exe.run(main_program, feed=feed, fetch_list=[loss, last_hidden, last_cell])

                    total_loss += out_loss
                    iters += num_steps

                    if batch_id > 0 and batch_id % log_interval == 0:
                        ppl = np.exp(total_loss / iters)
                        print(
                            "-- Epoch:[%d]; Batch:[%d]; loss: %.6f; ppl: %.5f"
                            % (epoch_id, batch_id, out_loss, ppl[0]))

                print("one epoch finished", epoch_id)

            fluid.io.save_inference_model(
                dirname='./infer_static_graph',
                feeded_var_names=[
                    x.name, y.name, init_hidden.name, init_cell.name
                ],
                target_vars=[loss],
                executor=exe)

    def train_dygraph():
        with fluid.dygraph.guard(get_place()):
            ptb_model = build_model()

            sgd = SGDOptimizer(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=bd, values=lr_arr),
                parameter_list=ptb_model.parameters())

            grad_clip = gradient_clip()

            backward_strategy = fluid.dygraph.BackwardStrategy()
            backward_strategy.sort_sum_gradient = True

            traced_layer = None

            for epoch_id in range(max_epoch):
                total_loss = 0.0
                iters = 0.0

                init_hidden_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')
                init_cell_data = np.zeros(
                    (num_layers, batch_size, hidden_size), dtype='float32')

                train_data_iter = reader.get_data_iter(train_data, batch_size,
                                                       num_steps)

                for batch_id, batch in enumerate(train_data_iter):
                    x_data, y_data = batch
                    x_data = x_data.reshape((-1, num_steps))
                    y_data = y_data.reshape((-1, 1))

                    # x_data = generate_unique_ids()

                    x = to_variable(x_data)
                    y = to_variable(y_data)

                    init_hidden = to_variable(init_hidden_data)
                    init_cell = to_variable(init_cell_data)

                    if traced_layer is None:
                        outs, traced_layer = TracedLayer.trace(
                            ptb_model, [x, y, init_hidden, init_cell])
                    else:
                        outs = ptb_model(x, y, init_hidden, init_cell)

                    dy_loss, last_hidden, last_cell = outs

                    out_loss = dy_loss.numpy()

                    init_hidden_data = last_hidden.numpy()
                    init_cell_data = last_cell.numpy()

                    dy_loss.backward(backward_strategy)
                    sgd.minimize(dy_loss, grad_clip=grad_clip)
                    ptb_model.clear_gradients()

                    total_loss += out_loss
                    iters += num_steps

                    if batch_id > 0 and batch_id % log_interval == 0:
                        ppl = np.exp(total_loss / iters)
                        print(
                            "-- Epoch:[%d]; Batch:[%d]; loss: %.6f; ppl: %.5f"
                            % (epoch_id, batch_id, out_loss, ppl[0]))

                print("one epoch finished", epoch_id)

            traced_layer.save_inference_model(
                dirname='./infer_dygraph', fetch=[0])

    if use_dygraph:
        print('Dygraph mode enabled')
        train_dygraph()
    else:
        print('Static graph mode enabled')
        train_static_graph()


def main():
    args = parse_args()
    print(args)
    train_data, _, _ = reader.get_ptb_data(args.data_path)
    train_ptb_model(train_data, use_dygraph=args.use_dygraph)


if __name__ == '__main__':
    main()
