# TracedLayer Test scripts  

This project is for testing `paddle.fluid.dygraph.TracedLayer` interface.
It would check whether the speed and accuracy of the inference model
saved by `paddle.fluid.dygraph.TracedLayer` is the same as the static graph
model. The process would be: 

- train the model and save the inference model in dygraph mode and static graph mode respectively.      
- load the saved models using C++ inference APIs to check the speed and accuracy.

## Preparation

- Install the C++ inference library. Please refer to [Inference library installation](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/build_and_install_lib_cn.html). Make sure that the C++ inference library is built when `WITH_GPU=ON` and `WITH_MKL=ON` .  
- Change 3 environment variables in `run.sh`.
  - `LIB_DIR` : directory where the C++ inference library is installed, i.e, `PADDLE_ROOT` in [Inference library installation](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/build_and_install_lib_cn.html).
  - `CUDA_LIB_DIR` : directory where CUDA is installed, i.e., where `libcudart.so` is located. 
  - `CUDNN_LIB_DIR`: directory where CUDNN is installed, i.e., where `libcudnn.so` is located.

## How to run models 
### 1. PTB Model

Please download dataset first by running the following command:

```
cd ptb && sh download_data.sh && cd - 
```

Then you can run the model: 

```
MODEL=ptb CUDA_VISIBLE_DEVICES=0 sh run.sh
```

If there is no error and you see the following outputs, the test is successful. 
```
Model ptb runs successfully!
```

### 2. ResNet50 model

Run the model:
```
MODEL=resnet50 CUDA_VISIBLE_DEVICES=0 sh run.sh
```

If there is no error and you see the following outputs, the test is successful.
```
Model resnet50 runs successfully!
```
