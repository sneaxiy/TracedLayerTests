set -ex

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

# Set the path of inference library, cuda and cudnn
LIB_DIR=/Paddle/Paddle/inference_install_dir
CUDA_LIB_DIR=/usr/local/cuda-9.0/targets/x86_64-linux/lib/
CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/


# STEP 1: train in dygraph and static graph mode
if [ -z "${MODEL}" ]; then
  echo "env MODEL must be set"
  exit 1
fi

cd ${MODEL}
python train.py --use_dygraph 1 && python train.py --use_dygraph 0
cd -

# STEP 2: compile inference code 
export FLAGS_dygraph_dirname=${MODEL}/infer_dygraph
export FLAGS_static_graph_dirname=${MODEL}/infer_static_graph

sh run_impl.sh ${LIB_DIR} ${MODEL} ${WITH_MKL} ${WITH_GPU} ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR} ${USE_TENSORRT}

# STEP 3: run inference code
./build/${MODEL}_inference

echo "Model ${MODEL} runs successfully!"
