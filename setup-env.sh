# export LD_LIBRARY_PATH=${pwd}/.venv/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:${pwd}/.venv/lib/python3.9/site-packages/nvidia/cudnn/lib:${pwd}/.venv/lib/python3.9/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/usr/lib/wsl/lib
export LD_LIBRARY_PATH=`python -c "import os;import nvidia.cuda_runtime.lib; print(os.path.dirname(nvidia.cuda_runtime.lib.__file__))"`:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=`python -c "import os;import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cudnn.lib.__file__))"`:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=`python -c "import os;import nvidia.cublas.lib; print(os.path.dirname(nvidia.cublas.lib.__file__))"`:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=`python -c "import os;import nvidia.curand.lib; print(os.path.dirname(nvidia.curand.lib.__file__))"`:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=`python -c "import os;import nvidia.cufft.lib; print(os.path.dirname(nvidia.cufft.lib.__file__))"`:${LD_LIBRARY_PATH}
