把 ecinstruct_sft解压后 放到 "./data" 文件夹下面

conda create -n kdd python=3.8
conda activate kdd

cd kdd_multilingual_sft


pip install torch==2.1.2  --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.39.1
pip install packaging
pip uninstall -y ninja && pip install ninja
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install peft bitsandbytes sentencepiece deepspeed datasets protobuf accelerate


TORCH_EXTENSIONS_DIR=./torch-extensions sh train_zero2.sh

# 如果不行试试不指定版本，能装最新装最新

## deepspeed bug
# pip install deepspeed
# ninja 问题 https://blog.csdn.net/qq_21768483/article/details/129853752
# /include/python3.8/Python.h:44:10: fatal error: crypt.h: No such file or directory   sudo cp /usr/include/crypt.h /home/{username}/anaconda3/envs/{venv name}/include/python3.8/
# error: 'timespec_get' has not been declared in '::'  \ async_io libaio缺失        conda upgrade -c conda-forge --all
# x86_64-conda-linux-gnu/bin/ld: cannot find -lcurand: not such file or directory       sudo apt install libaio-dev
# 新操作后尝试删除torch extension缓存  rm -rf ~/.cache/torch_extensions/
#  python setup.py egg_info did not run successfully.         pip install --upgrade setuptools

# 如果都不行考虑,  https://www.deepspeed.ai/tutorials/advanced-install/
# 源码安装
# git https://github.com/microsoft/DeepSpeed.git & cd DeepSpeed & conda env create -n deepspeed -f environment.yml & conda activate deepspeed & pip install .
# pip install transformers datasets protobuf accelerate tiktoken sentencepiece peft bitsandbytes

# 调参
# deepspeed配置简单介绍https://zhuanlan.zhihu.com/p/645627795,https://zhuanlan.zhihu.com/p/675360966 阶段越高并行程度越大
# 速度方面（左边比右边快）
# zero 0 (DDP) > zero 1 > zero 2 > zero 2 + offload > zero 3 > zero 3 + offload
# GPU 内存使用情况（右侧的 GPU 内存效率高于左侧）
# zero 0 (DDP) < zero 1 < zero 2 < zero 2 + offload < zero 3 < zero 3 + offload

# huggingface transformers的trainer 调参https://zhuanlan.zhihu.com/p/363670628
# deepspeed一些参数和trainer_zero2.sh一些重合，这种情况下在zero配置文件设置auto，在train_zero2.sh调参
# 如果有同一参数如果在train_zero2.sh的参数和zero配置的是不同配置会报错