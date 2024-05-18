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


sh train_zero2.sh

# 如果不行试试不指定版本，能装最新装最新