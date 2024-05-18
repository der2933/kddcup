## Installation

Git Clone MA-RLHF

```bash
git clone git@github.com:dhcode-cpp/MA-RLHF.git
cd MA-RLHF
```

Create Dev Environment

```bash
conda create -n llm python=3.11
conda activate llm
pip install -r requirements.txt
pip install flash-attn #
```

Setting Environment

```bash
export WANDB_API_KEY={YOU_WANDB_TOKEN} # from https://wandb.ai/authorize
export HF_ENDPOINT=https://hf-mirror.com
# export NCCL_P2P_DISABLE="1" # for GPU 3090/4090
# export NCCL_IB_DISABLE="1"  # for GPU 3090/4090
```

DeepSpeed Test

```bash
deepspeed ./test/test_QLoRA.py
```

- Deepspeed config json is `./config/ds.json`
