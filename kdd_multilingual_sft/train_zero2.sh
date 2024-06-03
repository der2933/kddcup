# --nnodes 1 --nproc_per_node 4 --master_port 25641


# --------------------------evaluate_predict_generate----------------------------
# python train_sft.py \
#     --model_name_or_path /root/amazon-kdd-cup-2024-starter-kit/models/vicuna-7b-v1.5 \
#     --use_lora false \
#     --do_train false \
#     --use_qlora4bit false \
#     --do_eval true \
#     --do_eval_lora_ckp /root/train_sft/output/ \
#     --eval_data_path /root/train_sft/data/ \
#     --bf16 false \
#     --fp16 true \
#     --output_dir /root/train_sft/output/ \
#     --per_device_eval_batch_size 1 \
#     --source_length 1024 \
#     --target_length 256 

# ------------------------python accelerate train----------------------------------
# python train_sft.py \
#     --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --use_deepspeed false \
#     --use_flashattn2 true \
#     --weight_decay 1e-2 \
#     --warmup_steps 300 \
#     --do_train true \
#     --do_eval false \
#     --train_data_path ./data \
#     --bf16 false \
#     --fp16 true \
#     --output_dir ./output/ecinstruct_sft \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 64 \
#     --evaluation_strategy no \
#     --save_strategy epoch \
#     --learning_rate 5e-5 \
#     --logging_steps 200 \
#     --source_length 4096


# ----------------deepspeed_train-------------------------------
deepspeed --include localhost:0,1,2,3 ./train_sft.py \
    --deepspeed  ./deepspeed_zero2_config.json \
    --model_name_or_path ../../../model/meta-llama-3-8B-Instruct \
    --use_deepspeed true \
    --weight_decay 1e-3 \
    --warmup_steps 1000 \
    --do_train true \
    --do_eval false \
    --train_data_path ./data/ \
    --bf16 false \
    --fp16 true \
    --output_dir ./output/24-6-1/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2  \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --learning_rate 3e-5 \
    --logging_steps 10 \
    --source_length 4096 \
    --neftune_noise_alpha 4 \
    --use_flashattn2 false \
    --use_lora true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_length 128



    # --use_lora true \
    # --lora_r 16 \
    # --lora_alpha 32 \
    # --lora_dropout 0.05 \
# --neftune_noise_alpha 4 \
#     --save_total_limit 6 \
# --use_loftq
#    --per_device_train_batch_size 10 \
#        --gradient_accumulation_steps 6 \
#    --auto_find_batch_size true \
#     --do_eval_lora_ckp ./output/checkpoint-2050 \
# deepspeed --include localhost:0 train_sft.py \
# --deepspeed ds_zero2_no_offload.json \
# --save_steps 1000 \