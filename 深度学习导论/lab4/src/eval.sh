# 原始模型
CUDA_VISIBLE_DEVCIES=0 swift eval --model_type qwen1half-1_8b-chat \
    --eval_dataset no \
    --infer_backend vllm \
    --custom_eval_config ../data/eval_config.json

# LoRA微调后
CUDA_VISIBLE_DEVICES=0 swift eval --ckpt_dir output/qwen1half-1_8b-chat/v0-20240627-170756/checkpoint-400 \
    --eval_dataset no --infer_backend vllm \
    --merge_lora true \
    --custom_eval_config ../data/eval_config.json