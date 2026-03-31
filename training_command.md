
 进行policy批量训练设置
 ```shell
source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --dataset.repo_id=local/libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=mint \
  --policy.pretrained_path=huangrm/pi05_base \
  --policy.vqvae_name_or_path=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.optimizer_lr=3e-5 \
  --policy.scheduler_decay_lr=3e-6 \
  --policy.optimizer_grad_clip_norm=0.5 \
  --steps=2000 \
  --batch_size=4 \
  --num_workers=2 \
  --log_freq=20 \
  --output_dir=./outputs/policy_2k_org_stable_try1 \
  --job_name=policy_2k_org_stable_try1 \
  --policy.repo_id=local/policy_2k_org_stable_try1 | tee outputs/policy_2k_org_stable_try1.log
 ```


运行多核训练命令

 ```shell
source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
  --dataset.repo_id=local/libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=mint \
  --policy.pretrained_path=huangrm/pi05_base \
  --policy.vqvae_name_or_path=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.optimizer_lr=3e-5 \
  --policy.scheduler_decay_lr=3e-6 \
  --policy.optimizer_grad_clip_norm=0.5 \
  --steps=2000 \
  --batch_size=4 \
  --num_workers=2 \
  --log_freq=20 \
  --output_dir=./outputs/policy_2k_org_stable_try1 \
  --job_name=policy_2k_org_stable_try1 \
  --policy.repo_id=local/policy_2k_org_stable_try1 | tee outputs/policy_2k_org_stable_try1.log

############ train_org

source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint

mkdir -p outputs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 你已验证单卡 conv 正常，这里保留库优先级避免多进程回退系统库
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
unset LD_PRELOAD

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=local/libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=mint \
  --policy.pretrained_path=huangrm/pi05_base \
  --policy.vqvae_name_or_path=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --steps=200000 \
  --batch_size=16 \
  --log_freq=500 \
  --save_freq=2000 \
  --output_dir=./outputs/policy_100k_org_stable_huangrm_pi05_test \
  --job_name=policy_100k_org_stable_huangrm_pi05_test \
  --policy.repo_id=local/policy_100k_org_stable_huangrm_pi05_test \
  2>&1 | tee outputs/policy_100k_org_stable_huangrm_pi05_test.log



#### train_raw

source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint

mkdir -p outputs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 你已验证单卡 conv 正常，这里保留库优先级避免多进程回退系统库
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
unset LD_PRELOAD

accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=local/libero \
  --dataset.root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero \
  --policy.type=mint \
  --policy.pretrained_path=huangrm/pi05_base \
  --policy.vqvae_name_or_path=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint/outputs/stageb_align_raw_600k/checkpoints/last/tokenizer.pt \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --steps=200000 \
  --batch_size=16 \
  --log_freq=500 \
  --output_dir=./outputs/policy_200k_raw_stable_huangrm_pi05 \
  --job_name=policy_200k_raw_stable_huangrm_pi05 \
  --policy.repo_id=local/policy_200k_raw_stable_huangrm_pi05 \
  2>&1 | tee outputs/policy_200k_raw_stable_huangrm_pi05.log
 ```

 库自检

 ```bash
source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint

# 自动找 PyTorch wheels 自带的 nvidia 库目录
eval "$(
python - <<'PY'
import site, os
keys = {
  "CUDNN_LIB":"nvidia/cudnn/lib",
  "CUBLAS_LIB":"nvidia/cublas/lib",
  "CURAND_LIB":"nvidia/curand/lib",
  "CUFFT_LIB":"nvidia/cufft/lib",
  "CUDA_RUNTIME_LIB":"nvidia/cuda_runtime/lib",
}
found={}
for sp in site.getsitepackages():
  for k,rel in keys.items():
    p=os.path.join(sp,rel)
    if k not in found and os.path.isdir(p):
      found[k]=p
for k,v in found.items():
  print(f'export {k}="{v}"')
PY
)"

export LD_LIBRARY_PATH="$CUDNN_LIB:$CUBLAS_LIB:$CURAND_LIB:$CUFFT_LIB:$CUDA_RUNTIME_LIB:$CONDA_PREFIX/lib"
unset LD_PRELOAD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 自检：能否正常调用 cudnn
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
x=torch.randn(8,3,224,224,device='cuda')
w=torch.randn(16,3,3,3,device='cuda')
y=torch.nn.functional.conv2d(x,w,padding=1)
print("conv ok:", float(y.mean()))
PY
```

## eval
```shell
lerobot-eval \
    --policy.path=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint/outputs/policy_200k_org_stable_huangrm_pi05/checkpoints/040000/pretrained_model \
    --env.type=libero \
    --env.task=libero_10,libero_object,libero_spatial,libero_goal \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --seed=42 \
    --policy.n_action_steps=4

lerobot-eval \
    --policy.path=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint/outputs/policy_200k_org_stable_try1/checkpoints/040000/pretrained_model \
    --env.type=libero \
    --env.task=libero_10,libero_object,libero_spatial,libero_goal \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --seed=42 \
    --policy.n_action_steps=4

lerobot-eval \
    --policy.path=huangrm/MINT-libero \
    --env.type=libero \
    --env.task=libero_10,libero_object,libero_spatial,libero_goal \
    --eval.batch_size=1 \
    --eval.n_episodes=2 \
    --seed=42 \
    --policy.n_action_steps=4
```