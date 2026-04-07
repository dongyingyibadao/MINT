# Stage-B: tokenizer 训练快速上手（train_tokenizer_libero.py）

本说明仅覆盖 tokenizer 训练，不包含 policy 训练。

当前文档与代码同步，默认采用 demo-balanced 全轨迹框架（`--episode_level_align`）。

---

## 1) 先做什么（最短路径）

建议顺序：

1. 先跑 2K 小规模验证，确认 loss 曲线与日志字段正常。
2. 再跑 1M 正式训练（下面三种框架命令）。
3. 从 `checkpoints/last` 或固定 step checkpoint 抽测 policy。

---

## 2) 环境准备

```bash
source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint

export DATASET_ROOT=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero
export CKPT=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth
export ALIGN_MODEL=BAAI/bge-large-en-v1.5

# 建议减少 tokenizer 并行警告
export TOKENIZERS_PARALLELISM=false
```

---

## 3) 当前目标函数（与代码一致）

`--episode_level_align` 模式下，训练由两部分组成：

1. chunk 重建分支：每条 demonstration 内先平均，再在 batch 内按 demonstration 平均（demo-balanced）。
2. 语义对齐分支：整条 demonstration 的 S1 token 序列（由各 chunk latent 拼接）与文本对齐。

总目标：

$$
L_{total} = L_{chunk,\,demo\_balanced} + L_{align}
$$

其中：

$$
L_{chunk} = L_{freq} + \lambda_{vq}L_{vq} + \alpha L_{aux}
$$

$$
L_{align} = w_gL_{global} + w_lL_{local} + w_oL_{order}
$$

说明：

1. $L_{freq}$：分尺度 DCT 频域重建。
2. $L_{vq}$：量化损失（codebook + commitment）。
3. $L_{aux}$：时域辅助重建（L1，含 gripper 兼容分支）。
4. $L_{global}$：demo-level action-text 双向 InfoNCE。
5. $L_{local}$：action token 与 text token 的局部匹配对齐。
6. $L_{order}$：时序单调约束。

频域项默认按尺度权重和归一化：

$$
L_{freq} = w_{spec}\cdot\frac{\sum_s w_s\,\mathrm{MSE}(\mathrm{DCT}(\hat{x}_s),\mathrm{DCT}(x))}{\sum_s w_s}
$$

---

## 4) 关键参数

### 4.1 基础 tokenizer 参数

1. `--spectral_weight`
2. `--spectral_scale_weights`
3. `--aux_l1_weight`
4. `--vq_weight`
5. `--vq_beta`
6. `--disable_spectral_scale_normalization`
7. `--include_gripper_in_spectral`

### 4.2 全轨迹模式参数

1. `--episode_level_align`
2. `--temporal_stride`
3. `--min_episode_len`
4. `--max_episode_len`
5. `--chunk_size`
6. `--stride`

### 4.3 语义对齐参数

1. `--align` / `--align_model`
2. `--align_global_weight`
3. `--align_local_weight`
4. `--align_order_weight`
5. `--align_order_margin`
6. `--align_warmup`
7. `--align_order_warmup`

---

## 5) 2K 小规模验证命令

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
  --dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
  --steps=2000 \
  --batch_size=4 \
  --num_workers=0 \
  --episode_level_align \
  --temporal_stride=2 \
  --min_episode_len=16 \
  --max_episode_len=64 \
  --chunk_size=16 \
  --stride=8 \
  --lr=1e-4 \
  --scheduler=cosine \
  --lr_peak_scale=0.5 \
  --lr_warmup_steps=200 \
  --lr_warmup_init_ratio=0.1 \
  --lr_min_ratio=0.1 \
  --weight_decay=1e-2 \
  --grad_clip=1.0 \
  --spectral_weight=0.15 \
  --spectral_scale_weights=1,0.7,0.4 \
  --aux_l1_weight=1.0 \
  --vq_weight=1.0 \
  --vq_beta=0.25 \
  --align \
  --align_model="$ALIGN_MODEL" \
  --align_weight=0.05 \
  --align_global_weight=0.05 \
  --align_local_weight=0.02 \
  --align_order_weight=0.01 \
  --align_order_margin=0.02 \
  --align_warmup=500 \
  --align_order_warmup=800 \
  --log_freq=20 \
  --save_freq=500 \
  --output_dir=outputs/stageb_episode_2k_demo_balanced \
  2>&1 | tee outputs/stageb_episode_2k_demo_balanced.log
```

---

## 6) 三种框架训练命令（预计约 200K 收敛，实际执行 1M）

下面三组命令都基于同一训练骨架（`--episode_level_align` + demo-balanced），并统一为：

1. `--steps=1000000`（预计约 150K-250K 进入收敛平台，实际跑满 1M）。
2. `--save_freq=2000`。
3. 稳定优先统一骨架：`--batch_size=4`、`--stride=8`、`--lr=2e-5`、`--lr_peak_scale=0.3`、`--lr_warmup_steps=20000`、`--grad_clip=0.5`。


### A. 无语义对齐（baseline）

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
  --dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
  --steps=1000000 \
  --batch_size=4 \
  --num_workers=4 \
  --episode_level_align \
  --temporal_stride=2 \
  --min_episode_len=16 \
  --max_episode_len=64 \
  --chunk_size=16 \
  --stride=8 \
  --lr=2e-5 \
  --scheduler=cosine \
  --lr_peak_scale=0.3 \
  --lr_warmup_steps=20000 \
  --lr_warmup_init_ratio=0.02 \
  --lr_min_ratio=0.02 \
  --weight_decay=1e-2 \
  --grad_clip=0.5 \
  --spectral_weight=0.15 \
  --spectral_scale_weights=1,0.7,0.4 \
  --aux_l1_weight=1.0 \
  --vq_weight=1.0 \
  --vq_beta=0.25 \
  --log_freq=500 \
  --save_freq=2000 \
  --output_dir=outputs/AAAstageb_framework_no_align_1M \
  2>&1 | tee outputs/AAAstageb_framework_no_align_1M.log
```

### B. 有语义对齐（同时作用 encoder + codebook）

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
  --dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
  --tokenizer_ckpt="outputs/AAAstageb_framework_align_both_1M/checkpoints/128000/tokenizer.pt" \
  --steps=172000 \
  --batch_size=4 \
  --num_workers=4 \
  --episode_level_align \
  --temporal_stride=2 \
  --min_episode_len=16 \
  --max_episode_len=64 \
  --chunk_size=16 \
  --stride=8 \
  --lr=1e-5 \
  --scheduler=cosine \
  --lr_peak_scale=0.3 \
  --lr_warmup_steps=0 \
  --lr_warmup_init_ratio=0.02 \
  --lr_min_ratio=0.02 \
  --weight_decay=1e-2 \
  --grad_clip=0.5 \
  --spectral_weight=0.15 \
  --spectral_scale_weights=1,0.7,0.4 \
  --aux_l1_weight=1.0 \
  --vq_weight=1.0 \
  --vq_beta=0.25 \
  --align \
  --align_model="BAAI/bge-large-en-v1.5" \
  --align_weight=0.05 \
  --align_global_weight=0.05 \
  --align_local_weight=0.02 \
  --align_local_mode=window \
  --align_local_action_window_sizes=1,2 \
  --align_local_text_window_sizes=1,2,4 \
  --align_local_topk=2 \
  --align_local_window_temperature=0.07 \
  --align_order_weight=0.01 \
  --align_order_mode=window \
  --align_order_window_temperature=0.07 \
  --align_order_margin=0.02 \
  --align_warmup=0 \
  --align_order_warmup=0 \
  --align_quant_weight=1.0 \
  --align_encoder_weight=1.0 \
  --log_freq=500 \
  --save_freq=4000 \
  --output_dir=outputs/AAAstageb_framework_align_both_resume_lr1e5_to300k \
  2>&1 | tee outputs/AAAstageb_framework_align_both_resume_lr1e5_to300k.log
```

### C. 有语义对齐（仅作用 encoder）

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
  --dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
  --tokenizer_ckpt="outputs/AAAstageb_framework_align_encoder_only_1M/checkpoints/168000/tokenizer.pt" \
  --steps=132000 \
  --batch_size=4 \
  --num_workers=4 \
  --episode_level_align \
  --temporal_stride=2 \
  --min_episode_len=16 \
  --max_episode_len=64 \
  --chunk_size=16 \
  --stride=8 \
  --lr=1e-5 \
  --scheduler=cosine \
  --lr_peak_scale=0.3 \
  --lr_warmup_steps=0 \
  --lr_warmup_init_ratio=0.02 \
  --lr_min_ratio=0.02 \
  --weight_decay=1e-2 \
  --grad_clip=0.5 \
  --spectral_weight=0.15 \
  --spectral_scale_weights=1,0.7,0.4 \
  --aux_l1_weight=1.0 \
  --vq_weight=1.0 \
  --vq_beta=0.25 \
  --align \
  --align_model="BAAI/bge-large-en-v1.5" \
  --align_weight=0.05 \
  --align_global_weight=0.05 \
  --align_local_weight=0.02 \
  --align_local_mode=window \
  --align_local_action_window_sizes=1,2 \
  --align_local_text_window_sizes=1,2,4 \
  --align_local_topk=2 \
  --align_local_window_temperature=0.07 \
  --align_order_weight=0.01 \
  --align_order_mode=window \
  --align_order_window_temperature=0.07 \
  --align_order_margin=0.02 \
  --align_warmup=0 \
  --align_order_warmup=0 \
  --align_quant_weight=0.0 \
  --align_encoder_weight=1.0 \
  --log_freq=500 \
  --save_freq=4000 \
  --output_dir=outputs/AAAstageb_framework_align_encoder_only_resume_lr1e5_to300k \
  2>&1 | tee outputs/AAAstageb_framework_align_encoder_only_resume_lr1e5_to300k.log
```

---

## 7) 结果文件如何使用

1. 最新 checkpoint: `outputs/<run_name>/checkpoints/last/tokenizer.pt`
2. 固定步 checkpoint: `outputs/<run_name>/checkpoints/<step>/tokenizer.pt`
3. 训练报告: `outputs/<run_name>/train_report.json`

建议：

1. 每隔固定步数保存（`--save_freq=2000`）。
2. 至少抽测一个中段和一个后段 checkpoint。
3. policy 训练时，将 `--policy.vqvae_name_or_path` 指向目标 `tokenizer.pt`。

---

## 8) 后台运行（可选）

tmux：

```bash
tmux new -s tokenizer_1m
# 在新会话中粘贴训练命令并回车
# 退出会话不停止训练：Ctrl-b 然后按 d
```

nohup（示例为 A 方法）：

```bash
nohup bash -lc 'cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint && PYTHONPATH=src python scripts/train_tokenizer_libero.py --dataset_root="$DATASET_ROOT" --tokenizer_ckpt="$CKPT" --steps=1000000 --batch_size=4 --num_workers=4 --episode_level_align --temporal_stride=2 --min_episode_len=16 --max_episode_len=64 --chunk_size=16 --stride=8 --lr=2e-5 --scheduler=cosine --lr_peak_scale=0.3 --lr_warmup_steps=20000 --lr_warmup_init_ratio=0.02 --lr_min_ratio=0.02 --weight_decay=1e-2 --grad_clip=0.5 --spectral_weight=0.15 --spectral_scale_weights=1,0.7,0.4 --aux_l1_weight=1.0 --vq_weight=1.0 --vq_beta=0.25 --log_freq=200 --save_freq=2000 --output_dir=outputs/stageb_framework_no_align_1M' > outputs/stageb_framework_no_align_1M.nohup.log 2>&1 &
```
