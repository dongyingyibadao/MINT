# Stage-B: train_tokenizer_libero.py 三种训练状态说明

本目录仅针对 tokenizer 训练，不涉及 policy 训练入口。

## 1) 环境准备

```bash
source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint
```

建议设置公共变量：

```bash
export DATASET_ROOT=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero
export CKPT=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth
export ALIGN_MODEL=BAAI/bge-large-en-v1.5

export STEPS=2000
export BATCH=16
export WORKERS=4
export STRIDE=4
export LR=1e-4
export WD=1e-2
export CLIP=1.0
export LOGF=20
export SAVEF=200

# 动态学习率参数（warmup + cosine）
export LR_PEAK_SCALE=0.5
export LR_WARMUP_STEPS=100
export LR_WARMUP_INIT_RATIO=0.1
export LR_MIN_RATIO=0.1
```

## 2) 三种状态如何设置

### A. baseline（无文本对齐）

核心开关：不传 `--align`。

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=2000 \
	--batch_size=16 \
	--num_workers=4 \
	--stride=4 \
	--lr=1e-4 \
	--scheduler=cosine \
	--lr_peak_scale=0.5 \
	--lr_warmup_steps=100 \
	--lr_warmup_init_ratio=0.1 \
	--lr_min_ratio=0.1 \
	--weight_decay=1e-2 \
	--grad_clip=1.0 \
	--log_freq=20 \
	--save_freq=200 \
	--tokenizer_ckpt="/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth" \
	--output_dir=outputs/stageb_baseline_2k
```

### B. pretrained_align（预训练初始化 + 文本对齐）

核心开关：传 `--align`，并传 `--tokenizer_ckpt`。

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=2000000 \
	--batch_size=16 \
	--num_workers=8 \
	--stride=4 \
	--lr=3e-5 \
    --align_weight=0.05 \
	--scheduler=cosine \
	--lr_peak_scale=0.3 \
	--lr_warmup_steps=50000 \
	--lr_warmup_init_ratio=0.02 \
	--lr_min_ratio=0.02 \
	--weight_decay=1e-2 \
	--grad_clip=0.5 \
	--log_freq=200 \
	--save_freq=2000 \
	--tokenizer_ckpt="/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth" \
	--output_dir=outputs/stageb_align_pretrained_2k \
	--align \
	--align_model="BAAI/bge-large-en-v1.5"
```

### C. raw_align（随机初始化 + 文本对齐）

核心开关：传 `--align`，但不传 `--tokenizer_ckpt`（或传空字符串）。

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=2000 \
	--batch_size=16 \
	--num_workers=4 \
	--stride=4 \
	--lr=1e-4 \
	--scheduler=cosine \
	--lr_peak_scale=0.5 \
	--lr_warmup_steps=100 \
	--lr_warmup_init_ratio=0.1 \
	--lr_min_ratio=0.1 \
	--weight_decay=1e-2 \
	--grad_clip=1.0 \
	--log_freq=20 \
	--save_freq=200 \
	--output_dir=outputs/stageb_align_raw_2k \
	--align \
	--align_model="BAAI/bge-large-en-v1.5"
```

### D. org（随机初始化）

核心开关：不传 `--align`，不传 `--tokenizer_ckpt`（或传空字符串）。

```bash
source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate && conda activate mint && cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint && PYTHONPATH=src python scripts/train_tokenizer_libero.py --dataset_root=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero --steps=1000000 --batch_size=16 --num_workers=8 --stride=4 --lr=3e-5 --scheduler=cosine --lr_peak_scale=0.5 --lr_warmup_steps=20000 --lr_warmup_init_ratio=0.02 --lr_min_ratio=0.01 --weight_decay=1e-2 --grad_clip=0.5 --log_freq=1000 --save_freq=5000 --output_dir=outputs/stageb_org_600k 2>&1 | tee outputs/stageb_org_600k.log
```

## 3) 代码设置原理

`scripts/train_tokenizer_libero.py` 中，三种状态本质由两个条件决定：

1. 是否开启 `--align`
2. 是否提供 `--tokenizer_ckpt`

对应关系：

- baseline: `align=False`，loss 只包含 `recon + vq`
- pretrained_align: `align=True` 且 `tokenizer_ckpt` 非空，loss 为 `recon + vq + align`
- raw_align: `align=True` 且 `tokenizer_ckpt` 为空，loss 为 `recon + vq + align`，但初始权重是随机

align 分支使用冻结文本编码器，计算双向 InfoNCE；总损失为：

$$
L_{total} = L_{recon} + L_{vq} + w(t) \cdot L_{align,raw}
$$

其中 $w(t)$ 是 warmup 权重，前期逐步增大，减少训练初期不稳定。

学习率默认采用 warmup + cosine：

1. 先从较小比例升到峰值学习率（`lr * lr_peak_scale`）
2. 再按 cosine 衰减到 `lr_min_ratio` 对应的下限

## 4) 是否需要保留 experiments 下启动脚本

这个目录下的 `run_tokenizer_align_short.sh` 和 `run_tokenizer_baseline_only.sh` 是纯启动包装脚本，不包含独立算法逻辑。
为避免误解三种状态由脚本名决定（而不是参数开关决定），建议直接使用上面的 Python 命令。

因此这里默认删除两个脚本，只保留本 README 作为权威入口。
