# Stage-B: tokenizer 训练快速上手（train_tokenizer_libero.py）

本说明仅覆盖 tokenizer 训练，不包含 policy 训练。

---

## 1) 先做什么（最短路径）

如果你只想最快开始，请按这三步：

1. 先跑 2k smoke，确认 loss 正常下降。
2. 再跑长训（org 或 align 版本）。
3. 从 checkpoints/last 或 checkpoints/xxxxxx 中挑选 tokenizer.pt 做 policy 测试。

---

## 2) 环境准备

```bash
source /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/bin/activate
conda activate mint
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/lerobot_policy_mint

export DATASET_ROOT=/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero
export CKPT=/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth
export ALIGN_MODEL=BAAI/bge-large-en-v1.5

# 建议减少 tokenizers fork 警告
export TOKENIZERS_PARALLELISM=false
```

---

## 3) 当前目标函数（已同步代码）

当前 tokenizer 总损失为：

$$
L_{total} = L_{freq} + \lambda_{vq} L_{vq} + \alpha L_{aux} + L_{align}
$$

其中：

1. $L_{freq}$：scale-wise DCT 频域重建损失（论文 SDAT 主项）。
2. $L_{vq}$：量化损失（codebook + commitment）。
3. $L_{aux}$：时间域辅助重建（默认 L1，含 gripper 兼容分支）。
4. $L_{align}$：可选双向 InfoNCE（冻结文本编码器）。

频域项默认按尺度权重和做归一化：

$$
L_{freq} = w_{spec} \cdot \frac{\sum_s w_s \, \text{MSE}(\mathrm{DCT}(\hat{x}_s), \mathrm{DCT}(x))}{\sum_s w_s}
$$

这样不会因为尺度数量变化而放大总梯度，和论文“分尺度频域监督”的描述一致。

日志里你会看到：loss / freq_loss / aux_l1_loss / vq_loss / vq_loss_raw / align_loss / align_loss_raw。

---

## 4) 新增关键参数（和旧版差异）

以下参数是新版 loss 对应新增项：

1. `--spectral_weight`：频域损失总权重。
2. `--spectral_scale_weights`：各尺度权重，逗号分隔，长度需等于 scales 数。
3. `--aux_l1_weight`：时间域辅助项权重。
4. `--vq_weight`：$L_{vq}$ 项系数（即 $\lambda_{vq}$）。
5. `--vq_beta`：VQ commitment 系数（量化器内部参数）。
6. `--disable_spectral_scale_normalization`：关闭默认的按尺度权重和归一化。
7. `--include_gripper_in_spectral`：是否把最后一维（常为 gripper）纳入频域监督。

推荐起步：

1. `--spectral_weight=1.0`
2. `--spectral_scale_weights=1,1,1`
3. `--aux_l1_weight=1.0`
4. `--vq_weight=1.0`
5. `--vq_beta=0.25`
6. 不加 `--disable_spectral_scale_normalization`（默认开启归一化）
7. 不加 `--include_gripper_in_spectral`（默认排除最后一维）

---

## 5) 四种训练模式

### A. baseline（有预训练初始化，无文本对齐）

适用：先看稳定收敛，不引入文本对齐变量。

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
	--spectral_weight=1.0 \
	--spectral_scale_weights=1,1,1 \
	--aux_l1_weight=1.0 \
	--vq_weight=1.0 \
	--vq_beta=0.25 \
	--log_freq=20 \
	--save_freq=200 \
	--tokenizer_ckpt="/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth" \
	--output_dir=outputs/stageb_baseline_2k
```

### B. pretrained_align（有预训练初始化，有文本对齐）

适用：你们当前主推版本（论文损失 + InfoNCE）。

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=2000000 \
	--batch_size=16 \
	--num_workers=8 \
	--stride=4 \
	--lr=3e-5 \
	--scheduler=cosine \
	--lr_peak_scale=0.3 \
	--lr_warmup_steps=50000 \
	--lr_warmup_init_ratio=0.02 \
	--lr_min_ratio=0.02 \
	--weight_decay=1e-2 \
	--grad_clip=0.5 \
	--spectral_weight=1.0 \
	--spectral_scale_weights=1,1,1 \
	--aux_l1_weight=1.0 \
	--vq_weight=1.0 \
	--vq_beta=0.25 \
	--align \
	--align_model="BAAI/bge-large-en-v1.5" \
	--align_weight=0.05 \
	--log_freq=200 \
	--save_freq=2000 \
	--tokenizer_ckpt="/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/MINT/checkpoints/MINT-tokenizer-libero/ms_vqvae.pth" \
	--output_dir=outputs/stageb_align_pretrained_2M \
	2>&1 | tee outputs/stageb_align_pretrained_2M.log
```

### C. raw_align（随机初始化，有文本对齐）

适用：做对照实验，观察 align 对随机初始化的作用。

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=1000000 \
	--batch_size=16 \
	--num_workers=8 \
	--stride=4 \
	--lr=3e-5 \
	--scheduler=cosine \
	--lr_peak_scale=0.3 \
	--lr_warmup_steps=50000 \
	--lr_warmup_init_ratio=0.02 \
	--lr_min_ratio=0.02 \
	--weight_decay=1e-2 \
	--grad_clip=0.5 \
	--spectral_weight=1.0 \
	--spectral_scale_weights=1,1,1 \
	--aux_l1_weight=1.0 \
	--vq_weight=1.0 \
	--vq_beta=0.25 \
	--align \
	--align_model="BAAI/bge-large-en-v1.5" \
	--align_weight=0.05 \
	--log_freq=1000 \
	--save_freq=2000 \
	--output_dir=outputs/stageb_align_rawtrained_1M_true \
	2>&1 | tee outputs/stageb_align_rawtrained_1M_true.log
```

### D. org（随机初始化，无文本对齐）

适用：纯 SDAT 路径对照（不加 InfoNCE）。

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=1000000 \
	--batch_size=16 \
	--num_workers=8 \
	--stride=4 \
	--lr=3e-5 \
	--scheduler=cosine \
	--lr_peak_scale=0.3 \
	--lr_warmup_steps=5000 \
	--lr_warmup_init_ratio=0.02 \
	--lr_min_ratio=0.02 \
	--weight_decay=1e-2 \
	--grad_clip=0.5 \
	--spectral_weight=0.2 \
	--spectral_scale_weights=1,1,1 \
	--aux_l1_weight=1.0 \
	--vq_weight=1.0 \
	--vq_beta=0.25 \
	--log_freq=1000 \
	--save_freq=2000 \
	--output_dir=outputs/stageb_org_1M_true \
	2>&1 | tee outputs/stageb_org_1M_true.log
```

---

## 6) 我到底该先跑哪条命令

建议顺序：

1. 新环境先跑 A（2k）检查 loss 曲线。
2. 如果你现在要直接挂大训练：优先跑 D（1M，纯 tokenizer），更容易解释收敛行为。
3. 稳定后再跑 B（2M，主版本）或 C（1M，带 align 对照）。

---

## 7) 结果文件如何使用

训练完成后，使用：

1. `outputs/<run_name>/checkpoints/last/tokenizer.pt`：最新 checkpoint（符号链接）。
2. `outputs/<run_name>/checkpoints/<step>/tokenizer.pt`：固定 step checkpoint（例如 020000）。
3. `outputs/<run_name>/train_report.json`：查看最终 loss 与超参快照。

常用抽测方式：

1. 每隔固定步数保存：设置 `--save_freq=2000`（会得到 002000、004000、...）。
2. 先挑一个中段和一个后段 checkpoint 做 policy 测试，再决定是否继续跑满。
3. policy 训练时把 `--policy.vqvae_name_or_path` 指向你要测试的 `tokenizer.pt`。

---

## 8) 备注

本目录下旧的 shell 包装脚本不是算法逻辑入口，推荐统一以本 README 的 Python 命令为准。

---

## 9) 1M 挂跑模板（可直接复制）

下面两条命令都可以直接跑，区别是是否使用文本对齐。

### 9.1 org 1M（推荐先跑）

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=1000000 \
	--batch_size=16 \
	--num_workers=8 \
	--stride=4 \
	--lr=3e-5 \
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
	--vq_weight=1.2 \
	--vq_beta=0.35 \
	--log_freq=200 \
	--save_freq=2000 \
	--output_dir=outputs/stageb_org_1M_true \
	2>&1 | tee outputs/stageb_org_1M_true.log
```

### 9.2 raw_align 1M（对照实验）

```bash
PYTHONPATH=src python scripts/train_tokenizer_libero.py \
	--dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" \
	--steps=1000000 \
	--batch_size=16 \
	--num_workers=8 \
	--stride=4 \
	--lr=3e-5 \
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
	--vq_weight=1.2 \
	--vq_beta=0.35 \
	--align \
	--align_model="BAAI/bge-large-en-v1.5" \
	--align_weight=0.05 \
	--log_freq=500 \
	--save_freq=2000 \
	--output_dir=outputs/stageb_raw_align_1M_run_true \
	2>&1 | tee outputs/stageb_raw_align_1M_run_true.log
```

### 9.3 挂后台运行（可选）

tmux 方式：

```bash
tmux new -s tokenizer_1m
# 在新会话里粘贴上面的训练命令后回车
# 退出会话不停止训练：Ctrl-b 再按 d
```

nohup 方式：

```bash
nohup bash -lc 'PYTHONPATH=src python scripts/train_tokenizer_libero.py --dataset_root="/inspire/hdd/project/robot-decision/public/datasets/HuggingFaceVLA_cus/libero" --steps=1000000 --batch_size=16 --num_workers=8 --stride=4 --lr=3e-5 --scheduler=cosine --lr_peak_scale=0.3 --lr_warmup_steps=20000 --lr_warmup_init_ratio=0.02 --lr_min_ratio=0.02 --weight_decay=1e-2 --grad_clip=0.5 --spectral_weight=0.15 --spectral_scale_weights=1,0.7,0.4 --aux_l1_weight=1.0 --vq_weight=1.2 --vq_beta=0.35 --log_freq=200 --save_freq=2000 --output_dir=outputs/stageb_org_1M_nohup' > outputs/stageb_org_1M_nohup.log 2>&1 &