# train.py
# ============================================================
# 训练主循环（正式版 - FFHQ）
# ============================================================
# 功能：
#   1. 断点续训（--resume 自动从最新 checkpoint 恢复）
#   2. 混合精度训练（bfloat16）
#   3. 梯度累积（显存不够时等效增大 batch）
#   4. TensorBoard 日志 + 对比图
#   5. 每 5 epoch 保存 checkpoint + 最优模型
#
# 用法：
#   首次训练：  python train.py
#   断点续训：  python train.py --resume
#   指定恢复：  python train.py --resume --ckpt checkpoints/epoch_30.pth
# ============================================================

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.nafnet import build_model
from losses.losses import CombinedLoss
from datapipe.dataset import FaceDataset


# ============================================================
# 超参数
# ============================================================

# --- 数据 ---
DATA_DIR = "data/ffhq"
PATCH_SIZE = 256
NUM_WORKERS = 4                    # Windows 报错就改 0

# --- 训练 ---
BATCH_SIZE = 8
ACCUMULATE_STEPS = 1
LR = 2e-4
EPOCHS = 80
WIDTH = 32

# --- 日志与保存 ---
SAVE_EVERY = 5                     # 每 5 epoch 保存，最多丢 1.5 小时
LOG_EVERY = 50
CKPT_DIR = "checkpoints"
LOG_DIR = "runs/ffhq_exp1"


def find_latest_checkpoint(ckpt_dir):
    """自动找到最新的 checkpoint 文件"""
    if not os.path.exists(ckpt_dir):
        return None
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    if not ckpts:
        return None
    # 按 epoch 数排序，取最大的
    ckpts.sort(key=lambda x: int(x.replace("epoch_", "").replace(".pth", "")))
    return os.path.join(ckpt_dir, ckpts[-1])


def main():
    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 恢复训练")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="指定恢复的 checkpoint 路径")
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("人脸复原训练 - NAFNet（FFHQ 正式版）")
    print("=" * 60)
    print(f"设备: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Batch Size: {BATCH_SIZE} × {ACCUMULATE_STEPS} 累积 "
          f"= 等效 {BATCH_SIZE * ACCUMULATE_STEPS}")

    # ============================================================
    # 数据
    # ============================================================
    dataset = FaceDataset(DATA_DIR, patch_size=PATCH_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    print(f"数据集: {len(dataset)} 张 | "
          f"每 epoch {len(dataloader)} 个 batch")

    # ============================================================
    # 模型 + 损失 + 优化器
    # ============================================================
    model = build_model(width=WIDTH).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数量: {params_m:.2f}M")

    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler('cuda')

    # ============================================================
    # 断点续训
    # ============================================================
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if args.resume:
        ckpt_path = args.ckpt or find_latest_checkpoint(CKPT_DIR)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\n🔄 恢复训练: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

            # 加载模型（处理 torch.compile 的 key）
            state_dict = checkpoint['model_state_dict']
            cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('loss', float('inf'))
            global_step = start_epoch * len(dataloader)

            print(f"   从 epoch {start_epoch} 继续，"
                  f"已有 best_loss={best_loss:.4f}")
        else:
            print("⚠ 未找到 checkpoint，从头开始训练")

    # TensorBoard（append 模式，续训不会覆盖旧数据）
    writer = SummaryWriter(LOG_DIR)

    # 预估剩余时间
    remaining_epochs = EPOCHS - start_epoch
    est_batch_sec = 0.15
    est_total_min = len(dataloader) * remaining_epochs * est_batch_sec / 60
    print(f"\n剩余 {remaining_epochs} epoch，"
          f"预估 ~{est_total_min:.0f} 分钟 ({est_total_min/60:.1f} 小时)")
    print("=" * 60)

    # ============================================================
    # 训练循环
    # ============================================================
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (lq, hq) in enumerate(dataloader):
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(lq)
                loss, loss_dict = criterion(pred, hq)
                loss = loss / ACCUMULATE_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATE_STEPS == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * ACCUMULATE_STEPS
            global_step += 1

            # ---------- TensorBoard ----------
            if global_step % LOG_EVERY == 0:
                writer.add_scalar("Loss/total", loss.item() * ACCUMULATE_STEPS, global_step)
                writer.add_scalar("Loss/l1", loss_dict["l1"], global_step)
                writer.add_scalar("Loss/perceptual", loss_dict["perc"], global_step)
                writer.add_scalar("Loss/fft", loss_dict["fft"], global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)

            if global_step % 500 == 0:
                with torch.no_grad():
                    vis_lq = lq[:4].clamp(0, 1)
                    vis_pred = pred[:4].float().clamp(0, 1)
                    vis_hq = hq[:4].clamp(0, 1)
                    writer.add_images("Images/1_input_LQ", vis_lq, global_step)
                    writer.add_images("Images/2_output_pred", vis_pred, global_step)
                    writer.add_images("Images/3_target_HQ", vis_hq, global_step)

            # ---------- 打印进度 ----------
            if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_start
                batches_done = batch_idx + 1
                batches_left = len(dataloader) - batches_done
                eta_sec = (elapsed / batches_done) * batches_left
                eta_min = eta_sec / 60

                print(f"  Epoch [{epoch+1}/{EPOCHS}] "
                      f"[{batches_done}/{len(dataloader)}] "
                      f"Loss: {loss.item()*ACCUMULATE_STEPS:.4f} "
                      f"(l1={loss_dict['l1']:.4f} "
                      f"perc={loss_dict['perc']:.4f} "
                      f"fft={loss_dict['fft']:.4f}) "
                      f"ETA: {eta_min:.1f}min")

        # ---------- Epoch 结束 ----------
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - epoch_start

        remaining = EPOCHS - epoch - 1
        est_remaining = elapsed * remaining / 60

        print(f"\n>>> Epoch {epoch+1}/{EPOCHS} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"用时: {elapsed:.0f}s | "
              f"剩余: ~{est_remaining:.0f}min\n")

        writer.add_scalar("Epoch/avg_loss", avg_loss, epoch)

        # ---------- 保存 checkpoint ----------
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"   💾 已保存: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CKPT_DIR, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"   ⭐ 最优模型更新: {best_path} (loss={best_loss:.4f})")

    writer.close()
    print("\n" + "=" * 60)
    print(f"训练完成！最优 Loss: {best_loss:.4f}")
    print(f"查看训练曲线: tensorboard --logdir runs/")
    print("=" * 60)


if __name__ == "__main__":
    main()