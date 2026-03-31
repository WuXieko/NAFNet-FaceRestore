# test.py
# ============================================================
# 推理脚本 - 三图对比（原图 / 退化图 / 复原图）
# ============================================================
# 用法：
#   python test.py --input my_face.jpg
#   python test.py --input test_images/
#   python test.py --input my_face.jpg --ckpt checkpoints/epoch_30.pth
# ============================================================

import os
import sys
import glob
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.nafnet import build_model
from datapipe.degradation import BlindDegradation


def calculate_psnr(img1, img2):
    """
    计算两张 PIL Image 之间的 PSNR (Peak Signal-to-Noise Ratio)
    PSNR 越高越好，> 28dB 算不错，> 30dB 算好
    """
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def load_model(ckpt_path, width=32, device='cuda'):
    """加载训练好的模型"""
    model = build_model(width=width).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', '?')
        loss = checkpoint.get('loss', '?')
        print(f"加载模型: {ckpt_path} (epoch={epoch}, loss={loss})")
    else:
        state_dict = checkpoint
        print(f"加载模型: {ckpt_path}")

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v

    model.load_state_dict(cleaned)
    model.eval()
    return model


def add_label(img, text):
    """在图片底部添加文字标签"""
    w, h = img.size
    label_h = 30
    labeled = Image.new('RGB', (w, h + label_h), (0, 0, 0))
    labeled.paste(img, (0, 0))
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, h + 5), text, fill=(255, 255, 255), font=font)
    return labeled


def restore_image(model, degradation, img_path, output_dir, device='cuda'):
    """三图对比：原图 → 退化图 → 复原图"""
    # 读取原图
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    print(f"  处理: {os.path.basename(img_path)} ({original_size[0]}x{original_size[1]})")

    # 生成退化图
    degraded_img = degradation(img)

    # 退化图 → Tensor → 模型推理
    to_tensor = transforms.ToTensor()
    x = to_tensor(degraded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(x)

    output = output.squeeze(0).float().clamp(0, 1).cpu()
    restored_img = transforms.ToPILImage()(output)

    if restored_img.size != original_size:
        restored_img = restored_img.resize(original_size, Image.BICUBIC)

    # 保存复原图
    basename = os.path.splitext(os.path.basename(img_path))[0]
    result_path = os.path.join(output_dir, f"{basename}_restored.png")
    restored_img.save(result_path)

    # 三图对比（原图 | 退化图 | 复原图）
    img_labeled = add_label(img, "Original")
    deg_labeled = add_label(degraded_img, "Degraded")
    res_labeled = add_label(restored_img, "Restored")

    w, h_labeled = img_labeled.size
    compare = Image.new('RGB', (w * 3 + 4, h_labeled), (40, 40, 40))
    compare.paste(img_labeled, (0, 0))
    compare.paste(deg_labeled, (w + 2, 0))
    compare.paste(res_labeled, (w * 2 + 4, 0))

    compare_path = os.path.join(output_dir, f"{basename}_compare.png")
    compare.save(compare_path)

    # ---------- 5. 计算 PSNR ----------
    # 退化图 vs 原图（退化有多严重）
    psnr_degraded = calculate_psnr(img, degraded_img)
    # 复原图 vs 原图（修复了多少）
    psnr_restored = calculate_psnr(img, restored_img)
    # 提升量
    psnr_gain = psnr_restored - psnr_degraded

    print(f"    → 对比图: {compare_path}")
    print(f"    → PSNR  退化: {psnr_degraded:.2f} dB | "
          f"复原: {psnr_restored:.2f} dB | "
          f"提升: {psnr_gain:+.2f} dB")

    return psnr_degraded, psnr_restored


def main():
    parser = argparse.ArgumentParser(description="人脸复原推理 - 三图对比")
    parser.add_argument("--input", type=str, required=True,
                        help="输入图片路径或文件夹路径")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth",
                        help="模型 checkpoint 路径")
    parser.add_argument("--output", type=str, default="results",
                        help="输出目录")
    parser.add_argument("--width", type=int, default=32,
                        help="NAFNet 通道数（必须和训练时一致）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    model = load_model(args.ckpt, width=args.width, device=device)
    degradation = BlindDegradation()

    # 收集输入图片
    if os.path.isdir(args.input):
        extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        image_paths.sort()
    else:
        image_paths = [args.input]

    print(f"\n共 {len(image_paths)} 张图片待处理\n")

    psnr_deg_list = []
    psnr_res_list = []

    for path in image_paths:
        try:
            psnr_deg, psnr_res = restore_image(model, degradation, path, args.output, device)
            psnr_deg_list.append(psnr_deg)
            psnr_res_list.append(psnr_res)
        except Exception as e:
            print(f"  ❌ 处理失败 {path}: {e}")

    # ---------- 汇总统计 ----------
    print(f"\n{'='*60}")
    print(f"✅ 全部完成！共处理 {len(psnr_res_list)} 张图片")
    print(f"结果保存在: {args.output}/")

    if psnr_res_list:
        avg_deg = np.mean(psnr_deg_list)
        avg_res = np.mean(psnr_res_list)
        avg_gain = avg_res - avg_deg
        print(f"\n📊 PSNR 汇总:")
        print(f"   平均退化 PSNR:  {avg_deg:.2f} dB")
        print(f"   平均复原 PSNR:  {avg_res:.2f} dB")
        print(f"   平均提升:       {avg_gain:+.2f} dB")
        print(f"\n   参考基准: > 28dB 良好 | > 30dB 优秀")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()