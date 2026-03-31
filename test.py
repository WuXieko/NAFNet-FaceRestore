# test.py
# ============================================================
# 推理脚本 - 测试人脸复原效果
# ============================================================
# 用法：
#   1. 单张图片：python test.py --input my_blurry_face.jpg
#   2. 整个文件夹：python test.py --input test_images/
#   3. 指定模型：python test.py --input xxx.jpg --ckpt checkpoints/epoch_30.pth
# ============================================================

import os
import sys
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.nafnet import build_model


def load_model(ckpt_path, width=32, device='cuda'):
    """加载训练好的模型"""
    model = build_model(width=width).to(device)

    # 兼容两种保存格式
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # train.py 保存的完整 checkpoint
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', '?')
        loss = checkpoint.get('loss', '?')
        print(f"加载模型: {ckpt_path} (epoch={epoch}, loss={loss})")
    else:
        # 直接保存的 state_dict（best.pth）
        state_dict = checkpoint
        print(f"加载模型: {ckpt_path}")

    # 处理 torch.compile 保存的 key（去掉 _orig_mod. 前缀）
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v

    model.load_state_dict(cleaned)
    model.eval()
    return model


def restore_image(model, img_path, output_dir, device='cuda'):
    """
    对单张图片进行复原

    流程：读图 → tensor → 送入模型 → 保存结果
    同时生成一张对比图（左：原图，右：复原图）
    """
    # 读取图片
    img = Image.open(img_path).convert('RGB')
    original_size = img.size  # (W, H)
    print(f"  处理: {os.path.basename(img_path)} ({original_size[0]}x{original_size[1]})")

    # PIL → Tensor [0, 1]
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # 推理
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model(x)

    # Tensor → PIL
    output = output.squeeze(0).float().clamp(0, 1).cpu()
    result_img = transforms.ToPILImage()(output)

    # 确保输出和输入同尺寸（模型可能因为 padding 改变了大小）
    if result_img.size != original_size:
        result_img = result_img.resize(original_size, Image.BICUBIC)

    # 保存复原图
    basename = os.path.splitext(os.path.basename(img_path))[0]
    result_path = os.path.join(output_dir, f"{basename}_restored.png")
    result_img.save(result_path)

    # 生成对比图（左原图，右复原图）
    compare = Image.new('RGB', (original_size[0] * 2, original_size[1]))
    compare.paste(img, (0, 0))
    compare.paste(result_img, (original_size[0], 0))
    compare_path = os.path.join(output_dir, f"{basename}_compare.png")
    compare.save(compare_path)

    print(f"    → 复原图: {result_path}")
    print(f"    → 对比图: {compare_path}")

    return result_path


def main():
    parser = argparse.ArgumentParser(description="人脸复原推理")
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

    # 加载模型
    model = load_model(args.ckpt, width=args.width, device=device)

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

    # 逐张处理
    for path in image_paths:
        try:
            restore_image(model, path, args.output, device)
        except Exception as e:
            print(f"  ❌ 处理失败 {path}: {e}")

    print(f"\n✅ 全部完成！结果保存在: {args.output}/")


if __name__ == "__main__":
    main()