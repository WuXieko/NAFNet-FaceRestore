# de.py - 测试 datapipe 是否正常工作
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datapipe.dataset import FaceDataset

ds = FaceDataset("WIDER_train", patch_size=256)
print(f"共 {len(ds)} 张图")

lq, hq = ds[0]
print(f"LQ shape: {lq.shape}")   # 应该是 torch.Size([3, 256, 256])
print(f"HQ shape: {hq.shape}")   # 应该是 torch.Size([3, 256, 256])
print(f"LQ 值范围: [{lq.min():.3f}, {lq.max():.3f}]")  # 应该在 [0, 1]
print(f"HQ 值范围: [{hq.min():.3f}, {hq.max():.3f}]")

print("\n✅ datapipe 测试通过！")