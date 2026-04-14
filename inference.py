import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from modules.models.model import PFNet


def predict_and_show(idx="0700"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '/home/pengfei/PFmodel/DataSets'

    # 1. 实例化当前的三分支模型
    model = PFNet(n_classes=14, pretrained_path=None).to(device)

    # 2. 加载权重 (使用 strict=False 避开新旧结构冲突)
    checkpoint_path = 'Checkpoint/best_model_0.412.pth'
    print(f"Loading weights from: {checkpoint_path}")

    # 重要：如果你还没用新代码训练完，加载旧权重必须加 strict=False
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. 读取图片
    rgb_raw = cv2.cvtColor(cv2.imread(f"{data_dir}/NYURGB/{idx}.png"), cv2.COLOR_BGR2RGB)
    hha_raw = cv2.cvtColor(cv2.imread(f"{data_dir}/HHA_datasets/hha_{idx}.png"), cv2.COLOR_BGR2RGB)

    # 4. 预处理 (确保与训练代码的 Transform 一致)
    img_size = (320, 320)
    rgb_resize = cv2.resize(rgb_raw, img_size)
    hha_resize = cv2.resize(hha_raw, img_size)

    # 转换为 Tensor 并在 0-1 之间标准化
    rgb_t = torch.from_numpy(rgb_resize).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    hha_t = torch.from_numpy(hha_resize).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = model(rgb_t, hha_t)
        # 5. 重点：检查输出分布
        # 如果全是一片颜色，可能是 argmax 拿到了全是 0 的背景类
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 6. 绘图优化
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_raw)
    plt.title("Original RGB")
    plt.axis('off')

    # 可视化 HHA 的 A 通道梯度，看看模型“几何眼”看到了什么
    plt.subplot(1, 3, 2)
    a_channel = hha_resize[:, :, 2]  # 假设 A 是第三通道
    grad_x = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(a_channel, cv2.CV_64F, 0, 1, ksize=3)
    grad_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
    plt.imshow(grad_map, cmap='magma')
    plt.title("Geometric Guidance (Gradient)")
    plt.axis('off')

    # 预测图：使用更有辨识度的 colormap
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='nipy_spectral', vmin=0, vmax=13)
    plt.title("Prediction (Semantic Seg)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"result_{idx}.png", dpi=200)
    print(f"Result saved as result_{idx}.png")


if __name__ == "__main__":
    predict_and_show("0700")
