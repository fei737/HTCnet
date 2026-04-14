import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models import PFNet


def predict_and_show(data_dir, checkpoint_path, idx="0700", n_classes=14):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PFNet(n_classes=n_classes, pretrained_path=None, return_aux=False).to(device)

    print(f"Loading weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    rgb_raw = cv2.cvtColor(cv2.imread(f"{data_dir}/NYURGB/{idx}.png"), cv2.COLOR_BGR2RGB)
    hha_raw = cv2.cvtColor(cv2.imread(f"{data_dir}/HHA_datasets/hha_{idx}.png"), cv2.COLOR_BGR2RGB)

    img_size = (320, 320)
    rgb_resize = cv2.resize(rgb_raw, img_size)
    hha_resize = cv2.resize(hha_raw, img_size)

    rgb_t = torch.from_numpy(rgb_resize).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    hha_t = torch.from_numpy(hha_resize).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = model(rgb_t, hha_t)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_raw)
    plt.title("Original RGB")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    a_channel = hha_resize[:, :, 2]
    grad_x = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(a_channel, cv2.CV_64F, 0, 1, ksize=3)
    grad_map = np.sqrt(grad_x ** 2 + grad_y ** 2)
    plt.imshow(grad_map, cmap='magma')
    plt.title("Geometric Guidance (Gradient)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='nipy_spectral', vmin=0, vmax=n_classes - 1)
    plt.title("Prediction (Semantic Seg)")
    plt.axis('off')

    plt.tight_layout()
    save_path = f"result_{idx}.png"
    plt.savefig(save_path, dpi=200)
    print(f"Result saved as {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--idx', type=str, default='0700')
    parser.add_argument('--n-classes', type=int, default=14)
    args = parser.parse_args()

    predict_and_show(args.data_dir, args.ckpt, args.idx, args.n_classes)
