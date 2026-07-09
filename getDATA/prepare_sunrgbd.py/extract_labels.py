import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm

# ==========================================
# ⚙️ 路径配置 (完全对齐你的服务器架构)
# ==========================================
MAT_FILE = '/home/pengfei/HTCnet/DataSets/SUNRGBD_Raw/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat'
OUT_DIR = '/home/pengfei/HTCnet/DataSets/SUNRGBD_Processed/Labels'

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. 检查文件是否存在
    if not os.path.exists(MAT_FILE):
        print(f"❌ 找不到官方标签文件: {MAT_FILE}")
        print("\n💡 不要慌！官方把这个文件单独拆分了。请在终端执行以下两条命令下载它：")
        print("cd /home/pengfei/HTCnet/DataSets/SUNRGBD_Raw/SUNRGBDtoolbox/Metadata/")
        print("wget http://rgbd.cs.princeton.edu/data/SUNRGBD2Dseg.mat")
        print("\n⏳ 下载完成后（大约 400MB），再次运行本脚本即可！")
        return

    # 2. 读取 v7.3 格式的超大 .mat 文件
    print("📦 正在加载 SUNRGBD2Dseg.mat (文件较大，可能需要 10-20 秒)...")
    f = h5py.File(MAT_FILE, 'r')
    
    # 定位到 seglabel 数据块
    seg_refs = f['SUNRGBD2Dseg']['seglabel']
    num_images = len(seg_refs)
    print(f"✅ 成功检测到 {num_images} 张高精度 Mask 掩码图！")

    print("🔥 正在疯狂提取并生成 37 类单通道 PNG...")
    for i in tqdm(range(num_images)):
        # 解析 HDF5 对象引用
        ref = seg_refs[i][0]
        label_matrix = np.array(f[ref])
        
        # ⚠️ 极其核心：MATLAB 的矩阵是“列优先”存储的，读到 Python 里必须转置！
        label = label_matrix.T.astype(np.uint8)
        
        # 保存为 00001.png 这种格式
        out_name = f"{i + 1:05d}.png"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), label)
        
    f.close()
    print(f"\n🎉 完美收工！所有的 Labels 已经整齐地保存在了: {OUT_DIR}")
    print("===============================================================")
    print("🚀 弹药库彻底装填完毕！现在你可以直接去执行:")
    print("CUDA_VISIBLE_DEVICES=2,3 python train.py")

if __name__ == '__main__':
    main()