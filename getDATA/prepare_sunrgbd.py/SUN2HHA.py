import os
import cv2
import numpy as np
import scipy.io as sio
import multiprocessing as mp
from tqdm import tqdm

# ==========================================
# ⚙️ 全局路径配置
# ==========================================
RAW_ROOT = '/home/pengfei/HTCnet/DataSets/SUNRGBD_Raw'
OUT_ROOT = '/home/pengfei/HTCnet/DataSets/SUNRGBD_Processed'

TOOLBOX_DIR = os.path.join(RAW_ROOT, 'SUNRGBDtoolbox')
META_FILE = os.path.join(TOOLBOX_DIR, 'Metadata/SUNRGBDMeta.mat')
SPLIT_FILE = os.path.join(TOOLBOX_DIR, 'traintestSUNRGBD/allsplit.mat')

# ==========================================
# 🛠️ 核心 HHA 生成器
# ==========================================
def generate_hha(depth, K, R_tilt):
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    points_3d = np.dstack((X, Y, Z))

    # 1. 视差 (Disparity)
    disparity = np.zeros_like(depth)
    valid = depth > 0
    disparity[valid] = 1.0 / depth[valid]
    disparity = np.clip(disparity / np.max(disparity) * 255.0, 0, 255).astype(np.uint8)

    # 2. 高度 (Height)
    points_3d_flat = points_3d.reshape(-1, 3)
    points_world = np.dot(points_3d_flat, R_tilt.T) 
    height = points_world[:, 2].reshape(h, w)
    height = np.clip((height + 2.0) / 4.0 * 255.0, 0, 255).astype(np.uint8)

    # 3. 角度 (Angle)
    normal_estimator = cv2.rgbd.RgbdNormals_create(h, w, cv2.CV_32F, K)
    normals = normal_estimator.apply(points_3d.astype(np.float32))
    
    gravity_vec = np.array([0, 0, 1])
    normals_flat = normals.reshape(-1, 3)
    dot_prod = np.dot(normals_flat, gravity_vec)
    angle = np.arccos(np.clip(dot_prod, -1.0, 1.0)).reshape(h, w)
    angle = np.clip(angle / (np.pi / 2) * 255.0, 0, 255).astype(np.uint8)

    return np.dstack((disparity, height, angle))

# ==========================================
# 📦 数据解析与单图处理
# ==========================================
def process_single_image(args):
    idx, rgb_rel, depth_rel, K, R_tilt, out_dirs = args
    
    # 构建在你服务器上的绝对路径
    rgb_path = os.path.join(RAW_ROOT, rgb_rel)
    depth_path = os.path.join(RAW_ROOT, depth_rel)
    
    if not os.path.exists(rgb_path):
        return None
    if not os.path.exists(depth_path):
        return None

    # 读取图像
    rgb = cv2.imread(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if depth_raw is None:
        return None
        
    depth_m = depth_raw.astype(np.float32) / 10000.0
    hha = generate_hha(depth_m, K, R_tilt)
    
    out_name = f"{idx:05d}.png"
    cv2.imwrite(os.path.join(out_dirs['rgb'], out_name), rgb)
    cv2.imwrite(os.path.join(out_dirs['hha'], out_name), hha)
    
    # 提取目录特征用于匹配划分 (例如: SUNRGBD/kv1/b/bedroom/sun_abc)
    folder_id = "/".join(rgb_rel.split('/')[:-2]) 
    return (out_name, folder_id)

# ==========================================
# 🚀 主控台
# ==========================================
def main():
    print("🚀 正在初始化 SUN RGB-D 数据处理引擎...")
    
    out_dirs = {
        'rgb': os.path.join(OUT_ROOT, 'RGB'),
        'hha': os.path.join(OUT_ROOT, 'HHA')
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"📦 正在读取 SUNRGBDMeta.mat ...")
    meta_data = sio.loadmat(META_FILE)
    sun_meta = meta_data['SUNRGBDMeta'][0]
    num_images = len(sun_meta)
    print(f"✅ 成功检测到 {num_images} 张图像！")

    tasks = []
    print("🔍 正在清洗路径并打包内参...")
    for i in range(num_images):
        # 清洗可能存在的括号和引号
        rgb_p = str(sun_meta[i]['rgbpath'][0]).strip("[]'")
        depth_p = str(sun_meta[i]['depthpath'][0]).strip("[]'")
        
        # 🚀 核心修复：丢掉普林斯顿的绝对路径，只保留 SUNRGBD/...
        if "SUNRGBD/" in rgb_p:
            rgb_p = rgb_p[rgb_p.find("SUNRGBD/"):]
        if "SUNRGBD/" in depth_p:
            depth_p = depth_p[depth_p.find("SUNRGBD/"):]
            
        K = sun_meta[i]['K']
        R_tilt = sun_meta[i]['Rtilt']
        
        tasks.append((i + 1, rgb_p, depth_p, K, R_tilt, out_dirs))

    print(f"🔥 启动多进程并发生成 HHA (这次是真的火力全开了！)...")
    results = []
    with mp.Pool(processes=max(1, mp.cpu_count() - 2)) as pool:
        for res in tqdm(pool.imap(process_single_image, tasks), total=len(tasks)):
            if res:
                results.append(res)

    print(f"✅ 成功生成 {len(results)} 张 HHA 图像！")

    # ==========================================
    # 🎯 提取官方 Train/Test 划分
    # ==========================================
    print("📊 正在比对官方 Train/Test 黄金划分...")
    split_data = sio.loadmat(SPLIT_FILE)
    
    # 清洗划分路径
    train_raw_paths = [str(p[0][0]).strip("[]'") for p in split_data['alltrain']]
    test_raw_paths = [str(p[0][0]).strip("[]'") for p in split_data['alltest']]
    
    train_list, test_list = [], []
    
    for out_name, folder_id in results:
        is_train = False
        # 只要官方路径包含在我们的实际文件夹路径中就算匹配成功
        for tp in train_raw_paths:
            if tp in folder_id:
                train_list.append(out_name)
                is_train = True
                break
                
        if not is_train:
            for tep in test_raw_paths:
                if tep in folder_id:
                    test_list.append(out_name)
                    break

    with open(os.path.join(OUT_ROOT, 'train.txt'), 'w') as f_train:
        f_train.write('\n'.join(sorted(train_list)))
    with open(os.path.join(OUT_ROOT, 'test.txt'), 'w') as f_test:
        f_test.write('\n'.join(sorted(test_list)))

    print(f"🎉 完美收工！")
    print(f"📈 最终匹配训练集: {len(train_list)} 张")
    print(f"📉 最终匹配测试集: {len(test_list)} 张")

if __name__ == '__main__':
    main()