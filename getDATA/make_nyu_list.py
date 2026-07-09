import os
import urllib.request
import scipy.io as sio

def main():
    # ================= 路径配置 =================
    # 指向你生成好的干净数据集 (RGB, Depth, Label 的根目录)
    nyu_root = '/home/pengfei/HTCnet/DataSets/NYU' 
    
    # 准备存放 splits.mat 的路径
    splits_mat_path = '/home/pengfei/HTCnet/DataSets/NYU/splits.mat' 
    # ============================================

    # 1. 检查并自动下载 splits.mat (使用 GitHub 上高赞仓库的稳定镜像)
    if not os.path.exists(splits_mat_path):
        print(f"未检测到 splits.mat，正在从可靠镜像自动下载...")
        os.makedirs(os.path.dirname(splits_mat_path), exist_ok=True)
        # 这是一个被学术界广泛使用的 NYUv2 metadata 备份仓库
        mirror_url = "https://raw.githubusercontent.com/ankurhanda/nyuv2-meta-data/master/splits.mat"
        try:
            urllib.request.urlretrieve(mirror_url, splits_mat_path)
            print("✅ splits.mat 下载成功！")
        except Exception as e:
            print(f"❌ 下载失败，请检查网络是否能访问 Github: {e}")
            return

    # 2. 读取官方 .mat 划分文件
    print("正在读取官方划分数据...")
    splits = sio.loadmat(splits_mat_path)

    # 3. 提取索引并转换 (⚠️ 核心陷阱)
    # MATLAB 的索引从 1 开始 (1~1449)，Python 的索引从 0 开始 (0~1448)
    # 所以必须减去 1 才能和我们之前提取的 0.jpg 到 1448.jpg 完美对应！
    train_ndxs = splits['trainNdxs'].flatten() - 1
    test_ndxs = splits['testNdxs'].flatten() - 1

    print(f"官方严格划分: Train {len(train_ndxs)} 张, Test {len(test_ndxs)} 张.")

    # 4. 写入 train.txt
    train_txt_path = os.path.join(nyu_root, 'train.txt')
    with open(train_txt_path, 'w') as f:
        for idx in train_ndxs:
            # 写入对应的文件名，如 0.jpg
            f.write(f"{idx}.jpg\n")
            
    # 5. 写入 test.txt
    test_txt_path = os.path.join(nyu_root, 'test.txt')
    with open(test_txt_path, 'w') as f:
        for idx in test_ndxs:
            f.write(f"{idx}.jpg\n")

    print(f"✅ 大功告成！符合顶会论文标准的 train.txt 和 test.txt 已保存在: {nyu_root}")

if __name__ == "__main__":
    main()