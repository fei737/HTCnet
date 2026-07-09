import os
import scipy.io as sio
import numpy as np

# 你的路径配置
RAW_ROOT = '/home/pengfei/HTCnet/DataSets/SUNRGBD_Raw'
OUT_ROOT = '/home/pengfei/HTCnet/DataSets/SUNRGBD_Processed'
TOOLBOX_DIR = os.path.join(RAW_ROOT, 'SUNRGBDtoolbox')
META_FILE = os.path.join(TOOLBOX_DIR, 'Metadata/SUNRGBDMeta.mat')
SPLIT_FILE = os.path.join(TOOLBOX_DIR, 'traintestSUNRGBD/allsplit.mat')

def get_clean_string(obj):
    """递归脱去 Numpy 外套"""
    while isinstance(obj, (np.ndarray, list)):
        if len(obj) > 0:
            obj = obj[0]
        else:
            return ""
    return str(obj).strip().replace('\\', '/').rstrip('/')

def get_core_id(path_str):
    """
    终极清洗器：把乱七八糟的前缀和后缀全部剁掉，只保留核心指纹
    例如：/n/fs/sun3d/data/SUNRGBD/kv1/b/bedroom/sun_abc/image/001.jpg
    变成：SUNRGBD/kv1/b/bedroom/sun_abc
    """
    p = get_clean_string(path_str)
    
    # 1. 统一截取 SUNRGBD/ 后面的核心路径
    if 'SUNRGBD/' in p:
        core = p[p.find('SUNRGBD/'):]
    else:
        # 极少数连 SUNRGBD 都没有写的奇葩路径，直接取最后4级目录
        parts = [x for x in p.split('/') if x and x not in ['image', 'depth']]
        core = "/".join(parts[-4:])
        
    # 2. 丢掉 /image 及其后面的内容
    core = core.split('/image')[0]
    # 3. 丢掉 /depth 及其后面的内容 (以防万一)
    core = core.split('/depth')[0]
    
    return core.strip('/')

def main():
    print("📦 正在启动终极数据挖掘机 (核心指纹精确对齐)...")
    sun_meta = sio.loadmat(META_FILE)['SUNRGBDMeta'][0]
    split_data = sio.loadmat(SPLIT_FILE)
    
    # 🚀 使用 Set 哈希表，提取所有黄金划分的“核心指纹”
    train_set = set([get_core_id(p) for p in split_data['alltrain'].flatten()])
    test_set = set([get_core_id(p) for p in split_data['alltest'].flatten()])
    
    train_list, test_list = [], []
    missed_examples = []
    
    print("🔍 正在为您进行 O(1) 极速匹配...")
    for i in range(len(sun_meta)):
        out_name = f"{i + 1:05d}.png"  
        
        # 提取当前图像的“核心指纹”
        rgb_p = get_clean_string(sun_meta[i]['rgbpath'])
        core_id = get_core_id(rgb_p)
        
        # 绝对精准匹配
        if core_id in train_set:
            train_list.append(out_name)
        elif core_id in test_set:
            test_list.append(out_name)
        else:
            if len(missed_examples) < 3:
                missed_examples.append((rgb_p, core_id))
                    
    # 写入最终的 txt 文件
    with open(os.path.join(OUT_ROOT, 'train.txt'), 'w') as f_train:
        f_train.write('\n'.join(sorted(train_list)))
    with open(os.path.join(OUT_ROOT, 'test.txt'), 'w') as f_test:
        f_test.write('\n'.join(sorted(test_list)))

    print(f"\n🎉 匹配完成！")
    print(f"📈 最终匹配训练集: {len(train_list)} 张 (官方黄金标准: 5285)")
    print(f"📉 最终匹配测试集: {len(test_list)} 张 (官方黄金标准: 5050)")

    # 如果还是没满，打印出到底长什么奇葩样
    if len(train_list) != 5285 or len(test_list) != 5050:
        print("\n💡 异常样本诊断分析 (截取前3个):")
        for raw, core in missed_examples:
            print(f"   > 原始路径 : {raw}")
            print(f"   > 提取的ID : {core}")
        print(f"\n   > Train集合里的样本长这样 : {list(train_set)[:2]}")

if __name__ == '__main__':
    main()