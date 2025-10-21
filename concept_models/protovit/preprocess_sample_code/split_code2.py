import torch
import os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from PIL import Image
import random

# 设置随机种子确保可复现性
random.seed(42)

# AWA2数据集路径
data_path = r"/data/yang/benchmark/data/AWA2"

# 读取类别信息
class_ref = {}
ref_loc = os.path.join(data_path, "classes.txt")
for line in open(ref_loc, 'r').readlines():
    word = line.split()
    key = int(word[0])
    class_ref[key] = word[1]

# 图像目录
img_base_path = os.path.join(data_path, "JPEGImages")

# 输出目录
output_base = r"/data/yang/benchmark/data/AWA2_processed"

# 训练测试分割比例
train_ratio = 0.8

print("开始处理AWA2数据集，按8:2比例分割每个类别...")

total_train_images = 0
total_test_images = 0
class_count = 0

# 遍历所有类别文件夹
for class_folder in tqdm(os.listdir(img_base_path), desc="处理类别"):
    class_path = os.path.join(img_base_path, class_folder)
    
    # 跳过非目录文件
    if not os.path.isdir(class_path):
        continue
    
    class_count += 1
    
    # 获取该类别下的所有图像文件
    image_files = []
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(img_file)
    
    # 随机打乱图像列表
    random.shuffle(image_files)
    
    # 计算分割点
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    
    # 分割图像列表
    train_images = image_files[:train_count]
    test_images = image_files[train_count:]
    
    # 创建输出目录
    train_class_path = os.path.join(output_base, "train", class_folder)
    test_class_path = os.path.join(output_base, "test", class_folder)
    
    if not os.path.exists(train_class_path):
        os.makedirs(train_class_path)
    if not os.path.exists(test_class_path):
        os.makedirs(test_class_path)
    
    # 处理训练集图像
    for img_file in train_images:
        img_path = os.path.join(class_path, img_file)
        output_img_path = os.path.join(train_class_path, img_file)
        
        try:
            img = Image.open(img_path)
            img.save(output_img_path)
            total_train_images += 1
        except Exception as e:
            print(f"Error processing train image {img_path}: {e}")
    
    # 处理测试集图像
    for img_file in test_images:
        img_path = os.path.join(class_path, img_file)
        output_img_path = os.path.join(test_class_path, img_file)
        
        try:
            img = Image.open(img_path)
            img.save(output_img_path)
            total_test_images += 1
        except Exception as e:
            print(f"Error processing test image {img_path}: {e}")
    
    print(f"类别 {class_folder}: 总计{total_images}张图像 -> 训练{len(train_images)}张, 测试{len(test_images)}张")

print("\nAWA2数据集预处理完成！")
print(f"总类别数: {class_count}")
print(f"训练图像总数: {total_train_images}")
print(f"测试图像总数: {total_test_images}")
print(f"训练/测试比例: {total_train_images/(total_train_images+total_test_images):.2f}/{total_test_images/(total_train_images+total_test_images):.2f}")
print(f"输出目录: {output_base}")

# 验证每个类别在训练和测试中都存在
train_classes = set(os.listdir(os.path.join(output_base, "train")))
test_classes = set(os.listdir(os.path.join(output_base, "test")))
print(f"\n验证结果:")
print(f"训练集类别数: {len(train_classes)}")
print(f"测试集类别数: {len(test_classes)}")
print(f"训练测试类别是否一致: {train_classes == test_classes}")