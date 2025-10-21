import os
import shutil
from tqdm import tqdm
from PIL import Image

# DeepFashion数据集路径
data_path = r"/data/yang/benchmark/data/DeepFashion"
output_base = r"/data/yang/benchmark/data/DeepFashion_processed_cropped"

def read_paired_files(img_file, cate_file, bbox_file):
    """读取配对的图像路径、类别ID和边界框文件"""
    with open(img_file, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]
    
    with open(cate_file, 'r') as f:
        categories = [int(line.strip()) for line in f if line.strip()]
    
    with open(bbox_file, 'r') as f:
        bbox_lines = [line.strip() for line in f if line.strip()]
    
    # 解析边界框数据
    bboxes = []
    for line in bbox_lines:
        parts = line.split()
        if len(parts) >= 4:
            x1, y1, x2, y2 = map(int, parts[:4])
            bboxes.append((x1, y1, x2, y2))
        else:
            bboxes.append(None)  # 无效的边界框
    
    # 确保三个文件行数一致
    assert len(img_paths) == len(categories) == len(bboxes), \
        f"文件行数不匹配: {len(img_paths)} vs {len(categories)} vs {len(bboxes)}"
    
    return list(zip(img_paths, categories, bboxes))

def read_category_names():
    """读取类别名称"""
    category_file = os.path.join(data_path, "Anno_fine", "list_category_cloth.txt")
    category_names = {}
    
    with open(category_file, 'r') as f:
        lines = f.readlines()[2:]  # 跳过前两行
        
    category_id = 1
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) >= 1:
                category_name = parts[0]
                category_names[category_id] = category_name
                category_id += 1
    
    return category_names

def generate_unique_filename(img_path, output_dir):
    """生成唯一的文件名，避免冲突"""
    unique_name = img_path.replace('/', '_').replace('\\', '_')
    
    if not unique_name.lower().endswith('.jpg'):
        unique_name += '.jpg'
    
    output_path = os.path.join(output_dir, unique_name)
    
    # 如果仍有冲突，添加计数器
    if os.path.exists(output_path):
        base_name = os.path.splitext(unique_name)[0]
        extension = os.path.splitext(unique_name)[1]
        counter = 1
        
        while os.path.exists(output_path):
            unique_name = f"{base_name}_{counter:04d}{extension}"
            output_path = os.path.join(output_dir, unique_name)
            counter += 1
    
    return unique_name, output_path

def crop_and_resize_image(image, bbox, target_size=(224, 224)):
    """
    根据边界框裁剪图像并调整大小
    
    Args:
        image: PIL Image对象
        bbox: (x1, y1, x2, y2) 边界框坐标
        target_size: 目标尺寸
    """
    if bbox is None:
        # 如果没有边界框，只调整大小
        return image.resize(target_size)
    
    x1, y1, x2, y2 = bbox
    
    # 确保坐标在图像范围内
    img_width, img_height = image.size
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(x1, min(x2, img_width))
    y2 = max(y1, min(y2, img_height))
    
    # 裁剪图像
    if x2 > x1 and y2 > y1:
        cropped_img = image.crop((x1, y1, x2, y2))
    else:
        # 如果边界框无效，使用原图
        cropped_img = image
    
    # 调整到目标尺寸
    return cropped_img.resize(target_size)

def process_split(img_cate_bbox_pairs, split_name, category_names):
    """处理训练或测试数据 - 包含裁剪功能"""
    img_base_path = os.path.join(data_path, "Img")
    processed_count = 0
    missing_count = 0
    error_count = 0
    bbox_count = 0
    no_bbox_count = 0
    category_count = {}
    
    print(f"\n开始处理{split_name}集（包含裁剪）...")
    
    for i, (img_path, category_id, bbox) in enumerate(tqdm(img_cate_bbox_pairs, desc=f"处理{split_name}")):
        # 统计边界框情况
        if bbox is not None:
            bbox_count += 1
        else:
            no_bbox_count += 1
        
        # 尝试多种可能的路径格式
        possible_paths = [
            os.path.join(img_base_path, img_path),
            os.path.join(img_base_path, "img_highres", img_path),
            os.path.join(img_base_path, img_path.replace("img/", "")),
            os.path.join(img_base_path, "img_highres", img_path.replace("img/", ""))
        ]
        
        found_path = None
        for attempt_path in possible_paths:
            if os.path.exists(attempt_path):
                found_path = attempt_path
                break
        
        if found_path:
            # 获取类别名称
            category_name = category_names.get(category_id, f"category_{category_id}")
            
            # 创建输出目录
            output_dir = os.path.join(output_base, split_name, f"{category_id:02d}_{category_name}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成唯一文件名
            unique_filename, output_path = generate_unique_filename(img_path, output_dir)
            
            try:
                # 处理图像：裁剪和调整大小
                img = Image.open(found_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 关键步骤：使用边界框裁剪图像
                cropped_img = crop_and_resize_image(img, bbox, target_size=(224, 224))
                cropped_img.save(output_path)
                
                processed_count += 1
                category_count[category_id] = category_count.get(category_id, 0) + 1
                
            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    print(f"处理图像失败 {found_path}: {e}")
        else:
            missing_count += 1
            if missing_count <= 10:
                print(f"图像文件未找到: {img_path}")
    
    # 处理统计
    print(f"\n{split_name}集处理统计:")
    print(f"  成功处理: {processed_count}")
    print(f"  文件缺失: {missing_count}")
    print(f"  处理错误: {error_count}")
    print(f"  有边界框: {bbox_count}")
    print(f"  无边界框: {no_bbox_count}")
    print(f"  边界框覆盖率: {bbox_count/(bbox_count+no_bbox_count)*100:.1f}%")
    print(f"  成功率: {processed_count/(processed_count+missing_count+error_count)*100:.1f}%")
    
    return processed_count, category_count

def verify_cropping_results():
    """验证裁剪结果"""
    print(f"\n=== 裁剪结果验证 ===")
    
    # 检查几个样本图像的尺寸
    sample_count = 0
    for split in ['train', 'test']:
        split_dir = os.path.join(output_base, split)
        if os.path.exists(split_dir):
            for class_dir in os.listdir(split_dir)[:3]:  # 检查前3个类别
                class_path = os.path.join(split_dir, class_dir)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path)[:2]:  # 每个类别检查2张图
                        if img_file.endswith('.jpg'):
                            img_path = os.path.join(class_path, img_file)
                            try:
                                with Image.open(img_path) as img:
                                    print(f"  {split}/{class_dir}/{img_file}: {img.size}")
                                    sample_count += 1
                                    if sample_count >= 6:  # 总共检查6张图像
                                        return
                            except Exception as e:
                                print(f"  检查{img_path}失败: {e}")

def main():
    print("开始处理DeepFashion数据集 - 带边界框裁剪...")
    print(f"输出目录: {output_base}")
    
    # 清理旧的输出目录（可选）
    if os.path.exists(output_base):
        response = input("输出目录已存在，是否清理？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_base)
            print("已清理旧的输出目录")
    
    # 确保输出目录存在
    os.makedirs(output_base, exist_ok=True)
    
    # 文件路径
    anno_dir = os.path.join(data_path, "Anno_fine")
    train_img_file = os.path.join(anno_dir, "train.txt")
    train_cate_file = os.path.join(anno_dir, "train_cate.txt")
    train_bbox_file = os.path.join(anno_dir, "train_bbox.txt")
    
    test_img_file = os.path.join(anno_dir, "test.txt")
    test_cate_file = os.path.join(anno_dir, "test_cate.txt")
    test_bbox_file = os.path.join(anno_dir, "test_bbox.txt")
    
    # 检查边界框文件是否存在
    for bbox_file in [train_bbox_file, test_bbox_file]:
        if not os.path.exists(bbox_file):
            print(f"警告: 边界框文件不存在 {bbox_file}")
            print("将进行无裁剪处理")
    
    # 读取训练数据
    print("读取训练数据...")
    train_pairs = read_paired_files(train_img_file, train_cate_file, train_bbox_file)
    print(f"训练样本数: {len(train_pairs)}")
    
    # 读取测试数据
    print("读取测试数据...")
    test_pairs = read_paired_files(test_img_file, test_cate_file, test_bbox_file)
    print(f"测试样本数: {len(test_pairs)}")
    
    # 读取类别名称
    print("读取类别名称...")
    category_names = read_category_names()
    print(f"类别数: {len(category_names)}")
    
    # 处理训练集
    train_count, train_category_count = process_split(train_pairs, "train", category_names)
    
    # 处理测试集
    test_count, test_category_count = process_split(test_pairs, "test", category_names)
    
    # 保存类别映射
    all_categories = set(train_category_count.keys()) | set(test_category_count.keys())
    mapping_file = os.path.join(output_base, "category_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("Category_ID\tCategory_Name\tTrain_Count\tTest_Count\n")
        for cat_id in sorted(all_categories):
            cat_name = category_names.get(cat_id, f"category_{cat_id}")
            train_num = train_category_count.get(cat_id, 0)
            test_num = test_category_count.get(cat_id, 0)
            f.write(f"{cat_id}\t{cat_name}\t{train_num}\t{test_num}\n")
    
    print(f"类别映射已保存: {mapping_file}")
    
    # 验证裁剪结果
    verify_cropping_results()
    
    print(f"\n=== 处理完成 ===")
    print(f"裁剪版输出目录: {output_base}")
    print(f"训练图像: {train_count}")
    print(f"测试图像: {test_count}")
    print("所有图像已裁剪并调整为 224x224 尺寸")
    print("建议使用此版本进行训练以获得更好效果")

if __name__ == "__main__":
    main()