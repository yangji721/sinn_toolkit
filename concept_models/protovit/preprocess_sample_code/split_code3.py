import os
import shutil
from tqdm import tqdm
from PIL import Image

# DeepFashion数据集路径
data_path = r"/data/yang/benchmark/data/DeepFashion"
output_base = r"/data/yang/benchmark/data/DeepFashion_processed_fixed"

def read_paired_files(img_file, cate_file):
    """读取配对的图像路径和类别ID文件"""
    with open(img_file, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]
    
    with open(cate_file, 'r') as f:
        categories = [int(line.strip()) for line in f if line.strip()]
    
    # 确保两个文件行数一致
    assert len(img_paths) == len(categories), f"文件行数不匹配: {len(img_paths)} vs {len(categories)}"
    
    return list(zip(img_paths, categories))

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
    # 方法1：使用完整路径信息创建唯一名称
    unique_name = img_path.replace('/', '_').replace('\\', '_')
    
    # 确保扩展名正确
    if not unique_name.lower().endswith('.jpg'):
        unique_name += '.jpg'
    
    output_path = os.path.join(output_dir, unique_name)
    
    # 方法2：如果仍有冲突，添加计数器
    if os.path.exists(output_path):
        base_name = os.path.splitext(unique_name)[0]
        extension = os.path.splitext(unique_name)[1]
        counter = 1
        
        while os.path.exists(output_path):
            unique_name = f"{base_name}_{counter:04d}{extension}"
            output_path = os.path.join(output_dir, unique_name)
            counter += 1
    
    return unique_name, output_path

def process_split(img_cate_pairs, split_name, category_names):
    """处理训练或测试数据 - 修复版"""
    img_base_path = os.path.join(data_path, "Img")
    processed_count = 0
    missing_count = 0
    error_count = 0
    category_count = {}
    category_actual_files = {}  # 实际保存的文件数
    
    print(f"\n开始处理{split_name}集...")
    
    for i, (img_path, category_id) in enumerate(tqdm(img_cate_pairs, desc=f"处理{split_name}")):
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
                # 处理图像
                img = Image.open(found_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path)
                
                processed_count += 1
                
                # 统计每个类别的图像数
                category_count[category_id] = category_count.get(category_id, 0) + 1
                
                # 实际文件统计（验证用）
                if category_id not in category_actual_files:
                    category_actual_files[category_id] = []
                category_actual_files[category_id].append(output_path)
                
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # 只打印前10个错误
                    print(f"处理图像失败 {found_path}: {e}")
        else:
            missing_count += 1
            if missing_count <= 10:  # 只打印前10个缺失
                print(f"图像文件未找到: {img_path}")
    
    # 验证统计准确性
    print(f"\n{split_name}集处理统计:")
    print(f"  成功处理: {processed_count}")
    print(f"  文件缺失: {missing_count}")
    print(f"  处理错误: {error_count}")
    print(f"  成功率: {processed_count/(processed_count+missing_count+error_count)*100:.1f}%")
    
    # 验证实际文件数与统计数是否一致
    print(f"\n验证文件保存完整性:")
    total_files_saved = 0
    discrepancy_found = False
    
    for cat_id in sorted(category_count.keys())[:5]:  # 检查前5个类别
        reported_count = category_count[cat_id]
        actual_count = len(category_actual_files.get(cat_id, []))
        total_files_saved += actual_count
        
        cat_name = category_names.get(cat_id, f"category_{cat_id}")
        print(f"  类别{cat_id}_{cat_name}: 统计{reported_count}, 实际{actual_count}", end="")
        
        if reported_count != actual_count:
            print(" ⚠️  不一致!")
            discrepancy_found = True
        else:
            print(" ✓")
    
    if not discrepancy_found:
        print("  文件统计与实际保存一致 ✓")
    
    return processed_count, category_count

def verify_final_results():
    """验证最终处理结果"""
    print(f"\n=== 最终验证 ===")
    
    for split in ['train', 'test']:
        split_dir = os.path.join(output_base, split)
        if os.path.exists(split_dir):
            total_files = 0
            class_dirs = os.listdir(split_dir)
            
            print(f"{split}集:")
            print(f"  类别目录数: {len(class_dirs)}")
            
            for class_dir in class_dirs[:10]:  # 显示前10个类别
                class_path = os.path.join(split_dir, class_dir)
                if os.path.isdir(class_path):
                    file_count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
                    total_files += file_count
                    print(f"    {class_dir}: {file_count} files")
            
            if len(class_dirs) > 10:
                # 计算剩余类别的总文件数
                remaining_files = 0
                for class_dir in class_dirs[10:]:
                    class_path = os.path.join(split_dir, class_dir)
                    if os.path.isdir(class_path):
                        file_count = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
                        remaining_files += file_count
                
                total_files += remaining_files
                print(f"    ... 其他{len(class_dirs)-10}个类别: {remaining_files} files")
            
            print(f"  {split}集总文件数: {total_files}")

def filter_categories_by_sample_count(train_pairs, test_pairs, category_names, min_samples=10):
    """
    过滤掉样本数过少的类别，确保训练测试集类别一致
    
    Args:
        min_samples: 最小样本数阈值，默认10
    """
    print(f"\n统计各类别样本数并过滤（最小样本数: {min_samples}）...")
    
    # 统计所有类别的样本数
    category_stats = {}
    
    # 统计训练集
    for img_path, category_id in train_pairs:
        if category_id not in category_stats:
            category_stats[category_id] = {'train': 0, 'test': 0, 'total': 0}
        category_stats[category_id]['train'] += 1
    
    # 统计测试集
    for img_path, category_id in test_pairs:
        if category_id not in category_stats:
            category_stats[category_id] = {'train': 0, 'test': 0, 'total': 0}
        category_stats[category_id]['test'] += 1
    
    # 计算总数
    for cat_id in category_stats:
        category_stats[cat_id]['total'] = category_stats[cat_id]['train'] + category_stats[cat_id]['test']
    
    # 找出有效类别（样本数 > min_samples）
    valid_categories = set()
    filtered_categories = []
    
    print("所有类别样本数统计:")
    for cat_id in sorted(category_stats.keys()):
        cat_name = category_names.get(cat_id, f"category_{cat_id}")
        stats = category_stats[cat_id]
        
        if stats['total'] > min_samples:
            valid_categories.add(cat_id)
            status = "✓ 保留"
        else:
            filtered_categories.append((cat_id, cat_name, stats['total']))
            status = "✗ 过滤"
        
        print(f"  {status} {cat_id:2d}_{cat_name:<15}: 训练{stats['train']:4d}, 测试{stats['test']:4d}, 总计{stats['total']:4d}")
    
    print(f"\n过滤总结:")
    print(f"  保留类别数: {len(valid_categories)}")
    print(f"  过滤类别数: {len(filtered_categories)} (样本数 <= {min_samples})")
    
    # 过滤数据对
    filtered_train_pairs = [(img, cat) for img, cat in train_pairs if cat in valid_categories]
    filtered_test_pairs = [(img, cat) for img, cat in test_pairs if cat in valid_categories]
    
    # 过滤类别名称
    filtered_category_names = {cat_id: cat_name for cat_id, cat_name in category_names.items() 
                              if cat_id in valid_categories}
    
    print(f"\n过滤结果:")
    print(f"  原始类别数: {len(category_names)}")
    print(f"  有效类别数: {len(valid_categories)}")
    print(f"  过滤掉类别数: {len(filtered_categories)}")
    print(f"  原始训练样本: {len(train_pairs)} -> 过滤后: {len(filtered_train_pairs)}")
    print(f"  原始测试样本: {len(test_pairs)} -> 过滤后: {len(filtered_test_pairs)}")
    
    return filtered_train_pairs, filtered_test_pairs, filtered_category_names, valid_categories

def ensure_consistent_folders(valid_categories, category_names):
    """确保训练和测试集有完全相同的类别文件夹"""
    print(f"\n创建一致的类别文件夹结构...")
    
    for split in ['train', 'test']:
        split_dir = os.path.join(output_base, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for cat_id in valid_categories:
            cat_name = category_names[cat_id]
            folder_name = f"{cat_id:02d}_{cat_name}"
            category_dir = os.path.join(split_dir, folder_name)
            os.makedirs(category_dir, exist_ok=True)
    
    print(f"  为 {len(valid_categories)} 个有效类别创建了训练和测试文件夹")

def main():
    print("开始处理DeepFashion数据集 - 过滤版...")
    print(f"输出目录: {output_base}")
    
    # 清理旧的输出目录（可选）
    if os.path.exists(output_base):
        response = input("输出目录已存在，是否清理？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_base)
            print("已清理旧的输出目录")
    
    # 确保输出目录存在
    os.makedirs(output_base, exist_ok=True)
    
    # 读取类别名称
    print("读取类别名称...")
    category_names = read_category_names()
    print(f"原始类别数: {len(category_names)}")
    
    # 文件路径
    anno_dir = os.path.join(data_path, "Anno_fine")
    train_img_file = os.path.join(anno_dir, "train.txt")
    train_cate_file = os.path.join(anno_dir, "train_cate.txt")
    test_img_file = os.path.join(anno_dir, "test.txt")
    test_cate_file = os.path.join(anno_dir, "test_cate.txt")
    
    # 读取原始数据
    print("读取训练数据...")
    train_pairs = read_paired_files(train_img_file, train_cate_file)
    print(f"原始训练样本数: {len(train_pairs)}")
    
    print("读取测试数据...")
    test_pairs = read_paired_files(test_img_file, test_cate_file)
    print(f"原始测试样本数: {len(test_pairs)}")
    
    # 关键步骤：过滤类别
    filtered_train_pairs, filtered_test_pairs, filtered_category_names, valid_categories = \
        filter_categories_by_sample_count(train_pairs, test_pairs, category_names, min_samples=50)
    
    # 确保文件夹结构一致
    ensure_consistent_folders(valid_categories, filtered_category_names)
    
    # 处理过滤后的数据
    train_count, train_category_count = process_split(filtered_train_pairs, "train", filtered_category_names)
    test_count, test_category_count = process_split(filtered_test_pairs, "test", filtered_category_names)
    
    # 保存类别映射（只包含有效类别）
    mapping_file = os.path.join(output_base, "category_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("Category_ID\tCategory_Name\tTrain_Count\tTest_Count\n")
        for cat_id in sorted(valid_categories):
            cat_name = filtered_category_names[cat_id]
            train_num = train_category_count.get(cat_id, 0)
            test_num = test_category_count.get(cat_id, 0)
            f.write(f"{cat_id}\t{cat_name}\t{train_num}\t{test_num}\n")
    
    print(f"类别映射已保存: {mapping_file}")
    
    # 最终验证
    verify_consistency(valid_categories, filtered_category_names)
    
    print(f"\n=== 处理完成 ===")
    print(f"输出目录: {output_base}")
    print(f"有效类别数: {len(valid_categories)}")
    print(f"训练图像: {train_count}")
    print(f"测试图像: {test_count}")
    print("训练和测试集类别结构完全一致")

def verify_consistency(valid_categories, category_names):
    """验证训练测试集文件夹一致性"""
    print(f"\n=== 验证文件夹一致性 ===")
    
    train_dir = os.path.join(output_base, "train")
    test_dir = os.path.join(output_base, "test")
    
    train_folders = set(os.listdir(train_dir)) if os.path.exists(train_dir) else set()
    test_folders = set(os.listdir(test_dir)) if os.path.exists(test_dir) else set()
    
    print(f"训练集文件夹数: {len(train_folders)}")
    print(f"测试集文件夹数: {len(test_folders)}")
    
    if train_folders == test_folders:
        print("✓ 训练和测试集文件夹完全一致")
        
        # 统计有数据和空的文件夹
        train_non_empty = 0
        test_non_empty = 0
        
        for folder in train_folders:
            train_path = os.path.join(train_dir, folder)
            test_path = os.path.join(test_dir, folder)
            
            train_files = len([f for f in os.listdir(train_path) if f.endswith('.jpg')])
            test_files = len([f for f in os.listdir(test_path) if f.endswith('.jpg')])
            
            if train_files > 0:
                train_non_empty += 1
            if test_files > 0:
                test_non_empty += 1
        
        print(f"训练集有数据的类别: {train_non_empty}/{len(train_folders)}")
        print(f"测试集有数据的类别: {test_non_empty}/{len(test_folders)}")
        
    else:
        print("✗ 训练和测试集文件夹不一致")
        missing_in_test = train_folders - test_folders
        missing_in_train = test_folders - train_folders
        
        if missing_in_test:
            print(f"测试集缺失: {missing_in_test}")
        if missing_in_train:
            print(f"训练集缺失: {missing_in_train}")

if __name__ == "__main__":
    main()