# DeepFashion数据集实验设置 - 修复版
# 基于处理结果：45个类别

import os

# 基础架构设置
base_architecture = 'deit_small_patch16_224'
radius = 1
img_size = 224

# 原型设置 - 使用更合理的原型数量
num_classes = 34
prototypes_per_class = 5  # 减少到每类5个原型

if base_architecture == 'deit_small_patch16_224':
    prototype_shape = (num_classes * prototypes_per_class, 384, 4)  # (225, 384, 4)
elif base_architecture == 'deit_tiny_patch16_224':
    prototype_shape = (num_classes * prototypes_per_class, 192, 4)  # (225, 192, 4)
elif base_architecture == 'cait_xxs24_224':
    prototype_shape = (num_classes * prototypes_per_class, 192, 4)  # (225, 192, 4)

dropout_rate = 0.1
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

# 实验标识
experiment_run = 'deepfashion_45class_5proto_fixed'

# 数据路径
data_path = r"/data/yang/benchmark/data/DeepFashion_processed/"
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'

# 批次大小
train_batch_size = 64
test_batch_size = 50
train_push_batch_size = 32

# 学习率设置 - 修复版
joint_optimizer_lrs = {
    'features': 5e-5,     # 提高特征学习率
    'prototype_vectors': 3e-3  # 保持原型学习率
}
joint_lr_step_size = 5

stage_2_lrs = {'patch_select': 5e-5}

warm_optimizer_lrs = {
    'features': 1e-7,
    'prototype_vectors': 3e-3
}

last_layer_optimizer_lr = 1e-4

# 损失函数系数 - 修复版
coefs = {
    'crs_ent': 1,
    'clst': -0.8,
    'sep': 0.1,
    'l1': 1e-5,      # 提高L1正则化
    'orth': 1e-3,    # 提高正交损失
    'coh': 3e-7      # 提高一致性损失
}

coefs_slots = {'coh': 1e-6}

# 训练设置
sum_cls = False
k = 1
sig_temp = 100

# 训练轮次
num_joint_epochs = 15        # 增加联合训练轮次
num_warm_epochs = 5
num_train_epochs = num_joint_epochs + num_warm_epochs
slots_train_epoch = 5
push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

# 针对类别不平衡的特殊设置
class_balanced_loss = True
focal_loss_alpha = 0.25
focal_loss_gamma = 2.0

# 数据增强设置
augmentation_config = {
    'use_augmentation': True,
    'random_horizontal_flip': True,
    'color_jitter': True,
    'random_rotation': 15,
    'random_crop': True,
    'mixup_alpha': 0.2,
    'cutmix_alpha': 1.0
}

# 模型保存设置
save_config = {
    'save_dir': f'./saved_models/{experiment_run}/',
    'save_every_epoch': 5,
    'save_best_model': True,
    'early_stopping_patience': 15  # 增加耐心度
}

# 评估设置
eval_config = {
    'eval_every_epoch': 1,
    'compute_class_accuracies': True,
    'save_confusion_matrix': True,
    'top_k_accuracy': [1, 3, 5]
}

# 打印配置摘要
def print_config_summary():
    print("=" * 60)
    print(f"DeepFashion实验配置摘要 - 修复版")
    print("=" * 60)
    print(f"实验名称: {experiment_run}")
    print(f"数据集: DeepFashion (45类服装分类)")
    print(f"模型架构: {base_architecture}")
    print(f"图像大小: {img_size}x{img_size}")
    print()
    print(f"原型配置:")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 每类原型数: {prototypes_per_class}")
    print(f"  - 总原型数: {prototype_shape[0]}")
    print(f"  - 原型形状: {prototype_shape}")
    print()
    print(f"训练配置:")
    print(f"  - 训练轮次: {num_train_epochs}")
    print(f"  - 联合训练轮次: {num_joint_epochs}")
    print(f"  - 预热轮次: {num_warm_epochs}")
    print(f"  - 训练批次大小: {train_batch_size}")
    print(f"  - 测试批次大小: {test_batch_size}")
    print()
    print(f"学习率:")
    print(f"  - 特征提取器: {joint_optimizer_lrs['features']}")
    print(f"  - 原型向量: {joint_optimizer_lrs['prototype_vectors']}")
    print(f"  - 最后一层: {last_layer_optimizer_lr}")
    print()
    print(f"损失函数系数:")
    for name, coef in coefs.items():
        print(f"  - {name}: {coef}")
    print()
    print(f"修复内容:")
    print(f"  - 减少原型数量: 450 -> 225")
    print(f"  - 提高特征学习率: 1e-5 -> 5e-5")
    print(f"  - 提高损失系数: L1, Orth, Coh")
    print(f"  - 增加训练轮次: 10 -> 15")
    print("=" * 60)

# 创建必要的目录
def create_directories():
    """创建实验所需的目录"""
    os.makedirs(save_config['save_dir'], exist_ok=True)
    os.makedirs(f"./logs/{experiment_run}/", exist_ok=True)
    os.makedirs(f"./results/{experiment_run}/", exist_ok=True)
    print(f"已创建实验目录: {save_config['save_dir']}")

# 验证配置
def validate_config():
    """验证配置参数的合理性"""
    issues = []
    
    # 检查原型数量
    if prototype_shape[0] < num_classes:
        issues.append(f"原型总数({prototype_shape[0]})少于类别数({num_classes})")
    
    # 检查批次大小
    if train_batch_size % 8 != 0:
        issues.append(f"建议训练批次大小为8的倍数，当前为{train_batch_size}")
    
    # 检查学习率
    if joint_optimizer_lrs['features'] >= joint_optimizer_lrs['prototype_vectors']:
        issues.append("特征学习率通常应小于原型学习率")
    
    # 检查路径
    if not os.path.exists(data_path):
        issues.append(f"数据路径不存在: {data_path}")
    
    if issues:
        print("配置验证发现问题:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        return False
    else:
        print("✅ 配置验证通过")
        return True

if __name__ == "__main__":
    print_config_summary()
    print()
    create_directories()
    print()
    validate_config()
