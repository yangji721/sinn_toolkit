# AWA2数据集实验配置 - 常规分类任务（8:2分割）
base_architecture = 'deit_small_patch16_224'
radius = 1 # unit of patches
img_size = 224

# 根据基础架构设置原型形状
if base_architecture == 'deit_small_patch16_224':
    prototype_shape = (500, 384, 4)  # AWA2有50个类别，设置合适的原型数量
elif base_architecture == 'deit_tiny_patch16_224':
    prototype_shape = (500, 192, 4)
elif base_architecture == 'cait_xxs24_224':
    prototype_shape = (500, 192, 4)

dropout_rate = 0.0
num_classes = 50  # AWA2有50个类别，所有类别都用于训练和测试
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = 'awa2_classification_exp1'

# AWA2数据路径（8:2分割后）
data_path = r"/data/yang/benchmark/data/AWA2_processed/"
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'

# 批次大小
train_batch_size = 80      # 根据AWA2数据量调整
test_batch_size = 50       
train_push_batch_size = 40 

# 学习率设置
joint_optimizer_lrs = {'features': 5e-5,
                      'prototype_vectors': 3e-3}
joint_lr_step_size = 5

stage_2_lrs = {'patch_select': 5e-5}

warm_optimizer_lrs = {'features': 1e-7,
                     'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

# 损失函数系数
coefs = {
    'crs_ent': 1,
    'clst': -0.8,
    'sep': 0.1,
    'l1': 1e-2,
    'orth': 1e-3,
    'coh': 3e-3
}

coefs_slots = {'coh': 1e-6}

# 其他参数
sum_cls = False
k = 1
sig_temp = 100

# 训练轮次设置
num_joint_epochs = 12      # 适当调整训练轮次
num_warm_epochs = 5
num_train_epochs = num_joint_epochs + num_warm_epochs
slots_train_epoch = 5
push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

print(f"AWA2分类任务实验配置:")
print(f"- 总类别数: {num_classes}")
print(f"- 训练测试分割: 8:2（每个类别内部分割）")
print(f"- 原型形状: {prototype_shape}")
print(f"- 数据路径: {data_path}")
print(f"- 训练轮次: {num_train_epochs}")
print(f"- 实验名称: {experiment_run}")
print(f"- 训练批次大小: {train_batch_size}")
print(f"- 测试批次大小: {test_batch_size}")