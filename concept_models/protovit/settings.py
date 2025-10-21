base_architecture = 'deit_small_patch16_224'
radius = 1 # unit of patches 
img_size = 224
if base_architecture == 'deit_small_patch16_224':
    prototype_shape = (2000, 384, 4)
elif base_architecture == 'deit_tiny_patch16_224':
    prototype_shape = (2000, 192, 4)
elif base_architecture == 'cait_xxs24_224':
    prototype_shape = (2000, 192, 4)

dropout_rate = 0.0
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'
experiment_run = 'exp1'

# data_path =  "./cub200_cropped/"
# train_dir = data_path + 'train_cropped_augmented/'
# test_dir = data_path + 'test_cropped/'
# train_push_dir = data_path + 'train_cropped/'
data_path = r"/data/yang/benchmark/data/CUB_processed/"
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'

train_batch_size = 128
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 5e-5,#2e-5,#1e-4,#1e-5,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5
 
stage_2_lrs = {'patch_select':5e-5}

warm_optimizer_lrs = {'features': 1e-7,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': -0.8,
    'sep': 0.1,
    'l1': 1e-2,
    'orth': 1e-3,
    'coh': 3e-3#,
}

coefs_slots = {'coh': 1e-6}
sum_cls = False
k = 1
sig_temp = 100
num_joint_epochs = 10
num_warm_epochs = 5
num_train_epochs = num_joint_epochs + num_warm_epochs

slots_train_epoch = 5
push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]
