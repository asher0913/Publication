#!/bin/bash

# CEM-main 标准化实验脚本
# 确保与对比项目完全相同的实验条件

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 实验参数（完全一致）
arch="vgg11_bn_sgm"
cutlayer=4
batch_size=128
num_client=1
num_epochs=240
learning_rate=0.05
lambd=16
dataset_portion=1.0
client_sample_ratio=1.0
noniid=1.0
local_lr=-1.0
dataset="cifar10"
scheme="V2_epoch"
regularization="Gaussian_kl"
regularization_strength=0.025
var_threshold=0.125
AT_regularization="SCA_new"
AT_regularization_strength=0.3
log_entropy=1
ssim_threshold=0.5
gan_AE_type="res_normN4C64"
gan_loss_type="SSIM"
bottleneck_option="noRELU_C8S1"
optimize_computation=1
random_seed=125

# 文件名生成
filename="pretrain_False_lambd_${lambd}_noise_${regularization_strength}_epoch_${num_epochs}_bottleneck_${bottleneck_option}_log_${log_entropy}_ATstrength_${AT_regularization_strength}_lr_${learning_rate}_varthres_${var_threshold}"
folder_name="saves/cifar10/${AT_regularization}_infocons_sgm_lg${log_entropy}_thre${var_threshold}"

echo "🚀 开始 CEM-main 实验..."
echo "📊 实验参数: λ=${lambd}, 正则化强度=${regularization_strength}, 训练轮数=${num_epochs}"

# 训练阶段
echo "🔥 阶段1: 训练模型..."
python main_MIA.py \
    --arch=${arch} \
    --cutlayer=${cutlayer} \
    --batch_size=${batch_size} \
    --filename=${filename} \
    --num_client=${num_client} \
    --num_epochs=${num_epochs} \
    --dataset=${dataset} \
    --scheme=${scheme} \
    --regularization=${regularization} \
    --regularization_strength=${regularization_strength} \
    --log_entropy=${log_entropy} \
    --AT_regularization=${AT_regularization} \
    --AT_regularization_strength=${AT_regularization_strength} \
    --random_seed=${random_seed} \
    --learning_rate=${learning_rate} \
    --lambd=${lambd} \
    --gan_AE_type ${gan_AE_type} \
    --gan_loss_type ${gan_loss_type} \
    --local_lr ${local_lr} \
    --bottleneck_option ${bottleneck_option} \
    --folder ${folder_name} \
    --ssim_threshold ${ssim_threshold} \
    --var_threshold ${var_threshold}

if [ $? -eq 0 ]; then
    echo "✅ CEM-main 训练完成"
else
    echo "❌ CEM-main 训练失败"
    exit 1
fi

# 攻击测试阶段  
echo "🔥 阶段2: 模型反演攻击测试..."
python main_test_MIA.py \
    --arch=${arch} \
    --cutlayer=${cutlayer} \
    --batch_size=${batch_size} \
    --filename=${filename} \
    --num_client=${num_client} \
    --num_epochs=${num_epochs} \
    --dataset=${dataset} \
    --scheme=${scheme} \
    --regularization=${regularization} \
    --regularization_strength=${regularization_strength} \
    --log_entropy=${log_entropy} \
    --AT_regularization=${AT_regularization} \
    --AT_regularization_strength=${AT_regularization_strength} \
    --random_seed=${random_seed} \
    --learning_rate=${learning_rate} \
    --lambd=${lambd} \
    --gan_AE_type ${gan_AE_type} \
    --gan_loss_type ${gan_loss_type} \
    --local_lr ${local_lr} \
    --bottleneck_option ${bottleneck_option} \
    --folder ${folder_name} \
    --ssim_threshold ${ssim_threshold} \
    --var_threshold ${var_threshold}

if [ $? -eq 0 ]; then
    echo "✅ CEM-main 攻击测试完成"
else
    echo "❌ CEM-main 攻击测试失败"
    exit 1
fi

echo "🎯 CEM-main 完整实验完成！"
