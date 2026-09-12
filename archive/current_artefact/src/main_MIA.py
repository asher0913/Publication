"""
main_MIA.py — training entry-point for the SCA-CEM defended split-learning run.

Called by run_exp.sh. Wraps the MIA_train class in
model_training_paral_pruning.py with an argparse layer so that all
hyper-parameters can be set from the shell script.

The actual training loop lives in model_training_paral_pruning.py;
this file is only the CLI shim.
"""

import logging

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # 跑在远端 GPU 机器上没有 X server，强制无显示后端
import matplotlib.pyplot as plt

import argparse

import model_training_paral_pruning
from datasets_torch import *
from utils import setup_logger


parser = argparse.ArgumentParser(description='Split-learning training with SCA-CEM defence')

# ── architecture / split / dataset ────────────────────────────────────────
parser.add_argument('--arch', default="vgg11_bn", type=str,
                    help='backbone, e.g. vgg11_bn_sgm')
parser.add_argument('--cutlayer', default=4, type=int,
                    help='split point — number of layers kept on the client')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--filename', required=True, type=str,
                    help='subfolder name used for saving checkpoints/logs')
parser.add_argument('--folder', default="saves", type=str,
                    help='top-level save directory')
parser.add_argument('--num_client', default=1, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float,
                    help='LR for the server-side model')
parser.add_argument('--lambd', default=1.0, type=float,
                    help='CEM weight λ')
parser.add_argument('--dataset_portion', default=1.0, type=float)
parser.add_argument('--client_sample_ratio', default=1.0, type=float)
parser.add_argument('--noniid', default=1.0, type=float,
                    help='non-iid ratio; 0.1 = 1 in 10 classes per client')
parser.add_argument('--local_lr', default=-1, type=float,
                    help='LR for the client-side model; -1 = use server LR')
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--scheme', default="V2_epoch", type=str)

# ── regularisation / CEM / attention surrogate ────────────────────────────
parser.add_argument('--regularization', default="None", type=str,
                    help='which input-space regulariser, e.g. Gaussian_kl')
parser.add_argument('--regularization_strength', default=0, type=float,
                    help='strength of the input-space regulariser (e.g. σ for Gaussian_kl)')
parser.add_argument('--var_threshold', default=0.1, type=float,
                    help='variance threshold used inside the CEM regulariser')
parser.add_argument('--AT_regularization', default="None", type=str,
                    help='attention-side regulariser; SCA_new = the dissertation method')
parser.add_argument('--AT_regularization_strength', default=0, type=float,
                    help='weight on the attention regulariser')
parser.add_argument('--log_entropy', default=0, type=float,
                    help='if >0, log the entropy term during training')
parser.add_argument('--ssim_threshold', default=0.0, type=float)
parser.add_argument('--gan_AE_type', default="custom", type=str,
                    help='AE used inside GAN_adv: custom / simple / simplest')
parser.add_argument('--gan_loss_type', default="SSIM", type=str,
                    help='SSIM or MSE for the defensive decoder')
parser.add_argument('--bottleneck_option', default="None", type=str,
                    help='bottleneck spec, e.g. noRELU_C16S1')
parser.add_argument('--optimize_computation', default=1, type=int)
parser.add_argument('--decoder_sync', action='store_true', default=False)

# Slot Attention / memory-bank hyper-parameters (the SCA_new module)
parser.add_argument('--attention_num_slots', default=8, type=int,
                    help='number of slots S')
parser.add_argument('--attention_num_heads', default=4, type=int,
                    help='cross-attention heads (must divide the slot dim)')
parser.add_argument('--attention_num_iterations', default=3, type=int,
                    help='slot refinement iterations T')
parser.add_argument('--attention_loss_scale', default=0.25, type=float,
                    help='multiplier on the attention-CEM loss term')
parser.add_argument('--attention_warmup_epochs', default=3, type=int,
                    help='epochs before the attention loss is switched on')
parser.add_argument('--attention_bank_size', default=64, type=int,
                    help='per-class memory bank size K')
parser.add_argument('--attention_slot_dim', default=128, type=int,
                    help='projected slot dim d_s (feature_dim → d_s)')

# ── checkpoint / transfer / misc ──────────────────────────────────────────
parser.add_argument('--load_from_checkpoint', action='store_true', default=False)
parser.add_argument('--load_from_checkpoint_server', action='store_true', default=False)
parser.add_argument('--transfer_source_task', default="cifar100", type=str)
parser.add_argument('--finetune_freeze_bn', action='store_true', default=False)
parser.add_argument('--save_more_checkpoints', action='store_true', default=False)
parser.add_argument('--initialize_different', action='store_true', default=False,
                    help='use different init per client (multi-client setting)')

parser.add_argument('--random_seed', default=123, type=int)

args = parser.parse_args()

# Seed both torch and numpy — most data-loader randomness goes through numpy
random_seed = args.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

batch_size = args.batch_size
cutting_layer = args.cutlayer
num_client = args.num_client
save_dir_name = "./{}/{}".format(args.folder, args.filename)

# All argparse values get forwarded straight into MIA_train. Keeping this
# as one big constructor call (rather than splitting into a config dict)
# matches the upstream CEM codebase so diffs against the original stay readable.
mi = model_training_paral_pruning.MIA_train(
    args.arch, cutting_layer, batch_size,
    lambd=args.lambd, n_epochs=args.num_epochs, scheme=args.scheme,
    num_client=num_client, dataset=args.dataset, save_dir=save_dir_name,
    random_seed=random_seed,
    regularization_option=args.regularization,
    regularization_strength=args.regularization_strength,
    AT_regularization_option=args.AT_regularization,
    AT_regularization_strength=args.AT_regularization_strength,
    log_entropy=args.log_entropy,
    initialize_different=args.initialize_different,
    learning_rate=args.learning_rate, local_lr=args.local_lr,
    gan_AE_type=args.gan_AE_type,
    load_from_checkpoint=args.load_from_checkpoint,
    bottleneck_option=args.bottleneck_option,
    optimize_computation=args.optimize_computation,
    decoder_sync=args.decoder_sync,
    finetune_freeze_bn=args.finetune_freeze_bn,
    gan_loss_type=args.gan_loss_type,
    ssim_threshold=args.ssim_threshold, var_threshold=args.var_threshold,
    attention_num_slots=args.attention_num_slots,
    attention_num_heads=args.attention_num_heads,
    attention_num_iterations=args.attention_num_iterations,
    attention_loss_scale=args.attention_loss_scale,
    attention_warmup_epochs=args.attention_warmup_epochs,
    attention_bank_size=args.attention_bank_size,
    attention_slot_dim=args.attention_slot_dim,
    source_task=args.transfer_source_task,
    load_from_checkpoint_server=args.load_from_checkpoint_server,
    save_more_checkpoints=args.save_more_checkpoints,
    dataset_portion=args.dataset_portion, noniid=args.noniid,
    client_sample_ratio=args.client_sample_ratio,
)
mi.logger.debug(str(args))

# log every 500 steps inside MIA_train.__call__
log_frequency = 500

LOG = mi(verbose=True, progress_bar=True, log_frequency=log_frequency)
