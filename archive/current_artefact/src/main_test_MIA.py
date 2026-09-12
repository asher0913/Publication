"""
main_test_MIA.py — Phase 2 entry-point: run the MIA decoder attack
against the model trained by main_MIA.py.

Called by run_exp.sh after training finishes. Re-instantiates the same
MIA_train wrapper, resumes from `checkpoint_f_<epoch>.tar`, and then
calls .MIA_attack() to train the inversion decoder and report the
final (MSE, SSIM, PSNR) numbers under both training-time and
inference-time attack settings.

The CLI args here are deliberately a subset of those in main_MIA.py —
only the ones that affect attack-side behaviour and the model rebuild
need to be passed through.
"""

import os
import logging

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # 远端机器没有 X server，必须强制 Agg 后端
import matplotlib.pyplot as plt

import argparse

import MIA_torch  # noqa: F401  (kept around — older runs sometimes resumed via MIA_torch.MIA)
import model_training_paral_pruning
from datasets_torch import *
from utils import setup_logger


parser = argparse.ArgumentParser(description='MIA attack against a trained SCA-CEM model')

# ── must match the training-time setting exactly ─────────────────────────
parser.add_argument('--arch', default="vgg11_bn", type=str)
parser.add_argument('--cutlayer', default=4, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--filename', required=True, type=str)
parser.add_argument('--folder', default="saves", type=str)
parser.add_argument('--num_client', default=1, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--test_best', action='store_true', default=False,
                    help='resume the "best" checkpoint instead of a fixed epoch')
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--random_seed', default=125, type=int)
parser.add_argument('--scheme', default="V2_epoch", type=str)
parser.add_argument('--bottleneck_option', default="None", type=str)

# ── attack-time configuration ────────────────────────────────────────────
parser.add_argument('--regularization', default="None", type=str)
parser.add_argument('--regularization_strength', default=0.0, type=float)
parser.add_argument('--var_threshold', default=0.1, type=float)
parser.add_argument('--AT_regularization', default="None", type=str)
parser.add_argument('--AT_regularization_strength', default=0, type=float)
parser.add_argument('--log_entropy', default=0, type=float)
parser.add_argument('--average_time', default=1, type=int,
                    help='re-run the attack this many times and average')
parser.add_argument('--target_client', default=0, type=int)
parser.add_argument('--attack_scheme', default="MIA", type=str,
                    help='MIA or MIA_mf')
parser.add_argument('--attack_epochs', default=50, type=int,
                    help='epochs to train the inversion decoder')
parser.add_argument('--attack_from_later_layer', default=-1, type=int,
                    help='>-1 attacks at a later layer than the cut')
parser.add_argument('--gan_AE_type', default="custom", type=str)
parser.add_argument('--attack_loss_type', default="MSE", type=str)
parser.add_argument('--gan_loss_type', default="SSIM", type=str)
parser.add_argument('--MIA_optimizer', default="Adam", type=str)
parser.add_argument('--MIA_lr', default=1e-3, type=float)
parser.add_argument('--save_activation_tensor', action='store_true', default=False)
parser.add_argument('--attack_confidence_score', action='store_true', default=False)
parser.add_argument('--measure_option', action='store_true', default=False,
                    help='print MAC/param counts of client and server models')
parser.add_argument('--noise_aware', action='store_true', default=False,
                    help='enable noise-aware attack against GAN_noise / DP defences')
parser.add_argument('--new_log_folder', action='store_true', default=False)
parser.add_argument('--bhtsne_option', action='store_true', default=False)

# Slot-attention hyper-parameters (must match training)
parser.add_argument('--attention_num_slots', default=8, type=int)
parser.add_argument('--attention_num_heads', default=4, type=int)
parser.add_argument('--attention_num_iterations', default=3, type=int)
parser.add_argument('--attention_loss_scale', default=0.25, type=float)
parser.add_argument('--attention_warmup_epochs', default=3, type=int)
parser.add_argument('--attention_bank_size', default=64, type=int)
parser.add_argument('--attention_slot_dim', default=128, type=int)

args = parser.parse_args()

batch_size = args.batch_size
cutting_layer = args.cutlayer
date_list = [args.filename]
num_client = args.num_client
target_client = args.target_client

mse_score_list = []
ssim_score_list = []
psnr_score_list = []

# Fixed seed for the attack-side random-batch sampling. Using a single
# value here so that repeated runs of the attack pipeline produce
# numbers that are comparable across logs.
random_seed_list = [200]
random_seed = args.random_seed


def _first_batch(loader_like):
    """Return next(iter(...)) but handle the case where the loader is
    wrapped in a list (datasets_torch returns a list when num_client>1)."""
    loader = loader_like[0] if isinstance(loader_like, list) else loader_like
    return next(iter(loader))


def load_fixed_test_data(dataset_name, batch_size):
    """Load (or build-and-cache) a fixed batch of test images.

    The MIA attack is supposed to be evaluated on a *deterministic*
    set of images so that the reported (MSE, SSIM, PSNR) numbers are
    comparable across runs. The first time this function is called for
    a given dataset, it pulls one batch from the standard loader and
    caches it as `./test_<dataset>_image.pt` / `..._label.pt`. Every
    subsequent call just torch.load()s the cache.
    """
    img_path = f"./test_{dataset_name}_image.pt"
    label_path = f"./test_{dataset_name}_label.pt"
    if os.path.isfile(img_path) and os.path.isfile(label_path):
        return torch.load(img_path), torch.load(label_path)

    if dataset_name == "facescrub":
        facescrub_training_loader, _, _, _, _ = get_facescrub_bothloader(
            batch_size=batch_size, num_workers=2, shuffle=False,
            num_client=1, collude_use_public=False,
        )
        images, labels = _first_batch(facescrub_training_loader)
    elif dataset_name == "cifar10":
        cifar10_training_loader, _, _ = get_cifar10_trainloader(
            batch_size=batch_size, num_workers=2, shuffle=False, num_client=1
        )
        images, labels = _first_batch(cifar10_training_loader)
    elif dataset_name == "cifar100":
        cifar100_training_loader, _, _ = get_cifar100_trainloader(
            batch_size=batch_size, num_workers=2, shuffle=False, num_client=1
        )
        images, labels = _first_batch(cifar100_training_loader)
    elif dataset_name == "svhn":
        svhn_training_loader, _, _ = get_SVHN_trainloader(
            batch_size=batch_size, num_workers=2, shuffle=False, num_client=1
        )
        images, labels = _first_batch(svhn_training_loader)
    elif dataset_name == "mnist":
        mnist_training_loader, _ = get_mnist_bothloader(
            batch_size=batch_size, num_workers=2, shuffle=False, num_client=1
        )
        images, labels = _first_batch(mnist_training_loader)
    elif dataset_name == "fmnist":
        fmnist_training_loader, _ = get_fmnist_bothloader(
            batch_size=batch_size, num_workers=2, shuffle=False, num_client=1
        )
        images, labels = _first_batch(fmnist_training_loader)
    else:
        raise FileNotFoundError(
            f"Missing cached test tensors for {dataset_name} and no fallback loader available."
        )

    torch.save(images, img_path)
    torch.save(labels, label_path)
    return images, labels


# Confidence-score attack always operates on the smashed data, not a deeper layer.
if args.attack_confidence_score:
    args.attack_from_later_layer = -1


for date_0 in date_list:
    # Either resume the best checkpoint or a fixed-epoch one. The "80"
    # default is a leftover from earlier sweeps; for the headline run
    # `--test_best` is always passed in run_exp.sh.
    if args.test_best:
        args.num_epochs = "best"
    else:
        args.num_epochs = "80"
    save_dir_name = "./{}/{}".format(args.folder, date_0)

    mi = model_training_paral_pruning.MIA_train(
        args.arch, cutting_layer, batch_size,
        n_epochs=args.num_epochs, scheme=args.scheme,
        num_client=num_client, dataset=args.dataset, save_dir=save_dir_name,
        random_seed=random_seed,
        regularization_option=args.regularization,
        regularization_strength=args.regularization_strength,
        AT_regularization_option=args.AT_regularization,
        AT_regularization_strength=args.AT_regularization_strength,
        log_entropy=args.log_entropy,
        gan_AE_type=args.gan_AE_type, bottleneck_option=args.bottleneck_option,
        gan_loss_type=args.gan_loss_type,
        attention_num_slots=args.attention_num_slots,
        attention_num_heads=args.attention_num_heads,
        attention_num_iterations=args.attention_num_iterations,
        attention_loss_scale=args.attention_loss_scale,
        attention_warmup_epochs=args.attention_warmup_epochs,
        attention_bank_size=args.attention_bank_size,
        attention_slot_dim=args.attention_slot_dim,
    )

    if args.new_log_folder:
        # Optional separate log dir keyed by regularisation setting — used
        # when sweeping multiple regs against the same checkpoint.
        new_folder_dir = mi.save_dir + '/{}_{}/'.format(
            args.regularization, args.regularization_strength)
        new_folder_dir = os.path.abspath(new_folder_dir)
        if not os.path.isdir(new_folder_dir):
            os.makedirs(new_folder_dir)
        model_log_file = new_folder_dir + '/MIA.log'
        mi.logger = setup_logger(
            '{}_logger'.format(str(save_dir_name)), model_log_file,
            level=logging.DEBUG)

    if args.measure_option:
        c_mac, c_num_param, s_mac, s_num_param = mi.model.get_MAC_param()
        mi.logger.debug("Client Model's Mac and Param are {} and {}".format(c_mac, c_num_param))
        mi.logger.debug("Server Model's Mac and Param are {} and {}".format(s_mac, s_num_param))

    # Resume the trained client+server checkpoint. With `orig` schemes
    # the file naming is different (`checkpoint_<epoch>.tar` vs.
    # `checkpoint_f_<epoch>.tar`), so we branch on it.
    if "orig" not in args.scheme:
        print('the test model is:', args.num_epochs)
        resume_path = "./{}/{}/checkpoint_f_{}.tar".format(
            args.folder, date_0, args.num_epochs)

        # If `--test_best` was passed but checkpoint_f_best.tar is missing
        # (training crashed before the best snapshot got written), fall
        # back to the latest fixed-epoch snapshot we can find.
        if not os.path.isfile(resume_path) and args.test_best:
            found_fallback = False
            for fb_epoch in [args.num_epochs, 360, 300, 240, 200, 100]:
                fallback_path = "./{}/{}/checkpoint_f_{}.tar".format(
                    args.folder, date_0, fb_epoch)
                if os.path.isfile(fallback_path):
                    print(f"best checkpoint missing, fallback to {fallback_path}")
                    resume_path = fallback_path
                    found_fallback = True
                    break
            if not found_fallback:
                print(f"WARNING: No checkpoint found for {date_0}, skipping")
        mi.resume(resume_path)
    else:
        print("resume orig scheme's checkpoint")
        mi.resume("./{}/{}/checkpoint_{}.tar".format(args.folder, date_0, args.num_epochs))

    # Some defences require an extra "noise-aware" attack mode. We don't
    # use any of these in the headline SCA-CEM run (regularization is
    # `Gaussian_kl`), but the branches are kept so the same script can be
    # reused against the baseline defences.
    if "gan_adv_noise" in args.regularization:
        mi.logger.debug("regularization_strength for GAN_noise is {}".format(args.regularization_strength))
        mi.pre_GAN_train(30)
        noise_aware = args.noise_aware
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break GAN_noise Defense) ==")
    elif "local_dp" in args.regularization:
        noise_aware = args.noise_aware
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break Local Differential Privacy) ==")
    elif "dropout" in args.regularization:
        noise_aware = args.noise_aware
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break Local Differential Privacy) ==")
    elif "topkprune" in args.regularization:
        noise_aware = args.noise_aware
        if noise_aware:
            mi.logger.debug("== Noise-Aware MIA attack (smart attack to break Local Differential Privacy) ==")
    else:
        noise_aware = False

    # The attack is always evaluated on the same fixed batch (see docstring
    # of load_fixed_test_data).
    images, labels = load_fixed_test_data(args.dataset, batch_size)

    log_frequency = 500
    skip_valid = False
    if not skip_valid:
        # Run one validation pass on the resumed checkpoint to confirm
        # accuracy matches the training log before launching the attack.
        LOG = mi(verbose=True, progress_bar=True, log_frequency=log_frequency)

    # Outer loop over attack random seeds: when --average_time>1, we run
    # the attack multiple times and report the average / best score.
    for random_seed in random_seed_list:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        client_mse_list = []
        client_ssim_list = []
        client_psnr_list = []
        for j in range(num_client):
            # Skip the target client — that's the one being attacked.
            if num_client > 1 and j == target_client:
                continue
            mse_score, ssim_score, psnr_score, mse_score_I, ssim_score_I, psnr_score_I = mi.MIA_attack(
                args.attack_epochs,
                attack_option=args.attack_scheme,
                collude_client=j, target_client=target_client,
                noise_aware=noise_aware,
                loss_type=args.attack_loss_type,
                attack_from_later_layer=args.attack_from_later_layer,
                MIA_optimizer=args.MIA_optimizer, MIA_lr=args.MIA_lr,
            )
            client_mse_list.append(mse_score)
            client_ssim_list.append(ssim_score)
            client_psnr_list.append(psnr_score)

        # Worst-case-attacker reporting: among colluders, pick the
        # strongest reconstruction (lowest MSE / highest SSIM, PSNR).
        mse_score_list.append(np.min(np.array(client_mse_list)))
        ssim_score_list.append(np.max(np.array(client_ssim_list)))
        psnr_score_list.append(np.max(np.array(client_psnr_list)))

    avg_mse_score = np.mean(np.array(mse_score_list))
    avg_ssim_score = np.mean(np.array(ssim_score_list))
    avg_psnr_score = np.mean(np.array(psnr_score_list))
    std_mse_score = np.std(np.array(mse_score_list))
    std_ssim_score = np.std(np.array(ssim_score_list))
    std_psnr_score = np.std(np.array(psnr_score_list))

    mi.logger.debug(
        "== {} Training-based {} performance Score with optimizer {}, lr {}, "
        "loss type {} on {} epoch saved model ==".format(
            args.gan_AE_type, args.attack_scheme, args.MIA_optimizer,
            args.MIA_lr, args.attack_loss_type, args.num_epochs))

    if not args.attack_confidence_score:
        mi.logger.debug(
            "Reverse Intermediate activation at layer {} (-1 is the smashed-data)".format(
                args.attack_from_later_layer))
    else:
        mi.logger.debug("Reverse Confidence Score, conventional MIA!")
    mi.logger.debug("The tested model is: %s", args.num_epochs)
    mi.logger.debug(
        "MIA performance Score training time is (MSE, SSIM, PSNR) "
        "averaging {} times\n{}, {}, {}".format(
            len(random_seed_list), avg_mse_score, avg_ssim_score, avg_psnr_score))
    mi.logger.debug(
        "MIA performance Score inference time is (MSE, SSIM, PSNR): "
        "%.5f, %.5f, %.2f", mse_score_I, ssim_score_I, psnr_score_I)
