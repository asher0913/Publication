"""
model_training_paral_pruning.py — main split-learning training pipeline.

This is the file that the training and attack CLI scripts load
(`main_MIA.py` and `main_test_MIA.py`). It contains:

  * The three classes that make up this dissertation's contribution:
        - SlotAttention            — slot-attention block, soft-clusters
                                     a set of class-conditional features
                                     into S slots in slot_dim space.
        - CrossAttention           — vanilla multi-head cross-attention,
                                     used as a re-binding step.
        - FeatureMemoryBank        — per-class ring buffer that lets us
                                     do CROSS-SAMPLE clustering even
                                     when a class shows up <1 time per
                                     batch (which is the common case at
                                     530 classes / batch=256).
        - SlotCrossAttentionCEM    — wraps the three above into the
                                     drop-in replacement for the upstream
                                     GMM-based CEM surrogate. This is
                                     what `--AT_regularization=SCA_new`
                                     selects in run_exp.sh.

  * The MIA_train class — the split-learning training/attack
    orchestrator that the upstream CEM code already provides; the only
    SCA-specific changes here are the wiring through the constructor
    arguments `attention_*` (see `main_MIA.py`) and the call site that
    invokes SlotCrossAttentionCEM when AT_regularization == "SCA_new".

The MIA_train class itself is a slightly extended version of the
upstream CEM training loop. Edits beyond the SCA wiring are limited to
logging/checkpointing.
"""

import os
import time
import logging
from datetime import datetime
from shutil import rmtree

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.serialization import save
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from torchsummary import summary

from thop import profile
from ptflops import get_model_complexity_info
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import torch_pruning as tp

import pytorch_ssim
import architectures_torch as architectures
from utils import setup_logger, accuracy, AverageMeter, WarmUpLR, apply_transform_test, apply_transform, TV, l2loss, dist_corr, get_PSNR
from utils import freeze_model_bn, average_weights, DistanceCorrelationLoss, spurious_loss, prune_top_n_percent_left, dropout_defense, prune_defense
from model_architectures.resnet_cifar import ResNet20, ResNet32
from model_architectures.resnet_imagenet import Imagenet_ResNet20
from model_architectures.mobilenetv2 import MobileNetV2
from model_architectures.vgg import vgg11, vgg13, vgg11_bn, vgg13_bn, vgg11_bn_sgm
from datasets_torch import get_cifar100_trainloader, get_cifar100_testloader, get_cifar10_trainloader, \
    get_cifar10_testloader, get_mnist_bothloader, get_facescrub_bothloader, get_SVHN_trainloader, get_SVHN_testloader, get_fmnist_bothloader, get_tinyimagenet_bothloader, get_imagenet_bothloader, get_celeba_trainloader, get_celeba_testloader


# =============================================================================
# Attention-CEM modules — the dissertation contribution.
#
# These four pieces (SlotAttention, CrossAttention, FeatureMemoryBank,
# SlotCrossAttentionCEM) replace the upstream GMM-based CEM surrogate
# with a fully differentiable slot-attention soft-clustering surrogate
# that operates over CROSS-SAMPLE features (i.e. multiple samples of
# the same class), pulled from a per-class memory bank rather than just
# from the current batch. The motivation and ablation evidence are
# discussed in the dissertation §4-5 and §6.7-6.9.
# =============================================================================
class SlotAttention(nn.Module):
    """Slot Attention block (Locatello et al., NeurIPS 2020) adapted to
    operate on a flat set of feature vectors rather than spatial tokens.

    Input:  a set of N feature vectors of dim `feature_dim` for one
            class (concatenation of current-batch features + memory-bank
            features for that class).
    Output: S "slots" (each of dim `slot_dim`) plus the soft-assignment
            attention map, used downstream by SlotCrossAttentionCEM to
            compute the geometric variance term.

    Notes
    -----
    * Slots are sampled from a learnable Gaussian (`slot_mu`,
      `slot_log_sigma`) at training time and held deterministic at eval
      time — this is what gives the surrogate access to class-conditional
      stochasticity without leaking it into the attack-time output.
    * `slot_mu` uses orthogonal initialisation. With random init the
      slots collapse to a single mode after a handful of iterations on
      FaceScrub (530 classes, sparse-per-class batches), which makes the
      variance term degenerate.
    * `slot_log_sigma` is initialised at -3.0 (i.e. σ ≈ 0.05) for the
      same reason — large initial σ blows up the GRU update and the
      whole module diverges in the first epoch.
    """

    def __init__(self, feature_dim, num_slots=8, num_iterations=3, eps: float = 1e-8, mlp_hidden_scale: int = 2, slot_dim: int = None):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.feature_dim = feature_dim
        self.eps = eps
        self.slot_dim = feature_dim if slot_dim is None else slot_dim

        # Orthogonal init on the slot prototype prevents mode collapse on
        # FaceScrub (see class docstring).
        self.slot_mu = nn.Parameter(torch.empty(1, 1, self.slot_dim))
        nn.init.orthogonal_(self.slot_mu)
        # log σ initialised to -3 (σ ≈ 0.05). Larger values cause the
        # GRU update to explode early in training.
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_dim) - 3.0)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, self.slot_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, self.slot_dim, bias=False)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim * mlp_hidden_scale),
            nn.ReLU(inplace=True),
            nn.Linear(self.slot_dim * mlp_hidden_scale, self.slot_dim),
        )
        self.norm_inputs = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)
        self.scale = self.slot_dim ** -0.5

    def forward(self, inputs: torch.Tensor):
        """Run T = `num_iterations` rounds of slot-attention refinement.

        Args
        ----
        inputs : Tensor of shape [B, N, feature_dim]
            B is usually 1 in our usage (one class group at a time),
            N is the cross-sample set size (current-batch + bank).

        Returns
        -------
        slots : [B, S, slot_dim] — refined slot prototypes
        attn  : [B, N, S]        — soft-assignment of each input to each
                                    slot from the LAST iteration. The
                                    CEM loss in SlotCrossAttentionCEM
                                    uses this map directly.
        """
        B, N, D = inputs.shape
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        # Sample slots from the learnable Gaussian during training; keep
        # them deterministic (mu only) at eval time so that attack-time
        # forward passes don't get extra randomness on top.
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        if self.training:
            sigma = self.slot_log_sigma.exp().expand(B, self.num_slots, -1)
            slots = mu + sigma * torch.randn_like(mu)
        else:
            slots = mu.clone()

        attn = None  # holds the last iteration's attention weights
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm) * self.scale
            attn_logits = torch.einsum('bnd,bsd->bns', k, q)
            attn = torch.softmax(attn_logits, dim=-1)

            # Clamp + renormalise to prevent exact zeros from causing
            # NaNs in the einsum below; without this the loss spikes to
            # NaN during the first few epochs on FaceScrub.
            attn = torch.clamp(attn, min=self.eps)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            attn_sum = torch.clamp(attn.sum(dim=1), min=self.eps)  # [B, S]
            updates = torch.einsum('bns,bnd->bsd', attn, v)
            updates = updates / attn_sum.unsqueeze(-1)

            # GRU update on (slot_prev, weighted update). Reshape to a
            # flat batch because nn.GRUCell expects a 2D input.
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim))
            slots = slots.reshape(B, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn

class CrossAttention(nn.Module):
    """Multi-head cross-attention with a small residual gate (`alpha_*`)
    in front of both the attention output and the FFN. The gating
    coefficients are learned but bounded to (-1, 1) by `tanh`, which
    keeps the residual connection from being completely overwritten in
    the early epochs.

    Used by SlotCrossAttentionCEM as the optional "re-binding" step that
    pushes batch features back through the slot summary; the headline
    SCA-CEM run does NOT use this re-binding (the loss is computed
    directly from the slot-attention map), but the class is kept here
    because earlier ablation runs did.
    """

    def __init__(self, feature_dim, num_heads=4, mlp_mult: int = 4):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        self.ln_q = nn.LayerNorm(feature_dim)
        self.ln_kv = nn.LayerNorm(feature_dim)
        self.ln_ff = nn.LayerNorm(feature_dim)
        self.to_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.to_v = nn.Linear(feature_dim, feature_dim, bias=False)
        self.proj = nn.Linear(feature_dim, feature_dim)
        self.alpha_xattn = nn.Parameter(torch.tensor(0.1))
        self.alpha_ffn = nn.Parameter(torch.tensor(0.1))
        hidden = feature_dim * mlp_mult
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )
        self.scale = (self.head_dim) ** -0.5

    def forward(self, query_features, kv_features):
        squeeze_q = False
        if query_features.dim() == 2:
            query_features = query_features.unsqueeze(1)
            squeeze_q = True
        if kv_features.dim() == 2:
            kv_features = kv_features.unsqueeze(1)

        B, Q, D = query_features.shape
        _, K, _ = kv_features.shape
        H, d = self.num_heads, self.head_dim

        q_in = self.ln_q(query_features)
        kv_in = self.ln_kv(kv_features)
        q = self.to_q(q_in).view(B, Q, H, d).transpose(1, 2)
        k = self.to_k(kv_in).view(B, K, H, d).transpose(1, 2)
        v = self.to_v(kv_in).view(B, K, H, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Q, D)
        out = self.proj(out)
        y = query_features + torch.tanh(self.alpha_xattn) * out
        y_ff = self.ffn(self.ln_ff(y))
        y = y + torch.tanh(self.alpha_ffn) * y_ff
        if squeeze_q:
            y = y.squeeze(1)
        return y


class FeatureMemoryBank:
    """Per-class ring buffer of recent encoder features.

    Why this exists
    ---------------
    With FaceScrub at 530 classes and batch_size 256, a typical batch
    contains every class either zero or one time. That makes
    same-class clustering on the batch alone meaningless — there is
    nothing to cluster against. The memory bank keeps the last
    `bank_size` (= 64 in the headline run) projected features for each
    class, so that even when the current batch only contains one
    sample of class C, we have up to 64 detached references for that
    class to soft-cluster against.

    Layout
    ------
    self.bank   : [num_classes, bank_size, feat_dim]  ring buffer
    self.ptr    : [num_classes]                       write head per class
    self.valid  : [num_classes]                       count of populated slots,
                                                      saturating at bank_size

    The bank is NOT a torch.nn.Module because we don't want its
    contents in the optimiser state or saved into checkpoints — it's
    a runtime cache that is rebuilt from scratch on resume. The custom
    `to()` method exists so that the wrapping nn.Module
    (SlotCrossAttentionCEM) can transparently move the bank to GPU
    along with the rest of the model.
    """

    def __init__(self, num_classes, bank_size, feat_dim, device='cpu'):
        self.num_classes = num_classes
        self.bank_size = bank_size
        self.feat_dim = feat_dim
        self.bank = torch.zeros(num_classes, bank_size, feat_dim, device=device)
        self.ptr = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.valid = torch.zeros(num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, features, labels):
        """Insert detached features into the ring buffer for each class.

        `features` is [B, D]; `labels` is [B]. Out-of-range labels are
        silently skipped (defensive against label-noise during early
        debug runs — should never trigger on a clean FaceScrub split).
        Detaching is critical: without it, gradients would flow back
        through ALL past stored features and the bank would effectively
        become an unbounded computation graph.
        """
        for i in range(features.size(0)):
            cls = labels[i].item()
            if cls < 0 or cls >= self.num_classes:
                continue
            idx = self.ptr[cls].item()
            self.bank[cls, idx] = features[i].detach()
            self.ptr[cls] = (idx + 1) % self.bank_size
            if self.valid[cls] < self.bank_size:
                self.valid[cls] += 1

    def get_class_features(self, cls_idx):
        """Return the populated slice of class `cls_idx`'s ring buffer.

        Returns a [valid_count, D] tensor, or None when no features have
        been stored for this class yet (typically only during the first
        batch where this class appears).
        """
        n = self.valid[cls_idx].item()
        if n == 0:
            return None
        return self.bank[cls_idx, :n]

    def to(self, device):
        self.bank = self.bank.to(device)
        self.ptr = self.ptr.to(device)
        self.valid = self.valid.to(device)
        return self


class SlotCrossAttentionCEM(nn.Module):
    """Slot-Cross-Attention CEM surrogate (the dissertation contribution).

    Drop-in replacement for the upstream KMeans-based CEM surrogate.
    Selected by `--AT_regularization=SCA_new` in `run_exp.sh`.

    What it does, end-to-end
    ------------------------
    Given a batch of encoder features `z_flat` and their labels:

    1. Project `z_flat` (which is `feature_dim` = 4096 in the headline
       C16S1 run) down to `slot_dim` = 64 with a learnable linear layer.
       Slot Attention is run in `slot_dim` space because the
       quadratic-cost cross-attention would be too expensive at full
       feature width. CEM gradients still reach all encoder dimensions
       through this differentiable projection.
    2. Push the projected features into the per-class memory bank
       (detached copies — see FeatureMemoryBank).
    3. For each unique class in the batch, concatenate its current-batch
       features (with gradient) and its bank features (detached) and run
       Slot Attention over the resulting cross-sample set.
    4. Compute per-slot soft centroids (DETACHED — see "Why detached"
       below) and a per-slot geometric variance.
    5. Apply the CEM-style log-variance penalty:
              relu( log(var + γ) - log(threshold) )
       This penalises slots whose intra-class variance exceeds the
       noise-scaled threshold. Minimising the term contracts residual
       within-slot variation until the threshold is reached; smaller
       variances receive no further penalty.

    Why centroids are detached
    --------------------------
    Centroids act as fixed geometric references during each backward
    pass. Detaching them prevents the loss from being reduced by moving
    the feature and its reference together. Gradients still reach the
    current features through their distances and the soft assignments.

    Threshold
    ---------
    `threshold = var_threshold * (reg_strength ** 2) + gamma` follows
    the upstream CEM convention: when the encoder noise σ
    (`reg_strength`) goes up, the variance threshold scales quadratically.
    `gamma = 0.01` is a small offset that keeps the log
    well-defined when the variance is genuinely close to zero.

    Returned values
    ---------------
    `rob_loss` — averaged across classes and scaled by
                 `attention_loss_scale`; the outer loop merges its
                 encoder gradient using weight λ and its LR schedule.
    `intra_class_mse` — averaged variance, used purely for logging /
                        early-stopping checks; not differentiable
                        (`.item()` is called per class).
    """

    def __init__(self, feature_dim=2048, slot_dim=128, num_slots=8,
                 num_iterations=3, var_threshold=0.1, reg_strength=0.0,
                 log_entropy=1, num_classes=530, bank_size=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.slot_dim = slot_dim
        self.var_threshold = var_threshold
        self.reg_strength = reg_strength
        self.log_entropy = log_entropy
        # Small constant offset inside log() so the loss stays defined
        # when the empirical variance approaches zero.
        self.gamma = 0.01

        # Linear projection from the full feature width down to
        # `slot_dim`, where slot-attention runs. The projection is
        # learned end-to-end with the rest of the model.
        self.proj_down = nn.Linear(feature_dim, slot_dim)

        self.slot_attention = SlotAttention(slot_dim, num_slots, num_iterations)

        # Memory bank stores already-projected features (slot_dim, not
        # feature_dim) so that we never have to re-project bank entries
        # at lookup time.
        self.memory_bank = FeatureMemoryBank(num_classes, bank_size, slot_dim)

    def to(self, *args, **kwargs):
        """Override `.to()` so that the (non-Module) memory bank tags
        along with the rest of the model when we move to GPU."""
        result = super().to(*args, **kwargs)
        try:
            device = next(result.parameters()).device
            result.memory_bank.to(device)
        except StopIteration:
            # Module had no parameters — nothing to move.
            pass
        return result

    def forward(self, z_flat, labels, unique_labels):
        """
        Args
        ----
        z_flat        : [B, feature_dim] — flattened encoder output for
                        the current batch.
        labels        : [B]              — class label per sample.
        unique_labels : 1D tensor of class IDs present in this batch
                        (passed in to avoid recomputing on every call).

        Returns
        -------
        rob_loss        : scalar Tensor — the CEM penalty, averaged
                          across classes that had ≥ 2 features available.
        intra_class_mse : Python float  — averaged variance, for logs.
        """
        device = z_flat.device

        # Down-project. Gradients flow back through proj_down to the
        # encoder.
        z_proj = self.proj_down(z_flat)  # [B, slot_dim]

        # Update bank BEFORE clustering so that the very first sample of
        # a class still gets clustered against its previous occurrences
        # from earlier batches. The detach happens inside .update().
        self.memory_bank.update(z_proj, labels)

        # CEM-style threshold scales with the encoder noise σ.
        strength = self.reg_strength if self.reg_strength > 0 else 1.0
        threshold = self.var_threshold * (strength ** 2) + self.gamma

        if unique_labels is None:
            unique_labels = torch.unique(labels)

        loss_accum = 0.0
        var_accum = 0.0
        count = 0

        for label in unique_labels:
            label_val = label.item()
            mask = (labels == label)

            # batch_feats: HAS gradient — these are the features whose
            # variance the encoder is being asked to manage.
            batch_feats = z_proj[mask]

            # bank_feats: NO gradient — historic context that lets the
            # variance estimate make sense even when batch_feats is just
            # one sample.
            bank_feats = self.memory_bank.get_class_features(label_val)

            if bank_feats is not None and bank_feats.size(0) > 0:
                all_feats = torch.cat([batch_feats, bank_feats], dim=0)
            else:
                all_feats = batch_feats

            # Need at least two samples to talk about variance at all.
            if all_feats.size(0) < 2:
                continue

            # Slot-attention expects a [B_set, N, D] tensor — we set
            # B_set = 1 because we're processing one class at a time.
            feats_input = all_feats.unsqueeze(0)
            slots, attn_weights = self.slot_attention(feats_input)

            # Normalise the soft-assignment attention map to a proper
            # per-slot weighting over the N inputs.
            attn_sum = attn_weights.sum(dim=1, keepdim=True) + 1e-8
            attn_norm = attn_weights / attn_sum  # [1, N, S]

            # Soft centroids — detached so that the encoder cannot cheat
            # by moving the centroids towards the features (see class
            # docstring).
            centroids = torch.einsum('bns,bnd->bsd', attn_norm, feats_input).detach()

            # Per-slot geometric variance: weighted squared deviation of
            # the inputs from each centroid, averaged over the slot_dim.
            diff = feats_input.unsqueeze(2) - centroids.unsqueeze(1)  # [1,N,S,D]
            sq_diff = diff ** 2
            var_per_slot = torch.einsum('bns,bnsd->bs', attn_norm, sq_diff) / (self.slot_dim + 1e-8)

            # CEM penalty: ReLU keeps it one-sided. Variance above the
            # noise-scaled threshold is penalised; variance at or below
            # the threshold yields zero. Minimising this term therefore
            # contracts within-slot variation without rewarding further
            # contraction once the threshold has been reached.
            per_slot_loss = F.relu(
                torch.log(var_per_slot + self.gamma)
                - torch.log(torch.tensor(threshold, device=device, dtype=z_flat.dtype))
            )
            class_loss = per_slot_loss.mean()

            loss_accum += class_loss
            var_accum += var_per_slot.mean().item()
            count += 1

        rob_loss = loss_accum / max(count, 1)
        intra_class_mse = var_accum / max(count, 1)
        return rob_loss, intra_class_mse
def init_weights(m): # weight initialization
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()

def denormalize(x, dataset): # normalize a zero mean, std = 1 to range [0, 1]
    
    if dataset == "mnist" or dataset == "fmnist":
        return torch.clamp((x + 1)/2, 0, 1)
    elif dataset == "cifar10":
        std = [0.247, 0.243, 0.261]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "facescrub":
        std = (0.2058, 0.2275, 0.2098)
        mean = (0.5708, 0.5905, 0.4272)
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    elif dataset == "tinyimagenet":
        mean = (0.5141, 0.5775, 0.3985)
        std = (0.2927, 0.2570, 0.1434)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = (tensor[t]).mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)

def normalize(x, dataset): 
    if dataset == "mnist" or dataset == "fmnist":
        return torch.clamp(2 * x - 1, -1, 1)
    elif dataset == "cifar10":
        std = [0.247, 0.243, 0.261]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "facescrub":
        std = (0.2058, 0.2275, 0.2098)
        mean = (0.5708, 0.5905, 0.4272)
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = tensor[t].sub_(m).div_(s)
    # B, 3, H, W
    return torch.clamp(tensor, -1, 1).permute(3, 0, 1, 2)

def test_denorm(): # test function for denorm
    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    cifar10_training = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    cifar10_training_loader_iter = iter(DataLoader(cifar10_training, shuffle=False, num_workers=1, batch_size=128))
    transform_orig = transforms.Compose([
        transforms.ToTensor()
    ])
    cifar10_original = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_orig)
    cifar10_original_loader_iter = iter(DataLoader(cifar10_original, shuffle=False, num_workers=1, batch_size=128))
    images, _ = next(cifar10_training_loader_iter)
    orig_image, _  = next(cifar10_original_loader_iter)
    recovered_image = denormalize(images, "cifar100")
    return torch.isclose(orig_image, recovered_image)

def save_images(input_imgs, output_imgs, epoch, path, offset=0, batch_size=64): # saved image from tensor to jpg
    """
    """
    input_prefix = "inp_"
    output_prefix = "out_"
    out_folder = "{}/{}".format(path, epoch)
    out_folder = os.path.abspath(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for img_idx in range(output_imgs.shape[0]):
        inp_img_path = "{}/{}{}.jpg".format(out_folder, input_prefix, offset * batch_size + img_idx)
        out_img_path = "{}/{}{}.jpg".format(out_folder, output_prefix, offset * batch_size + img_idx)

        if input_imgs is not None:
            save_image(input_imgs[img_idx], inp_img_path)
        if output_imgs is not None:
            save_image(output_imgs[img_idx], out_img_path)
            



class MIA_train: # main class for every thing

    def __init__(self, arch, cutting_layer, batch_size, n_epochs, lambd=1, scheme="V2_epoch", num_client=1, dataset="cifar10",
                 logger=None, save_dir=None, regularization_option="None", regularization_strength=0, AT_regularization_option="None", AT_regularization_strength=0, log_entropy=0,
                 collude_use_public=False, initialize_different=False, learning_rate=0.1, local_lr=-1,
                 gan_AE_type="custom", random_seed=123, client_sample_ratio=1.0,
                 load_from_checkpoint=False, bottleneck_option="None", measure_option=False,
                 optimize_computation=1, decoder_sync=False, bhtsne_option=False, gan_loss_type="SSIM", attack_confidence_score=False,
                 ssim_threshold=0.0, var_threshold=0.1, attention_num_slots=8, attention_num_heads=4, attention_num_iterations=3,
                 attention_loss_scale=0.25, attention_warmup_epochs=3, attention_bank_size=64, attention_slot_dim=128, finetune_freeze_bn=False,
                 load_from_checkpoint_server=False, source_task="cifar100",
                 save_activation_tensor=False, save_more_checkpoints=False, dataset_portion=1.0, noniid=1.0):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.arch = arch
        self.bhtsne = bhtsne_option
        self.batch_size = batch_size
        self.lr = learning_rate
        self.finetune_freeze_bn = finetune_freeze_bn

        if local_lr == -1: # if local_lr is not set
            self.local_lr = self.lr
        else:
            self.local_lr = local_lr
        self.lambd=lambd
        self.n_epochs = n_epochs
        self.measure_option = measure_option
        self.optimize_computation = optimize_computation
       # self.client_sample_ratio = client_sample_ratio
       # self.dataset_portion = dataset_portion
       # self.noniid_ratio = noniid
        self.save_more_checkpoints = save_more_checkpoints

        # setup save folder
        if save_dir is None:
            self.save_dir = "./saves/{}/".format(datetime.today().strftime('%m%d%H%M'))
        else:
            self.save_dir = str(save_dir) + "/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # setup tensorboard
        tensorboard_path = str(save_dir) + "/tensorboard"
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        self.writer = SummaryWriter(log_dir=tensorboard_path)
        
        self.save_activation_tensor = save_activation_tensor

        # setup logger
        model_log_file = self.save_dir + '/MIA.log'
        if logger is not None:
            self.logger = logger
        else:
            self.logger = setup_logger('{}_logger'.format(str(save_dir)), model_log_file, level=logging.DEBUG)
        
        self.warm = 1
        self.current_epoch = 0
        self.scheme = scheme

        # migrate old naming:
        # if self.scheme == "V1" or self.scheme == "V2" or self.scheme == "V3" or self.scheme == "V4":
            # self.scheme = self.scheme + "_batch"

        # self.num_client = num_client
        self.num_client=num_client
        self.dataset = dataset
        self.call_resume = False

        self.load_from_checkpoint = load_from_checkpoint
        self.load_from_checkpoint_server = load_from_checkpoint_server
        self.source_task = source_task
        self.cutting_layer = cutting_layer

        if self.cutting_layer == 0:
            self.logger.debug("Centralized Learning Scheme:")
        if "resnet20" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/10".format(self.cutting_layer))
        if "vgg11" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/13".format(self.cutting_layer))
        if "mobilenetv2" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/9".format(self.cutting_layer))
        
        self.confidence_score = attack_confidence_score
        self.collude_use_public = collude_use_public
        self.initialize_different = initialize_different
        
        # from here on, use potentially updated self.arch consistently
        arch = self.arch

        if "C" in bottleneck_option or "S" in bottleneck_option:
            self.adds_bottleneck = True
            self.bottleneck_option = bottleneck_option
        else:
            self.adds_bottleneck = False
            self.bottleneck_option = bottleneck_option
        
        self.decoder_sync = decoder_sync

        ''' Activation Defense ''' ##this is kind of defense method
        self.regularization_option = regularization_option
        self.AT_regularization_option = AT_regularization_option
        
        # If strength is 0.0, then there is no regularization applied, train normally.
        self.regularization_strength = regularization_strength
        self.AT_regularization_strength = AT_regularization_strength
        
        self.log_entropy= log_entropy
        if self.regularization_strength == 0.0:
            self.regularization_option = "None"
        if self.AT_regularization_strength == 0.0:
            self.AT_regularization_option = "None"       

        # setup nopeek regularizer
        if "nopeek" in self.AT_regularization_option:
            self.nopeek = True
        else:
            self.nopeek = False

        if "SCA" in self.AT_regularization_option:
            self.SCA = True
        else:
            self.SCA = False   
        
        self.alpha1 = AT_regularization_strength  # set to 0.1 # 1000 in Official NoteBook https://github.com/tremblerz/nopeek/blob/master/noPeekCifar10%20(1)-Copy2.ipynb


        # setup gan_adv regularizer
        ## this is used to trian a robust model
        self.gan_AE_activation = "sigmoid"
        self.gan_AE_type = gan_AE_type
        self.gan_loss_type = gan_loss_type
        self.gan_decay = 0.2
        self.alpha2 = AT_regularization_strength  # set to 1~10
        self.pretrain_epoch = 100

        self.ssim_threshold = ssim_threshold
        self.var_threshold = var_threshold
        if "gan_adv" in self.AT_regularization_option:
            self.gan_regularizer = True
            if "step" in self.AT_regularization_option:
                try:
                    self.gan_num_step = int(self.AT_regularization_option.split("step")[-1])
                except:
                    print("Auto extract step fail, geting default value 3")
                    self.gan_num_step = 3
            else:
                self.gan_num_step = 3
            if "noise" in self.AT_regularization_option:
                self.gan_noise = True
            else:
                self.gan_noise = False
        else:
            self.gan_regularizer = False
            self.gan_noise = False
            self.gan_num_step = 1

        # setup local dp (noise-injection defense)
        if "local_dp" in self.AT_regularization_option:
            self.local_DP = True
        else:
            self.local_DP = False

        self.dp_epsilon = AT_regularization_strength

        if "dropout" in self.AT_regularization_option:
            self.dropout_defense = True
            try: 
                self.dropout_ratio = float(self.AT_regularization_option.split("dropout")[1].split("_")[0])
            except:
                self.dropout_ratio = AT_regularization_strength
                print("Auto extract dropout ratio fail, use regularization_strength input as dropout ratio")
            print('dropout_ratio is:',self.dropout_ratio)
        else:
            self.dropout_defense = False
            self.dropout_ratio = AT_regularization_strength

        if "topkprune" in self.AT_regularization_option:
            self.topkprune = True
            try: 
                self.topkprune_ratio = float(self.AT_regularization_option.split("topkprune")[1].split("_")[0])
            except:
                self.topkprune_ratio = AT_regularization_strength
                print("Auto extract topkprune ratio fail, use regularization_strength input as topkprune ratio")
        else:
            self.topkprune = False
            self.topkprune_ratio = AT_regularization_strength

        if "pruning" in self.AT_regularization_option:
            self.double_local_layer = True
        else:
            self.double_local_layer = False
        
        ''' Activation Defense (end)'''

        # Always use attention CEM surrogate to replace GMM (only module difference)
        self.use_attention_cem = True
        self.attention_cem = None
        self._attention_cem_registered = False
        self.attention_warmup_epochs = attention_warmup_epochs
        self.attention_loss_scale = attention_loss_scale
        self.attention_num_slots = attention_num_slots
        self.attention_num_heads = attention_num_heads
        self.attention_num_iterations = attention_num_iterations
        self.attention_bank_size = attention_bank_size
        self.attention_slot_dim = attention_slot_dim
        self._cem_fail_count = 0   # cumulative RuntimeError failures in CEM
        self._cem_nan_count = 0    # cumulative NaN/Inf failures in CEM
        self._cem_fail_this_epoch = 0  # reset each epoch for per-epoch summary


        # client sampling: dividing datasets to actual number of clients, self.num_clients is fake num of clients for ease of simulation.
        # multiplier = 1/self.client_sample_ratio #100
        # actual_num_users = int(multiplier * self.num_client)
        # self.
        actual_num_users = 1
        self.collude_use_public=False
        num_workers=8
        # setup dataset
        if self.dataset == "cifar10":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar10_trainloader(batch_size=self.batch_size,
                                                                                                        num_workers=num_workers,
                                                                                                        shuffle=True,
                                                                                                        num_client=actual_num_users,
                                                                                                        collude_use_public=self.collude_use_public
                                                                                                        )
            self.client_dataloader_rob, self.mem_trainloader_rob, self.mem_testloader_rob = get_cifar10_trainloader(batch_size=self.batch_size*20,
                                                                                                        num_workers=num_workers,
                                                                                                        shuffle=True,
                                                                                                        num_client=actual_num_users,
                                                                                                        collude_use_public=self.collude_use_public
                                                                                                        )
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar10_testloader(batch_size=self.batch_size,
                                                                                                        num_workers=num_workers,
                                                                                                        shuffle=False)
            self.orig_class = 10
            self.feature_size = 8
        elif self.dataset == "cifar100":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar100_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public)
            self.client_dataloader_rob, self.mem_trainloader_rob, self.mem_testloader_rob = get_cifar100_trainloader(batch_size=self.batch_size*20,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public)                                                                                                    
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar100_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=False)
            self.orig_class = 100
            self.feature_size = 8
        elif self.dataset == "svhn":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_SVHN_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_SVHN_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=False)
            self.orig_class = 10

        elif self.dataset == "facescrub":
            self.client_dataloader, self.pub_dataloader,_,_,_ = get_facescrub_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 530
            self.feature_size = 12

        elif self.dataset == "tinyimagenet":
            self.client_dataloader, self.pub_dataloader,self.AT_trainloader,_,_ = get_tinyimagenet_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 200
            self.feature_size = 16
            
        elif self.dataset == "imagenet":        
            self.client_dataloader, self.pub_dataloader,self.AT_trainloader,_,_ = get_imagenet_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 1000
        

        elif self.dataset == "mnist":
            self.client_dataloader, self.pub_dataloader = get_mnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10
        elif self.dataset == "fmnist":
            self.client_dataloader, self.pub_dataloader = get_fmnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=num_workers,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10

        elif self.dataset == "celeba":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_celeba_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_celeba_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=num_workers,
                                                                                                         shuffle=False)
            self.orig_class = 1000
            self.feature_size = 16
        else:
            raise ("Dataset {} is not supported!".format(self.dataset))
        self.num_class = self.orig_class
        self.num_batches = len(self.client_dataloader[0])
        # self.num_batches_rob = len(self.client_dataloader_rob[0])
        print("Total number of batches per epoch for each client is ", self.num_batches)
        for images, labels in self.client_dataloader[0]:
            sample_image = images[0]
            self.sample_image=sample_image
            # print("Sample image shape:", sample_image.shape)
            self.recons_dim=sample_image.shape[-1]
            if sample_image.shape[-1]>63:
                self.upsize=True
            else:
                self.upsize=False
            print(sample_image.shape[-1])
            break
        self.model = None

        # print('dose self upsize?:',self.upsize)
        # Initialze all client, server side models.

        self.initialize_different = False

        if arch == "imagenet_resnet20":
            model = Imagenet_ResNet20(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer,upsize=self.upsize)
        elif arch == "resnet20":
            model = ResNet20(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer,upsize=self.upsize,SCA=self.SCA)
        elif arch == "resnet32":
            model = ResNet32(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer,upsize=self.upsize)
        elif arch == "vgg13":
            model = vgg13(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                          initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer)
        elif arch == "vgg11":
            # print('dose upsize?',upsize)
            model = vgg11(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                          initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer,upsize=self.upsize)
        elif arch == "vgg13_bn":
            model = vgg13_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer)
        elif arch == "vgg11_bn":
            model = vgg11_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer)
        elif arch == "vgg11_bn_sgm":
            # print('the value of SCA is:',self.SCA)
            model = vgg11_bn_sgm(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option, double_local_layer=self.double_local_layer,upsize=self.upsize,SCA=self.SCA,feature_size=self.feature_size)
        elif arch == "mobilenetv2":
            model = MobileNetV2(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                             initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
        else:
            raise ("No such architecture!")
        self.model = model

        self.f = model.local_list[0]
        self.f_tail = model.cloud
        self.classifier = model.classifier
        self.f.cuda()
        self.f_tail.cuda()
        self.classifier.cuda()
        self.params = list(self.f_tail.parameters()) + list(self.classifier.parameters())
        self.local_params = []
        if cutting_layer > 0:
            self.local_params.append(self.f.parameters())

    


        # setup optimizers
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if self.load_from_checkpoint:
            milestones = [30, 50, 70]
        else:
            milestones = [60, 120, 180, 210, 260] #optimize rob from 210 epoch
        self.save_freq=50
        if n_epochs==120:
            self.save_more_checkpoints=True
            milestones = [30, 60, 90, 105]


        self.local_optimizer_list = []
        self.train_local_scheduler_list = []
        self.warmup_local_scheduler_list = []
        for i in range(len(self.local_params)):
            self.local_optimizer_list.append(torch.optim.SGD(list(self.local_params[i]), lr=self.local_lr, momentum=0.9, weight_decay=5e-4))
            self.train_local_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.local_optimizer_list[i], milestones=milestones,
                                                                    gamma=0.2))  # learning rate decay
            self.warmup_local_scheduler_list.append(WarmUpLR(self.local_optimizer_list[i], self.num_batches * self.warm))

        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                    gamma=0.2)  # learning rate decay
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.num_batches * self.warm)
        
        # Set up GAN_ADV, used to training the encoder
        self.local_AE_list = []
        self.gan_params = []
        if self.gan_regularizer:
            self.feature_size = self.model.get_smashed_data_size()

            if self.gan_AE_type == "custom":
                self.local_AE_list.append(
                    architectures.custom_AE(input_nc=self.feature_size[1], output_nc=3, input_dim=self.feature_size[2],
                                            output_dim=self.recons_dim, activation=self.gan_AE_activation))
                # print('this is custom')
                # raise LinAlgError
            elif "conv_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("conv_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from conv_normN failed, set N to default 0")
                    N = 0
                    internal_C = 64
                self.local_AE_list.append(architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=self.feature_size[1], output_nc=3,
                                                         input_dim=self.feature_size[2]*(sample_image.shape[-1]/32), output_dim=sample_image.shape[-1],
                                                         activation=self.gan_AE_activation))
                print('this is conv_norm')
            elif "res_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("res_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from res_normN failed, set N to default 0")
                    N = 0
                    internal_C = 64
                self.local_AE_list.append(architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=self.feature_size[1], output_nc=3,
                                                         input_dim=self.feature_size[2]*(sample_image.shape[-1]/32), output_dim=sample_image.shape[-1],
                                                         activation=self.gan_AE_activation))
                # print('this is res_norm')
            else:
                raise ("No such GAN AE type.")
            self.gan_params.append(self.local_AE_list[i].parameters())
            self.local_AE_list[i].apply(init_weights)
            self.local_AE_list[i].cuda()
            
            self.gan_optimizer_list = []
            self.gan_scheduler_list = []
            if self.load_from_checkpoint:
                milestones = [30, 50, 70]
            else:
                milestones =  [60, 150, 240, 260]

            if n_epochs==120:
                milestones = [30, 60, 90]


            for i in range(len(self.gan_params)):
                self.gan_optimizer_list.append(torch.optim.Adam(list(self.gan_params[i]), lr=1e-3))
                self.gan_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.gan_optimizer_list[i], milestones=milestones,
                                                                      gamma=self.gan_decay))  # learning rate decay
            
    def optimizer_step(self, set_client = False, client_id = 0):
        
        for i in range(len(self.local_optimizer_list)):
            self.local_optimizer_list[i].step()
        self.optimizer.step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
        for i in range(len(self.local_optimizer_list)):
            self.local_optimizer_list[i].zero_grad()

    def _flatten_for_classifier(self, output):
        if output.dim() == 4:
            output = F.adaptive_avg_pool2d(output, 1)
        return output.view(output.size(0), -1)

    def scheduler_step(self, epoch = 0, warmup = False):
        if warmup:
            self.warmup_scheduler.step()
            for i in range(len(self.warmup_local_scheduler_list)):
                self.warmup_local_scheduler_list[i].step()
        else:
            self.train_scheduler.step(epoch)
            for i in range(len(self.train_local_scheduler_list)):
                self.train_local_scheduler_list[i].step(epoch)

    def gan_scheduler_step(self, epoch = 0):
        for i in range(len(self.gan_scheduler_list)):
            self.gan_scheduler_list[i].step(epoch)
    def kmeans_plusplus_init(self, X, num_clusters):
        """
        KMeans++ initialization algorithm.
        
        Args:
        - X (torch.Tensor): Input tensor of shape (N, D)
        - num_clusters (int): Number of clusters
        
        Returns:
        - centroids (torch.Tensor): Initial centroids
        """
        N, D = X.shape
        centroids = torch.empty((num_clusters, D), device=X.device)
        centroids[0] = X[torch.randint(0, N, (1,))]

        for i in range(1, num_clusters):
            distances = torch.cdist(X, centroids[:i]).min(dim=1)[0]
            probs = distances / distances.sum()
            centroids[i] = X[torch.multinomial(probs, 1)]

        return centroids
    def kmeans_cuda(self,X, num_clusters,centroids,random_ini_centers, num_iterations=10, tol=1e-4):
        N, D = X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]
        X_flat = X.reshape(N, D)  # flatten

        # random initialize 
        cluster_start_time= time.time()
        if random_ini_centers :
            print('randomized selected centroids')
        # centroids = X_flat[torch.randperm(N)[:num_clusters]].clone()
            centroids = self.kmeans_plusplus_init(X_flat, num_clusters).clone()
        # if torch.isnan(centroids).any() > 0:
        #     centroids = X_flat[torch.randperm(N)[:num_clusters]].clone()
        # else:            
        #     # print(torch.isnan(centroids))
        #     if len(centroids) > num_clusters:
        #         indices = torch.randperm(len(centroids))[:num_clusters]
        #         centroids = centroids[indices]
        #     elif len(centroids) < num_clusters:
        #         indices = torch.randint(low=0, high=len(centroids), size=(num_clusters,))
        #         centroids = centroids[indices]
                # else:
                #     # 如果 n == k，直接使用原 tensor
                #     result_tensor = tensor
        random_initial_time= time.time()
        # print('the random ini time is:',random_initial_time-cluster_start_time)
        labels = torch.zeros(N, dtype=torch.long, device=X.device)

        for _ in range(num_iterations):
            #calculate distance
            distances = torch.cdist(X_flat, centroids)
            new_labels = distances.argmin(dim=1)

            unique_cluster_assignments = torch.unique(new_labels)
            if torch.equal(labels, new_labels):
                break
            labels = new_labels
            unique_cluster_assignments = torch.unique(labels)
            
            
            for i in unique_cluster_assignments:
                centroids[i] = X_flat[labels == i].mean(dim=0)
        
        cluster_time= time.time()

        # print('the cluster_time is:',cluster_time-random_initial_time)
        # calculate variance
        # cluster_variances = torch.tensor([((X_flat[labels == i] - centroids[i])**2).mean() for i in range(num_clusters)], device=X.device)
        
        cluster_variances = torch.tensor([((X_flat[labels == i] - centroids[i])**2).mean() for i in unique_cluster_assignments], device=X_flat.device)
        average_variance = cluster_variances.mean().item()
        cluster_covariances = []
        cluster_weights = []
        for i in unique_cluster_assignments:
            cluster_data = X_flat[labels == i]
            if cluster_data.size(0) > 1:
                mean = cluster_data.mean(dim=0)
                cov = torch.mm((cluster_data - mean).t(), (cluster_data - mean)) / (cluster_data.size(0) - 1)
                weight = torch.tensor(cluster_data.size(0) / N)

                cluster_covariances.append(cov)
                cluster_weights.append(weight)       
        if torch.isnan(cluster_variances.mean()).any():
            print(cluster_variances,unique_cluster_assignments,centroids.mean()) 
        cluster_covariances=torch.stack(cluster_covariances)
        cluster_weights = torch.stack(cluster_weights)

        if cluster_variances.size(0) < centroids.size(0):
            last_variance = cluster_variances[-1]
            num_to_add = centroids.size(0) - cluster_variances.size(0)
            cluster_variances = torch.cat([cluster_variances, last_variance.repeat(num_to_add)])
        
        if cluster_weights.size(0) < centroids.size(0):
            num_to_add = centroids.size(0) - cluster_weights.size(0)
            additional_weights = torch.full((num_to_add,), 0.01)
            cluster_weights = torch.cat([cluster_weights, additional_weights])
        
        variance_calculating_time= time.time()
        # print('the variance_time is:',variance_calculating_time-cluster_time)
        return centroids, cluster_variances, cluster_covariances,cluster_weights

    # def compute_class_means(self, features, labels, unique_labels, centroids_list):
    #     class_means = []

    #     # Sort labels and get sorted indices
    #     sorted_indices = torch.argsort(labels)
    #     sorted_labels = labels[sorted_indices]
    #     sorted_features = features[sorted_indices]

    #     unique_labels = unique_labels.cpu().numpy()
    #     sorted_labels = sorted_labels.cpu().numpy()

    #     N, D = sorted_features.shape[0], sorted_features[0].numel()
    #     sorted_features_flat = sorted_features.view(N, D)  # Flatten
    #     num_centroids = len(centroids_list[0])
    #     centroids = torch.zeros((N, num_centroids, D), dtype=sorted_features_flat.dtype)

    #     # Assign the corresponding centroids to the centroids tensor
    #     for i in range(len(sorted_labels)):
    #         label = sorted_labels[i]
    #         centroids[i] = centroids_list[label]
    #     distances = torch.cdist(sorted_features_flat.unsqueeze(1), centroids, p=2)
    #     cluster_sorted_indices = distances.argmin(dim=2).squeeze()


    #     label_it = 0
    #     current_index = 0
    #     for i in unique_labels:
    #         centroids = centroids_list[i]
    #         num_clusters = centroids.size(0)

    #         # Get the range of indices corresponding to the current label
    #         start_index = current_index
    #         while current_index < len(sorted_labels) and sorted_labels[current_index] == i:
    #             current_index += 1
    #         end_index = current_index

    #         # Get the features for the current class
    #         class_features = sorted_features[start_index:end_index]

    #         class_mean = class_features.mean(dim=0)

    #         N, D = class_features.shape[0], class_features[0].numel()
    #         class_features_flat = class_features.view(N, D)  # Flatten

    #         distances = torch.cdist(class_features_flat, centroids).detach().cpu().numpy()
    #         cluster_assignments = np.argmin(distances, axis=1)

    #         # Sort cluster assignments and get sorted indices
    #         cluster_sorted_indices = np.argsort(cluster_assignments)
    #         sorted_cluster_assignments = cluster_assignments[cluster_sorted_indices]
    #         sorted_class_features_flat = class_features_flat[cluster_sorted_indices]

    #         unique_cluster_assignments = np.unique(sorted_cluster_assignments)

    #         num = 0
    #         cluster_current_index = 0
    #         for j in unique_cluster_assignments:
    #             # Get the range of indices corresponding to the current cluster
    #             cluster_start_index = cluster_current_index
    #             while (cluster_current_index < len(sorted_cluster_assignments) and
    #                 sorted_cluster_assignments[cluster_current_index] == j):
    #                 cluster_current_index += 1
    #             cluster_end_index = cluster_current_index

    #             # Get the features for the current cluster
    #             indice_cluster = range(cluster_start_index, cluster_end_index)
    #             weight = (cluster_end_index - cluster_start_index) / (end_index - start_index)
    #             variances = torch.mean((sorted_class_features_flat[indice_cluster] - centroids[j]) ** 2, dim=0).cuda()

    #             reg_variances = variances + 0.001
    #             mean_reg_variances = reg_variances.mean() * weight

    #             mutual_infor = F.relu(torch.log(reg_variances + 0.0001) - torch.log(torch.tensor(0.001)))
    #             reg_mutual_infor = mutual_infor.mean() * weight

    #             if num == 0:
    #                 average_variance = reg_mutual_infor if self.log_entropy == 1 else mean_reg_variances
    #             else:
    #                 average_variance += reg_mutual_infor if self.log_entropy == 1 else mean_reg_variances

    #             num += 1

    #         if label_it == 0:
    #             intra_class_mse = average_variance
    #         else:
    #             intra_class_mse += average_variance

    #         class_means.append(class_mean)
    #         label_it += 1

    #     intra_class_mse /= len(unique_labels)
    #     class_means = torch.stack(class_means)
    #     class_mean_overall = class_means.mean(dim=0)
    #     inter_class_mse = F.mse_loss(class_means, class_mean_overall.expand_as(class_means))
    #     loss = intra_class_mse

    #     return loss, intra_class_mse

    def compute_class_means(self, features, labels, unique_labels, centroids_list, weights_list,cluster_variances_list):
        class_means = []

        len_dataset= len(self.client_dataloader[0])*self.batch_size
        class_length = len_dataset/self.num_class
        adaptive_avg_pool = nn.AdaptiveAvgPool2d((16, 16))
        if self.pooling:
            features=adaptive_avg_pool(features)*3.4
        # Sort labels and get sorted indices
        sorted_indices = torch.argsort(labels)
        sorted_labels = labels[sorted_indices]
        sorted_features = features[sorted_indices]

        unique_labels = unique_labels.cpu().numpy()
        sorted_labels = sorted_labels.cpu().numpy()

        N, D = sorted_features.shape[0], sorted_features[0].numel()
        sorted_features_flat = sorted_features.view(N, D)  # Flatten
        num_centroids = len(centroids_list[0])
        assert len(centroids_list[0])==len(weights_list[0]), print('the weights and centroids not match')
        # centroids_all = torch.zeros((N, num_centroids, D), dtype=sorted_features_flat.dtype)

        # # Assign the corresponding centroids to the centroids tensor
        # for i in range(len(sorted_labels)):
        #     label = sorted_labels[i]
        #     centroids_all[i] = centroids_list[label].cuda()

        label_indices = torch.tensor(sorted_labels).long()
# 
        centroids_all = torch.stack([centroids_list[label] for label in label_indices], dim=0)
        weights_all = torch.stack([weights_list[label] for label in label_indices], dim=0)
        cluster_variances_all = torch.stack([cluster_variances_list[label] for label in label_indices], dim=0)

        distances = torch.cdist(sorted_features_flat.unsqueeze(1), centroids_all.cuda(), p=2)

        distances=distances.squeeze(1).detach().cpu().numpy()

        all_cluster_assignments = np.argmin(distances, axis=1)
        nearest_centroids = centroids_all[torch.arange(N), all_cluster_assignments].cuda()
        # nearest_centroids = torch.zeros_like(sorted_features_flat)
        # for i in range(N):
        #     nearest_centroids[i] = centroids_all[i, all_cluster_assignments[i]]

        # Calculate the variance matrix
        variance_matrix = (sorted_features_flat - nearest_centroids) ** 2

        label_it = 0
        current_index = 0
        lb_count=0
        for i in unique_labels:
            weights_label= weights_all[lb_count]
            variance_label= cluster_variances_all[lb_count]
            # Get the range of indices corresponding to the current label
            start_index = current_index
            while current_index < len(sorted_labels) and sorted_labels[current_index] == i:
                current_index += 1
            end_index = current_index
            lb_count+=1
            # Get the features for the current class
            # class_features_flat = sorted_features_flat[start_index:end_index]
            
            # class_mean = class_features_flat.mean(dim=0)

            class_variance_matrix = variance_matrix[start_index:end_index]
            cluster_assignments = all_cluster_assignments[start_index:end_index]

            # Sort cluster assignments and get sorted indices
            cluster_sorted_indices = np.argsort(cluster_assignments)
            sorted_cluster_assignments = cluster_assignments[cluster_sorted_indices]
            sorted_class_variance_matrix = class_variance_matrix[cluster_sorted_indices]
# 
            unique_cluster_assignments = np.unique(sorted_cluster_assignments)

            num = 0
            cluster_current_index = 0
            for j in unique_cluster_assignments:
                # Get the range of indices corresponding to the current cluster
                cluster_start_index = cluster_current_index
                while (cluster_current_index < len(sorted_cluster_assignments) and sorted_cluster_assignments[cluster_current_index] == j):
                    cluster_current_index += 1
                cluster_end_index = cluster_current_index

                # Get the features for the current cluster
                
                # weight = (cluster_end_index - cluster_start_index) / (end_index - start_index)
                weight= weights_label[j]
                variance = variance_label[j]
                totol_number= weight*class_length
                num_samples= (cluster_end_index - cluster_start_index)
                scaling_lambda= 10
                var_lambda= (num_samples/totol_number)*scaling_lambda
                if var_lambda>=1:
                    var_lambda=0.99

                variances = torch.mean(sorted_class_variance_matrix[cluster_start_index:cluster_end_index]).cuda() #variance is the mean of the cluster distance
                variances = var_lambda*variances + (1-var_lambda)*variance

                if self.log_entropy == 0:
                    reg_variances = variances + 0.001
                    reg_mutual_infor = reg_variances.mean() * weight
                else:
                    gamma=0.01
                    mutual_infor = F.relu(torch.log(variances + gamma) - torch.log(self.var_threshold*torch.tensor(self.regularization_strength**2)+ 1*gamma))
                    reg_mutual_infor = mutual_infor.mean() * weight
                # print(totol_number,num_samples,weight) 
                if torch.isnan(reg_mutual_infor):
                    print(variances,num_samples,totol_number,var_lambda,variance) 
                    print('rob_loss is nan')


                if num == 0:
                    average_variance = reg_mutual_infor 
                else:
                    average_variance += reg_mutual_infor 

                num += 1

            if label_it == 0:
                intra_class_mse = average_variance
            else:
                intra_class_mse += average_variance

            # class_means.append(class_mean)
            label_it += 1

        intra_class_mse /= len(unique_labels)
        if torch.isnan(intra_class_mse):
            print(len(unique_labels)) 
            print('rob_loss is nan')
        # class_means = torch.stack(class_means)
        # class_mean_overall = class_means.mean(dim=0)
        # inter_class_mse = F.mse_loss(class_means, class_mean_overall.expand_as(class_means))
        loss = intra_class_mse

        return loss, intra_class_mse

        # return torch.tensor(0.01), torch.tensor(0.01)


    '''Main training function, the communication between client/server is implicit to keep a fast training speed'''
    def train_target_step(self, x_private, label_private, adding_noise, random_ini_centers, centroids_list, weights_list, cluster_variances_list, client_id=0):
        self.optimizer_zero_grad()
        self.f_tail.train()
        self.classifier.train()
        self.f.train()
        x_private = x_private.cuda()
        label_private = label_private.cuda()

        # Freeze batchnorm parameter of the client-side model.
        if self.load_from_checkpoint and self.finetune_freeze_bn:
            freeze_model_bn(self.f)


        z_private = self.f(x_private)

        # print(z_private.shape)

        unique_labels = torch.unique(label_private)

        device = z_private.device
        use_attention = (
            self.use_attention_cem
            and not random_ini_centers
            and self.lambd > 0
            and self.current_epoch > self.attention_warmup_epochs
        )

        # Flatten to sample-level features: [B, feature_dim] (e.g. 2048 for FaceScrub 64x64)
        feats_flat = z_private.view(z_private.size(0), -1)

        # Eagerly create attention_cem on FIRST training step (not just post-warmup)
        # so the memory bank can be pre-warmed during warmup epochs
        if self.attention_cem is None and self.use_attention_cem and self.lambd > 0:
            reg_strength = self.regularization_strength if self.regularization_strength > 0 else 1.0
            self.attention_cem = SlotCrossAttentionCEM(
                feature_dim=feats_flat.size(-1),
                slot_dim=self.attention_slot_dim,
                num_slots=self.attention_num_slots,
                num_iterations=self.attention_num_iterations,
                var_threshold=self.var_threshold,
                reg_strength=reg_strength,
                log_entropy=self.log_entropy,
                num_classes=self.num_class,
                bank_size=self.attention_bank_size,
            ).to(device)  # .to() override also moves memory_bank to device
            if not self._attention_cem_registered:
                base_group = self.optimizer.param_groups[0]
                self.optimizer.add_param_group({
                    'params': self.attention_cem.parameters(),
                    'lr': base_group.get('lr', self.lr),
                    'momentum': base_group.get('momentum', 0.0),
                    'weight_decay': base_group.get('weight_decay', 0.0),
                })
                self._attention_cem_registered = True
            self.logger.debug(f"Attention CEM created: feature_dim={feats_flat.size(-1)}, "
                              f"slot_dim={self.attention_slot_dim}, device={device}")

        # Pre-warm memory bank during warmup (forward() also calls update, so skip here post-warmup)
        if self.attention_cem is not None and not use_attention:
            with torch.no_grad():
                z_proj_warmup = self.attention_cem.proj_down(feats_flat)
                self.attention_cem.memory_bank.update(z_proj_warmup, label_private)

        if use_attention:
            if torch.isnan(feats_flat).any() or torch.isinf(feats_flat).any():
                print("Warning: NaN/Inf detected in features, skipping CEM calculation")
                rob_loss = torch.zeros((), device=device)
                intra_class_mse = torch.zeros((), device=device)
            else:
                # Update dynamic params each step
                reg_strength = self.regularization_strength if self.regularization_strength > 0 else 1.0
                self.attention_cem.var_threshold = self.var_threshold
                self.attention_cem.reg_strength = reg_strength
                self.attention_cem.log_entropy = self.log_entropy
                self.attention_cem.train()
                try:
                    rob_loss, intra_class_mse = self.attention_cem(feats_flat, label_private, unique_labels)
                except RuntimeError as e:
                    import traceback
                    self._cem_fail_count += 1
                    self._cem_fail_this_epoch += 1
                    tb_str = traceback.format_exc()
                    # Always log full traceback at debug level
                    self.logger.debug(f"CEM RuntimeError #{self._cem_fail_count}: {e}\n{tb_str}")
                    # Prominent WARNING on first failure and every 50 thereafter
                    if self._cem_fail_count == 1 or self._cem_fail_count % 50 == 0:
                        self.logger.warning(
                            "\n" + "=" * 70 + "\n"
                            f"  *** CEM FAILURE #{self._cem_fail_count} ***  epoch={self.current_epoch}\n"
                            f"  RuntimeError: {e}\n"
                            f"  rob_loss has been set to 0 — CEM defense IS NOT ACTIVE this step!\n"
                            f"  Check debug log for full traceback.\n"
                            + "=" * 70
                        )
                    rob_loss = torch.zeros((), device=device, requires_grad=False)
                    intra_class_mse = torch.zeros((), device=device)
                else:
                    if not isinstance(rob_loss, torch.Tensor):
                        rob_loss = torch.tensor(rob_loss, device=device, dtype=torch.float32)
                    if not isinstance(intra_class_mse, torch.Tensor):
                        intra_class_mse = torch.tensor(intra_class_mse, device=device, dtype=torch.float32)
                    rob_loss = rob_loss * self.attention_loss_scale
                    if torch.isnan(rob_loss) or torch.isinf(rob_loss):
                        self._cem_nan_count += 1
                        self._cem_fail_this_epoch += 1
                        if self._cem_nan_count == 1 or self._cem_nan_count % 50 == 0:
                            self.logger.warning(
                                "\n" + "=" * 70 + "\n"
                                f"  *** CEM NaN/Inf in rob_loss #{self._cem_nan_count} ***  epoch={self.current_epoch}\n"
                                f"  rob_loss has been set to 0 — CEM defense IS NOT ACTIVE this step!\n"
                                + "=" * 70
                            )
                        rob_loss = torch.zeros((), device=device)
                    if torch.isnan(intra_class_mse) or torch.isinf(intra_class_mse):
                        intra_class_mse = torch.zeros((), device=device)
        else:
            rob_loss = torch.zeros((), device=device)
            intra_class_mse = torch.zeros((), device=device)
        # assert 1==0, print(x_private.shape,label_private.shape,unique_values)
        # Final Prediction Logits (complete forward pass)
        if "Gaussian" in self.regularization_option: # and adding_noise:
            if not random_ini_centers:
                if intra_class_mse<1:
                    sigma = self.regularization_strength#*intra_class_mse
                else:
                    sigma = self.regularization_strength
            else:
                sigma = self.regularization_strength
            noise = sigma * torch.randn_like(z_private).cuda()
            # z_private_c=z_private
            z_private_n =z_private + noise
        else:
            z_private_n=z_private
        # Perform various activation defenses, default no defense
        if self.local_DP:
            if "laplace" in self.AT_regularization_option:
                noise = torch.from_numpy(
                    np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=z_private_n.size())).cuda()
                z_private_n = z_private_n + noise.detach().float()
            else:  # apply gaussian noise
                delta = 10e-5
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                noise = sigma * torch.randn_like(z_private_n).cuda()
                z_private_n = z_private_n + noise.detach().float()
        if self.dropout_defense:
            z_private_n = dropout_defense(z_private_n, self.dropout_ratio)
        if self.topkprune:
            z_private_n = prune_defense(z_private_n, self.topkprune_ratio)
        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private_n.clone()
            grad = torch.zeros_like(z_private_n).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)
            x_private = denormalize(x_private, self.dataset)
            
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)  
            z_private_n = z_private_n - grad.detach() * epsilon

        output = self.f_tail(z_private_n)

        # print(output.shape)

        if "mobilenetv2" in self.arch:
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        elif self.arch == "resnet20" or self.arch == "resnet32":
            # output = F.avg_pool2d(output, 8)
            output = F.adaptive_avg_pool2d(output,(1,1))
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        elif self.arch == "imagenet_resnet20":
            output = F.adaptive_avg_pool2d(output, 1)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        else:
            output = self._flatten_for_classifier(output)
            output = self.classifier(output)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        if not random_ini_centers:
            total_loss = f_loss#+2*rob_loss
        else:    
            total_loss = f_loss#+0*rob_loss


        # perform nopeek regularization
        if self.nopeek:
            #
            if "ttitcombe" in self.AT_regularization_option:
                dc = DistanceCorrelationLoss()
                dist_corr_loss = 0.1*self.alpha1 * dc(x_private, z_private)
            else:
                dist_corr_loss = 0.1*self.alpha1 * dist_corr(x_private, z_private).sum()
            if self.dataset=='cifar10':
                total_loss = total_loss + 10*dist_corr_loss
            else:
                total_loss = total_loss + dist_corr_loss
            # print (dist_corr_loss)
        
        # perform our proposed attacker-aware training
        if self.gan_regularizer and not self.gan_noise:
            self.local_AE_list[client_id].eval()
            output_image = self.local_AE_list[client_id](z_private)
            
            x_private = denormalize(x_private, self.dataset)
            
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                ssim_term = ssim_loss(output_image, x_private)
                
                if self.ssim_threshold > 0.0:
                    if ssim_term > self.ssim_threshold:
                        gan_loss = self.alpha2 * (ssim_term - self.ssim_threshold) # Let SSIM approaches 0.4 to avoid overfitting
                    else:
                        gan_loss = 0.0 # Let SSIM approaches 0.4 to avoid overfitting
                else:
                    gan_loss = self.alpha2 * ssim_term  
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                mse_term = mse_loss(output_image, x_private)
                gan_loss = - self.alpha2 * mse_term  
            
            total_loss = total_loss + gan_loss
            
        # print(total_loss, f_loss)
       
        attention_params = []
        if use_attention and self.attention_cem is not None:
            attention_params = list(self.attention_cem.parameters())
        attention_gradients = None
        encoder_gradients = {}

        if not random_ini_centers and self.lambd>0 and rob_loss.requires_grad:
            try:
                # print(rob_loss)
                rob_loss.backward(retain_graph=True)
                for name, param in self.f.named_parameters():
                    if param.grad is not None:
                        encoder_gradients[name] = param.grad.clone()
                if attention_params:
                    attention_gradients = []
                    for param in attention_params:
                        if param.grad is not None:
                            attention_gradients.append(param.grad.clone())
                        else:
                            attention_gradients.append(None)
                # optimizer.zero_grad()
                self.optimizer_zero_grad()
            except Exception as e:
                print(f"Error in rob_loss backward: {e}")
                encoder_gradients = {}
                attention_gradients = None

        total_loss.backward()

        if not random_ini_centers and self.lambd>0 and encoder_gradients:
            for name, param in self.f.named_parameters():
                if param.grad is None or name not in encoder_gradients:
                    continue
                try:
                    if self.load_from_checkpoint:
                        param.grad += self.lambd*encoder_gradients[name]
                    else:
                        if (self.train_scheduler.get_last_lr()[0])<0.00041: #strat to enhance rob when lr is small and acc is high
                            param.grad += self.lambd*encoder_gradients[name]
                        else:
                            param.grad +=self.lambd*encoder_gradients[name]*(0.001/self.train_scheduler.get_last_lr()[0])
                except Exception as e:
                    print(f"Error adding rob gradients for {name}: {e}")

        if attention_gradients is not None and attention_params:
            for param, grad in zip(attention_params, attention_gradients):
                if grad is None:
                    continue
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad += grad

        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        return total_losses, f_losses, z_private

    # Main function for validation accuracy, is also used to get statistics
    def validate_target(self, client_id=0):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        val_loader = self.pub_dataloader
        # val_loader = self.AT_trainloader

        # switch to evaluate mode
        # self.f.eval()
        if client_id == 0:
            self.f.eval()
        elif client_id == 1:
            self.c.eval()
        elif client_id > 1:
            self.model.local_list[client_id].eval()
        self.f_tail.eval()
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss()

        activation_0 = {}

        def get_activation_0(name):
            def hook(model, input, output):
                activation_0[name] = output.detach()

            return hook
            # with torch.no_grad():

            #     count = 0
            #     for name, m in self.model.cloud.named_modules():
            #         if attack_from_later_layer == count:
            #             m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
            #             valid_key = "ACT-{}".format(name)
            #             break
            #         count += 1
            #     output = self.model.cloud(ir)

            # ir = activation_4[valid_key]

        # for name, m in self.model.local_list[client_id].named_modules():
        #     m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))

        # for name, m in self.f_tail.named_modules():
        #     m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))


        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            activation_0 = {}
            # compute output
            with torch.no_grad():

                output = self.model.local_list[client_id](input)
                # output = self.f(input)
                # code for save the activation of cutlayer
                

                if self.bhtsne:
                    self.save_activation_bhtsne(output, target, client_id)
                    exit()
                if "Gaussian" in self.regularization_option:
                    sigma = self.regularization_strength
                    noise = sigma * torch.randn_like(output).cuda()
                    output += noise
                '''Optional, Test validation performance with local_DP/dropout (apply DP during query)'''
                if self.local_DP:
                    if "laplace" in self.AT_regularization_option:
                        noise = torch.from_numpy(
                            np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=output.size())).cuda()
                    else:  # apply gaussian noise
                        delta = 10e-5
                        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                        noise = sigma * torch.randn_like(output).cuda()
                    output += noise
                if self.dropout_defense:
                    output = dropout_defense(output, self.dropout_ratio)
                if self.topkprune:
                    output = prune_defense(output, self.topkprune_ratio)
            
            '''Optional, Test validation performance with gan_noise (apply gan_noise during query)'''
            if self.gan_noise:
                epsilon = self.alpha2
                
                self.local_AE_list[client_id].eval()
                fake_act = output.clone()
                grad = torch.zeros_like(output).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = self.local_AE_list[client_id](fake_act)
                
                input = denormalize(input, self.dataset)

                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, input)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(x_recon, input)
                    loss.backward()
                    grad += torch.sign(fake_act.grad) 

                output = output - grad.detach() * epsilon
            
            with torch.no_grad():
                output = self.f_tail(output)

                if "mobilenetv2" in self.arch:
                    output = F.avg_pool2d(output, 4)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    # output = F.avg_pool2d(output, 8)
                    output = F.adaptive_avg_pool2d(output,(1,1))
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                elif self.arch == "imagenet_resnet20":
                    output = F.adaptive_avg_pool2d(output, 1)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                else:
                    output = self._flatten_for_classifier(output)
                    output = self.classifier(output)
                loss = criterion(output, target)


            # Get statistics of server/client's per-layer activation
            # if i == 0:
            #     try:
            #         if not os.path.exists(self.save_dir):
            #             os.makedirs(self.save_dir)

            #         # setup tensorboard
            #         if self.save_activation_tensor:
            #             save_tensor_path = self.save_dir + "/saved_tensors"
            #             if not os.path.isdir(save_tensor_path):
            #                 os.makedirs(save_tensor_path)
            #         for key, value in activation_0.items():
            #             if "client" in key:
            #                 self.writer.add_histogram("local_act/{}".format(key), value.clone().cpu().data.numpy(), i)
            #                 if self.save_activation_tensor:
            #                     np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
            #             if "server" in key:
            #                 self.writer.add_histogram("server_act/{}".format(key), value.clone().cpu().data.numpy(), i)
            #                 if self.save_activation_tensor:
            #                     np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
                    
            #         for name, m in self.model.local_list[client_id].named_modules():
            #             handle = m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))
            #             handle.remove()
            #         for name, m in self.f_tail.named_modules():
            #             handle = m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))
            #             handle.remove()
            #     except:
            #         print("something went wrong adding histogram, ignore it..")

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            # prec1 = accuracy(output.data, target, compress_V4shadowlabel=self.V4shadowlabel, num_client=self.num_client)[0] #If V4shadowlabel is activated, add one extra step to process output back to orig_class
            prec1 = accuracy(output.data, target)[
                0]  # If V4shadowlabel is activated, add one extra step to process output back to orig_class
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time

            if i % 50 == 0:
                self.logger.debug('Epoch {ep}\t''Test (client-{0}):\t'
                                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    client_id, ep=i, loss=losses,
                    top1=top1))
        # for name, param in self.model.local_list[client_id].named_parameters():
        #     self.writer.add_histogram("local_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        # for name, param in self.model.cloud.named_parameters():
        #     self.writer.add_histogram("server_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        self.logger.debug(' * Prec@1 {top1.avg:.3f}'
                          .format(top1=top1))

        return top1.avg, losses.avg

    # auto complete model's name, since we have many
    def infer_path_list(self, path_to_infer):
        split_list = path_to_infer.split("checkpoint_f")
        first_part = split_list[0]
        second_part = split_list[1]
        model_path_list = []
        for i in range(self.num_client):
            if i == 0:
                model_path_list.append(path_to_infer)
            elif i == 1:
                model_path_list.append(first_part + "checkpoint_c" + second_part)
            else:
                model_path_list.append(first_part + "checkpoint_local{}".format(i) + second_part)

        return model_path_list

    # resume all client and server model from checkpoint
    def resume(self, model_path_f=None):

        # if "pruning" in self.AT_regularization_option:
            # for i in range(self.num_client):
            #     print("load client {}'s local".format(i))
            #     checkpoint_i = torch.load(model_path_list[i])
            #     self.model.local_list[i]=torch.load(model_path_list[i])
            #     self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)
        #     try:
        #         print(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
        #         self.f=torch.load(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
        #         print('load local in pruning way')
        #     except:
        #         print("No valid Checkpoint Found!")
        #         return
        # else:
        # if "pruning" in self.AT_regularization_option:
            # self.f = self.model_pruning(self.f,ratio=0.1)
            
        # print(model_path_f)
        if model_path_f is None:

            try:
                if "V" in self.scheme:
                    checkpoint = torch.load(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                    model_path_list = self.infer_path_list(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                else:
                    checkpoint = torch.load(self.save_dir + "checkpoint_{}.tar".format(self.n_epochs))
                    # model_path_list = self.infer_path_list(self.save_dir + "checkpoint_200.tar")
            except:
                print("No valid Checkpoint Found!")
                print(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                return
        else:
            if "V" in self.scheme:
                model_path_list = self.infer_path_list(model_path_f)

        if "V" in self.scheme:
            for i in range(self.num_client):
                print("load client {}'s local".format(i))
                checkpoint_i = torch.load(model_path_list[i])
                self.model.local_list[i].cuda()
                if "pruning" in self.AT_regularization_option:
                    self.model.local_list[i]=torch.load(model_path_list[i])
                    
                    self.f = self.model.local_list[0].cuda()
                  
                else:
                    print(model_path_list[i])
                    # print("pruning" in self.AT_regularization_option)
                    self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)
        else:
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.f = self.model.local
            self.f.cuda()

        try:
            self.call_resume = True
            print("load cloud")
            checkpoint = torch.load(self.save_dir + "checkpoint_cloud_{}.tar".format(self.n_epochs))
            self.f_tail.cuda()
            self.f_tail.load_state_dict(checkpoint, strict = False)
            print("load classifier")
            checkpoint = torch.load(self.save_dir + "checkpoint_classifier_{}.tar".format(self.n_epochs))
            self.classifier.cuda()
            self.classifier.load_state_dict(checkpoint, strict = False)
        except:
            print("might be old style saving, load entire model")
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.call_resume = True
            self.f = self.model.local
            self.f.cuda()
            self.f_tail = self.model.cloud
            self.f_tail.cuda()
            self.classifier = self.model.classifier
            self.classifier.cuda()
        # print(self.save_dir)
        self.f.eval()
        self.f_tail.eval()
        self.classifier.eval()



    # client-side model sync
    def sync_client(self):
        # update global weights
        global_weights = average_weights(self.model.local_list)

        # update global weights
        for i in range(self.num_client):
            self.model.local_list[i].load_state_dict(global_weights)

    # decoder sync
    def sync_decoder(self):
        # update global weights
        global_weights = average_weights(self.local_AE_list)

        # update global weights
        for i in range(self.num_client):
            self.local_AE_list[i].load_state_dict(global_weights)

    # train local inversion model
    def gan_train_step(self, input_images, client_id, loss_type="SSIM"):
        device = next(self.model.local_list[client_id].parameters()).device

        input_images = input_images.to(device)

        self.model.local_list[client_id].eval()

        z_private = self.model.local_list[client_id](input_images)

        self.local_AE_list[client_id].train()

        x_private, z_private = Variable(input_images).to(device), Variable(z_private)

        x_private = denormalize(x_private, self.dataset)

        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private.clone()
            grad = torch.zeros_like(z_private).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)
            if loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif loss_type == "MSE":
                MSE_loss = torch.nn.MSELoss()
                loss = MSE_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)
            else:
                raise ("No such loss_type for gan train step")
            
            z_private = z_private - grad.detach() * epsilon
            
            self.local_AE_list[client_id].train()

        output = self.local_AE_list[client_id](z_private.detach())

        if loss_type == "SSIM":
            ssim_loss = pytorch_ssim.SSIM()
            loss = -ssim_loss(output, x_private)
        elif loss_type == "MSE":
            MSE_loss = torch.nn.MSELoss()
            loss = MSE_loss(output, x_private)
        else:
            raise ("No such loss_type for gan train step")
        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].zero_grad()

        loss.backward()

        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].step()

        losses = loss.detach().cpu().numpy()
        del loss

        return losses


    # Main function for controlling training and testing, soul of ResSFL
    def __call__(self, log_frequency=100, verbose=False, progress_bar=True):
        

        self.logger.debug("Model's smashed-data size is {}".format(str(self.model.get_smashed_data_size())))
        
        best_avg_accu = 0.0
        if not self.call_resume:
            LOG = np.zeros((self.n_epochs * self.num_batches, self.num_client))
            client_iterator_list = []
            for client_id in range(self.num_client):
                client_iterator_list.append(iter(self.client_dataloader[client_id]))
                # if self.dataset== 'cifar100':
                # client_iterator_list.append(iter(self.client_dataloader_rob[client_id]))
            # self.pruning_ep = int(AT_regularization_option.split("pruning")[-1])
            if "pruning" in self.AT_regularization_option:
                try:
                    self.pruning_ep = int(self.AT_regularization_option.split("pruning")[-1])

                except:
                    self.pruning_ep=140
                    # print('pruning_ep not defined')
                print('the pruning epoch is:',self.pruning_ep)                
            #load pre-train models
            if self.load_from_checkpoint:
                checkpoint_dir = "new_saves/imagenet/imagenet_resnet20_None_infocons_sgm_lg1_thre0.125_512_ganthre0.6/pretrain_False_lambd_0_noise_0_epoch_120_bottleneck_noRELU_C4S1_log_1_ATstrength_1_lr_0.05/"
                try:
                    checkpoint_i = torch.load(checkpoint_dir + "checkpoint_f_90.tar")
                except:
                    print("No valid Checkpoint Found!")
                    return
                # print('load encoder')
                self.model.cuda()
                model_dict = self.f.state_dict() 
                filtered_dict = {k: v for k, v in checkpoint_i.items() if k in model_dict and model_dict[k].size() == v.size()}
    
                # 更新模型权重
                model_dict.update(filtered_dict)
                
                # 加载到模型
                # model.load_state_dict(model_dict, strict=False)
                self.model.local.load_state_dict(model_dict, strict = False)
                self.f = self.model.local
                self.f.cuda()
                
                load_classfier = True
                if self.load_from_checkpoint_server:
                    print("load cloud")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_cloud_90.tar")
                    self.f_tail.cuda()
                    self.f_tail.load_state_dict(checkpoint, strict = False)
                if load_classfier:
                    print("load classifier")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_classifier_90.tar")
                    self.classifier.cuda()
                    # self.classifier.load_state_dict(checkpoint, strict = False)
                if self.dataset != "imagenet":
                    accu, loss = self.validate_target(client_id=0)

            # summary(self.f.cuda(), input_size=(3,32, 32))
            if self.gan_regularizer:
                self.pre_GAN_train(30, range(self.num_client))


            self.logger.debug("Real Train Phase: done by all clients, for total {} epochs".format(self.n_epochs))

            if self.save_more_checkpoints:
                epoch_save_list = [30, 90]
            else:
                epoch_save_list = []
            # If optimize_computation, set GAN updating frequency to 1/5.
            ssim_log = 0.
            
            interval = self.optimize_computation
            self.logger.debug("GAN training interval N (once every N step) is set to {}!".format(interval))
            
            adding_noise=False
            centroids_list= [torch.tensor(float('nan')) for _ in range(self.num_class)]
            weights_list= [torch.tensor(float('nan')) for _ in range(self.num_class)]
            cluster_variances_list=[torch.tensor(float('nan')) for _ in range(self.num_class)]
            #Main Training
            lambd_start= self.lambd 
            lambd_end=lambd_start*2
            acc_list=[]
            rob_list= []
            for epoch in range(1, self.n_epochs+1):
                self.current_epoch = epoch
                ep_start_time = time.time() 
                if epoch > self.warm:
                    self.scheduler_step(epoch)
                    if self.gan_regularizer:
                        self.gan_scheduler_step(epoch)
                
                # if epoch > 0.3*self.n_epochs:
                #     print('start to adding noise')
                #     adding_noise=True


                self.logger.debug("Train in {} style".format(self.scheme))
                # print("adding noise:",adding_noise)
                Z_all = []
                label_all = [] 
                if epoch ==1:
                    random_ini_centers = True
                else: 
                    random_ini_centers = False
                train_loss_list=[]
                f_loss_list= []
                self._cem_fail_this_epoch = 0  # reset per-epoch counter
                # 重新初始化各客户端迭代器，避免沿用上一轮末尾位置
                client_iterator_list = []
                for _cid in range(self.num_client):
                    client_iterator_list.append(iter(self.client_dataloader[_cid]))
                model_train_stime= time.time()
                self.pooling = False
                if "pruning" in self.AT_regularization_option and epoch==self.pruning_ep:
                    self.f = self.model_pruning(self.f,ratio=0.05)
                    summary(self.f.cuda(), input_size=(3,32, 32))
                    
                    self.logger.debug('the model is pruned at epoch:%d',epoch)    
                    
                if "epoch" in self.scheme:
                    for batch in range(self.num_batches):

                        # shuffle_client_list = range(self.num_client)
                        for client_id in range(self.num_client):
                            batch_data_read_start_time = time.time()

                            # try:
                            images, labels = next(client_iterator_list[client_id])
                            if images.size(0) != self.batch_size:
                                client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                                images, labels = next(client_iterator_list[client_id])
                            # except StopIteration:
                            #     client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                            #     images, labels = next(client_iterator_list[client_id])

                            batch_data_read_time = time.time() 

                            # Train the AE decoder if self.gan_regularizer is enabled:
                            if self.gan_regularizer and batch % interval == 0:
                                for i in range(self.gan_num_step): #equals 1 or 3 
                                    ssim_log = -self.gan_train_step(images, client_id, loss_type=self.gan_loss_type)  # orig_epoch_gan_train

                            self.optimizer_zero_grad()
                            
                            # Train step (client/server)
                            # print(images.shape)
                            train_loss, f_loss, z_private = self.train_target_step(images, labels, adding_noise,random_ini_centers,centroids_list,weights_list,cluster_variances_list,client_id)

                            train_loss_list.append(torch.tensor(train_loss))
                            f_loss_list.append(torch.tensor(f_loss))
                            self.optimizer_step()
                            
                            # Logging
                            # LOG[batch, client_id] = train_loss
                            
                            if verbose and (batch==0 or batch==self.num_batches-1):
                                train_loss_mean=torch.stack(train_loss_list).mean()
                                f_loss_mean=torch.stack(f_loss_list).mean()
                                self.logger.debug(
                                    "log--[{}/{}][{}/{}][client-{}] train loss: {:1.4f} cross-entropy loss: {:1.4f}".format(
                                        epoch, self.n_epochs, batch, self.num_batches, client_id, train_loss_mean, f_loss_mean))
                                if self.gan_regularizer:
                                    self.logger.debug(
                                        "log--[{}/{}][{}/{}][client-{}] Adversarial Loss of local AE: {:1.4f}".format(epoch,
                                                                                                                self.n_epochs,
                                                                                                                batch,
                                                                                                                self.num_batches,
                                                                                                                client_id,
                                                                                                                ssim_log))
                            if batch == 0:
                                self.writer.add_scalar('train_loss/client-{}/total'.format(client_id), train_loss,
                                                        epoch)
                                self.writer.add_scalar('train_loss/client-{}/cross_entropy'.format(client_id), f_loss,
                                                        epoch)
                            # if batch%20 ==0:
                        
                                # print(f"data_readtime:{batch_data_read_time - batch_data_read_start_time} s",f"model_infer_time:{model_train_time - batch_data_read_time} s")   
                # Validate and get average accu among clients
                avg_accu = 0
                val_start_time= time.time()
                for client_id in range(self.num_client):
                    if self.dataset != "imagenet":
                        accu, loss = self.validate_target(client_id=client_id)
                    else:
                        accu,loss = 0,0 
                    self.writer.add_scalar('valid_loss/client-{}/cross_entropy'.format(client_id), loss, epoch)
                    avg_accu += accu
                avg_accu = avg_accu / self.num_client
                acc_list.append(torch.tensor(avg_accu))
                val_time=time.time()
                # print(f"val_one_ep_time:{val_time-val_start_time} s")
                # Save the best model
                if avg_accu > best_avg_accu:
                    self.save_model(epoch, is_best=True)
                    print('best model saved at:',epoch)
                    best_avg_accu = avg_accu
                    best_rob_loss= train_loss_mean
                if epoch==40 or epoch==100 or epoch==200 :
                    best_avg_accu=0

                # if epoch==2:
                #     summary(self.f.cuda(), input_size=(3,32, 32))
                # (Removed) GMM/KMeans-based feature clustering analysis; attention-only pipeline keeps training simple and stable.

                # kmeans_cuda(self,X, num_clusters,centroids, num_iterations=10, tol=1e-4):
                
                # V1/V2 synchronization
                if self.scheme == "V1_epoch" or self.scheme == "V2_epoch":
                    self.sync_client()
                    if self.gan_regularizer and self.decoder_sync:
                        self.sync_decoder()

                # Step the warmup scheduler
                if epoch <= self.warm:
                    self.scheduler_step(warmup=True)



                
                # if epoch ==150 or epoch ==170:
                # self.lambd = lambd_end + 0.5 * (lambd_start - lambd_end) * (1 + np.cos(np.pi * epoch / self.n_epochs))
                # Per-epoch CEM failure summary
                total_cem_fails = self._cem_fail_count + self._cem_nan_count
                if self._cem_fail_this_epoch > 0:
                    self.logger.warning(
                        "\n" + "=" * 70 + "\n"
                        f"  *** CEM EPOCH {epoch} SUMMARY: {self._cem_fail_this_epoch} step(s) failed ***\n"
                        f"  Cumulative failures: RuntimeError={self._cem_fail_count}, NaN/Inf={self._cem_nan_count} (total={total_cem_fails})\n"
                        f"  CEM defense was INACTIVE for these steps — check logs!\n"
                        + "=" * 70
                    )
                else:
                    self.logger.debug(f"CEM epoch {epoch} OK — 0 failures this epoch (cumulative total={total_cem_fails})")
                if (self.train_scheduler.get_last_lr()[0])<0.00041: #strat to enhance rob when lr is small and acc is high
                    print('lambd value is:', self.lambd, 'learning rate is:', self.train_scheduler.get_last_lr()[0] )
                else:
                    print('lambd value is:',self.lambd*(0.001/self.train_scheduler.get_last_lr()[0]), 'learning rate is:', self.train_scheduler.get_last_lr()[0] )
                # Save Model regularly
                if epoch % 50 == 0 or epoch == self.n_epochs or epoch in epoch_save_list:  # save model
                    self.save_model(epoch)
            
                epochs = list(range(1, len(acc_list) + 1))
                logimg_path = str(self.save_dir) + "/log_img"
                if not os.path.isdir(logimg_path):
                    os.makedirs(logimg_path)
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, acc_list, label='Accuracy', marker='o', color='blue')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Accuracy over Epochs')
                plt.legend()
                plt.grid(True)
                acc_img=logimg_path+'/acc.png'
                plt.savefig(acc_img)
                plt.close()

                # 绘制并保存 Robustness 的折线图
                epochs = list(range(1, len(rob_list) + 1))
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, rob_list, label='Robustness', marker='o', color='orange')
                plt.xlabel('Epoch')
                plt.ylabel('Robustness')
                plt.title('Robustness over Epochs')
                plt.legend()
                plt.grid(True)
                rob_img=logimg_path+'/rob.png'
                plt.savefig(rob_img)
                plt.close()

        if not self.call_resume:

            self.logger.debug("Best Average Validation Accuracy is {}".format(best_avg_accu))
        else:
        

            self.f.eval()
            # summary(self.f.cuda(), input_size=self.sample_image.shape)
            # summary(self.model.cuda(), input_size=self.sample_image.shape)
            with torch.cuda.device(0):
                flops, params = get_model_complexity_info(self.f.cuda(), (3, 32, 32), as_strings=True, print_per_layer_stat=True)
                print(f"FLOPs: {flops}, Parameters: {params}")        
                flops, params = get_model_complexity_info(self.model.cuda(), (3, 32, 32), as_strings=True, print_per_layer_stat=True)
                print(f"FLOPs: {flops}, Parameters: {params}")    
            model1_params = sum(p.numel() for p in self.f.parameters())
            model2_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model 1 Params: {model1_params}, Model 2 Params: {model2_params}")
            self.model.eval()
            with torch.no_grad():
                
                input_tensor = torch.randn(128, 3, 32, 32)  
                
                if torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                    self.model.cuda()
                    self.f.cuda()
                    torch.cuda.synchronize()  # 确保所有CUDA核心同步
                

                start_time = time.time()
                for i in range(100):
                    outputs = self.model(input_tensor)
                
                # 如果使用CUDA，需要同步确保所有计算完成
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # 停止计时
                end_time = time.time()

                total_time = end_time - start_time
                print(f"推理时间: {total_time * 1000:.2f} ms")  # 输出以毫秒为单位

            
            LOG = None
            avg_accu = 0
            for client_id in range(self.num_client):
                # client_iterator_list.append(iter(self.client_dataloader[client_id]))
                accu, loss = self.validate_target(client_id=client_id)
                avg_accu += accu
            avg_accu = avg_accu / self.num_client
            self.logger.debug("Best Average Validation Accuracy is {}".format(avg_accu))
        return LOG

    def save_model(self, epoch, is_best=False):
        if is_best:
            epoch = "best"

        if "V" in self.scheme:
            if 'pruning' in self.AT_regularization_option:
                torch.save(self.f, self.save_dir + 'checkpoint_f_{}.tar'.format(epoch))
            else:
                torch.save(self.f.state_dict(), self.save_dir + 'checkpoint_f_{}.tar'.format(epoch))
            if self.num_client > 1:
                torch.save(self.c.state_dict(), self.save_dir + 'checkpoint_c_{}.tar'.format(epoch))
            torch.save(self.f_tail.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))
            if self.num_client > 2:
                for i in range(2, self.num_client):
                    torch.save(self.model.local_list[i].state_dict(),
                                self.save_dir + 'checkpoint_local{}_{}.tar'.format(i, epoch))
        else:
            torch.save(self.model.state_dict(), self.save_dir + 'checkpoint_{}.tar'.format(epoch))
            torch.save(self.f_tail.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))


    def gen_ir(self, val_single_loader, local_model, img_folder="./tmp", intermed_reps_folder="./tmp", all_label=True,
               select_label=0, attack_from_later_layer=-1, attack_option = "MIA"):
        """
        Generate (Raw Input - Intermediate Representation) Pair for Training of the AutoEncoder
        """

        # switch to evaluate mode
        local_model.eval()
        file_id = 0
        for i, (input, target) in enumerate(val_single_loader):
            # input = input.cuda(async=True)
            input = input.cuda()
            # print(input.shape,target)
            target = target.item()
            if not all_label:
                if target != select_label:
                    continue

            img_folder = os.path.abspath(img_folder)
            intermed_reps_folder = os.path.abspath(intermed_reps_folder)
            if not os.path.isdir(intermed_reps_folder):
                os.makedirs(intermed_reps_folder)
            if not os.path.isdir(img_folder):
                os.makedirs(img_folder)

            # compute output
            with torch.no_grad():
                ir = local_model(input)
            
            if self.confidence_score:
                self.model.cloud.eval()
                ir = self.model.cloud(ir)
                if "mobilenetv2" in self.arch:
                    ir = F.avg_pool2d(ir, 4)
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    ir = F.avg_pool2d(ir, 8)
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
                else:
                    ir = self._flatten_for_classifier(ir)
                    ir = self.classifier(ir)
            
            if attack_from_later_layer > -1 and (not self.confidence_score):
                self.model.cloud.eval()

                activation_4 = {}

                def get_activation_4(name):
                    def hook(model, input, output):
                        activation_4[name] = output.detach()

                    return hook

                with torch.no_grad():
                    activation_4 = {}
                    count = 0
                    for name, m in self.model.cloud.named_modules():
                        if attack_from_later_layer == count:
                            m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
                            valid_key = "ACT-{}".format(name)
                            break
                        count += 1
                    output = self.model.cloud(ir)
                try:
                    ir = activation_4[valid_key]
                except:
                    print("cannot attack from later layer, server-side model is empty or does not have enough layers")
            ir = ir.float()

            if "truncate" in attack_option:
                try:
                    percentage_left = int(attack_option.split("truncate")[1])
                except:
                    print("auto extract percentage fail. Use default percentage_left = 20")
                    percentage_left = 20
                ir = prune_top_n_percent_left(ir, percentage_left)

            inp_img_path = "{}/{}.jpg".format(img_folder, file_id)
            out_tensor_path = "{}/{}.pt".format(intermed_reps_folder, file_id)
            
            input = denormalize(input, self.dataset)
            save_image(input, inp_img_path)
            torch.save(ir.cpu(), out_tensor_path)
            file_id += 1
        print("Overall size of Training/Validation Datset for AE is {}: {}".format(int(file_id * 0.9),
                                                                                   int(file_id * 0.1)))

    def gen_inp_feat_pair(self,input,local_model):
        local_model.eval()
        input = input.cuda()
        with torch.no_grad():
            ir = local_model(input)
        ir = ir.float()
        input = denormalize(input, self.dataset)
        return input,ir

# pre-train a GAN with local data before SFL training
    def pre_GAN_train(self, num_epochs, select_client_list=[0]):

        # Generate latest images/activation pair for all clients:
        client_iterator_list = []
        for client_id in range(self.num_client):
            client_iterator_list.append(iter(self.client_dataloader[client_id]))
        try:
            images, labels = next(client_iterator_list[client_id])
            if images.size(0) != self.batch_size:
                client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                images, labels = next(client_iterator_list[client_id])
        except StopIteration:
            client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
            images, labels = next(client_iterator_list[client_id])

        for client_id in select_client_list:
            self.save_image_act_pair(images, labels, client_id, 0, clean_option=True)

        for client_id in select_client_list:

            attack_batchsize = 256
            attack_num_epochs = num_epochs
            model_log_file = self.save_dir + '/MIA_attack_{}_{}.log'.format(client_id, client_id)
            logger = setup_logger('{}_{}to{}_attack_logger'.format(str(self.save_dir), client_id, client_id),
                                  model_log_file, level=logging.DEBUG)
            # pass
            image_data_dir = self.save_dir + "/img"
            tensor_data_dir = self.save_dir + "/img"

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)

            if self.dataset == "cifar100":
                val_loader, val_train_loader, val_test_loader= get_cifar100_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "cifar10":
                val_loader, val_train_loader, val_test_loader = get_cifar10_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "svhn":
                val_single_loader, _, _ = get_SVHN_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "mnist":
                _, val_single_loader = get_mnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "fmnist":
                _, val_single_loader = get_fmnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "facescrub":
                _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)
                # _, val_single_loader = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "tinyimagenet":
                _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_tinyimagenet_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "imagenet":
                _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_imagenet_bothloader(batch_size=1, num_workers=4, shuffle=False)
            attack_path = self.save_dir + '/MIA_attack_{}to{}'.format(client_id, client_id)
            if not os.path.isdir(attack_path):
                os.makedirs(attack_path)
                os.makedirs(attack_path + "/train")
                os.makedirs(attack_path + "/test")
                os.makedirs(attack_path + "/tensorboard")
                os.makedirs(attack_path + "/sourcecode")
            train_output_path = "{}/train".format(attack_path)
            test_output_path = "{}/test".format(attack_path)
            tensorboard_path = "{}/tensorboard/".format(attack_path)
            model_path = "{}/model.pt".format(attack_path)
            path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                         "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

            logger.debug("Generating IR ...... (may take a while)")

            self.gen_ir(val_loader, self.model.local_list[client_id], image_data_dir, tensor_data_dir)

            if self.dataset == "cifar100":
                val_loader, val_train_loader, val_test_loader= get_cifar100_testloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
            elif self.dataset == "cifar10":
                val_loader, val_train_loader, val_test_loader = get_cifar10_testloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
            elif self.dataset == "facescrub":
                _, _,_,val_train_loader,val_test_loader = get_facescrub_bothloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
            elif self.dataset == "tinyimagenet":
                _, _,_,val_train_loader,val_test_loader = get_tinyimagenet_bothloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
            elif self.dataset == "imagenet":
                _, _,_,val_train_loader,val_test_loader = get_imagenet_bothloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
            decoder = self.local_AE_list[client_id]

            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            scheduler = CosineAnnealingLR(optimizer, T_max=attack_num_epochs,eta_min=1e-5)
            # Construct a dataset for training the decoder
            trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)

            # Do real test on target's client activation (and test with target's client ground-truth.)
            sp_testloader = apply_transform_test(1,
                                                 self.save_dir + "/save_activation_client_{}_epoch_{}".format(client_id,
                                                                                                             0),
                                                 self.save_dir + "/save_activation_client_{}_epoch_{}".format(client_id,
                                                                                                             0))

            # Perform Input Extraction Attack
            # self.attack(attack_num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict,
            #             attack_batchsize)
            self.attack(attack_num_epochs,self.model.local_list[client_id], decoder, optimizer,scheduler, val_train_loader, val_test_loader, logger, path_dict,
                            attack_batchsize, noise_aware=False, loss_type='MSE')
            # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False
            mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs,self.model.local_list[client_id], decoder, val_test_loader, logger,
                                                                 path_dict, attack_batchsize,
                                                                 num_classes=self.num_class,sp_not=0)
            # mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs, decoder, trainloader, logger,
            #                                                      path_dict, attack_batchsize,
            #                                                      num_classes=self.num_class)

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)
    def MIA_attack(self, num_epochs, attack_option="MIA", collude_client=1, target_client=0, noise_aware=False,
                   loss_type="MSE", attack_from_later_layer=-1, MIA_optimizer = "Adam", MIA_lr = 1e-3):
        attack_option = attack_option
        MIA_optimizer = MIA_optimizer
        MIA_lr = MIA_lr
        attack_batchsize = 256
        attack_num_epochs = num_epochs
        model_log_file = self.save_dir + '/{}_attack_{}_{}.log'.format(attack_option, collude_client, target_client)
        logger = setup_logger('{}_{}to{}_attack_logger'.format(str(self.save_dir), collude_client, target_client),
                              model_log_file, level=logging.DEBUG)
        # pass
        image_data_dir = self.save_dir + "/img"
        tensor_data_dir = self.save_dir + "/img"

        # Clear content of image_data_dir/tensor_data_dir
        if os.path.isdir(image_data_dir):
            rmtree(image_data_dir)
        if os.path.isdir(tensor_data_dir):
            rmtree(tensor_data_dir)

        if self.dataset == "cifar100":
            _, train_single_loader, train_single_loader_irg = get_cifar100_trainloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "cifar10":
            _, train_single_loader, train_single_loader_irg = get_cifar10_trainloader(batch_size=1, num_workers=4, shuffle=True)
            # val_loader, val_train_loader, val_test_loader = get_cifar10_testloader(batch_size=1, num_workers=4, shuffle=True)
        elif self.dataset == "svhn":
            val_single_loader, _, _ = get_SVHN_testloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "mnist":
            _, val_single_loader = get_mnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "fmnist":
            _, val_single_loader = get_fmnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "facescrub":
            
            _, _,_,_,train_single_loader_irg = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)
            # facescrub_training_loader, facescrub_testing_loader,facescrub_training_loader_AT, facescrub_testing_loader_AT,facescrub_testing_loader_val
        elif self.dataset == "tinyimagenet":
            _, _,_,_,train_single_loader_irg = get_tinyimagenet_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "imagenet":
            _, _,_,_,train_single_loader_irg = get_imagenet_bothloader(batch_size=1, num_workers=4, shuffle=False)

        attack_path = self.save_dir + '/{}_attack_{}to{}'.format(attack_option, collude_client, target_client)
        if not os.path.isdir(attack_path):
            os.makedirs(attack_path)
            os.makedirs(attack_path + "/train")
            os.makedirs(attack_path + "/test")
            os.makedirs(attack_path + "/tensorboard")
            os.makedirs(attack_path + "/sourcecode")
        train_output_path = "{}/train".format(attack_path)
        test_output_path = "{}/test".format(attack_path)
        tensorboard_path = "{}/tensorboard/".format(attack_path)
        model_path = "{}/model.pt".format(attack_path)
        path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                     "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

        if ("MIA" in attack_option) and ("MIA_mf" not in attack_option):
            logger.debug("Generating IR ...... (may take a while)")

            # self.gen_ir(train_single_loader_irg, self.model.local_list[collude_client], image_data_dir, tensor_data_dir,
            #         attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
            # print(image_data_dir,tensor_data_dir)

            # for filename in os.listdir(tensor_data_dir):
            #     if ".pt" in filename:
            #         sampled_tensor = torch.load(tensor_data_dir + "/" + filename)
            #         input_nc = sampled_tensor.size()[1]
            #         try:
            #             input_dim = sampled_tensor.size()[2]
            #         except:
            #             print("Extract input dimension fialed, set to 0")
            #             input_dim = 0
            #         break
            input_nc=8
            input_dim=32
            if self.dataset == "cifar10":
                _, train_single_loader, train_single_loader_irg = get_cifar10_trainloader(batch_size=attack_batchsize, num_workers=4, shuffle=True)
                val_loader, val_train_loader, val_test_loader = get_cifar10_testloader(batch_size=attack_batchsize, num_workers=4, shuffle=True)
            elif self.dataset == "cifar100":
                _, train_single_loader, train_single_loader_irg = get_cifar100_trainloader(batch_size=attack_batchsize, num_workers=4, shuffle=True)
                val_loader, val_train_loader, val_test_loader = get_cifar100_testloader(batch_size=attack_batchsize, num_workers=4, shuffle=True)
            elif self.dataset == "facescrub":
                input_nc=8
                input_dim=64  # 64x64 images (was 48, but datasets_torch loads at 64x64)
                _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_facescrub_bothloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
            elif self.dataset == "tinyimagenet":
                _, val_loader,train_single_loader,val_train_loader,val_test_loader  = get_tinyimagenet_bothloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
            elif self.dataset == "imagenet":
                _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_imagenet_bothloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
                input_nc=8
                input_dim=64
                # _, val_loader,train_single_loader,val_train_loader,val_test_loader = get_facescrub_bothloader(batch_size=attack_batchsize, num_workers=4, shuffle=False)
                # facescrub_training_loader, facescrub_testing_loader,facescrub_training_loader_AT, facescrub_testing_loader_AT,facescrub_testing_loader_val
            attack_mode= ['train_time','infer_time']
            for mode in attack_mode:
                if mode=='train_time':
                    logger.debug("run the attack for training time")
                    # print(input_dim)
                    mse_score_t, ssim_score_t, psnr_score_t=self.train_infer_attack(train_single_loader,val_loader,MIA_optimizer,attack_num_epochs,noise_aware,attack_batchsize,loss_type,MIA_lr,logger,path_dict,input_nc,input_dim)
                elif mode=='infer_time':
                    logger.debug("run the attack for inference time")
                    mse_score_I, ssim_score_I, psnr_score_I=self.train_infer_attack(val_train_loader,val_test_loader,MIA_optimizer,attack_num_epochs,noise_aware,attack_batchsize,loss_type,MIA_lr,logger,path_dict,input_nc,input_dim)
            return mse_score_t, ssim_score_t, psnr_score_t,mse_score_I, ssim_score_I, psnr_score_I
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)

        elif attack_option == "MIA_mf":  # Stands for Model-free MIA, does not need a AE model, optimize each fake image instead.

            lambda_TV = 0.0
            lambda_l2 = 0.0
            num_step = attack_num_epochs * 60

            sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
                target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                        self.n_epochs))
            criterion = nn.MSELoss().cuda()
            ssim_loss = pytorch_ssim.SSIM()
            all_test_losses = AverageMeter()
            ssim_test_losses = AverageMeter()
            psnr_test_losses = AverageMeter()
            fresh_option = True
            for num, data in enumerate(sp_testloader, 1):
                img, ir, _ = data

                # optimize a fake_image to (1) have similar ir, (2) have small total variance, (3) have small l2
                img = img.cuda()
                if not fresh_option:
                    ir = ir.cuda()
                self.model.local_list[collude_client].eval()
                self.model.local_list[target_client].eval()

                fake_image = torch.zeros(img.size(), requires_grad=True, device="cuda")
                optimizer = torch.optim.Adam(params=[fake_image], lr=8e-1, amsgrad=True, eps=1e-3)
                # optimizer = torch.optim.Adam(params = [fake_image], lr = 1e-2, amsgrad=True, eps=1e-3)
                for step in range(1, num_step + 1):
                    optimizer.zero_grad()

                    fake_ir = self.model.local_list[collude_client](fake_image)  # Simulate Original

                    if fresh_option:
                        ir = self.model.local_list[target_client](img)  # Getting fresh ir from target local model

                    featureLoss = criterion(fake_ir, ir)

                    TVLoss = TV(fake_image)
                    normLoss = l2loss(fake_image)

                    totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

                    totalLoss.backward()

                    optimizer.step()
                    # if step % 100 == 0:
                    if step == 0 or step == num_step:
                        logger.debug("Iter {} Feature loss: {} TVLoss: {} l2Loss: {}".format(step,
                                                                                            featureLoss.cpu().detach().numpy(),
                                                                                            TVLoss.cpu().detach().numpy(),
                                                                                            normLoss.cpu().detach().numpy()))
                imgGen = fake_image.clone()
                imgOrig = img.clone()

                mse_loss = criterion(imgGen, imgOrig)
                ssim_loss_val = ssim_loss(imgGen, imgOrig)
                psnr_loss_val = get_PSNR(imgOrig, imgGen)
                all_test_losses.update(mse_loss.item(), ir.size(0))
                ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
                psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
                if not os.path.isdir(test_output_path + "/{}".format(attack_num_epochs)):
                    os.mkdir(test_output_path + "/{}".format(attack_num_epochs))
                
                torchvision.utils.save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(attack_num_epochs,
                                                                                                num * attack_batchsize + attack_batchsize))
                
                torchvision.utils.save_image(imgOrig, test_output_path + '/{}/inp_{}.jpg'.format(attack_num_epochs,
                                                                                                num * attack_batchsize + attack_batchsize))
            logger.debug("MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                all_test_losses.avg))
            logger.debug("SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                ssim_test_losses.avg))
            logger.debug("PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                psnr_test_losses.avg))
            return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

    def train_infer_attack(self,train_data,test_data,MIA_optimizer,attack_num_epochs,noise_aware,attack_batchsize,loss_type,MIA_lr,logger,path_dict,input_nc,input_dim):
            self.feature_size = self.model.get_smashed_data_size()
            if self.gan_AE_type == "custom":
                decoder = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=self.recons_dim,
                                                    activation=self.gan_AE_activation).cuda()
                # *(sample_image.shape[-1]/32)
            
            elif "conv_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("conv_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from conv_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=self.feature_size[1], output_nc=3,
                                                         input_dim=self.feature_size[2]*(self.sample_image.shape[-1]/32), output_dim=self.sample_image.shape[-1],
                                                         activation=self.gan_AE_activation).cuda()

            elif "res_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("res_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from res_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                if "c3" in self.regularization_option:
                    feature_dim=16
                else:
                    feature_dim=8
                print((self.sample_image.shape[-1]/32),self.feature_size)
                decoder = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=self.feature_size[1], output_nc=3,
                                                         input_dim=8*(self.sample_image.shape[-1]/32), output_dim=self.sample_image.shape[-1],
                                                         activation=self.gan_AE_activation).cuda()
            
            else:
                raise ("No such GAN AE type.")

            if self.measure_option:
                noise_input = torch.randn([1, input_nc, input_dim, input_dim])
                device = next(decoder.parameters()).device
                noise_input = noise_input.to(device)
                macs, num_param = profile(decoder, inputs=(noise_input,))
                self.logger.debug(
                    "{} Decoder Model's Mac and Param are {} and {}".format(self.gan_AE_type, macs, num_param))
                
                '''Uncomment below to also get decoder's inference and training time overhead.'''

            '''Setting attacker's learning algorithm'''
            # optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            if MIA_optimizer == "Adam":
                # optimizer = torch.optim.Adam(decoder.parameters(), lr=MIA_lr)
                optimizer = torch.optim.Adam(decoder.parameters(), lr=MIA_lr, weight_decay=1e-4)
            elif MIA_optimizer == "SGD":
                optimizer = torch.optim.SGD(decoder.parameters(), lr=MIA_lr)
            else:
                raise("MIA optimizer {} is not supported!".format(MIA_optimizer))
            # Construct a dataset for training the decoder

            # trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)
            scheduler = CosineAnnealingLR(optimizer, T_max=attack_num_epochs,eta_min=1e-5)
            # Do real test on target's client activation (and test with target's client ground-truth.)
            # sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
                # target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                        #   self.n_epochs))
            # print(self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client, self.n_epochs))                                                                                              
            # print('the length of the test and sp test is:',len(testloader),len(sp_testloader))
            if "gan_adv_noise" in self.regularization_option and noise_aware:
                
                print("create a second decoder")
                if self.gan_AE_type == "custom":
                    decoder2 = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                       output_dim=self.recons_dim, activation=self.gan_AE_activation).cuda()
                    
                elif "conv_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("conv_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from conv_normN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    decoder2 = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=self.feature_size[1], output_nc=3,
                                                         input_dim=self.feature_size[2]*(sample_image.shape[-1]/32), output_dim=sample_image.shape[-1],
                                                         activation=self.gan_AE_activation).cuda()
                    
                elif "res_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("res_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from res_normN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    if "c3" in self.regularization_option:
                        decoder2 = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=self.feature_size[1], output_nc=3,
                                                            input_dim=self.feature_size[2]*2*(sample_image.shape[-1]/32), output_dim=sample_image.shape[-1],
                                                            activation=self.gan_AE_activation).cuda()
                    else:
                        decoder2 = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=self.feature_size[1], output_nc=3,
                                    input_dim=self.feature_size[2]*(sample_image.shape[-1]/32), output_dim=sample_image.shape[-1],
                                    activation=self.gan_AE_activation).cuda()
                        
                else:
                    raise ("No such GAN AE type.")
                # optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
                optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
                self.attack(attack_num_epochs, decoder2, optimizer2, trainloader, testloader, logger, path_dict,
                            attack_batchsize, pretrained_decoder=self.local_AE_list[collude_client], noise_aware=noise_aware)
                decoder = decoder2  # use decoder2 for testing

                        

            client_id=0
            self.attack(attack_num_epochs,self.model.local_list[client_id], decoder, optimizer,scheduler, train_data, test_data, logger, path_dict,
                            attack_batchsize, noise_aware=noise_aware, loss_type=loss_type)


            mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs,self.model.local_list[client_id], decoder, test_data, logger,
                                                                 path_dict, attack_batchsize,
                                                                 num_classes=self.num_class,sp_not=0)
            # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False


            # Clear content of image_data_dir/tensor_data_dir

            return mse_score, ssim_score, psnr_score
        
        

    # This function means performing training of the attacker's inversion model, is used in MIA_attack function.
    def attack(self, num_epochs, local_model, decoder, optimizer,scheduler, trainloader, testloader, logger, path_dict, batch_size,
               loss_type="MSE", pretrained_decoder=None, noise_aware=False):
        round_ = 0
        min_val_loss = 999.
        max_val_loss = 0.
        train_output_freq = 10


        # Optimize based on MSE distance
        if loss_type == "MSE":
            criterion = nn.MSELoss()
        elif loss_type == "L1":
            criterion = nn.L1Loss()
        elif loss_type == "SSIM":
            criterion = pytorch_ssim.SSIM()
        elif loss_type == "PSNR":
            criterion = None
        else:
            raise ("No such loss in self.attack")
        criterion_l1=nn.L1Loss()
        
        device = next(decoder.parameters()).device
        decoder.train()
        
        # summary(decoder.cuda(), input_size=(8,8, 8))
        # flops, params = get_model_complexity_info(decoder, (8,8, 8), as_strings=True, print_per_layer_stat=True)
        # summary(local_model.cuda(), input_size=(3,32, 32))
        # print(f"FLOPs: {flops}")
        # print(f"参数数量: {params}")
        
        for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
            train_losses = AverageMeter()
            val_losses = AverageMeter()
            val_losses_white = AverageMeter()
            for i, (input, target) in enumerate(trainloader):
                input = input.cuda()
                img, ir=self.gen_inp_feat_pair(input,local_model)
                # img, ir = data
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)
                # print(img)
                # Use local DP for training the AE.
                if "Gaussian" in self.regularization_option:
                    sigma = self.regularization_strength
                    noise = sigma * torch.randn_like(ir).cuda()
                    ir += noise

                if self.local_DP and noise_aware:
                    with torch.no_grad():
                        if "laplace" in self.regularization_option:
                            ir += torch.from_numpy(
                                np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=ir.size())).cuda()
                        else:  # apply gaussian noise
                            delta = 10e-5
                            sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                            ir += sigma * torch.randn_like(ir).cuda()
                if self.dropout_defense and noise_aware:
                    ir = dropout_defense(ir, self.dropout_ratio)
                if self.topkprune and noise_aware:
                    ir = prune_defense(ir, self.topkprune_ratio)
                if pretrained_decoder is not None and "gan_adv_noise" in self.regularization_option and noise_aware:
                    epsilon = self.alpha2
                    
                    pretrained_decoder.eval()
                    fake_act = ir.clone()
                    grad = torch.zeros_like(ir).cuda()
                    fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                    x_recon = pretrained_decoder(fake_act)
                    if self.gan_loss_type == "SSIM":
                        ssim_loss = pytorch_ssim.SSIM()
                        loss = ssim_loss(x_recon, img)
                        loss.backward()
                        grad -= torch.sign(fake_act.grad)
                    else:
                        mse_loss = nn.MSELoss()
                        loss = mse_loss(x_recon, img)
                        loss.backward()
                        grad += torch.sign(fake_act.grad)
                    ir = ir + grad.detach() * epsilon
                # print(ir.size())
                output = decoder(ir)
                # print(output.shape,img.shape)
                if loss_type == "MSE":
                    reconstruction_loss = criterion(output, img)
                elif loss_type == "L1":
                    reconstruction_loss = criterion(output, img)
                elif loss_type == "SSIM":
                    reconstruction_loss = -criterion(output, img)
                elif loss_type == "PSNR":
                    reconstruction_loss = -1 / 10 * get_PSNR(img, output)
                else:
                    raise ("No such loss in self.attack")
                train_loss = reconstruction_loss#+criterion_l1(output, img)-1 / 10 * get_PSNR(img, output)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_losses.update(train_loss.item(), ir.size(0))
                # if i %50 ==0:
                #     print('iter:',i,'loss:',train_loss.item())
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}, Current LR: {current_lr}")
            if (epoch + 1) % train_output_freq == 0:
                save_images(img, output, epoch, path_dict["train_output_path"], offset=0, batch_size=batch_size)
            top1 = AverageMeter()
            # if epoch == 1:
            rec_inputs_list = []
            targets_list = []
            for i, (input, target) in enumerate(testloader):
                input = input.cuda()
                img, ir=self.gen_inp_feat_pair(input,local_model)
                # decoder.eval()
                # img, ir = data
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)
                if "Gaussian" in self.regularization_option:
                    sigma = self.regularization_strength
                    noise = sigma * torch.randn_like(ir).cuda()
                    ir += noise
                output = decoder(ir)
                criterion_test = nn.MSELoss()
                reconstruction_loss = criterion_test(output, img)

                ####test_reconstruction acc 
                # if epoch==10:
                #     self.save_dir = "new_saves/cifar10/Aresult_model/None_infocons_sgm_lg1_thre0.125/pretrain_False_lambd_0_noise_0.01_epoch_240_bottleneck_noRELU_C8S1_log_1_ATstrength_0.3_lr_0.05_varthres_0.125/"
                #     self.resume(model_path_f=None)
                rec_inputs = normalize(output,self.dataset)
                rec_inputs_list.append(rec_inputs.clone().cpu())  # 保存 inputs
                targets_list.append(target.clone().cpu())
                
                pred = self.model.local_list[0](rec_inputs.cuda())
                with torch.no_grad():
                    pred = self.f_tail(pred)
                    # if "Gaussian" in self.regularization_option:
                    #     sigma = self.regularization_strength
                    #     noise = sigma * torch.randn_like(pred).cuda()
                    #     pred += noise
                    if "mobilenetv2" in self.arch:
                        pred = F.avg_pool2d(pred, 4)
                        pred = pred.view(pred.size(0), -1)
                        pred = self.classifier(pred)
                    elif self.arch == "resnet20" or self.arch == "resnet32":
                        pred = F.avg_pool2d(pred, 8)
                        pred = pred.view(pred.size(0), -1)
                        pred = self.classifier(pred)
                    else:
                        pred = self._flatten_for_classifier(pred)
                        pred = self.classifier(pred)
                prec1 = accuracy(pred.data.cpu(), target.cpu())[
                0] 
                top1.update(prec1.item(), input.size(0))
               
                
                
                whitebox_con=0
                # whitebox_con=
                if whitebox_con==1:
                    loss,output_w = self.whitebox(self.model.local_list[0], img, ir, num_steps =5, X = output)
                    val_loss_white=criterion_test(output_w, img)
                    val_losses_white.update(val_loss_white.item(), ir.size(0))
                val_loss = reconstruction_loss

                if loss_type == "MSE" and val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                elif loss_type == "SSIM" and val_loss > max_val_loss:
                    max_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                elif loss_type == "PSNR" and val_loss > max_val_loss:
                    max_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                val_losses.update(val_loss.item(), ir.size(0))

                self.writer.add_scalar('decoder_loss/val', val_loss.item(), len(testloader) * epoch + i)
                self.writer.add_scalar('decoder_loss/val_loss/reconstruction', reconstruction_loss.item(),
                                       len(testloader) * epoch + i)
            # [DEBUG BLOCK REMOVED] was: epoch==49 hardcoded cifar10 path override
            # This was leftover debug code that would crash FaceScrub runs by
            # overwriting self.save_dir with a non-existent CIFAR10 path.
            print('the pred acc is:',top1.avg)
            for name, param in decoder.named_parameters():
                self.writer.add_histogram("decoder_params/{}".format(name), param.clone().cpu().data.numpy(), epoch)

            # torch.save(decoder.state_dict(), path_dict["model_path"])
            if whitebox_con==1:
                logger.debug(
                    "epoch [{}/{}], train_loss {train_losses.val:.4f} ({train_losses.avg:.4f}), val_loss {val_losses.val:.4f} ({val_losses.avg:.4f}),val_loss_white {val_losses_white.val:.4f} ({val_losses_white.avg:.4f})".format(
                        epoch + 1,
                        num_epochs, train_losses=train_losses, val_losses=val_losses,val_losses_white=val_losses_white))
            else:
                logger.debug(
                    "epoch [{}/{}], train_loss {train_losses.val:.4f} ({train_losses.avg:.4f}), val_loss {val_losses.val:.4f} ({val_losses.avg:.4f})".format(
                        epoch + 1,
                        num_epochs, train_losses=train_losses, val_losses=val_losses))                
        if loss_type == "MSE":
            logger.debug("Best Validation Loss is {}".format(min_val_loss))
        elif loss_type == "SSIM":
            logger.debug("Best Validation Loss is {}".format(max_val_loss))
        elif loss_type == "PSNR":
            logger.debug("Best Validation Loss is {}".format(max_val_loss))

    def denormalize(self, img):
        std = (0.247, 0.243, 0.261)
        mean = (0.4914, 0.4822, 0.4465)
        for i in range(3):
            img[:,i,:,:] = img[:,i,:,:] * std[i] + mean[i]
        return img

    def normalize(self, img):
        std = (0.247, 0.243, 0.261)
        mean = (0.4914, 0.4822, 0.4465)

        for i in range(3):
            img[:,i,:,:] = (img[:,i,:,:] - mean[i])/ std[i]
        return img


    def whitebox(self, model, img, feature, num_steps=20, X = None):
        shape=img.shape
        X_rec = torch.rand(shape, requires_grad=True, device = 'cuda').float()
        model.eval()
        if X is not None:
            X_rec = X.detach().clone()
        # print(shape)
        # X_rec.data = normalize(X_rec.data,'cifar10')
        X_rec = self.normalize(X_rec).requires_grad_(True)
        # X_rec = X_rec.detach().clone().requires_grad_(True)
        
        with torch.no_grad():
            img_n= self.normalize(img.clone())
            out_feature= model(img_n)

        # opt = torch.optim.LBFGS([X_rec], lr=1, line_search_fn = 'strong_wolfe')
        # opt = torch.optim.LBFGS([X_rec], lr=1)
        opt = torch.optim.Adam([X_rec], lr=8e-1, eps=5e-3) #amsgrad=True,
        # opt = torch.optim.RMSprop([X_rec], lr=0.05, alpha=0.5)
        num = 50
        noise = 0.1 * torch.randn_like(X_rec).cuda()
        X_rec.data=X_rec.data+noise.data
        for i in range(num):
            # print (i)
            # def closure():
            with torch.enable_grad():
                opt.zero_grad()
                model.eval()
                gen_ir= model(X_rec)
                if "Gaussian" in self.regularization_option:
                    sigma = self.regularization_strength
                    noise = sigma * torch.randn_like(gen_ir).cuda()
                    gen_ir = gen_ir + noise

                # loss = nn.MSELoss()(10*gen_ir.cuda(), 10*feature)
                loss = nn.L1Loss()(gen_ir.cuda(), feature)
                TVLoss = TV(X_rec)
                normLoss = l2loss(X_rec)

                totalLoss = 5*loss # + 0.1 * TVLoss + 0.1 * normLoss
                with torch.no_grad():
                    rec=self.denormalize(X_rec.clone().data).cuda()
                    loss_mse = nn.MSELoss()(rec, img.clone())
               
                totalLoss.backward()
                # if loss<1:
                #     scaling_factor = 10/loss
                # else:
                scaling_factor=10
                for param in opt.param_groups[0]['params']:
                    if param.grad is not None:
                        param.grad *= scaling_factor
                X_rec_old=X_rec.detach().clone()
                opt.step()
                upd = nn.MSELoss()(X_rec.clone(), X_rec_old.clone())
                if i%(num/1)==0:
                    # print('loss (white-box): ', loss.detach())
                    print('loss of feature (white-box): ', loss.detach().item(),'loss of input (mse): ', loss_mse.detach().item(), 'input upd:',upd.item())
                # return loss

                # print (X_rec.grad.mean())
            # opt.step()
            

            X_rec.data = self.normalize(self.denormalize(X_rec.data).clamp(0, 1))
            X_rec.data = X_rec.data.clamp(0, 1)

        # step_size = 1e-3  # 固定步长

        loss = nn.MSELoss()(model(X_rec), out_feature)
        loss_clean=nn.MSELoss()(model(img_n), out_feature)
        print('loss (white-box): ', loss.detach(), 'org_loss:',loss_clean.detach() )
        X_rec.data = self.denormalize(X_rec.data).clamp(0, 1)
        return loss,X_rec

    # This function means testing of the attacker's inversion model
    def test_attack(self, num_epochs, local_model,decoder, sp_testloader, logger, path_dict, batch_size, num_classes=10,
                    select_label=0,sp_not=0):
        device = next(decoder.parameters()).device
        # # print("Load the best Decoder Model...")
        # new_state_dict = torch.load(path_dict["model_path"])
        # decoder.load_state_dict(new_state_dict)
        decoder.eval()
        # test_losses = []
        all_test_losses = AverageMeter()
        ssim_test_losses = AverageMeter()
        psnr_test_losses = AverageMeter()
        ssim_loss = pytorch_ssim.SSIM()

        criterion = nn.MSELoss()

        for i, (input, target) in enumerate(sp_testloader):
            input = input.cuda()
            img, ir=self.gen_inp_feat_pair(input,local_model)
            decoder.eval()
            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)
            if "Gaussian" in self.regularization_option:
                sigma = self.regularization_strength
                noise = sigma * torch.randn_like(ir).cuda()
                ir += noise
            output_imgs = decoder(ir)
            reconstruction_loss = criterion(output_imgs, img)
            ssim_loss_val = ssim_loss(output_imgs, img)
            psnr_loss_val = get_PSNR(img, output_imgs)
            all_test_losses.update(reconstruction_loss.item(), ir.size(0))
            ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
            psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
            if (i + 1) % 100 == 0:
                save_images(img, output_imgs, num_epochs, path_dict["test_output_path"], offset=i, batch_size=batch_size)

        logger.debug(
            "MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(all_test_losses.avg))
        logger.debug(
            "SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(ssim_test_losses.avg))
        logger.debug(
            "PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(psnr_test_losses.avg))
        return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

    # used for bhtsne
    def save_activation_bhtsne(self, save_activation, target, client_id):
        """
            Run one train epoch
        """

        path_dir = os.path.join(self.save_dir, 'save_activation_cutlayer')
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)

        save_activation = save_activation.float()
        save_activation = save_activation.cpu().numpy()
        save_activation = save_activation.reshape(self.batch_size, -1)
        np.savetxt(os.path.join(path_dir, "{}.txt".format(client_id)), save_activation, fmt='%.2f')

        target = target.float()
        target = target.cpu().numpy()
        target = target.reshape(self.batch_size, -1)
        np.savetxt(os.path.join(path_dir, "{}target.txt".format(client_id)), target, fmt='%.2f')

    #Generate test set for MIA decoder
    def save_image_act_pair(self, input, target, client_id, epoch, clean_option=False, attack_from_later_layer=-1, attack_option = "MIA"):
        """
            Run one train epoch
        """
        path_dir = os.path.join(self.save_dir, 'save_activation_client_{}_epoch_{}'.format(client_id, epoch))
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)
        else:
            rmtree(path_dir)
            os.makedirs(path_dir)
        input = input.cuda()

        for j in range(input.size(0)):
            img = input[None, j, :, :, :]
            label = target[None, j]
            with torch.no_grad():
                if client_id == 0:
                    self.f.eval()
                    save_activation = self.f(img)
                elif client_id == 1:
                    self.c.eval()
                    save_activation = self.c(img)
                elif client_id > 1:
                    self.model.local_list[client_id].eval()
                    save_activation = self.model.local_list[client_id](img)
                if self.confidence_score:
                    self.model.cloud.eval()
                    save_activation = self.model.cloud(save_activation)
                    if "mobilenetv2" in self.arch:
                        save_activation = F.avg_pool2d(save_activation, 4)
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
                    elif self.arch == "resnet20" or self.arch == "resnet32":
                        save_activation = F.avg_pool2d(save_activation, 8)
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
                    else:
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
            

            if attack_from_later_layer > -1 and (not self.confidence_score):
                self.model.cloud.eval()

                activation_3 = {}

                def get_activation_3(name):
                    def hook(model, input, output):
                        activation_3[name] = output.detach()

                    return hook

                with torch.no_grad():
                    activation_3 = {}
                    count = 0
                    for name, m in self.model.cloud.named_modules():
                        if attack_from_later_layer == count:
                            m.register_forward_hook(get_activation_3("ACT-{}".format(name)))
                            valid_key = "ACT-{}".format(name)
                            break
                        count += 1
                    output = self.model.cloud(save_activation)
                try:
                    save_activation = activation_3[valid_key]
                except:
                    print("cannot attack from later layer, server-side model is empty or does not have enough layers")
            # if "Gaussian" in self.regularization_option:
            #     print('adding noise in the test feature')
            #     sigma = self.regularization_strength
            #     noise = sigma * torch.randn_like(save_activation).cuda()
            #     # noise = 10000*sigma * torch.randn_like(save_activation).cuda()
            #     save_activation += noise
            if self.local_DP and not clean_option:  # local DP or additive noise
                if "laplace" in self.regularization_option:
                    save_activation += torch.from_numpy(
                        np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=save_activation.size())).cuda()
                    # the addtive work uses scale in (0.1 0.5 1.0) -> (1 2 10) regularization_strength (self.dp_epsilon)
                else:  # apply gaussian noise
                    delta = 10e-5
                    sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                    save_activation += sigma * torch.randn_like(save_activation).cuda()
            if self.dropout_defense and not clean_option:  # activation dropout defense
                save_activation = dropout_defense(save_activation, self.dropout_ratio)
            if self.topkprune and not clean_option:
                save_activation = prune_defense(save_activation, self.topkprune_ratio)
            
            img = denormalize(img, self.dataset)
                
            if self.gan_noise and not clean_option:
                epsilon = self.alpha2
                self.local_AE_list[client_id].eval()
                fake_act = save_activation.clone()
                grad = torch.zeros_like(save_activation).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = self.local_AE_list[client_id](fake_act)
                
                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, img)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(x_recon, img)
                    loss.backward()
                    grad += torch.sign(fake_act.grad)  

                save_activation = save_activation - grad.detach() * epsilon
            if "truncate" in attack_option:
                save_activation = prune_top_n_percent_left(save_activation)
            
            save_activation = save_activation.float()
            
            save_image(img, os.path.join(path_dir, "{}.jpg".format(j)))
            torch.save(save_activation.cpu(), os.path.join(path_dir, "{}.pt".format(j)))
            torch.save(label.cpu(), os.path.join(path_dir, "{}.label".format(j)))

    def model_pruning(self,model,ratio=0.5):
        example_inputs = torch.randn(1, 3, 32, 32).cuda()
        # 1. Importance criterion
        imp = tp.importance.GroupTaylorImportance()
        ignored_layers = []
        for m in model.modules():
            # if isinstance(m, torch.nn.BatchNorm2d):
            #     ignored_layers.append(m) # DO NOT prune the final classifier!
            if isinstance(m, torch.nn.Conv2d):
                if m.out_channels<=32:
                    ignored_layers.append(m)
        pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
            ignored_layers=ignored_layers,
        )
        # 3. Prune & finetune the model
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        if isinstance(imp, tp.importance.GroupTaylorImportance):
            # Taylor expansion requires gradients for importance estimation
            loss = model(example_inputs).sum() # A dummy loss, please replace this line with your loss function and data!
            loss.backward() # before pruner.step()
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        
        summary(model.cuda(), input_size=(3,32, 32))
        print('the param before pruning is:',base_nparams,'the param after pruning is:', nparams)
        return model
