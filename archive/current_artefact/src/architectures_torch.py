"""
architectures_torch.py — generic conv blocks, encoders, decoders, and the
inversion-attack autoencoder families used throughout the pipeline.

Most of these helpers come from the upstream CEM codebase. The two
families that the MIA attack actually relies on are `conv_normN_AE`
and `res_normN_AE` (e.g. `res_normN4C64`, `res_normN8C64` in
run_exp.sh) — the rest are kept for completeness so that older
ablation configs can still be reconstructed.
"""

import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Plain pre-activation residual block, used as a building block by
    most of the encoder/decoder factories below.

    `bn=True` switches on BatchNorm before each conv; the upstream code
    keeps `bn=False` in some places for the unbatched generators.
    """

    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1)

        # Identity shortcut by default; if the spatial size or channel
        # count changes we use a 1x1 conv with BN to match dims.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out


def resnet(input_shape, level):
    """Generic ResNet-style cloud-side encoder, levels 1..4.

    The depth grows with `level`, mirroring the cut-layer choice in
    the upstream split-learning protocol.
    """
    net = []
    net += [nn.Conv2d(input_shape[0], 64, 3, 1, 1)]
    net += [nn.BatchNorm2d(64)]
    net += [nn.ReLU()]
    net += [nn.MaxPool2d(2)]
    net += [ResBlock(64, 64)]
    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(64, 128, stride=2)]
    if level == 2:
        return nn.Sequential(*net)

    net += [ResBlock(128, 128)]
    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock(128, 256, stride=2)]
    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def resnet_tail(level, num_class=10):
    """Server-side ResNet tail. Picks up where `resnet(level)` left off
    and finishes with the classifier head."""
    print(level)
    net = []
    if level <= 1:
        net += [ResBlock(64, 128, stride=2)]
    if level <= 2:
        net += [ResBlock(128, 128)]
    if level <= 3:
        net += [ResBlock(128, 256, stride=2)]
    net += [ResBlock(256, 256, stride=1)]
    net += [ResBlock(256, 512, stride=2)]
    net += [ResBlock(512, 512, stride=1)]
    net += [ResBlock(512, 1024, stride=2)]
    net += [ResBlock(1024, 1024, stride=1)]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(num_class)]
    return nn.Sequential(*net)


def pilot(input_shape, level):
    """Plain conv "pilot" encoder used as a baseline against ResNet."""
    net = []
    act = None
    print("[PILOT] activation: ", act)

    net += [nn.Conv2d(input_shape[0], 64, 3, 2, 1)]
    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]
    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(128, 256, 3, 2, 1)]
    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


class View(nn.Module):
    """Reshape helper so that nn.Sequential pipelines can include a
    .view() without breaking the Module-only contract."""

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def make_generator(latent_size):
    """GAN generator used by some of the upstream defence baselines
    (gan_adv_*). Not used in the headline SCA-CEM run."""
    net = []
    net += [torch.nn.Linear(latent_size, 8 * 8 * 256, bias=False)]
    net += [torch.nn.BatchNorm1d(8 * 8 * 256)]
    net += [torch.nn.LeakyReLU()]
    net += [View((-1, 256, 8, 8))]
    net += [torch.nn.ConvTranspose2d(256, 128, 3, 1, padding=1, bias=False)]
    net += [torch.nn.BatchNorm2d(128)]
    net += [torch.nn.LeakyReLU()]

    net += [torch.nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1, bias=False)]
    net += [torch.nn.BatchNorm2d(64)]
    net += [torch.nn.LeakyReLU()]

    net += [torch.nn.ConvTranspose2d(64, 3, 3, 2, padding=1, output_padding=1, bias=False)]
    net += [torch.nn.Tanh()]
    return nn.Sequential(*net)


def multihead_buffer(feature_size):
    """Plain 3-block conv stack used as a buffer between encoder and
    multi-head outputs."""
    assert len(feature_size) == 4
    net = []
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    return nn.Sequential(*net)


def multihead_buffer_res(feature_size):
    """Same idea as `multihead_buffer` but using two ResBlocks. Lets
    you swap a deeper residual buffer in without changing the call
    site."""
    assert len(feature_size) == 4
    net = []
    net += [ResBlock(feature_size[1], feature_size[1])]
    net += [ResBlock(feature_size[1], feature_size[1])]
    return nn.Sequential(*net)


def cifar_pilot(output_dim, level):
    """CIFAR-shaped variant of `pilot`. The output spatial size is
    derived from `output_dim[2]` rather than from `level`, so the same
    function works for the 32/16/8/4 cut-layer choices."""
    net = []
    act = None
    print("[PILOT] activation: ", act)
    print(output_dim)
    if output_dim[2] == 32:
        net += [nn.Conv2d(3, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(3, 64, 3, 2, 1)]
    net += [nn.Conv2d(64, 64, 3, 1, 1)]
    if output_dim[2] == 16:
        net += [nn.Conv2d(64, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]
    net += [nn.Conv2d(128, 128, 3, 1, 1)]
    if output_dim[2] == 8:
        net += [nn.Conv2d(128, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(128, 256, 3, 2, 1)]
    if output_dim[2] == 4:
        net += [nn.Conv2d(256, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def decoder(input_shape, level, channels=3):
    """Generic upsampling decoder used by the GAN-style defences.

    The 1/3/4 branches mirror the encoder levels above; each level
    halves the channel count and doubles the spatial size.
    """
    net = []
    act = None
    print("[DECODER] activation: ", act)

    net += [nn.ConvTranspose2d(input_shape[0], 256, 3, 2, 1, output_padding=1)]
    if level == 1:
        net += [nn.Conv2d(256, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]
    if level <= 3:
        net += [nn.Conv2d(128, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

    net += [nn.ConvTranspose2d(128, channels, 3, 2, 1, output_padding=1)]
    net += [nn.Tanh()]
    return nn.Sequential(*net)


def cifar_decoder(input_shape, channels=3):
    """CIFAR-shaped decoder. Branches on the input spatial size (16/8/4)
    rather than on a level int, like `cifar_pilot`."""
    net = []
    act = None
    print("[DECODER] activation: ", act)

    if input_shape[2] == 16:
        net += [nn.Conv2d(input_shape[0], 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

    elif input_shape[2] == 8:
        net += [nn.Conv2d(input_shape[0], 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    elif input_shape[2] == 4:
        net += [nn.Conv2d(input_shape[0], 256, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    else:
        raise Exception('No Dim %d' % input_shape[2])


class inference_model(nn.Module):
    """Membership-inference classifier: takes (logits, one-hot label) and
    predicts whether the sample was in the training set.

    Used by the membership-inference attack baseline; not active in the
    SCA-CEM headline run.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(inference_model, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.labels = nn.Sequential(
            nn.Linear(num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(128 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.output = nn.Sigmoid()

    def forward(self, x, l):
        out_x = self.features(x)
        out_l = self.labels(l)
        is_member = self.combine(torch.cat((out_x, out_l), 1))
        return self.output(is_member)


def discriminator(input_shape, level):
    """GAN discriminator used by the gan_adv defence family."""
    net = []
    if level == 1:
        net += [nn.Conv2d(input_shape[0], 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level <= 3:
        net += [nn.Conv2d(input_shape[0], 256, 3, 2, 1)]
    elif level <= 4:
        net += [nn.Conv2d(input_shape[0], 256, 3, 1, 1)]

    bn = False
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(1)]
    return nn.Sequential(*net)


# ──────────────────────────────────────────────────────────────────────
# AE families used as the attacker's inversion model. The MIA pipeline
# instantiates one of these (with the chosen depth N and channel count
# C) to invert the smashed feature back to image space.
# ──────────────────────────────────────────────────────────────────────


class custom_AE_bn(nn.Module):
    """Simple BN+ReLU upsampling AE. Used by older configs; superseded
    by `conv_normN_AE` for the parameterised-depth experiments."""

    def __init__(self, input_nc=256, output_nc=3,
                 input_dim=8, output_dim=32, activation="sigmoid"):
        super(custom_AE_bn, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            model += [nn.Conv2d(nc, int(nc / 2), kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(int(nc / 2))]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc / 2), int(nc / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(int(nc / 2))]
            model += [nn.ReLU()]
            nc = int(nc / 2)
        if upsampling_num >= 1:
            mid = int(input_nc / (2 ** (upsampling_num - 1)))
            model += [nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(mid)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(mid, output_nc, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(input_nc)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class custom_AE(nn.Module):
    """Same shape as `custom_AE_bn` but without BatchNorm. Not used by
    the final run; kept as a reference baseline."""

    def __init__(self, input_nc=256, output_nc=3,
                 input_dim=8, output_dim=32, activation="sigmoid"):
        super(custom_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            model += [nn.Conv2d(nc, int(nc / 2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc / 2), int(nc / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc / 2)
        if upsampling_num >= 1:
            mid = int(input_nc / (2 ** (upsampling_num - 1)))
            model += [nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(mid, output_nc, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class conv_normN_AE(nn.Module):
    """Plain-conv decoder with parameterised depth N and width
    `internal_nc`. Picked via the `--gan_AE_type=conv_normN<N>C<C>`
    string in run_exp.sh.

    `input_dim=0` is a special signal that the decoder is being fed a
    confidence-score vector rather than a 2D feature map; in that case
    we treat the input as a (B, num_classes, 1, 1) tensor.
    """

    def __init__(self, N=0, internal_nc=64, input_nc=256, output_nc=3,
                 input_dim=8, output_dim=32, activation="sigmoid"):
        super(conv_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim))
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
        # First 1x1-channel-mix layer: feature_dim → internal_nc.
        model += [nn.Conv2d(input_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
        model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        # N "middle" same-width conv blocks. Most of the decoder's
        # capacity sits here.
        for _ in range(N):
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
            model += [nn.ReLU()]
        model += [nn.Dropout(0.25)]

        # Two upsampling stages: each ×2. If we don't actually need
        # that many ×2s (small `upsampling_num`), substitute plain
        # convs to keep the parameter count comparable.
        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        # Any extra ×2s beyond the first two (for very small input
        # spatial sizes, e.g. 4×4 → 32×32 needs 3 stages).
        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3,
                                             stride=2, padding=1, output_padding=1)]
                model += [nn.BatchNorm2d(internal_nc)]
                model += [nn.ReLU()]

        # Final 3-channel projection + activation (sigmoid keeps the
        # output in [0,1] for normalised images).
        model += [nn.Conv2d(internal_nc, output_nc, kernel_size=3, stride=1, padding=1)]
        model += [nn.BatchNorm2d(output_nc)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            # Reshape (B, num_classes) → (B, num_classes, 1, 1) so the
            # conv stack can treat it as a 1×1 feature map.
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output


class res_normN_AE(nn.Module):
    """Residual variant of `conv_normN_AE` — same structural layout but
    each "middle" block is a ResBlock. This is the family used by the
    headline run (`res_normN4C64` for training-time, `res_normN8C64`
    for inference-time)."""

    def __init__(self, N=0, internal_nc=64, input_nc=256, output_nc=3,
                 input_dim=8, output_dim=32, activation="sigmoid"):
        super(res_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim))
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
        model += [ResBlock(input_nc, internal_nc, bn=True, stride=1)]
        model += [nn.ReLU()]

        for _ in range(N):
            model += [ResBlock(internal_nc, internal_nc, bn=True, stride=1)]
            model += [nn.ReLU()]

        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn=True, stride=1)]
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3,
                                         stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn=True, stride=1)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3,
                                             stride=2, padding=1, output_padding=1)]
                model += [nn.BatchNorm2d(internal_nc)]
                model += [nn.ReLU()]

        model += [ResBlock(internal_nc, output_nc, bn=True, stride=1)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output


def classifier_binary(input_shape, class_num):
    """Tiny linear classifier head used by the binary-class baselines."""
    net = []
    net += [nn.ReLU()]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(256)]
    net += [nn.ReLU()]
    net += [nn.Linear(256, 128)]
    net += [nn.ReLU()]
    net += [nn.Linear(128, class_num)]
    return nn.Sequential(*net)


def pilotClass(input_shape, level):
    """Pilot encoder that uses SiLU instead of ReLU. Kept as a sweep
    target; not used in the headline run."""
    net = []
    net += [nn.Conv2d(input_shape[0], 64, 3, 2, 1)]
    net += [nn.SiLU]
    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]
    net += [nn.SiLU]
    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(128, 256, 3, 2, 1)]
    net += [nn.SiLU]
    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


# Pre-built (encoder, pilot, decoder, discriminator, tail) tuples for
# levels 1..5, plus two extra binary-class entries at levels 4 and 3.
SETUPS = [
    (functools.partial(resnet, level=i), functools.partial(pilot, level=i),
     functools.partial(decoder, level=i), functools.partial(discriminator, level=i),
     functools.partial(resnet_tail, level=i))
    for i in range(1, 6)
]

l = 4
SETUPS += [(functools.partial(resnet, level=l),
            functools.partial(pilot, level=l),
            classifier_binary,
            functools.partial(discriminator, level=l),
            functools.partial(resnet_tail, level=l))]

l = 3
SETUPS += [(functools.partial(resnet, level=l),
            functools.partial(pilot, level=l),
            classifier_binary,
            functools.partial(discriminator, level=l),
            functools.partial(resnet_tail, level=l))]
