from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SplitClassifier(nn.Module):
    """Explicit edge/cloud classifier with an auditable transmitted tensor."""

    def __init__(
        self,
        local: nn.Module,
        bottleneck_encoder: nn.Module,
        bottleneck_decoder: nn.Module,
        cloud: nn.Module,
        classifier: nn.Module,
        channel_noise: Optional[PowerConstrainedChannelNoise] = None,
        margin_head: Optional[AngularMarginHead] = None,
    ) -> None:
        super().__init__()
        self.local = local
        self.bottleneck_encoder = bottleneck_encoder
        self.bottleneck_decoder = bottleneck_decoder
        self.cloud = cloud
        self.classifier = classifier
        self.channel_noise = channel_noise
        self.margin_head = margin_head

    def encode(self, images: Tensor) -> Tensor:
        return self.bottleneck_encoder(self.local(images))

    def decode(self, smashed_features: Tensor) -> Tensor:
        hidden = self.cloud(self.bottleneck_decoder(smashed_features))
        hidden = F.adaptive_avg_pool2d(hidden, output_size=1).flatten(start_dim=1)
        return self.classifier(hidden)

    def transmit(
        self, features: Tensor, noise_std: float, generator=None
    ) -> Tensor:
        if self.channel_noise is None:
            return add_gaussian_noise(features, noise_std, generator)
        return self.channel_noise(features, noise_std, generator)

    def channel_noise_variance(self, noise_std: float) -> Optional[Tensor]:
        if self.channel_noise is None:
            return None
        return self.channel_noise.channel_std(noise_std).square()

    def auxiliary_margin_logits(self, features: Tensor, labels: Tensor) -> Tensor:
        if self.margin_head is None:
            raise RuntimeError("the model has no angular-margin head")
        return self.margin_head(features, labels)

    def forward(
        self, images: Tensor, noise_std: float = 0.0, generator=None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        clean = self.encode(images)
        transmitted = self.transmit(clean, noise_std, generator)
        return self.decode(transmitted), clean, transmitted

    @torch.no_grad()
    def smashed_shape(self, input_shape: Sequence[int]) -> Tuple[int, ...]:
        device = next(self.parameters()).device
        training_states = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            sample = torch.zeros(1, *input_shape, device=device)
            return tuple(self.encode(sample).shape[1:])
        finally:
            for module, was_training in training_states:
                module.training = was_training


@torch.no_grad()
def _infer_local_shape(local: nn.Module, image_size: int) -> torch.Size:
    training_states = [(module, module.training) for module in local.modules()]
    try:
        local.eval()
        return local(torch.zeros(1, 3, image_size, image_size)).shape
    finally:
        for module, was_training in training_states:
            module.training = was_training


class PowerConstrainedChannelNoise(nn.Module):
    """Learn a channel allocation while keeping total Gaussian power fixed."""

    def __init__(
        self,
        channels: int,
        max_log_ratio: float = math.log(2.0),
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if max_log_ratio < 0:
            raise ValueError("max_log_ratio must be non-negative")
        self.channels = channels
        self.max_log_ratio = float(max_log_ratio)
        self.allocation_logits = nn.Parameter(
            torch.zeros(channels), requires_grad=learnable
        )

    def channel_multipliers(self) -> Tensor:
        bounded = self.max_log_ratio * torch.tanh(self.allocation_logits)
        multipliers = torch.exp(bounded)
        return multipliers / multipliers.square().mean().sqrt()

    def channel_std(self, base_std: float) -> Tensor:
        if base_std < 0:
            raise ValueError("base_std must be non-negative")
        return self.channel_multipliers() * base_std

    def forward(self, features: Tensor, base_std: float, generator=None) -> Tensor:
        if features.ndim < 2 or features.shape[1] != self.channels:
            raise ValueError(
                f"expected channel dimension {self.channels}, got {tuple(features.shape)}"
            )
        if base_std == 0:
            return features
        channel_std = self.channel_std(base_std)
        shape = [1, self.channels, *([1] * (features.ndim - 2))]
        noise = torch.randn(
            features.shape,
            device=features.device,
            dtype=features.dtype,
            generator=generator,
        )
        return features + channel_std.reshape(shape) * noise


class AngularMarginHead(nn.Module):
    """Training-only identity head that protects inter-class angular margins."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        embedding_dim: int = 128,
        scale: float = 30.0,
        margin: float = 0.20,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_classes <= 0 or embedding_dim <= 0:
            raise ValueError("margin-head dimensions must be positive")
        if scale <= 0 or not 0 <= margin < math.pi / 2:
            raise ValueError("invalid angular-margin scale or margin")
        self.scale = float(scale)
        self.margin = float(margin)
        self.embedding = nn.Sequential(
            nn.Linear(2 * channels, embedding_dim, bias=False),
            nn.LayerNorm(embedding_dim),
        )
        self.class_weights = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.class_weights)

    def summarise(self, features: Tensor) -> Tensor:
        flattened = features.flatten(start_dim=2)
        means = flattened.mean(dim=2)
        deviations = flattened.var(dim=2, unbiased=False).clamp_min(1e-8).sqrt()
        return self.embedding(torch.cat((means, deviations), dim=1))

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        embeddings = F.normalize(self.summarise(features), dim=1)
        weights = F.normalize(self.class_weights, dim=1)
        cosine = F.linear(embeddings, weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        target_cosine = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        one_hot = F.one_hot(labels, num_classes=weights.shape[0]).to(cosine.dtype)
        return self.scale * (one_hot * target_cosine + (1.0 - one_hot) * cosine)


class ImageNetNormalise(nn.Module):
    """Apply ImageNet normalisation inside the model, keeping images in [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images: Tensor) -> Tensor:
        return (images - self.mean) / self.std


class SpatialSlotTokenizer(nn.Module):
    """Compress a spatial feature map into unordered semantic identity tokens."""

    def __init__(
        self,
        input_channels: int,
        token_dim: int,
        num_slots: int = 4,
        iterations: int = 3,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        side = math.isqrt(num_slots)
        if side * side != num_slots:
            raise ValueError("num_slots must be a perfect square")
        if input_channels <= 0 or token_dim <= 0 or iterations <= 0:
            raise ValueError("tokenizer dimensions and iterations must be positive")
        self.num_slots = num_slots
        self.token_dim = token_dim
        self.side = side
        self.iterations = iterations
        hidden_dim = max(token_dim, int(token_dim * mlp_ratio))

        self.input_projection = nn.Conv2d(input_channels, token_dim, kernel_size=1)
        self.input_norm = nn.LayerNorm(token_dim)
        self.slot_norm = nn.LayerNorm(token_dim)
        self.mlp_norm = nn.LayerNorm(token_dim)
        self.keys = nn.Linear(token_dim, token_dim, bias=False)
        self.values = nn.Linear(token_dim, token_dim, bias=False)
        self.queries = nn.Linear(token_dim, token_dim, bias=False)
        self.slots = nn.Parameter(torch.empty(1, num_slots, token_dim))
        self.gru = nn.GRUCell(token_dim, token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self.output_norm = nn.LayerNorm(token_dim)
        nn.init.normal_(self.slots, std=token_dim**-0.5)

    def forward(self, feature_map: Tensor) -> Tensor:
        inputs = self.input_projection(feature_map).flatten(start_dim=2).transpose(1, 2)
        inputs = self.input_norm(inputs)
        keys = self.keys(inputs)
        values = self.values(inputs)
        slots = self.slots.expand(feature_map.shape[0], -1, -1)

        for _ in range(self.iterations):
            previous = slots
            queries = self.queries(self.slot_norm(slots)) * self.token_dim**-0.5
            logits = torch.einsum("bsd,bnd->bsn", queries, keys)
            # Each spatial token competes across slots; rows are then normalised
            # so a slot update remains stable when the input resolution changes.
            attention = logits.softmax(dim=1).clamp_min(1e-8)
            attention = attention / attention.sum(dim=2, keepdim=True)
            updates = torch.einsum("bsn,bnd->bsd", attention, values)
            slots = self.gru(
                updates.reshape(-1, self.token_dim),
                previous.reshape(-1, self.token_dim),
            ).view_as(previous)
            slots = slots + self.mlp(self.mlp_norm(slots))

        slots = self.output_norm(slots)
        return slots.transpose(1, 2).reshape(
            feature_map.shape[0], self.token_dim, self.side, self.side
        )


class PermutationInvariantTokenCloud(nn.Module):
    """Classify token sets without exposing or relying on a spatial ordering."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if token_dim % heads:
            raise ValueError("token_dim must be divisible by attention heads")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        sequence = tokens.flatten(start_dim=2).transpose(1, 2)
        pooled = self.encoder(sequence).mean(dim=1)
        return self.output(pooled).unsqueeze(-1).unsqueeze(-1)


def add_gaussian_noise(
    features: Tensor, noise_std: float | Tensor, generator=None
) -> Tensor:
    if isinstance(noise_std, Tensor):
        if torch.any(noise_std < 0):
            raise ValueError("noise_std must be non-negative")
    elif noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not isinstance(noise_std, Tensor) and noise_std == 0:
        return features
    noise = torch.randn(
        features.shape,
        device=features.device,
        dtype=features.dtype,
        generator=generator,
    )
    if isinstance(noise_std, Tensor) and noise_std.ndim == 1:
        if features.ndim < 2 or noise_std.numel() != features.shape[1]:
            raise ValueError("channel noise_std does not match feature channels")
        shape = [1, noise_std.numel(), *([1] * (features.ndim - 2))]
        noise_std = noise_std.reshape(shape)
    return features + noise_std * noise


def _build_channel_modules(
    bottleneck_channels: int,
    num_classes: int,
    adaptive_channel_noise: Optional[Mapping[str, Any]],
    margin_head: Optional[Mapping[str, Any]],
) -> tuple[Optional[PowerConstrainedChannelNoise], Optional[AngularMarginHead]]:
    noise_module = None
    if adaptive_channel_noise and adaptive_channel_noise.get("enabled", False):
        noise_module = PowerConstrainedChannelNoise(
            bottleneck_channels,
            max_log_ratio=float(
                adaptive_channel_noise.get("max_log_ratio", math.log(2.0))
            ),
            learnable=bool(adaptive_channel_noise.get("learnable", True)),
        )
    margin_module = None
    if margin_head and margin_head.get("enabled", False):
        margin_module = AngularMarginHead(
            bottleneck_channels,
            num_classes,
            embedding_dim=int(margin_head.get("embedding_dim", 128)),
            scale=float(margin_head.get("scale", 30.0)),
            margin=float(margin_head.get("margin", 0.20)),
        )
    return noise_module, margin_module


def build_vgg11_bn_split(
    num_classes: int,
    image_size: int = 64,
    cut_index: int = 8,
    bottleneck_channels: int = 16,
    pretrained: bool = False,
    adaptive_channel_noise: Optional[Mapping[str, Any]] = None,
    margin_head: Optional[Mapping[str, Any]] = None,
) -> SplitClassifier:
    """Build a controlled VGG11-BN split after the second pooling block.

    The default cut produces a 16x16 spatial map for 64x64 inputs. LCA is not
    present; it must be introduced as an explicit ablation if required.
    """

    try:
        from torchvision.models import VGG11_BN_Weights, vgg11_bn
    except ImportError as exc:
        raise RuntimeError("torchvision is required to build VGG11-BN") from exc

    weights = VGG11_BN_Weights.DEFAULT if pretrained else None
    backbone = vgg11_bn(weights=weights)
    layers = list(backbone.features.children())
    if not 1 <= cut_index < len(layers):
        raise ValueError("cut_index must split the VGG feature sequence")
    local = nn.Sequential(*layers[:cut_index])
    cloud = nn.Sequential(*layers[cut_index:])

    local_shape = _infer_local_shape(local, image_size)
    local_channels = int(local_shape[1])
    bottleneck_encoder = nn.Conv2d(
        local_channels, bottleneck_channels, kernel_size=1, bias=False
    )
    bottleneck_decoder = nn.Conv2d(
        bottleneck_channels, local_channels, kernel_size=1, bias=False
    )
    classifier = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )
    noise_module, margin_module = _build_channel_modules(
        bottleneck_channels,
        num_classes,
        adaptive_channel_noise,
        margin_head,
    )
    return SplitClassifier(
        local,
        bottleneck_encoder,
        bottleneck_decoder,
        cloud,
        classifier,
        noise_module,
        margin_module,
    )


def build_resnet18_split(
    num_classes: int,
    image_size: int = 64,
    cut_index: int = 2,
    bottleneck_channels: int = 16,
    pretrained: bool = False,
    adaptive_channel_noise: Optional[Mapping[str, Any]] = None,
    margin_head: Optional[Mapping[str, Any]] = None,
) -> SplitClassifier:
    """Build a ResNet-18 split at a stage boundary.

    ``cut_index`` selects how many chunks run at the edge: 1 is the stem,
    2 is stem+layer1, and 3 is stem+layer1+layer2.
    """

    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError("torchvision is required to build ResNet-18") from exc
    if cut_index not in {1, 2, 3}:
        raise ValueError("ResNet-18 cut_index must be one of 1, 2, or 3")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    chunks = [
        nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    ]
    local = nn.Sequential(*chunks[:cut_index])
    cloud = nn.Sequential(*chunks[cut_index:])
    local_shape = _infer_local_shape(local, image_size)
    local_channels = int(local_shape[1])
    bottleneck_encoder = nn.Conv2d(
        local_channels, bottleneck_channels, kernel_size=1, bias=False
    )
    bottleneck_decoder = nn.Conv2d(
        bottleneck_channels, local_channels, kernel_size=1, bias=False
    )
    noise_module, margin_module = _build_channel_modules(
        bottleneck_channels,
        num_classes,
        adaptive_channel_noise,
        margin_head,
    )
    return SplitClassifier(
        local,
        bottleneck_encoder,
        bottleneck_decoder,
        cloud,
        nn.Linear(512, num_classes),
        noise_module,
        margin_module,
    )


def build_mobilenet_slot_split(
    num_classes: int,
    image_size: int = 48,
    bottleneck_channels: int = 128,
    pretrained: bool = True,
    num_slots: int = 4,
    slot_iterations: int = 3,
    cloud_hidden_dim: int = 256,
    cloud_layers: int = 2,
    cloud_heads: int = 4,
    dropout: float = 0.1,
    backbone_variant: str = "large",
    adaptive_channel_noise: Optional[Mapping[str, Any]] = None,
    margin_head: Optional[Mapping[str, Any]] = None,
) -> SplitClassifier:
    """Build a semantic token bottleneck on a lightweight MobileNetV3 encoder."""

    try:
        from torchvision.models import (
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
            mobilenet_v3_large,
            mobilenet_v3_small,
        )
    except ImportError as exc:
        raise RuntimeError("torchvision is required to build MobileNetV3") from exc
    if backbone_variant == "large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
    elif backbone_variant == "small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
    else:
        raise ValueError("backbone_variant must be 'small' or 'large'")
    local = nn.Sequential(ImageNetNormalise(), backbone.features)
    local_shape = _infer_local_shape(local, image_size)
    tokenizer = SpatialSlotTokenizer(
        input_channels=int(local_shape[1]),
        token_dim=bottleneck_channels,
        num_slots=num_slots,
        iterations=slot_iterations,
    )
    cloud = PermutationInvariantTokenCloud(
        token_dim=bottleneck_channels,
        hidden_dim=cloud_hidden_dim,
        layers=cloud_layers,
        heads=cloud_heads,
        dropout=dropout,
    )
    noise_module, margin_module = _build_channel_modules(
        bottleneck_channels,
        num_classes,
        adaptive_channel_noise,
        margin_head,
    )
    return SplitClassifier(
        local=local,
        bottleneck_encoder=tokenizer,
        bottleneck_decoder=nn.Identity(),
        cloud=cloud,
        classifier=nn.Linear(cloud_hidden_dim, num_classes),
        channel_noise=noise_module,
        margin_head=margin_module,
    )


def build_split_model(
    model_config: Mapping[str, Any],
    num_classes: int,
    image_size: int,
) -> SplitClassifier:
    name = model_config["name"]
    if name == "mobilenet_slot_split":
        return build_mobilenet_slot_split(
            num_classes=num_classes,
            image_size=image_size,
            bottleneck_channels=int(model_config.get("bottleneck_channels", 128)),
            pretrained=bool(model_config.get("pretrained", True)),
            num_slots=int(model_config.get("num_slots", 4)),
            slot_iterations=int(model_config.get("slot_iterations", 3)),
            cloud_hidden_dim=int(model_config.get("cloud_hidden_dim", 256)),
            cloud_layers=int(model_config.get("cloud_layers", 2)),
            cloud_heads=int(model_config.get("cloud_heads", 4)),
            dropout=float(model_config.get("dropout", 0.1)),
            backbone_variant=str(model_config.get("backbone_variant", "large")),
            adaptive_channel_noise=model_config.get("adaptive_channel_noise"),
            margin_head=model_config.get("margin_head"),
        )
    common = {
        "num_classes": num_classes,
        "image_size": image_size,
        "cut_index": int(model_config["cut_index"]),
        "bottleneck_channels": int(model_config["bottleneck_channels"]),
        "pretrained": bool(model_config.get("pretrained", False)),
        "adaptive_channel_noise": model_config.get("adaptive_channel_noise"),
        "margin_head": model_config.get("margin_head"),
    }
    if name == "vgg11_bn_split":
        return build_vgg11_bn_split(**common)
    if name == "resnet18_split":
        return build_resnet18_split(**common)
    raise ValueError(f"unsupported split model: {name}")


class InversionDecoder(nn.Module):
    """Architecture-neutral decoder attack for transmitted feature maps."""

    def __init__(
        self,
        input_channels: int,
        output_size: int = 64,
        width: int = 128,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 4, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, smashed_features: Tensor) -> Tensor:
        hidden = self.stem(smashed_features)
        hidden = F.interpolate(
            hidden,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.reconstruct(hidden)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.gelu(inputs + self.layers(inputs))


class ResidualInversionDecoder(nn.Module):
    """Higher-capacity residual decoder matching the public CEM attack family."""

    def __init__(
        self,
        input_channels: int,
        output_size: int = 64,
        width: int = 128,
        blocks: int = 6,
    ) -> None:
        super().__init__()
        groups = min(8, width)
        while width % groups:
            groups -= 1
        self.output_size = output_size
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, smashed_features: Tensor) -> Tensor:
        hidden = self.blocks(self.stem(smashed_features))
        hidden = F.interpolate(
            hidden,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.head(hidden)


class PatchDiscriminator(nn.Module):
    """Small image discriminator used only by the GAN-style inversion attack."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        layers = []
        channels = 3
        for multiplier in (1, 2, 4):
            output_channels = width * multiplier
            layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        output_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = output_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)


def build_inversion_decoder(
    attack_type: str,
    input_channels: int,
    output_size: int,
    width: int,
) -> nn.Module:
    if attack_type == "conv_decoder":
        return InversionDecoder(input_channels, output_size, width)
    if attack_type in {"residual_decoder", "gan", "adaptive"}:
        blocks = 8 if attack_type == "adaptive" else 6
        return ResidualInversionDecoder(
            input_channels,
            output_size,
            width,
            blocks=blocks,
        )
    raise ValueError(f"unsupported attack type: {attack_type}")
