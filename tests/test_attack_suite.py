import torch
import torch.nn.functional as F

from publication_cem.models import (
    PatchDiscriminator,
    build_inversion_decoder,
    build_split_model,
)
from scripts.train_attack import convergence_record


def test_every_attack_decoder_has_expected_shape_and_gradients():
    smashed = torch.randn(2, 4, 16, 16)
    target = torch.rand(2, 3, 64, 64)
    for attack_type in ("conv_decoder", "residual_decoder", "gan", "adaptive"):
        decoder = build_inversion_decoder(attack_type, 4, 64, 16)
        reconstruction = decoder(smashed)
        assert reconstruction.shape == target.shape
        assert torch.all((0 <= reconstruction) & (reconstruction <= 1))
        F.mse_loss(reconstruction, target).backward()
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in decoder.parameters()
        )


def test_patch_discriminator_supports_real_and_fake_backward_passes():
    discriminator = PatchDiscriminator(width=16)
    real = torch.rand(2, 3, 64, 64)
    fake = torch.rand(2, 3, 64, 64, requires_grad=True)
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)
    loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    loss += F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    loss.backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()


def test_convergence_is_only_declared_after_ten_stable_epochs():
    assert convergence_record([1.0] * 9) == (False, None)
    converged, relative_change = convergence_record([1.0] * 10)
    assert converged
    assert relative_change == 0.0


def test_vgg_and_resnet_share_the_split_classifier_contract():
    for name, cut_index in (("vgg11_bn_split", 8), ("resnet18_split", 2)):
        model = build_split_model(
            {
                "name": name,
                "cut_index": cut_index,
                "bottleneck_channels": 4,
                "pretrained": False,
            },
            num_classes=5,
            image_size=64,
        )
        images = torch.rand(2, 3, 64, 64)
        smashed = model.encode(images)
        logits = model.decode(smashed)
        assert smashed.shape[1] == 4
        assert logits.shape == (2, 5)


def test_shape_inference_preserves_batch_norm_state_and_module_modes():
    model = build_split_model(
        {
            "name": "vgg11_bn_split",
            "cut_index": 8,
            "bottleneck_channels": 4,
            "pretrained": False,
        },
        num_classes=5,
        image_size=64,
    )
    model.train()
    model.local[1].eval()
    batch_norms = [
        module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)
    ]
    running_state = [
        (module.running_mean.clone(), module.running_var.clone())
        for module in batch_norms
    ]
    training_state = [module.training for module in model.modules()]

    assert model.smashed_shape((3, 64, 64)) == (4, 16, 16)

    assert training_state == [module.training for module in model.modules()]
    for module, (running_mean, running_var) in zip(batch_norms, running_state):
        assert torch.equal(module.running_mean, running_mean)
        assert torch.equal(module.running_var, running_var)


def test_adaptive_noise_keeps_power_fixed_and_margin_head_is_trainable():
    model = build_split_model(
        {
            "name": "vgg11_bn_split",
            "cut_index": 8,
            "bottleneck_channels": 4,
            "pretrained": False,
            "adaptive_channel_noise": {
                "enabled": True,
                "max_log_ratio": 0.7,
            },
            "margin_head": {
                "enabled": True,
                "embedding_dim": 16,
                "scale": 12.0,
                "margin": 0.1,
            },
        },
        num_classes=5,
        image_size=64,
    )
    with torch.no_grad():
        model.channel_noise.allocation_logits.copy_(
            torch.tensor([-2.0, -0.5, 0.5, 2.0])
        )
    multipliers = model.channel_noise.channel_multipliers()
    assert torch.allclose(multipliers.square().mean(), torch.tensor(1.0))

    images = torch.rand(2, 3, 64, 64)
    labels = torch.tensor([1, 3])
    smashed = model.encode(images)
    transmitted = model.transmit(smashed, 0.025)
    assert transmitted.shape == smashed.shape
    margin_logits = model.auxiliary_margin_logits(smashed, labels)
    assert margin_logits.shape == (2, 5)
    F.cross_entropy(margin_logits, labels).backward()
    assert model.margin_head.class_weights.grad is not None
