import torch

from publication_cem.models import PowerConstrainedChannelNoise
from scripts.calibrate_channel_noise import (
    calibrated_multipliers,
    channel_scores,
    multipliers_to_logits,
)


def test_calibration_round_trip_preserves_fixed_noise_power() -> None:
    ratios = torch.tensor([0.5, 1.0, 2.0, 3.0])
    expected = calibrated_multipliers(ratios, 0.3, 0.5, 2.0)
    module = PowerConstrainedChannelNoise(4, max_log_ratio=0.7)
    with torch.no_grad():
        module.allocation_logits.copy_(multipliers_to_logits(expected, 0.7))
    realised = module.channel_multipliers()
    assert torch.allclose(realised, expected, atol=1e-6)
    assert torch.allclose(realised.square().mean(), torch.tensor(1.0))


def test_channel_scores_support_final_rule_and_component_ablations() -> None:
    channels = [
        {
            "task_gradient_share": 0.25,
            "reconstruction_gradient_share": 0.5,
            "reconstruction_to_task_ratio": 2.0,
        },
        {
            "task_gradient_share": 0.75,
            "reconstruction_gradient_share": 0.5,
            "reconstruction_to_task_ratio": 2.0 / 3.0,
        },
    ]
    ratio = channel_scores(channels, "reconstruction_to_task")
    reconstruction = channel_scores(channels, "reconstruction_only")
    inverse_task = channel_scores(channels, "inverse_task")
    assert ratio[0] > ratio[1]
    assert torch.allclose(reconstruction, torch.ones(2))
    assert inverse_task[0] > inverse_task[1]
