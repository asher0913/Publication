import torch

from publication_cem.prototype_bank import ClassPrototypeBank


def test_first_sample_is_stored_once() -> None:
    bank = ClassPrototypeBank(3, 2, 4, momentum=0.9)
    sample = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    bank.update_class(1, sample)
    assert bank.valid_count(1) == 1
    assert bank.counts[1].sum().item() == 1.0
    assert torch.equal(bank.get(1)[0], sample[0])


def test_bank_state_round_trip() -> None:
    source = ClassPrototypeBank(2, 2, 3, momentum=0.8)
    source.update_class(0, torch.tensor([[1.0, 0.0, -1.0]]))
    target = ClassPrototypeBank(2, 2, 3, momentum=0.8)
    target.load_state_dict(source.state_dict())
    assert torch.equal(source.prototypes, target.prototypes)
    assert torch.equal(source.valid, target.valid)
    assert torch.equal(source.counts, target.counts)
