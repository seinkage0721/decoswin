"""Small synthetic checks; no dataset or network access required."""

import torch

from decoswin.model import AOPA, DMF, DeCoSwin
from decoswin.train import class_balanced_weights


def test_aopa_sampling_and_gradient():
    module = AOPA(channels=8, num_prototypes=4, topk=2)
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    sampled, offsets = module.sampled_features(x)
    assert sampled.shape == (2, 8, 8, 8)
    assert offsets.shape == (2, 2, 8, 8)
    y = module(x)
    assert y.shape == x.shape
    (y.mean() + module.orthogonality_loss()).backward()
    assert module.prototypes.grad is not None
    assert module.offset[-1].weight.grad is not None


def test_dmf_and_weights():
    module = DMF(channels=8, hidden=4)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)
    assert module(x).shape == x.shape
    module(x).mean().backward()
    assert module.paths[0].offset.weight.grad is not None
    weights = class_balanced_weights([1, 10])
    assert weights[0] > weights[1]
    assert torch.isclose(weights.mean(), torch.tensor(1.0))


def test_full_model_shape():
    model = DeCoSwin(num_classes=3, num_prototypes=8, topk=2).eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 128, 128))
    assert y.shape == (1, 3)
    assert torch.isfinite(y).all()
