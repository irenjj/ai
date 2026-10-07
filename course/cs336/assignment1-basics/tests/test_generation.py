import pytest
import torch
from torch import nn

from cs336_basics.nn_utils import generate


class NextTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def forward(self, tokens):
        assert not self.training and not torch.is_grad_enabled()
        self.contexts.append(tokens.clone())
        logits = torch.full((*tokens.shape, 8), -100.0)
        return logits.scatter(-1, ((tokens + 1) % 8).unsqueeze(-1), 100.0)


def test_generate_context_and_eos():
    model = NextTokenModel()
    prompt = torch.tensor([1, 2, 3])
    result = generate(model, prompt, 10, 6, 2, 1.0, 0.9)
    assert result.tolist() == [1, 2, 3, 4, 5, 6]
    assert [x.tolist() for x in model.contexts] == [[[2, 3]], [[3, 4]], [[4, 5]]]
    assert prompt.tolist() == [1, 2, 3]
    assert model.training


def test_generate_limit_and_zero_tokens():
    model = NextTokenModel().eval()
    prompt = torch.tensor([1])
    assert generate(model, prompt, 2, 7, 4, 1.0, 1.0).tolist() == [1, 2, 3]
    assert not model.training
    assert torch.equal(generate(model, prompt, 0, 7, 4, 1.0, 1.0), prompt)
    assert len(model.contexts) == 2


def test_top_p_distribution(monkeypatch):
    class FixedModel(nn.Module):
        def forward(self, tokens):
            # 最高概率 token ID 为 1；阈值 0.8 需要保留前三项。
            logits = torch.tensor([0.07, 0.45, 0.03, 0.30, 0.15]).log()
            return logits.expand(1, tokens.shape[-1], -1)

    def sample(probs, num_samples):
        torch.testing.assert_close(probs, torch.tensor([0.5, 1 / 3, 1 / 6, 0., 0.]))
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", sample)
    result = generate(FixedModel(), torch.tensor([0]), 1, 2, 4, 1.0, 0.8)
    assert result.tolist() == [0, 4]


def test_generate_restores_mode_on_error():
    class BrokenModel(nn.Module):
        def forward(self, tokens):
            raise RuntimeError("forward failed")

    model = BrokenModel()
    with pytest.raises(RuntimeError, match="forward failed"):
        generate(model, torch.tensor([0]), 1, 1, 4, 1.0, 1.0)
    assert model.training
