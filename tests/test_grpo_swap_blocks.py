import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1] / "src" / "grpo_attention_tuning"


def load_swap():
    """Import only swap_adjacent_blocks (the module imports trl and the latent processor)."""
    source = (ROOT / "grpo_trainer.py").read_text()
    start = source.index("def swap_adjacent_blocks")
    end = source.index("\nclass ", start)
    namespace = {"torch": torch}
    exec(source[start:end], namespace)
    return namespace["swap_adjacent_blocks"]


swap_adjacent_blocks = load_swap()


def test_single_group_is_unchanged():
    x = torch.arange(8.0)
    torch.testing.assert_close(swap_adjacent_blocks(x, 8), x)


def test_two_groups_are_swapped_as_before():
    x = torch.arange(16.0)
    out = swap_adjacent_blocks(x, 8)
    torch.testing.assert_close(out, torch.cat([x[8:], x[:8]]))


def test_odd_number_of_groups_keeps_last_in_place():
    x = torch.arange(24.0)
    out = swap_adjacent_blocks(x, 8)
    torch.testing.assert_close(out, torch.cat([x[8:16], x[:8], x[16:]]))


@pytest.mark.parametrize("groups", [1, 2, 3, 4])
def test_mean_is_preserved(groups):
    x = torch.randn(groups * 8)
    assert torch.isclose(swap_adjacent_blocks(x, 8).mean(), x.mean())
