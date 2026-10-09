import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "prototype_grounding"))

from vrec_latent_compare import eta2, geometry, probe  # noqa: E402
from vrec_latent_extract import per_row_nll  # noqa: E402


def test_geometry_collapsed_vs_isotropic():
    torch.manual_seed(0)
    iso = torch.randn(4000, 32).half()
    collapsed = (torch.ones(4000, 32) * 10 + 0.01 * torch.randn(4000, 32)).half()
    g_iso, g_col = geometry(iso, pairs=5000, pca_rows=4000), geometry(collapsed, pairs=5000, pca_rows=4000)
    assert g_iso["effective_rank"] > 25 and abs(g_iso["random_pair_cosine"]) < 0.05
    assert g_col["cosine_to_mean"] > 0.99 and g_col["sample_specific_share"] < 0.01
    templates = torch.zeros(4000, 32).half()
    templates[:2000, 0] = 1
    g_t = geometry(templates, pairs=5000, pca_rows=4000)
    assert g_t["unique_fp16_fraction"] == 2 / 4000 and g_t["top10_template_coverage"] == 1.0


def test_eta2_and_probe_detect_group_structure():
    torch.manual_seed(0)
    codes = np.repeat(np.arange(5), 400)
    centers = torch.randn(5, 16) * 3
    x = centers[codes] + torch.randn(len(codes), 16)
    assert eta2(x, codes) > 0.8
    assert eta2(torch.randn(len(codes), 16), codes) < 0.02
    codes_missing = codes.copy()
    codes_missing[:10] = -1
    res = probe(x, codes_missing, x, codes, steps=300)
    assert res["test_acc"] > 0.95 and res["classes"] == 5 and abs(res["majority_baseline"] - 0.2) < 0.01


def test_per_row_nll_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[-100, -100, 3, 4, 1], [-100, 2, 2, -100, -100]])
    per_row, mean = per_row_nll(logits, labels)
    ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert abs(mean - float(ref)) < 1e-5
    row0 = torch.nn.functional.cross_entropy(logits[0, 1:4], labels[0, 2:5])
    assert abs(float(per_row[0]) - float(row0)) < 1e-5
