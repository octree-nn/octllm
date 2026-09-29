"""Language-quality metrics for 3D-understanding predictions.

Provides:
- Lexical metrics: BLEU-1/2/3/4, ROUGE-1/2/L (F1), METEOR.
- Embedding similarities: SBERT (all-mpnet-base-v2) and SimCSE
  (princeton-nlp/sup-simcse-roberta-large).

Public API:
- ``compute_lexical_metrics(pred_text, ref_text) -> dict[str, float]``
- ``compute_metrics(pred_text, ref_text) -> dict[str, float]`` (alias; lexical only)
- ``compute_embedding_metrics(pred_texts, ref_texts, ...) -> dict[str, list[float]]``
- Constants: ``LEXICAL_METRIC_KEYS``, ``EMBEDDING_METRIC_KEYS``, ``ALL_METRIC_KEYS``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score


LEXICAL_METRIC_KEYS: tuple[str, ...] = (
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "meteor",
)
EMBEDDING_METRIC_KEYS: tuple[str, ...] = ("sbert_similarity", "simcse_similarity")
ALL_METRIC_KEYS: tuple[str, ...] = LEXICAL_METRIC_KEYS + EMBEDDING_METRIC_KEYS


# NLTK resources required for word_tokenize + METEOR. We never trigger a
# download: the data is expected to be pre-installed under one of the paths
# below (override via the NLTK_DATA env var if needed).
_NLTK_REQUIRED_PATHS: tuple[str, ...] = (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/wordnet",
    "corpora/omw-1.4",
)
_NLTK_LOCAL_PATHS: tuple[str, ...] = (
    os.environ.get("NLTK_DATA", ""),
    os.path.expanduser("~/nltk_data"),
)


def _configure_nltk_data_path() -> None:
    for path in _NLTK_LOCAL_PATHS:
        if path and os.path.isdir(path) and path not in nltk.data.path:
            nltk.data.path.insert(0, path)


def _verify_nltk_resources() -> None:
    missing: list[str] = []
    for resource in _NLTK_REQUIRED_PATHS:
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource)
    if missing:
        raise LookupError(
            "Missing NLTK resources: "
            + ", ".join(missing)
            + f". Searched paths: {nltk.data.path}. Install them locally "
            "(e.g. under your home directory) or set NLTK_DATA."
        )


_configure_nltk_data_path()
_verify_nltk_resources()

_SMOOTHING = SmoothingFunction().method1


def tokenize(text: str) -> list[str]:
    """Standard NLTK word tokenization with lowercasing.

    Pure-punctuation tokens are dropped so that n-gram metrics focus on content
    words, matching common practice for lexical metrics.
    """
    tokens = [tok.lower() for tok in word_tokenize(text or "")]
    return [tok for tok in tokens if any(ch.isalnum() for ch in tok)]


def _safe_rouge_scores(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> dict[str, float]:
    """Compute ROUGE-1/2/L F1 via the `rouge` package; zeros on degenerate input."""
    out = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    if not pred_tokens or not ref_tokens:
        return out
    try:
        from rouge import Rouge
    except ImportError as exc:
        raise ImportError(
            "The `rouge` package is required for ROUGE metrics. "
            "Install it with `pip install rouge`."
        ) from exc
    try:
        scores = Rouge().get_scores(" ".join(pred_tokens), " ".join(ref_tokens))[0]
        out["rouge_1"] = float(scores["rouge-1"]["f"])
        out["rouge_2"] = float(scores["rouge-2"]["f"])
        out["rouge_l"] = float(scores["rouge-l"]["f"])
    except (ValueError, ZeroDivisionError):
        # `rouge` raises ValueError for hypothesis/reference that become empty
        # after its internal filtering. Treat as zero rather than crashing.
        pass
    return out


def compute_lexical_metrics(pred_text: str, ref_text: str) -> dict[str, float]:
    """BLEU-1..4, ROUGE-1/2/L F1, METEOR for a single (pred, ref) pair."""
    metrics = {key: 0.0 for key in LEXICAL_METRIC_KEYS}
    pred_tokens = tokenize(pred_text)
    ref_tokens = tokenize(ref_text)
    if not pred_tokens or not ref_tokens:
        return metrics

    metrics["bleu_1"] = sentence_bleu(
        [ref_tokens], pred_tokens, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=_SMOOTHING
    )
    metrics["bleu_2"] = sentence_bleu(
        [ref_tokens], pred_tokens, weights=(0.5, 0.5, 0.0, 0.0), smoothing_function=_SMOOTHING
    )
    metrics["bleu_3"] = sentence_bleu(
        [ref_tokens], pred_tokens, weights=(1 / 3, 1 / 3, 1 / 3, 0.0), smoothing_function=_SMOOTHING
    )
    metrics["bleu_4"] = sentence_bleu(
        [ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTHING
    )

    metrics.update(_safe_rouge_scores(pred_tokens, ref_tokens))
    metrics["meteor"] = float(meteor_score([ref_tokens], pred_tokens))
    return metrics


def compute_metrics(pred_text: str, ref_text: str) -> dict[str, float]:
    """Backward-compatible alias that returns lexical metrics only."""
    return compute_lexical_metrics(pred_text, ref_text)


# ---------------------------------------------------------------------------
# Embedding similarities (SBERT, SimCSE)
# ---------------------------------------------------------------------------

_SBERT_MODEL = None
_SIMCSE_CACHE: Optional[tuple] = None  # (tokenizer, model, device)

SBERT_MODEL_NAME = "all-mpnet-base-v2"
SIMCSE_MODEL_NAME = "princeton-nlp/sup-simcse-roberta-large"


def _resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_sbert_model(device: str):
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "`sentence-transformers` is required for SBERT similarity. "
                "Install it with `pip install sentence-transformers`."
            ) from exc
        _SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME, device=device)
    return _SBERT_MODEL


def _get_simcse(device: str):
    global _SIMCSE_CACHE
    if _SIMCSE_CACHE is None or _SIMCSE_CACHE[2] != device:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "`transformers` is required for SimCSE similarity. "
                "Install it with `pip install transformers`."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(SIMCSE_MODEL_NAME)
        model = AutoModel.from_pretrained(SIMCSE_MODEL_NAME).to(device).eval()
        _SIMCSE_CACHE = (tokenizer, model, device)
    return _SIMCSE_CACHE[0], _SIMCSE_CACHE[1]


def _cosine_similarity_rows(a, b) -> list[float]:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0.0, 1e-12, denom)
    return ((a * b).sum(axis=1) / denom).tolist()


def compute_sbert_similarities(
    pred_texts: Sequence[str],
    ref_texts: Sequence[str],
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> list[float]:
    if len(pred_texts) != len(ref_texts):
        raise ValueError("pred_texts and ref_texts must have the same length.")
    if not pred_texts:
        return []
    device = _resolve_device(device)
    model = _get_sbert_model(device)
    pred_emb = model.encode(
        list(pred_texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    ref_emb = model.encode(
        list(ref_texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    return _cosine_similarity_rows(pred_emb, ref_emb)


def _simcse_encode(texts: Sequence[str], batch_size: int, device: str):
    import numpy as np
    import torch

    tokenizer, model = _get_simcse(device)
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        chunks.append(out.detach().cpu().to(torch.float32).numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0))


def compute_simcse_similarities(
    pred_texts: Sequence[str],
    ref_texts: Sequence[str],
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> list[float]:
    if len(pred_texts) != len(ref_texts):
        raise ValueError("pred_texts and ref_texts must have the same length.")
    if not pred_texts:
        return []
    device = _resolve_device(device)
    pred_emb = _simcse_encode(pred_texts, batch_size, device)
    ref_emb = _simcse_encode(ref_texts, batch_size, device)
    return _cosine_similarity_rows(pred_emb, ref_emb)


def compute_embedding_metrics(
    pred_texts: Sequence[str],
    ref_texts: Sequence[str],
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
    include_sbert: bool = True,
    include_simcse: bool = True,
) -> dict[str, list[float]]:
    """Batch-compute SBERT/SimCSE cosine similarities for aligned text pairs.

    Pairs where either side is empty get a similarity of 0.0; the empty strings
    are replaced by a placeholder before encoding to avoid tokenizer errors.
    """
    placeholder = "##"

    def _is_empty(text: str) -> bool:
        return not text or not text.strip()

    empty_mask = [_is_empty(p) or _is_empty(r) for p, r in zip(pred_texts, ref_texts)]
    safe_pred = [placeholder if _is_empty(t) else t for t in pred_texts]
    safe_ref = [placeholder if _is_empty(t) else t for t in ref_texts]

    output: dict[str, list[float]] = {}
    if include_sbert:
        sims = compute_sbert_similarities(safe_pred, safe_ref, batch_size=batch_size, device=device)
        output["sbert_similarity"] = [0.0 if mask else s for s, mask in zip(sims, empty_mask)]
    if include_simcse:
        sims = compute_simcse_similarities(safe_pred, safe_ref, batch_size=batch_size, device=device)
        output["simcse_similarity"] = [0.0 if mask else s for s, mask in zip(sims, empty_mask)]
    return output


# ---------------------------------------------------------------------------
# Standalone CLI (single-directory evaluation)
# ---------------------------------------------------------------------------


def _evaluate_directory(
    pred_dir: str,
    gt_json: str,
    include_embedding: bool,
    batch_size: int,
    device: Optional[str],
) -> dict[str, float]:
    with open(gt_json, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_map = {item["name"]: item["description"] for item in gt_data}

    pred_texts: list[str] = []
    ref_texts: list[str] = []
    per_sample_lexical: list[dict[str, float]] = []
    skipped = 0

    for filename in sorted(os.listdir(pred_dir)):
        if not filename.endswith(".txt"):
            continue
        asset_id = os.path.splitext(filename)[0]
        if asset_id not in gt_map:
            skipped += 1
            continue
        with open(os.path.join(pred_dir, filename), "r", encoding="utf-8") as f:
            pred_text = f.read().strip()
        ref_text = gt_map[asset_id]
        pred_texts.append(pred_text)
        ref_texts.append(ref_text)
        per_sample_lexical.append(compute_lexical_metrics(pred_text, ref_text))

    count = len(per_sample_lexical)
    if count == 0:
        print("No matched files found.")
        return {}

    totals = {key: 0.0 for key in LEXICAL_METRIC_KEYS}
    for metrics in per_sample_lexical:
        for key in LEXICAL_METRIC_KEYS:
            totals[key] += metrics[key]

    if include_embedding:
        emb = compute_embedding_metrics(pred_texts, ref_texts, batch_size=batch_size, device=device)
        for key, sims in emb.items():
            totals[key] = sum(sims)
        metric_keys = ALL_METRIC_KEYS
    else:
        metric_keys = LEXICAL_METRIC_KEYS

    averages = {key: totals.get(key, 0.0) / count for key in metric_keys}

    print(f"Matched files: {count}")
    print(f"Skipped files: {skipped}")
    for key in metric_keys:
        print(f"{key}: {averages[key]:.6f}")
    return averages


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute lexical + embedding language metrics.")
    parser.add_argument(
        "--pred_dir",
        default="outputs/shapenet/octllm/captions",
        help="Directory containing prediction txt files.",
    )
    parser.add_argument(
        "--gt_json",
        default="datasets/shapenet/description/description-02691156.json",
        help="JSON file containing ground-truth descriptions.",
    )
    parser.add_argument(
        "--no_embedding",
        action="store_true",
        help="Skip SBERT/SimCSE similarity computation (lexical metrics only).",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for SBERT/SimCSE encoding.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device for embedding models (e.g. 'cuda', 'cuda:0', 'cpu'). Auto-detected if unset.",
    )
    args = parser.parse_args()

    _evaluate_directory(
        pred_dir=args.pred_dir,
        gt_json=args.gt_json,
        include_embedding=not args.no_embedding,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
