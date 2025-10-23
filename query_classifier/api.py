from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union
import sys

import torch
from transformers import logging as hf_logging

from .classifier import QueryClassifier, tokenizer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "best"
ID2LABEL = {
    0: "chitchat",
    1: "general_knowledge",
    2: "internal_knowledge",
    3: "unsafe",
}


def _find_weights(model_dir: Path):
    safe_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safe_path.exists():
        return "safetensors", safe_path
    if bin_path.exists():
        return "pt", bin_path
    return None, None


def _load_state_dict(model: torch.nn.Module, model_dir: Path = MODEL_DIR):
    kind, path = _find_weights(model_dir)
    if path is None:
        raise FileNotFoundError(
            f"No weights found in {model_dir} (expected model.safetensors or pytorch_model.bin)"
        )

    if kind == "safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(str(path), map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.stderr.write(
            "Warning: state dict mismatch:\n"
            f"   Missing: {missing}\n"
            f"   Unexpected: {unexpected}\n"
        )
    return model


def _set_logging_level(level: int):
    prev_level = hf_logging.get_verbosity()
    hf_logging.set_verbosity(level)
    return prev_level


@lru_cache(maxsize=8)
def _load_model(device_key: str):
    prev_level = _set_logging_level(hf_logging.ERROR)
    try:
        model = QueryClassifier()
    finally:
        hf_logging.set_verbosity(prev_level)

    _load_state_dict(model)
    model.eval()

    if device_key != "cpu":
        model.to(device_key)
    return model


def _normalize_device(device: Optional[str]) -> str:
    if not device or device == "cpu":
        return "cpu"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    return device


def _to_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _broadcast_to_batch(value, batch_size: int):
    value = _to_list(value)

    if isinstance(value, (int, float)):
        return [float(value)] * batch_size

    if isinstance(value, list):
        if len(value) == batch_size and all(not isinstance(i, list) for i in value):
            return [float(i) for i in value]

        flat = [i[0] if isinstance(i, list) else i for i in value]
        if len(flat) == batch_size:
            return [float(i) for i in flat]

    raise ValueError(f"Cannot broadcast value of type {type(value)} to batch={batch_size}")


def _extract_aux_scores(aux, batch_size: int):
    if isinstance(aux, dict):
        amb = _broadcast_to_batch(aux.get("ambiguity_score"), batch_size)
        over = _broadcast_to_batch(aux.get("is_overview"), batch_size)
        comp = _broadcast_to_batch(aux.get("is_compositional"), batch_size)
        alias = _broadcast_to_batch(aux.get("is_alias_or_underspecified"), batch_size)
        return [
            {
                "ambiguity_score": a,
                "is_overview": o,
                "is_compositional": c,
                "is_alias_or_underspecified": al,
            }
            for a, o, c, al in zip(amb, over, comp, alias)
        ]

    matrix = _to_list(aux)
    if isinstance(matrix, list) and len(matrix) == batch_size:
        out = []
        for row in matrix:
            row = _to_list(row)
            if not isinstance(row, list) or len(row) < 4:
                raise ValueError("Auxiliary score rows must contain at least four elements.")
            out.append(
                {
                    "ambiguity_score": float(row[0]),
                    "is_overview": float(row[1]),
                    "is_compositional": float(row[2]),
                    "is_alias_or_underspecified": float(row[3]),
                }
            )
        return out

    raise TypeError(f"Unrecognized auxiliary score format: {type(aux)}")


def _ensure_sequence(texts: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(texts, str):
        return [texts]
    if isinstance(texts, Iterable):
        items = list(texts)
        if all(isinstance(t, str) for t in items):
            return items
    raise TypeError("texts must be a string or an iterable of strings.")


def classify(
    texts: Union[str, Sequence[str]],
    max_length: int = 256,
    device: Optional[str] = None,
) -> List[dict]:
    items = _ensure_sequence(texts)
    if not items:
        return []

    model_device = _normalize_device(device)
    model = _load_model(model_device)
    runner_device = next(model.parameters()).device

    encodings = tokenizer(
        items,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encodings = {k: v.to(runner_device) for k, v in encodings.items()}

    with torch.no_grad():
        logits, aux = model(**encodings)
        probs = torch.softmax(logits, dim=-1).cpu().tolist()

    aux_list = _extract_aux_scores(aux, batch_size=len(items))

    results = []
    for text, pvec, aux_dict in zip(items, probs, aux_list):
        pred_id = max(range(len(pvec)), key=lambda idx: pvec[idx])
        result = {
            "text": text,
            "label": ID2LABEL[pred_id],
            "probs": {ID2LABEL[i]: float(pvec[i]) for i in range(len(pvec))},
            "ambiguity_score": float(aux_dict["ambiguity_score"]),
            "is_overview": float(aux_dict["is_overview"]),
            "is_compositional": float(aux_dict["is_compositional"]),
            "is_alias_or_underspecified": float(aux_dict["is_alias_or_underspecified"]),
        }
        results.append(result)

    return results
