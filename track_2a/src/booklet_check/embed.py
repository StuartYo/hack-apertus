"""Multilingual sentence embeddings, run locally on CPU with onnxruntime.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), the
8-bit ONNX export `onnx/model_quint8_avx2.onnx` (118 MB). It is copied into the Docker image
at build time (EMBED_DIR) and only read from disk at run time, so evaluation never
downloads anything. Mean pooling + L2 normalisation, as in the sentence-transformers config.
If the model cannot be loaded the pipeline falls back to BM25 only and says so on stderr.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np

EMBED_DIR = Path(os.environ.get("EMBED_DIR", "/app/models/minilm"))
ONNX_FILE = os.environ.get("EMBED_ONNX", "onnx/model_quint8_avx2.onnx")
MAX_LEN = 256
# e5 models expect "query: " / "passage: " prefixes; paraphrase-MiniLM expects none.
QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "")
PASSAGE_PREFIX = os.environ.get("EMBED_PASSAGE_PREFIX", "")

_state: dict = {}
_lock = threading.Lock()


def _load():
    with _lock:
        if "ok" in _state:
            return _state.get("session")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(EMBED_DIR / "tokenizer.json"))
            tok.enable_truncation(MAX_LEN)
            tok.enable_padding(pad_id=1, pad_token="<pad>")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.environ.get("EMBED_THREADS", "0")) or os.cpu_count() or 1
            sess = ort.InferenceSession(str(EMBED_DIR / ONNX_FILE), opts, providers=["CPUExecutionProvider"])
            _state.update(ok=True, session=sess, tok=tok, inputs={i.name for i in sess.get_inputs()})
        except Exception as exc:  # pragma: no cover - depends on the environment
            _state.update(ok=False, session=None)
            print(f"[booklet-check] embedding model unavailable ({EMBED_DIR}), BM25 only: {exc}", file=sys.stderr)
        return _state.get("session")


def available() -> bool:
    return _load() is not None


def encode(texts: list[str], batch_size: int = 32, kind: str = "passage") -> np.ndarray | None:
    sess = _load()
    if sess is None or not texts:
        return None
    prefix = QUERY_PREFIX if kind == "query" else PASSAGE_PREFIX
    texts = [prefix + t for t in texts]
    tok, names = _state["tok"], _state["inputs"]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok.encode_batch(texts[i:i + batch_size])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feed)[0]  # (batch, seq, dim)
        m = mask[..., None].astype(np.float32)
        out.append((hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9))
    vecs = np.concatenate(out).astype(np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
