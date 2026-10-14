"""Download the embedding model into the image at build time (pinned revision + sha256)."""

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Apache-2.0
REV = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
FILES = {
    "onnx/model_quint8_avx2.onnx": "98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f",  # gitleaks:allow (sha256 of a public file)
    "tokenizer.json": "2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8",  # gitleaks:allow (sha256 of a public file)
}


def main(dest: str):
    for name, digest in FILES.items():
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "booklet-check-build"}),
                                      timeout=600).read()
        got = hashlib.sha256(data).hexdigest()
        if got != digest:
            sys.exit(f"sha256 mismatch for {name}: {got}")
        path = Path(dest) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"{name}: {len(data)} bytes ok")


if __name__ == "__main__":
    main(sys.argv[1])
