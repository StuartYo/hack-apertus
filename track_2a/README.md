# booklet-check (Track 2A, OST challenge)

Multilingual natural language inference over Swiss official voting booklets, built with
Apertus v1.5. Given a claim about a federal vote (German, French or Italian) and either the
whole booklet PDF (task A) or one passage of it (task B), the system answers entailment,
neutral or contradiction and, for task A, returns evidence passages with their page numbers.
The write-up is [`technical_report.md`](technical_report.md) (PDF in `docs/`), and every
reported number is in [`eval/RESULTS.md`](eval/RESULTS.md).

Challenge description and judging criteria: https://hackapertus.notion.site/getting-started-guide-onlinehack

## Run it

From this directory (`track_2a/`):

```bash
make run                                   # build the image, run the two example cases
BASE_URL=... API_KEY=... make run          # the same with real Apertus calls
make run INPUT_DIR=/path/to/data           # your own case file, see below
```

`make run` builds the Docker image and runs it on `$(INPUT_DIR)/$(INPUT)` (default
`data/examples/cases.jsonl`). Predictions are written to `output/predictions.jsonl`.
Without `BASE_URL` and `API_KEY` the container runs in a clearly labelled DRY MODE: no model
calls, placeholder labels, so only use it to check that everything starts. The variables
`LLM_BASE_URL` / `LLM_API_KEY` and `APERTUS_BASE_URL` / `APERTUS_API_KEY` are accepted too, and
`LLM_NAME` picks a model (default: the endpoint's Apertus v1.5 8B instruct model). See
`.env.example`.

Make variables: `INPUT_DIR` (directory mounted read-only as `/data`), `INPUT` (case file name
inside it, default `cases.jsonl`), `OUTPUT_DIR` (default `output`), `IMAGE`.

### Input format

One JSON object per line, as written by the official `prepare_cases.py`:

- task A: `{"id", "vote", "claim": {"text", "language"}, "booklet": {"path", "language"}}`
- task B: `{"id", "vote", "claim": {"text", "language"}, "reference": {"text", "language"}}`

`booklet.path` is relative to `INPUT_DIR`, for example `booklets/2024_09_22_de.pdf`. Task B
needs no PDF. Each output line has `id`, `label` (0 entailment, 1 neutral, 2 contradiction),
`label_name`, `evidence` (up to five `{"page", "text"}` items; `page` is null for task B)
and `metrics` (token counts and time).

### Your own booklets

Put the case file and a `booklets/` folder side by side and point `INPUT_DIR` at it:

```
my-data/
  cases.jsonl                 # booklet.path = "booklets/<file>.pdf"
  booklets/<file>.pdf
```

```bash
make run INPUT_DIR=/path/to/my-data INPUT=cases.jsonl
```

To get all official cases and booklets, run `prepare_cases.py --download-booklets` from the
official starter (https://gitlab.com/ifsoftware/hackapertus-starter, commit `5e69577`); see
section 7 of the technical report for the full evaluation steps.

### Tests

The unit tests run outside Docker and need the development requirements first:

```bash
pip install -r requirements-dev.txt        # Python 3.12
make test                                  # or: python -m pytest -q tests
python scripts/make_results.py --check     # report numbers and PDF match the data files
```

## Requirements

- runtime: Docker (image is `linux/amd64`, Python 3.12, CPU only); `make run` builds it and
  runs the two example cases in `data/examples/`
- hardware: any x86-64 CPU with AVX2, about 1 GB RAM; no GPU. The embedding model is the
  8-bit AVX2 ONNX export. If it cannot be loaded (for example on a CPU without AVX2), the
  container only prints a warning on stderr (`embedding model unavailable ... BM25 only`)
  and keeps running with BM25 retrieval alone, which is noticeably worse for claims in
  another language than the booklet. Check stderr for that line.
- API keys: `BASE_URL` and `API_KEY` for an OpenAI-compatible Apertus v1.5 endpoint
  (optional `LLM_NAME`). Without them the container runs in DRY MODE
- model weights: Apertus is called over the API; the only local model is the 118 MB
  multilingual embedding model, downloaded at image build time (pinned revision,
  sha256-checked) and never at run time
- languages: German, French and Italian. Romansh is not supported and was not tested (no
  Romansh cases in the public data; the BM25 stop words and section rules do not cover it)

## Data

The `data/` directory stays under 100 MB. It holds the two official example cases with their
booklet (`data/examples/`) and the frozen development / held-out split (`data/split.json`).

## Submission

Submissions go through http://hackapertus.ch/online-hack/submissions (not Devpost). All Hack
Apertus projects are open source: https://hackapertus.ch/terms-and-conditions. Questions:
Discord or hello@hackapertus.ch.
