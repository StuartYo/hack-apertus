# booklet-check

Hack Apertus 2026, online hack, **Track 2A, OST challenge: multilingual NLI over Swiss
official voting booklets**. Code is in [`track_2a/`](track_2a/); the write-up is
[`track_2a/technical_report.md`](track_2a/technical_report.md).

Given a claim about a federal vote (in German, French or Italian) and either the whole
voting booklet PDF (task A) or one passage of it (task B), the system answers
entailment / neutral / contradiction and, for task A, cites the booklet pages it relied on.

How it works in one line: retrieval and evidence are plain code (PDF text extraction, a
small multilingual embedding model baked into the Docker image, BM25), and Apertus v1.5
only reads the few passages that matter and picks the label. Evidence quotes are chunks of
the PDF's pypdf text (NFKC-normalised, blank lines dropped) with their real page numbers, so
the model cannot invent a quote or a page.

Languages: German, French and Italian. Romansh is not supported and was not tested.

```bash
cd track_2a
make run                       # builds the linux/amd64 image, runs the two example cases
BASE_URL=... API_KEY=... make run   # same, with Apertus calls (otherwise a labelled DRY MODE)
```

Evaluation numbers, all generated from one file: [`track_2a/eval/RESULTS.md`](track_2a/eval/RESULTS.md).

## Licences

- Code: Apache-2.0 ([LICENSE](LICENSE))
- Documentation and report: CC-BY-4.0
- Data we add (frozen split, evaluation outputs): CDLA-Permissive-2.0
- The OST dataset (`OSTswiss/MNLIoverSwissVotingBooklets`) is MIT-licensed and is not copied
  here except for the two example cases from the official starter. The example booklet PDF
  is an official publication of the Swiss Federal Chancellery.
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0,
  downloaded at image build time at a pinned revision with sha256 checks.

Hack Apertus rules: https://hackapertus.ch/terms-and-conditions
