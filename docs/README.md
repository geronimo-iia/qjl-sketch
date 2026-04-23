# Documentation

## Study

Source material and research analysis.

| Document | What it covers |
|----------|----------------|
| [study.md](study.md) | TurboQuant overview, DKR vs RAG thesis, source article |

## Design

Architecture and implementation decisions.

| Document | What it covers |
|----------|----------------|
| [design/algorithms.md](design/algorithms.md) | 7 algorithms extracted from QJL reference, Rust data structures |
| [design/pipeline.md](design/pipeline.md) | Search query → attention goal → ranked results (no LLM) |
| [design/persistence.md](design/persistence.md) | Two-store file format, mmap loading, compaction |
| [design/store.md](design/store.md) | Store API usage, lifecycle, error handling, crash safety |
| [design/testing.md](design/testing.md) | 4-layer test strategy |

## Guides

- [roadmap.md](roadmap.md) — Phased development plan
- [release.md](release.md) — How to cut a release
