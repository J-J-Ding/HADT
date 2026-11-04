# Architecture study implementation plan

> **For agentic workers:** Execute inline using superpowers:executing-plans. The user explicitly authorized experiments and manuscript updates; proceed without an additional approval gate.

**Goal:** Select an architecture using validation data, run missing experiments on GPU3, and synchronize measured evidence into LaTeX.

**Architecture:** Separate model factory and study runner, immutable study directory, atomic progress records, artifact-driven incremental manuscript exporter.

**Tech Stack:** Existing PyTorch, NumPy, pandas, matplotlib and local Tectonic; djjTF.

**Spec:** docs/superpowers/specs/2026-11-05-architecture-study-design.md

## Global Constraints

- CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=3; use djjTF only.
- Do not modify historical model/trainer code or existing evidence.
- Selection uses validation only; complete three-seed groups for aggregate claims.
- Preserve full checkpoints and run provenance; fail on incompatible cache reuse.

## Review Focus

- Encoder/decoder combinations preserve output shape and quantile order, including constant inputs and gradients.
- Selection is independent of arbitrary test scores, missing seeds cannot enter ranking.
- Source/data/config changes cannot silently reuse completed records.
- Interrupted runs are not marked complete; duplicate workers cannot share the study.
- Paper includes only completed evidence and labels validation selection and transfer separately.

### Task 1: Candidate model factory

Files: create models/architecture_search.py, tests/test_architecture_search.py.

- [x] Write and run failing tests for all candidates' shape, order, channel independence, gradients, selection missing seeds and test-score independence.
- [x] Implement residual MLP/CNN/hybrid encoders and direct heads, retaining existing model construction.
- [x] Run tests on CPU in djjTF, then full existing suite with CUDA hidden.

### Task 2: Durable experiment queue

Files: create scripts/run_architecture_study.py, study protocol and progress in outputs/architecture_study/.

- [x] Implement screening, repeated confirmation, frozen validation selection, and transfer matrix.
- [x] Check source/data/config digests, GPU3/environment guard, process lock and atomic writes.
- [x] Save validation-only artifacts before any selected-model test evaluation; export after every job.
- [x] Start detached worker and inspect first completed real run and GPU process.

### Task 3: Manuscript synchronization

Files: create scripts/export_architecture_study.py; modify main.tex and add results/architecture_* artifacts plus figures.

- [x] Export candidate validation scores, selection protocol, completed test groups and truthful pending status.
- [x] Draw candidate comparison, selected architecture and fixed-window forecasts.
- [x] Independently compare aggregates/checkpoints/provenance, compile PDF and inspect pages (current completed evidence).
- [x] Record measured conclusions and outstanding matrix, commit source and manuscript updates.

## Experiment completion (separate from implementation)

- [x] Twelve screening runs and twelve confirmation runs; selection frozen before test.
- [x] ETTh1 H96: selected variant, original Transformer and Linear-NQ, three seeds each.
- [ ] Remaining transfer matrix. Live progress: outputs/architecture_study/progress.json. Do not infer experiment completion from implementation checkboxes.
