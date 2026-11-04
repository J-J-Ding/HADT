# Reproduction and evidence

The study uses PyTorch and the package versions in results/software_environment.json. Place the same CSV versions at dataset/ETT-small/ETTh1.csv, dataset/ETT-small/ETTh2.csv and dataset/traffic/traffic.csv; verify the SHA256 values in results/datasets.json. Dataset files and checkpoints are not included in this compact source archive.

From the project root, with the required CUDA-enabled Python environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python scripts/run_paper_experiments.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python scripts/run_paper_sampling_sensitivity.py
OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/hadt_paper_mpl python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/hadt_paper_mpl python scripts/export_paper_evidence.py
python scripts/validate_paper_artifacts.py
```

GPU3 was used in the recorded study. For a machine exposing only GPU0, add PAPER_EXPECTED_GPU=0 and set CUDA_VISIBLE_DEVICES=0. This changes the local device guard, not the model.

The experiment runner falls back to fresh training when old pilot outputs are absent. Completed cache entries require matching configuration/data/implementation signatures. Two imported historical records lack original source hashes and are labeled accordingly. Canonical provenance signs the pre-fit configuration; generated scaler/feature-column fields are retained separately in resolved_config.json. Raw original markers are preserved.

Software unit tests contain synthetic scoring/shape examples only; these are never loaded as paper forecast results. Tables use measured completed runs, not synthetic scores. Three-seed groups require exactly2024/2025/2026; sample SD usesddof1. Independent-head interval metrics are excluded from proper-score comparisons.

The primary run queue additionally provides single-seed Traffic controls. Figures of the earlier depth pilot require the saved CSVs in results/depth_pilot_etth1.csv and depth_pilot_traffic.csv; local full-project scripts can reproduce those pilots. The source archive includes the related scripts. Exporter paths refer to the full project layout, while the LaTeX directory is independently compilable.

A fresh run can take much longer than the original delivery window, depending on hardware, shared GPU load and cache reuse. The recorded deadline is a scheduling limit; no model group is padded with unfinished seeds.
