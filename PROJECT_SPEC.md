# HADT-NQ V1.1 — Frozen Implementation Specification

## A. Tensor conventions

| Symbol | Meaning | Source |
|---|---|---|
| B | batch size | training/evaluation config |
| L | historical context length | data config |
| C | number of numeric variables | detected from CSV |
| H | prediction horizon | experiment matrix |
| P | patch length | model config |
| S | patch stride | model config |
| N | number of patches | derived |
| D | embedding dimension | model config |
| M | attention heads | model config |
| Dh | head dimension | D/M |
| K | deformable sampling points | derived from N |
| Q | number of quantiles | fixed at 7 |

## B. Frozen default parameters

- `L=336`
- `H=[96,192,336,720]`
- `P=16`
- `S=8`
- `D=128`
- `M=8`
- `Dh=16`
- `encoder_layers=3`
- `d_ff=256`
- `dropout=0.1`
- `max_offset=1.5` Patch tokens
- `horizon_chunk_size=64`
- `gradient_checkpointing=true`
- quantiles `[.05,.10,.25,.50,.75,.90,.95]`

Derived patch count:

`N = floor((L-P)/S) + 1`

Auto sampling-point rule:

`K = min(N, max(4, min(8, ceil(N/5))))`

Examples:
- L=96 => N=11, K=4
- L=192 => N=23, K=5
- L=336 => N=41, K=8
- L=512 => N=63, K=8

## C. Dimension trace (ETTh1 example)

For B=8, L=336, C=7, H=720:
- input: `[8,336,7]`
- transpose: `[8,7,336]`
- unfold: `[8,7,41,16]`
- patch projection: `[8,7,41,128]`
- channel flatten: `[56,41,128]`
- encoder: `[56,41,128]`
- context: `[56,128]`
- horizon queries: `[56,720,128]`
- Q: `[56,720,8,16]`
- K/V: `[56,41,8,16]`
- offsets/positions: `[56,720,8,8]`
- sampled K/V per decoder chunk: `[56,chunk_H,8,8,16]`
- decoder: `[56,720,128]`
- quantiles: `[56,720,7]`
- reshape/permute: `[8,720,7,7]`

The two final dimensions both happen to equal 7 in this example, but the third axis is variables and the fourth is quantiles.

## D. Data and scale semantics

1. The CSV loader identifies numeric feature columns and excludes the timestamp column.
2. The scaler is fitted only on the training time range.
3. The full series is transformed using the training scaler.
4. Dataset windows return `x:[B,L,C]`, `y:[B,H,C]`.
5. Model-side RevIN normalizes each input instance/channel and reverses that normalization on the predicted quantiles.
6. Therefore, model outputs remain in globally standardized units.
7. The evaluator converts predictions and targets back to raw units using the training scaler before calculating interval widths and Winkler scores.

## E. Split semantics

Split ratios: ETT datasets use 60% / 20% / 20%; Electricity, Traffic, and Weather use 70% / 10% / 20%. For chronological split boundaries `[0, train_end, val_end, T]`:
- train target starts: `L ... train_end-H`
- validation target starts: `train_end ... val_end-H`
- test target starts: `val_end ... T-H`

Input context is the L observations immediately before each target start. Targets never cross the corresponding split boundary.

## F. Channel handling

The model uses shared channel-independent weights:
- training: random variable subset if C exceeds `channel_sample_size`;
- validation/test: process every variable in channel chunks;
- no learned cross-variable joint distribution is claimed.

PICP is marginal elementwise coverage, not simultaneous multivariate joint coverage.

## G. Reproducibility

- seeds are explicit in the experiment matrix;
- all resolved configuration and feature names are saved;
- best checkpoint selected by validation pinball loss;
- test metrics are computed only after checkpoint selection;
- mean and standard deviation are calculated across seeds.
