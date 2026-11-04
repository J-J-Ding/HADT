# Reference audit

Access date: 2026-10-30. Bibliography: `HADT_NQ_ICCRD2027_LaTeX/references.bib`. **15 authentic references**; all ten existing citation keys remain stable.

Crossref DOI metadata, publisher pages, official author repositories/project pages and author-submitted arXiv records. Browser search used only to locate primary sources. OpenReview direct page/API access returned challenges/403 and was cross-checked against primary author sources.

## Findings and manuscript implications

- Cite `shu2025deformtime` when discussing temporal deformable attention. Avoid first-in-field claims; distinguish horizon-conditioned marginal quantile decoding from DeformTime’s variable-dependency focus.
- Cite `cannon2018noncrossing` for prior neural non-crossing quantile estimation. The specific median-anchored gaps may be described, but guaranteed neural ordering is established prior art.
- Cite `zeng2023dlinear` for the decomposition-linear baseline and `nie2023patchtst` for patching. Describe the implemented quantile versions as original adaptations; their scores are not reproductions of the papers’ point baselines.
- Cite `zhou2021informer` for ETT provenance: the official dataset README requests that paper. It does not by itself verify the experiment’s train/validation/test split.
- Cite `loshchilov2019adamw` at the optimizer description.
- RevIN’s original `cGDAkQo1C0p` URL is correct. Direct OpenReview page/API access is blocked by a browser check/403 in this environment; the authors’ project page reproduces the exact ID and bibliographic record.
- Corrections: Bassett Jr.; Sercan Ö. Arık; Łukasz Kaiser; PatchTST source capitalization and forum URL; official NeurIPS URL.

## Per-reference evidence

### `koenker1978regression`

Added suffix Jr. from JSTOR issue contents; Crossref has a malformed suffix and only the first page, so full pages use JSTOR.

**Supported scope:** Regression quantiles and asymmetric absolute-error estimation.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.2307/1913643
- https://doi.org/10.2307/1913643
- https://www.jstor.org/stable/i332765

Crossref final snapshot HTTP status: 200.

### `gneiting2007probabilistic`

Existing bibliographic fields confirmed.

**Supported scope:** Calibration and sharpness as complementary evaluation goals.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1111/j.1467-9868.2007.00587.x
- https://doi.org/10.1111/j.1467-9868.2007.00587.x

Crossref final snapshot HTTP status: 200.

### `gneiting2007scoring`

Existing bibliographic fields confirmed.

**Supported scope:** Proper scoring rules for probabilistic forecasts and interval evaluation.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1198/016214506000001437
- https://doi.org/10.1198/016214506000001437

Crossref final snapshot HTTP status: 200.

### `winkler1972decision`

Existing bibliographic fields confirmed.

**Supported scope:** Decision-theoretic interval estimation and width/miss penalties.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1080/01621459.1972.10481224
- https://doi.org/10.1080/01621459.1972.10481224

Crossref final snapshot HTTP status: 200.

### `salinas2020deepar`

2020 volume publication retained despite 2019 DOI substring.

**Supported scope:** Autoregressive recurrent probabilistic forecasting across related time series.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1016/j.ijforecast.2019.07.001
- https://doi.org/10.1016/j.ijforecast.2019.07.001
- https://www.sciencedirect.com/science/article/pii/S0169207019301888

Crossref final snapshot HTTP status: 200.

### `lim2021tft`

Corrected Sercan Ö. Arık using Crossref spelling; other fields confirmed.

**Supported scope:** Interpretable multi-horizon neural forecasting, including quantile outputs.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1016/j.ijforecast.2021.03.012
- https://doi.org/10.1016/j.ijforecast.2021.03.012

Crossref final snapshot HTTP status: 200.

### `kim2022revin`

Original OpenReview ID confirmed by authors project BibTeX, retained unchanged. Direct OpenReview and API access challenged/403.

**Supported scope:** Reversible instance normalization and denormalization against distribution shift. The original includes learnable affine parameters; the present non-affine version is a stated adaptation.

Sources (accessed 2026-10-30):
- https://openreview.net/forum?id=cGDAkQo1C0p
- https://seharanul17.github.io/RevIN/

### `nie2023patchtst`

Normalized title to source capitalization and forum URL; retained correct Nam H. Nguyen author order. OpenReview direct access challenged; arXiv and official repository verify authors/title/ICLR2023.

**Supported scope:** Patch tokenization and shared channel-independent Transformer weights. Does not establish reproduction of original point forecasting results by a quantile adaptation.

Sources (accessed 2026-10-30):
- https://openreview.net/forum?id=Jbdc0vTOcol
- https://arxiv.org/abs/2211.14730
- https://github.com/yuqinie98/PatchTST

### `zhu2021deformable`

Authors/title/ICLR2021 confirmed by author-submitted arXiv record.

**Supported scope:** Sparse learned sampling around reference points for object detection. This is antecedent attention machinery, not evidence that this paper first applies deformable attention to time series.

Sources (accessed 2026-10-30):
- https://arxiv.org/abs/2010.04159

### `vaswani2017attention`

Replaced preprint URL with official NeurIPS proceedings and preserved Łukasz Kaiser diacritic.

**Supported scope:** Transformer multi-head attention architecture. This does not attribute later pre-layer-normalized variants to the 2017 paper.

Sources (accessed 2026-10-30):
- https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

### `shu2025deformtime`

New entry. TMLR April2025 and two authors confirmed by official indexed venue PDF and authors repository BibTeX. Forum challenged directly.

**Supported scope:** Prior deformable attention for multivariable time-series forecasting and variable dependencies. Requires positioning the contribution as a specific horizon-conditioned marginal quantile combination, not first temporal deformable attention.

Sources (accessed 2026-10-30):
- https://openreview.net/forum?id=M62P7iOT7d
- https://openreview.net/pdf?id=M62P7iOT7d
- https://github.com/ClaudiaShu/DeformTime
- https://lampos.net/papers/shu2025deformtime.pdf

### `cannon2018noncrossing`

New entry verified against Springer and Crossref.

**Supported scope:** MCQRNN simultaneously estimates multiple non-crossing nonlinear conditional quantiles. It establishes prior guaranteed neural quantile ordering, without asserting identity to the present median-anchored Softplus gaps.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1007/s00477-018-1573-6
- https://doi.org/10.1007/s00477-018-1573-6
- https://link.springer.com/article/10.1007/s00477-018-1573-6

Crossref final snapshot HTTP status: 200.

### `zeng2023dlinear`

New entry verified against AAAI publisher and Crossref; publication is2023, not arXiv2022.

**Supported scope:** Linear and decomposition-based direct long-horizon point forecasting baselines. Quantile DLinear-style variants are adaptations, not original published baseline reproduction.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1609/aaai.v37i9.26317
- https://doi.org/10.1609/aaai.v37i9.26317
- https://ojs.aaai.org/index.php/AAAI/article/view/26317

Crossref final snapshot HTTP status: 200.

### `zhou2021informer`

New entry verified against AAAI/Crossref and official dataset README. No separate undated dataset BibTeX invented.

**Supported scope:** Informer architecture and provenance citation requested by the official ETT dataset README. Repository supports ETT column and recording descriptions; actual experiment split boundaries require code/data inspection.

Sources (accessed 2026-10-30):
- https://api.crossref.org/works/10.1609/aaai.v35i12.17325
- https://doi.org/10.1609/aaai.v35i12.17325
- https://ojs.aaai.org/index.php/AAAI/article/view/17325
- https://github.com/zhouhaoyi/ETDataset

Crossref final snapshot HTTP status: 200.

### `loshchilov2019adamw`

New ICLR2019 entry; arXiv first submission2017 is not venue year. Direct OpenReview challenged; author record/repository confirm title/authors/year.

**Supported scope:** Decoupled weight decay (AdamW) used in optimization. Does not support paper-specific tuning efficacy.

Sources (accessed 2026-10-30):
- https://openreview.net/forum?id=Bkg6RiCqY7
- https://arxiv.org/abs/1711.05101
- https://github.com/loshchil/AdamW-and-SGDW

## Limits

This audit confirms bibliographic identity and appropriate attribution scope. It does not validate reported model results, reproduce baselines, or establish novelty by exhaustive literature review. Direct OpenReview access limitations are disclosed above. No conference DOI or dataset publication year was fabricated. The JSON companion stores available Crossref metadata for inspection.
