# Validation-selected architecture study

The user authorizes architecture and hyperparameter exploration, completing missing experiments, and synchronizing the LaTeX manuscript. Latest resource constraint: physical GPU 3 only, `/home/dingjunjie/anaconda3/envs/djjTF`. Earlier four-GPU permission is superseded. Existing evidence must remain intact.

Compare the existing Transformer with identity, residual per-token MLP, residual temporal CNN, and CNN followed by Transformer encoders. Include direct linear/MLP/CNN/patch controls to test whether the horizon decoder is useful. Keep ordered quantiles, train-only scaling, chronological partitions, and the original 30-epoch/7-patience policy in confirmation and evaluation.

Approaches considered: tune only the original model (cannot address the requested module search); replace it immediately (unsupported); compare explicit candidates on validation only and freeze a winner before test evaluation (chosen by agent under user's explicit authorization to explore and replace modules).

Screen a fixed, recorded candidate list on ETTh1 H96, seed 2024, eight epochs. Confirm the top three plus the original with 30 epochs and seeds 2024/2025/2026. Select the minimum mean validation pinball, with candidate ID as deterministic tie-breaker. Test results are never inputs to selection. Retain the original default and distinguish it from a selected variant; do not claim global optimality or novelty of component substitutions.

After selection, evaluate the selected model, original HADT, and Linear-NQ on all seven datasets at H96/192/336/720, three seeds. Prioritize ETTh1/ETTh2 and then new datasets at H96 before their long horizons. Avoid duplicate selected-model jobs if the winner is the original or linear model. This is an explicit finite study, not unlimited architecture search. Record pending, running, completed, and failed jobs. Incrementally export only finished records; aggregate only complete three-seed groups.

Use a separate output namespace and immutable source/data/config fingerprints. Never rewrite old provenance. Save checkpoints, histories, validation selections, resolved configurations, full-test metrics and fixed first/middle/last forecast windows. Generate new search/transfer tables, validation curves, selected architecture schematic, and forecast plots from artifacts. Add corresponding method/protocol/results text without presenting running jobs as measured results. Compile LaTeX and independently verify new evidence. Long jobs must survive terminal disconnection; no overlapping study workers.
