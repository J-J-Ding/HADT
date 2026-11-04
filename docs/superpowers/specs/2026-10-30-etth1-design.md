# ETTh1 training and diagnostics
User authorization: implement progress, best weights, forecast plots and component contribution diagnostics, then train/test using djjTF on physical GPU 3.

Use the existing PyTorch pipeline with tqdm and headless Matplotlib PNG/PDF figures and NPZ/CSV source data. Preserve default architecture, seven quantiles and chronological 60/20/20 split; choose H=96, L=336, seed=2024. Plot predetermined first/middle/last test windows for every variable in original units. Save training history and best checkpoint with scaler, scheduler and AMP states. No test-based selection.

Approaches: logging alone cannot show forecast behavior; activation plots alone cannot establish contribution. Chosen approach combines visual diagnostics with full-test inference interventions and separately retrained single-component ablations (offsets, encoder, RevIN), all with identical data and training budget. Interventions measure dependence of a fitted model; retraining measures one-seed architecture evidence, not statistical proof.

Use a GPU-specific configuration with batch size 32, evaluation batch 32, 30 maximum epochs, validation early stopping 7, no activation checkpointing, CUDA explicitly required. No available git repository, so edit in the user workspace and preserve artifacts there.
