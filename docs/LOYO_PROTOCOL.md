# Supplemental LOYO protocol

Role: core comparison / advanced emission verification. Use the eight frozen train-year folds 2016–2023. All held-out samples remain in the original physical train directory. Train/validation/test lists are never modified. Each training fold plus its evaluation fold must partition the 805 frozen training samples with no overlap.

Fixed 44 epochs, no early stopping, no held-out-year checkpoint selection. The budget originates from the formal D run's 2024-validation-selected epoch, which is a weak dependence on 2024 and must be disclosed. AdamW LR=1e-3, weight decay=1e-4, effective batch=16, seed=42. D and controls use the shared weighted-MSE loss; v2/v3 are the explicit focal-loss exceptions. v3 additionally uses the documented augmentation and random 6h/12h/18h spans. Use final weights, never a fold's diagnostic best epoch.

Aggregate MD/DR3 by unweighted mean of the eight fold values, with frame weighting inside each fold. Pooling all frames is a different secondary aggregation. Decoder and mask constants are frozen separately from training; no test data may select them.
