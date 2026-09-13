# Appearance-prior training stage for the full-posterior supplement

This stage is declared before extracting these features or observing any prior
training results. It leaves the existing main experiment unchanged. Its output
is a frozen visual prior for a separate matched evidence/projection factorial;
it is not itself evidence of improved dynamics understanding.

Use data/projection_appearance_v1/LOCK.json:12,000first-image mappings and2,538
unique image hashes. The exact image/prompt contract is already frozen in
PROJECTION_APPEARANCE_INPUT.md. Extract each original Qwen final token state
once in BF16, batch1, evaluation mode, no generation/cache/adapter. Original
revision89644892e4d85e24eaac8bacfd4f463576704203, local saved model, SDPA. Check
eight duplicated full forwards for bit-exact equality. Keep feature IDs, image
and total token counts, elapsed extraction time, peak GPU allocation and all
feature hashes. No model answers or physical metadata form the prior input.

Apply nonlearned per-vector layer normalization in float32 (epsilon1e-5), then
a single linear6-output appearance head:3transformed means and3softplus scales
with minimum.001. A diagonal Gaussian starts the prior; the later projected
posterior retains its full covariance. Inputs encode appearance/visible pose;
pretraining contamination or geometry priors are not ruled out.

Three seeds17/43/101,100epochs, batch32, CPUthreads4, AdamW learning rate.0003,
weight decay.01, gradient norm cap1. Same existing8,400training and1,200validation
episode rows and fixed groups, including their repeated appearances. Each row
has equal loss weight as in the main study. Initialize weights to zero and
biases to training-only transformed mean/std (sample std). Thus randomness
affects minibatch order, not invented random prior samples. Shuffle seed+997;
select minimum validation Gaussian NLL. Run all100epochs, with epoch-level
optimizer/RNG/resume state. Save actual optimizer steps, time, every validation
loss and the selected model's complete validation predictions. No rollout loss
is used for the appearance-only prior. Do not interpret an unchanged or worse
prior as failure to execute or silently increase the budget.

Report NLL per physical coordinate (joint diagonal Gaussian NLL divided by3),
explicitly separated from the later joint covariance score. Only train and
validation target values are parsed. The source target file is hashed, and IDs
are scanned to select permitted rows before JSON decoding their target values.
Calibration/test values do not participate in fitting/selection. Input features
may be extracted for all splits without target access.

After all three runs, reload each selected checkpoint and reproduce its
validation outputs before sealing checkpoint files and histories. Generate
all12,000prior means/scales per seed from those frozen checkpoints without
reading any targets. Verify all600hidden-mass same-image pairs have exactly
equal predictions. Preserve per-seed outputs; do not ensemble away variability.
These outputs are prerequisites for subsequent local-Jacobian preparation and
matched dense/sparse by protocol/projected training. Those later trained and
evaluated comparisons remain required for goal completion.

Execution is finite and serialized after external_physbench_chain_complete.
The feature extractor is the only GPU work in this stage; the tiny prior heads
train on CPU afterward. Stop on predecessor failure, missing output, replay
mismatch, nonfinite loss or broken source hash. Do not change the frozen
protocol to work around results. Sources/hyperparameters are sealed in
results/projection_prior/PLAN.json before starting the dependent process.
