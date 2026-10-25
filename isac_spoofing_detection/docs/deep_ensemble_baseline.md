# Deep Ensemble baseline

This module adapts the Deep-Ensemble-Learning-Based GPS Spoofing Detection
method to the saved ISAC spoofing episodes. It is isolated from the proposed
detector and does not change the existing feature or GRU pipeline.

## Data provenance

The offline derivation uses the following saved fields.

- true_position_m generates the actual A2G path-loss observations.
- gps_observed_m generates the GPS-implied theoretical path-loss observations.
- labels provides the time-slot training target.
- has_attack, attack_start_step, and attack_end_step_exclusive provide the
  event definitions used by the common evaluator.
- time_s provides the sampling interval used for detection delay and FAR.
- BS anchors and carrier frequency are read from manifest.json.

The actual path-loss samples never depend on gps_observed_m. This prevents a
spoofed report from modifying the physical observation used as the reference.
The two packet-level path-loss sets share a configurable fraction of the
environmental fluctuation and retain independent measurement error.

The deterministic A2G mean follows Eqs. (1)-(4) in the source paper. The
reference path-loss term is adjusted from the source 2 GHz setting to the
carrier stored in the local manifest. Each time slot contains E packet samples.
The default E is 150, matching the source experiment.

The ten inputs of every BS MLP are the absolute differences in mean, variance,
skewness, kurtosis, Q0, Q1, Q2, Q3, and Q4, followed by the second-order
Wasserstein distance. One MLP is trained for each BS. The four BS-level MLPs
are trained on the train split and use the validation split for early stopping.
After their parameters are frozen, their continuous probabilities form the
four inputs of one 4-4-1 meta MLP. The meta MLP traverses every UAV-time-slot
sample from all 4000 train episodes through sharded mini-batches. All 500
validation episodes are used to compute validation BCE for early stopping.
The test split is not read during either training stage. Both MLP training
stages use a maximum of 500 epochs and a patience of 100 validation checks.
The step-level probability threshold is fixed at 0.5.

## Installation

Install the optional dependencies from the repository root.

    python -m pip install -e ".[external-baselines,test]"

## Resumable execution

Use a dedicated feature directory and run directory for each configuration.

    isac-deep-ensemble generate --config configs/external_baselines/deep_ensemble.json --dataset-dir dataCubic/d1_5000_scurve_cubic_rcs6_seed20260714_interuav --feature-dir outputs/deep_ensemble/features --run-dir outputs/deep_ensemble/run

    isac-deep-ensemble train --config configs/external_baselines/deep_ensemble.json --dataset-dir dataCubic/d1_5000_scurve_cubic_rcs6_seed20260714_interuav --feature-dir outputs/deep_ensemble/features --run-dir outputs/deep_ensemble/run

    isac-deep-ensemble evaluate --config configs/external_baselines/deep_ensemble.json --dataset-dir dataCubic/d1_5000_scurve_cubic_rcs6_seed20260714_interuav --feature-dir outputs/deep_ensemble/features --run-dir outputs/deep_ensemble/run

Existing BS-level MLP checkpoints can be reused while training only the
full-train meta MLP by passing --pretrained-per-bs-dir to both the train and
evaluate stages.

The following PowerShell script reuses the completed E=150 feature sidecars,
reuses the four completed BS-level MLPs, trains the streaming meta MLP, and
then evaluates it on validation and test episodes.

    powershell -ExecutionPolicy Bypass -File scripts/run_deep_ensemble_train_test.ps1

Feature derivation writes every source shard atomically and records a
configuration-aware completion marker. Meta-probability caching preserves all
train and validation samples in compact four-probability shards. Repeating a
stage reuses completed artifacts. The training_complete.json file marks a
finished training stage.

Use --quick and --max-episodes-per-split for a small verification run. Quick
mode reduces E to 16, limits the episode counts, and uses two MLP epochs.
Quick outputs must not be reported as final experimental results.

## Outputs

- generation_summary.json records generated and reused feature shards.
- per_bs_models contains four MLP checkpoints, standardizers, and histories.
- meta_probability_cache contains the four BS probabilities and labels for
  every train and validation slot.
- selection/selected_ensemble.pt contains the full-train 4-4-1 meta MLP.
- selection/training_history.json records its train and validation BCE.
- selection/training_state.pt and training_progress.json support epoch-level
  recovery after interruption.
- selection/selection.json records the best validation BCE and the fixed 0.5
  decision threshold.
- evaluation_validation.json and evaluation_test.json use the common K-step
  alarm confirmation and event matching implementation.

The final paper table should use evaluation_test.json only after the
configuration has been fixed using validation data.
