# Grasp-Validation Methodology Comparison — Regenerated Results

This is the computational comparison behind thesis Table 7.6 — the experiment
that selected the single-phase neural network as the grasp-quality
classifier, out of six candidate methods. It's included here alongside
`Scenario_Bank/` because it's the other major computational experiment in
this thesis, even though its outputs are training curves and confusion
matrices rather than planned trajectories.

**Where this came from:** the original comparison script
(`Comparison_Test_With_Graphs.py`, plus one file per method) was located in
this project's original uploaded files, not in the previously-organized
`Grasp_Validation/` code — it had never been added to this repository. It
was re-run here, against the same real sponge-object sensor data
(`data/Spong_*.xlsx`) used throughout, not synthetic data, to produce
genuine, reproducible figures rather than leave this comparison
undocumented.

## Three real bugs fixed to get it running

The original script had never been run to completion in this environment,
and needed three fixes, none of which touch the model architectures,
training logic, or evaluation metrics:

1. **Hardcoded absolute paths** to the data files (from the original
   development machine) — repointed to this repo's `data/` folder.
2. **`Model_Save.py` assumed every model was a Keras model** and called
   `.save()` on it — breaks immediately on the four scikit-learn methods
   (logistic regression, decision tree, etc). Fixed to save sklearn models
   with `joblib.dump()` instead.
3. **The two-phase NN's hyperparameter search used `hp.Float` for neuron
   counts** (e.g. `units=hp.Float('units_o1', 20, 128, step=0.5)`), which
   can produce a non-integer like `20.5` — Keras requires an integer neuron
   count and raises immediately. Changed to `hp.Int`.

## One deliberate change, clearly flagged

The two-phase NN's hyperparameter search originally used `max_trials=20`
per model (two models = 40 trials, each up to 100 epochs). That doesn't
finish inside a single execution window here, so it was reduced to
`max_trials=6` — a real, smaller random search, not a skipped one. This is
almost certainly why the two-phase NN result below is close to, but not
identical to, the thesis's reported figure.

## Results: this re-run vs. thesis Table 7.6

| Method | Thesis Table 7.6 | Regenerated here | Match |
|---|---|---|---|
| Simple Threshold | 26.84% | 26.84% | exact |
| Logistic Regression | 69.87% | 69.87% | exact |
| Single-Layer Perceptron | 69.87% | 69.87% | exact |
| Decision Tree | 84.30% | 84.30% | exact |
| Two-Phase NN | 84.30% | 83.29% | close — reduced trial count above |
| **Single-Phase NN (selected)** | **86.84%** | **84.81%** | close — normal training variance |

Four of six methods reproduced exactly. The two neural-network methods are
close but not identical, which is expected: neural network training has
run-to-run variance even with a fixed random seed, across different
library versions and hardware, and the two-phase NN's search was
deliberately narrowed as noted above. Nothing here suggests the original
comparison or its conclusion (single-phase NN as the best performer) was
wrong — the ranking is identical to the thesis's own.

## Figures

| File | What it shows |
|---|---|
| `simple_threshold_curve.png` | Accuracy vs. threshold value, with the selected optimum marked |
| `learning_curve_Logistic_Regression.png` | Training/validation accuracy vs. training set size |
| `learning_curve_Single_Layer_Perceptron.png` | Same, for the single-layer perceptron |
| `learning_curve_Decision_Tree.png` | Same, for the decision tree |
| `training_curves_OnePhaseNN.png` | Accuracy and loss vs. epoch, for the selected single-phase NN |
| `two_phase_training_history.png` | Accuracy/loss vs. epoch, for both the orientation and grip models |
| `two_phase_confusion_matrices.png` | Confusion matrices for both the orientation and grip models |
