# Grasp-Validation Methodology Comparison

The code behind thesis Table 7.6: trains and compares six approaches to
grip-quality classification on the same real sensor data, to justify why
the single-phase neural network was selected over the alternatives.

This folder is not split into `final/`/`testing/` like the other
Grasp_Validation subfolders — there's no separate deployed system here to
test against; running the comparison itself *is* the result. Generated
figures from an actual run are in
`Simulation_Results/Grasp_Validation_Methodology_Comparison/`, alongside a
results writeup comparing them against the thesis's published numbers.

## Running it

```bash
cd Grasp_Validation/methodology_comparison
python Comparison_Test_With_Graphs.py
```

Trains and evaluates, in order: a simple threshold, logistic regression, a
single-layer perceptron, a decision tree, the single-phase NN, and the
two-phase NN (orientation detection feeding a grip-quality stage). Prints
each method's accuracy and saves every generated figure into a
`results_figures/` folder created alongside these scripts.

**Runtime note:** the two-phase NN's hyperparameter search
(`max_trials=6` per model, two models) is the slow part — the other five
methods together take well under a minute on the bundled data; the full
run with the search included takes several minutes.

## Fixes applied to the original scripts

These were bugs in the original code, not something introduced by
reorganizing it — each one made the script fail outright before reaching
that method:

- **Hardcoded absolute paths** to the training data, from the original
  development machine — repointed to the `data/` folder in this
  directory (a copy of the same sponge-object recordings used
  throughout `Grasp_Validation/`).
- **`Model_Save.py` called `.save()` on every trained model**, which
  only exists on Keras models — the four scikit-learn methods
  (threshold, logistic regression, perceptron, decision tree) errored
  immediately. Now checks the model type and uses `joblib.dump()` for
  scikit-learn models.
- **The two-phase NN's hyperparameter search specified neuron counts
  with `hp.Float(..., step=0.5)`**, which can produce a non-integer
  value — Keras requires an integer `units` count for a `Dense` layer
  and raises immediately. Changed to `hp.Int`.

## Files

| File | Role |
|---|---|
| `Comparison_Test_With_Graphs.py` | Orchestrator — loads data, runs all six methods, saves every figure |
| `Simple_Threshold_Test.py`, `Logistic_Regression_Test.py`, `Single_Layer_Perceptron_Test.py`, `Decision_Tree_Test.py` | One function each — trains that method, returns accuracy/predictions (and a learning curve for the four scikit-learn methods) |
| `Two_Phase_NN_Test.py` | The two-phase NN, including its own data loader and Keras-Tuner search |
| `Model_Save.py` | Saves a trained model with a timestamped filename, Keras or scikit-learn |
| `data/` | The same real sponge-object sensor recordings used throughout `Grasp_Validation/` |
