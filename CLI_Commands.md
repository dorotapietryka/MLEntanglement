entanglement_ml_pipeline.py

A reusable pipeline for studying whether SVMs and other machine-learning models
can learn the separability boundary for bipartite quantum states.

Supported systems
-----------------
1. 2x2 qubit-qubit states:
   * Entangled metric states are generated with Bures or Hilbert-Schmidt sampling.
   * Labels are assigned with the PPT criterion, which is necessary and sufficient
     for 2x2 separability.
   * Separable examples are generated as known separable convex mixtures of
     random product pure states.

2. 3x3 qutrit-qutrit states:
   * Entangled metric states are generated with Bures or Hilbert-Schmidt sampling.
   * NPT states are accepted as entangled immediately.
   * PPT states are delegated to the attached 3x3_svm.py labelling routine, which
     contains the DPS/Gilbert logic. Inconclusive labels are rejected.
   * Separable examples are generated as known separable convex mixtures of
     random product pure states.

Labels
------
y = -1  entangled
y = +1  separable

The saved datasets never contain raw density matrices. They contain only feature
columns and y. The optional NPZ bundle stores exactly:
    SU_features, Moment_features, RMInvariant_features, y

Example CLI usage
-----------------
```bash
# Method 1: existing logic. Known separable mixtures plus metric entangled search.

python entanglement_ml_pipeline.py generate --system 2x2 --metric bures --generation-method 2 --n-entangled 1000 --n-separable 1000 --out data/qubit_bures.csv

# Method 2: metric-consistent rejection sampling for both labels.
python entanglement_ml_pipeline.py generate --system 3x3 --metric hs --generation-method 2 --n-entangled 500 --n-separable 500 --qutrit-script 3x3_svm.py --out data/qutrit_hs.csv

# Method 3: metric-local separable mixtures; entangled rows fall back to Method 1.
python entanglement_ml_pipeline.py generate --system 2x2 --metric bures --generation-method 3 --sep-mixture-terms 16 --n-entangled 1000 --n-separable 1000 --out data/qubit_method3_bures.csv

# Method 4: controlled depolarization after a rejection threshold.
python entanglement_ml_pipeline.py generate --system 3x3 --metric bures --generation-method 4
    --method4-depolarize-after 250 --method4-depolarize-step 0.01 --n-entangled 500 --n-separable 500 --qutrit-script 3x3_svm.py --out data/qutrit_method4_bures.csv


python entanglement_ml_pipeline.py generate --system 3x3 --metric bures --generation-method 3 --sep-mixture-terms 81 --n-entangled 5000 --n-separable 5000 --qutrit-script 3x3_svm.py --out data/qutrit_method3_bures_5k.csv
---

# Evaluate feature groups and models.
python entanglement_ml_pipeline.py evaluate --dataset data2x2/qubit_bures_2.csv --out-dir data2x2/results/qubit_bures

python entanglement_ml_pipeline.py evaluate --dataset data3x3/qubit_method3_bures_5k_81c.csv --out-dir data3x3/results/qutrit_bures

# Plot t-SNE for a chosen feature scenario.
# 2x2
python entanglement_ml_pipeline.py tsne --dataset data2x2/qubit_bures_method2_5k.csv --feature-set ALL --out data2x2/results/qubit_bures_5k/tsne_ALL.png

python entanglement_ml_pipeline.py tsne --dataset data2x2/qubit_bures_method2_5k.csv --feature-set SU --out data2x2/results/qubit_bures_5k/tsne_SU.png

python entanglement_ml_pipeline.py tsne --dataset data2x2/qubit_bures_method2_5k.csv --feature-set Moment --out data2x2/results/qubit_bures_5k/tsne_Moments.png
```
--feature-set RMInvariant
--feature-set SU+Moment

```bash
# 3x3
python entanglement_ml_pipeline.py tsne --dataset data3x3/qubit_method3_bures_5k_81c.csv --feature-set ALL --out data3x3/results/qutrit_bures/tsne_ALL.png

python entanglement_ml_pipeline.py tsne --dataset data3x3/qubit_method3_bures_5k_81c.csv --feature-set SU --out data3x3/results/qutrit_bures/tsne_SU.png

python entanglement_ml_pipeline.py tsne --dataset data3x3/qubit_method3_bures_5k_81c.csv --feature-set Moment --out data3x3/results/qutrit_bures/tsne_Moment.png

python entanglement_ml_pipeline.py tsne --dataset data3x3/qubit_method3_bures_5k_81c.csv --feature-set RMInvariant --out data3x3/results/qutrit_bures/tsne_RMInvariant.png



# Plot held-out accuracy degradation after removing top-ranked RFE features.
python entanglement_ml_pipeline.py ts_rfe_ablation --dataset data2x2/qubit_bures_method2_5k.csv --feature-set SU --model LinearSVC --cv-folds 5 --repeats 5 --out data2x2/results/rfe_ablation_SU.png --out-csv data2x2/results/rfe_ablation_SU.csv

python entanglement_ml_pipeline.py ts_rfe_ablation --dataset data3x3/qubit_method3_bures_5k_81c.csv --feature-set SU --model LinearSVC --cv-folds 5 --repeats 5 --out data3x3/results/rfe_ablation_SU.png --out-csv data3x3/results/rfe_ablation_SU.csv


# Plots RocCurve ResidualPlot MarginDistributionPlot 2x2
python entanglement_ml_pipeline.py --dataset 2x2 --dataset_path data2x2/qubit_bures_method2_5k.csv --features All --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data2x2/results/qubit_bures_5k_plots/All

python entanglement_ml_pipeline.py --dataset 2x2 --dataset_path data2x2/qubit_bures_method2_5k.csv --features SU --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data2x2/results/qubit_bures_5k_plots/SU

python entanglement_ml_pipeline.py --dataset 2x2 --dataset_path data2x2/qubit_bures_method2_5k.csv --features Moments --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data2x2/results/qubit_bures_5k_plots/Moments

python entanglement_ml_pipeline.py --dataset 2x2 --dataset_path data2x2/qubit_bures_method2_5k.csv --features RM --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data2x2/results/qubit_bures_5k_plots/RM
```
# Plots SHAPP 2x2
SHAP is slower, especially for SVMs and for the 3×3 Gell-Mann feature set.

For 2×2:

```bash
python entanglement_ml_pipeline.py --dataset 2x2 --dataset_path data2x2/qubit_bures_method2_5k.csv --features All --plot SHAPPlot --output_dir data2x2/results/qubit_bures_5k_plots/All/shap
```

For a faster SHAP run:

```bash
python entanglement_ml_pipeline_v2.py --dataset 2x2 --features Moments --plot SHAPPlot --output_dir plots_2x2_shap_fast --max-shap-background 30 --max-shap-samples 50 --shap-nsamples 100
```
# Important note about SHAP

`SHAPPlot` is the slowest option.

The updated script handles:

```text
Random Forest → tree-aware SHAP explainer
SVM models    → SHAP KernelExplainer / black-box path
MLP           → model-agnostic SHAP path
```
# Plots 3x3
```bash
python entanglement_ml_pipeline.py --dataset 3x3 --dataset_path data3x3/qutrit_method3_bures_5k_81c.csv --features All --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data3x3/results/qutrit_bures_5k_plots/All

python entanglement_ml_pipeline.py --dataset 3x3 --dataset_path data3x3/qutrit_method3_bures_5k_81c.csv --features Moments --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data3x3/results/qutrit_bures_5k_plots/Moments

python entanglement_ml_pipeline.py --dataset 3x3 --dataset_path data3x3/qutrit_method3_bures_5k_81c.csv --features SU --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data3x3/results/qutrit_bures_5k_plots/SU

python entanglement_ml_pipeline.py --dataset 3x3 --dataset_path data3x3/qutrit_method3_bures_5k_81c.csv --features RM --plot RocCurve ResidualPlot MarginDistributionPlot --output_dir data3x3/results/qutrit_bures_5k_plots/RM
```

# Plots Shapp 3x3
```bash
python entanglement_ml_pipeline.py --dataset 3x3 --dataset_path data3x3/qutrit_method3_bures_5k_81c.csv --features All --plot SHAPPlot --output_dir data3x3/results/qutrit_bures_5k_plots/All/shap
```
### PURITY ANALISYS

We test whether the trained model maintains strong performance when the **training data is restricted to states with extreme purity values**, while evaluation remains general. 
We can use the same database or generate a new one.

```bash
python entanglement_ml_pipeline_v3.py generate --system 2x2 --metric bures --generation-method 2 --n-entangled 5000 --n-separable 5000 --purity-filter --eta 0.02 --out data2x2/results/qubit_purity_filtered.csv
```

```bash
+++
python entanglement_ml_pipeline.py purity_experiment --system 2x2 --metric bures generation-method 2 --eta-grid 0.005 0.01 0.02 0.05 0.10 --n-train 2000 --n-test 3000 --random-state 42 --out-dir data2x2/results/purity_test

--dataset_path data3x3/qutrit_method3_bures_5k_81c.csv --features All --plot SHAPPlot --output_dir data3x3/results/qutrit_bures_5k_plots/All/shap
+++
python entanglement_ml_pipeline.py purity_experiment --system 2x2 --metric bures --eta 0.02 --n-train 1000 --n-test 1000 --out-dir results/purity_test
```
--n-train and --n-test are total row counts. The script splits them as evenly as possible between entangled and separable examples.

sweep multiple η values:
```bash
python entanglement_ml_pipeline.py purity_experiment --system 2x2 --metric bures --eta-grid 0.005 0.01 0.02 0.05 0.10 --n-train 2000 --n-test 3000 --out-dir data2x2/results/purity_test --random-state 42
```
the generate command also supports:
--purity-filter
--eta 0.02
--purity-sampling-mode targeted
--no-purity-column


The new purity_experiment command saves:
results/purity_test/
├── baseline/
│   ├── classification_results.json
│   ├── classification_reports.txt
│   └── rfe_rankings.csv
├── eta_0p02/
│   └── purity_constrained/
│       ├── classification_results.json
│       ├── classification_reports.txt
│       └── rfe_rankings.csv
├── datasets/
│   ├── baseline_train.csv
│   ├── test_unrestricted.csv
│   └── eta_0p02/
│       └── purity_constrained_train.csv
├── purity_distributions/
│   ├── baseline_train_unrestricted_purity_distribution.csv
│   ├── baseline_train_unrestricted_purity_distribution.png
│   ├── test_unrestricted_purity_distribution.csv
│   ├── test_unrestricted_purity_distribution.png
│   └── eta_0p02/
│       ├── purity_constrained_train_purity_distribution.csv
│       ├── purity_constrained_train_purity_distribution.png
│       ├── test_unrestricted_purity_distribution.csv
│       └── test_unrestricted_purity_distribution.png
├── performance_vs_eta/
│   ├── performance_vs_eta_Linear_SVM.png
│   ├── performance_vs_eta_RBF_SVM_optimized.png
│   ├── performance_vs_eta_Random_Forest.png
│   └── performance_vs_eta_MLP.png
├── performance_summary.csv
├── comparison_summary.txt
└── purity_experiment_summary.json

smoke test:
```bash
python entanglement_ml_pipeline_v3.py purity_experiment --system 2x2 --metric bures --eta 0.02 --n-train 4 --n-test 4 --features Moments --no-rfe --n-jobs 1 --no-npz --out-dir /mnt/data/purity_smoke_eta002
```


### 3x3

    python entanglement_ml_pipeline_v3.py purity_experiment --system 3x3 --metric bures --generation-method 2 --eta-grid 0.005 0.01 0.025 0.05 0.10 \
  --n-train 2000 \
  --n-test 4000 \
  --qutrit-script 3x3_svm.py \
  --out-dir results/purity_3x3_bures \
  --random-state 42
---

python -m py_compile entanglement_ml_pipeline.py
python entanglement_ml_pipeline.py --help
python entanglement_ml_pipeline.py generate --help
python entanglement_ml_pipeline.py evaluate --help
python entanglement_ml_pipeline.py tsne --help
python entanglement_ml_pipeline.py ts_rfe_ablation --help
python entanglement_ml_pipeline.py purity_experiment --help