# Gains beyond 0.51 macro-F1 on 16-class ordinal tabular data

Context: 10962 train rows, 311 numeric features, 16 ordered age bands, macro-F1 metric, stratified 5-fold seed 42, CPU only, champion is Frank and Hall cumulative XGB plus gated pairwise override at CV 0.5089 and public 0.51.

## Ordinal losses not yet tried here

### 1. Frank and Hall with rank-consistent decoding (do first)

Mechanism: train 15 cumulative binary classifiers P(y > k) and reconstruct class probabilities by differencing, then enforce monotonicity by isotonic regression or cumulative max. Why it can beat the champion: the current recipe already uses this structure but independent XGBs can emit non-monotonic cumulative curves, and inconsistent curves concentrate errors off by one. Expected gain: +0.002 to +0.006. Cost on 11k by 311: 15 small XGBs, minutes on CPU. First experiment: reuse champion binaries, apply isotonic calibration per row on out-of-fold predictions, compare macro-F1 before and after on identical folds. Primary source: Frank and Hall simple approach to ordinal classification (https://link.springer.com/chapter/10.1007/3-540-44795-4_13).

### 2. CORAL and CORN formulations ported to GBDT

Mechanism: CORAL shares one weight vector across K-1 binaries with separate biases, which guarantees rank consistency; CORN replaces this with conditional training subsets and chain-rule decoding. Why it may help: same inconsistency argument as above, but CORN removes the restrictive weight-sharing constraint. Expected gain: +0.002 to +0.008 for CORN-style decoding, near zero for strict CORAL weight sharing on trees. Cost: same 15 binaries plus decoding, minutes. First experiment: keep 15 XGBs, decode with CORN chain rule P(y > k) = prod P(y > j | y > j-1) estimated on conditional subsets versus plain differencing. Primary sources: CORAL (https://arxiv.org/abs/1901.07884) and CORN (https://arxiv.org/abs/2111.08851); code reference (https://github.com/Raschka-research-group/coral-pytorch).

### 3. SORD soft labels with metric spacing

Mechanism: replace one-hot targets with a discretised exponential-decay distribution over rank distance, controlled by a distance metric and sharpness parameter. Why it may help: penalises far errors more than near errors, matching the observed 75 to 80 percent off-by-one error mass. Expected gain: 0 to +0.004; prior soft-label attempt failed, so treat as low priority. Cost: one multiclass XGB with soft targets, minutes. First experiment: single XGB multiclass with SORD targets, grid over sharpness on out-of-fold predictions only. Primary source: Diaz and Marathe soft labels for ordinal regression (https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf).

### 4. Label distribution learning with expectation decoding

Mechanism: train against a Gaussian label distribution centred on the true band and predict by expected value rounded to the nearest class. Why it may help: adjacent-band uncertainty is explicitly supervised instead of emerging accidentally. Expected gain: 0 to +0.004 on tabular survey data; face-age gains do not transfer automatically. Cost: one regressor or 16-output model, minutes. First experiment: Gaussian sigma 1.0 targets with XGB multi-output regression, expectation plus per-class bias tuned on out-of-fold data. Primary sources: label distribution learning overview (https://palm.seu.edu.cn/xgeng/LDL/index.htm) and expectation of label distribution learning (https://ijcai.org/proceedings/2018/99).

### 5. Custom ordinal objective in XGBoost

Mechanism: implement a twice-differentiable ordinal loss, for example weighted kappa or earth mover distance surrogate, through the custom objective interface returning gradient and hessian. Why it may help: trains trees directly on an ordinal cost instead of fixing it at decoding. Expected gain: +0.002 to +0.01 if the surrogate is well conditioned, negative if hessians are unstable. Cost: one custom XGB, tens of minutes with debugging. First experiment: replicate multi softprob gradients first, then add distance-weighted off-diagonal penalty. Primary source: XGBoost custom objective and metric tutorial (https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html).

### 6. Ordered logit as calibrator, not champion

Mechanism: fit a proportional-odds cumulative logit on a small feature subset or on model scores. Why it helps only modestly: strong parametric assumption rarely beats GBDTs on 311 heterogeneous survey features. Expected gain: 0 to +0.002 as a stacking input. Cost: seconds. First experiment: cumulative logit on top of 15 cumulative XGB scores. Primary source: McCullagh regression models for ordinal data (https://www.stat.uchicago.edu/~pmcc/pubs/paper2.pdf).

## Direct macro-F1 optimisation at decision time

### 7. Consistent plug-in rule for multiclass macro-F1

Mechanism: estimate calibrated posteriors, then choose the confusion-matrix-optimal decision on validation data; theory shows consistent algorithms exist for arbitrary functions of the confusion matrix. Why it matters here: argmax optimises accuracy, not macro-F1, and rare classes need lower effective thresholds. Expected gain: +0.003 to +0.01, the largest honest decision-time lever left. Cost: seconds on stored out-of-fold posteriors. First experiment: per-class additive bias grid tuned by inner-fold macro-F1 inside each outer fold. Primary source: Narasimhan et al consistent multiclass algorithms for complex performance measures (https://proceedings.mlr.press/v37/narasimhanb15.pdf).

### 8. Cost-sensitive reduction for F-measures

Mechanism: reduce F-optimisation to a series of cost-sensitive classification problems over a discretised cost grid. Why it matters: gives a principled alternative to ad hoc sample weights, which already failed here four times. Expected gain: +0.002 to +0.006. Cost: tens of model fits across the cost grid, under one hour. First experiment: per-class one-versus-rest XGB over a small cost grid, thresholds picked on inner folds. Primary source: Parambath et al optimising F-measures by cost-sensitive classification (https://papers.neurips.cc/paper_files/paper/2014/file/5c0314ec1b57fcd36bbb013f3f025868-Paper.pdf).

### 9. Read the F1 threshold theorems narrowly

Mechanism: for calibrated binary posteriors the optimal F1 threshold equals half the achievable F1, and uninformative rare-label models collapse to predict-all-positive under macro averaging. Why it matters: published threshold results are mostly binary or multilabel, not single-label multiclass with argmax over 16 classes, so naive per-class thresholding can inflate rare classes pathologically. Expected gain from correct use: +0.002 to +0.005; from naive use: negative. Cost: seconds. First experiment: apply bias tuning only, never independent per-class argmax overrides, with nested folds. Primary source: Lipton et al optimal thresholding of classifiers to maximise F1 (https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/).

### 10. Expected-F1 and ordinal-cost decoding

Mechanism: decode argmax over expected macro-F1 or over an ordinal cost matrix rather than over posterior probability. Why it may help: directly exploits the off-by-one structure by shifting borderline mass toward the neighbour with better class-wise F1. Expected gain: +0.002 to +0.005. Cost: seconds on stored posteriors. First experiment: posterior smoothing over adjacent classes with weight alpha tuned on inner folds, then Bayes-cost decoding. No new model needed.

## Foundation models and CPU tabular methods

### 11. TabPFN: version decides legality

Mechanism: in-context tabular foundation model used as an extra diversity member. Fact check: TabPFNv2, 2.5 and 2.6 support at most 10 classes, so they cannot take 16 classes natively (https://docs.priorlabs.ai/models). TabPFN-3 and 3.5 support up to 160 classes natively (https://docs.priorlabs.ai/models). CPU caveat: hosted docs cap CPU fits far below this task size and large fits need subsampling or the hosted API. Verdict: legal only on v3 or later, likely slow on CPU at 11k by 311, expected gain 0 to +0.003 as ensemble diversity. First experiment: TabPFN-3 base, subsampled context, out-of-fold posteriors blended at low weight. Needs torch and conda install, which breaks the NB01 to NB07 old-stack rule.

### 12. TabICL: hierarchical wrapper is not free ordinal signal

Fact check: TabICL is pretrained on up to 10 classes natively and handles more only through built-in hierarchical classification (https://pypi.org/project/tabicl/0.1.1/). Paper context: pretrained on synthetic sets up to 60k samples with larger-scale handling (https://arxiv.org/abs/2502.05564). Verdict: runs on CPU but hierarchical splits ignore order, torch dependency, expected gain 0 to +0.003. First experiment only if TabPFN v3 is already in place: TabICL as one more blender member, never as standalone champion.

### 13. Autogluon: do not propose under current constraints

Autogluon tabular is pip-first with heavy optional dependencies (https://auto.gluon.ai/stable/install.html) and CPU tuning on 11k by 311 with 16 classes is slower than the XGB champion without an ordinal advantage. Verdict: skip unless the conda and CPU rules are relaxed.

## Honest validation discipline

Use nested folds: all thresholds, biases, smoothing weights and blender weights tuned on inner folds, judged once on outer out-of-fold predictions (https://scikit-learn.org/stable/modules/cross_validation.html). Never tune on the public leaderboard; with 3 submissions per day and a 30/70 public-hidden split, leaderboard hill-climbing is the main overfitting risk. Reject anything transductive that touches test features or test-derived statistics before prediction, including test-based relabelling, test-mean imputation parameters, or semi-supervised retraining with test pseudo-labels evaluated on the same test split. Monotonic constraints use LightGBM native support (https://lightgbm.readthedocs.io/en/latest/Parameters.html) and stay honest because constraints are declared before fitting.

## Kill list with reasons

SMOTE: synthesises ordinal survey rows that break Likert-scale coherence; already banned by analysis. PCA and t-SNE: discard supervised ordinal signal on wide survey data. RFECV: unstable under 16 imbalanced classes, already rejected. Plain MLP: scored 0.39 here and CORN-style MLPs need torch outside the frozen stack. Vanilla seed averaging: already shown flat. Unnested threshold tuning and public-LB tuning: winner's curse artefacts documented in the F1 thresholding literature (https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/). Test-label relabelling or quarantine with leakage: invalid by construction.
