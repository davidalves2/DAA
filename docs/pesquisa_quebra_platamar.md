# Pesquisa — quebras de platamar ~0.51 macro-F1 (16 classes ordinais, tabular)

> **Status:** research-only (no experiments executed), 2026-11-10. Deliverable of a
> background research task. Scope: credible, untried (or insufficiently tried) ways to
> break the honest CV ceiling of **0.504–0.509 macro-F1**, constrained to the conda env
> `daa` (CPU-only) and to the existing 5-fold protocol.
> **Method:** primary sources only — official library docs (scikit-learn 1.9.1,
> LightGBM 4.6, imbalanced-learn 0.14, statsmodels 0.14, cleanlab 2.9, PriorLabs docs),
> papers on arXiv/journal (cited inline), package registries (PyPI/conda-forge/GitHub),
> and the repo's own experiment logs. Every non-obvious claim carries a citation;
> things I could not verify are flagged in §12. Env facts were verified by execution
> (imports, versions, checkpoint filenames), not by memory.

## 0. Pergunta e regras do jogo (baseline para todas as estimativas)

- Metric: **Macro F1 over 16 classes** (`docs/info_kaggle.md:56`); 3 valid
  submissions/day (`docs/info_kaggle.md:94`); final = 30% public + 70% private split
  (`docs/contexto.md:7`).
- Protocol: `StratifiedKFold(5, shuffle, seed=42)` on `data/train_nb02.csv` (311
  features), same row order → OOFs comparable (`HANDOFF.md:11-13`); promotion =
  NB18 holdout screen (bar **+0.010**; C3 row notes `barra +0.015`) → honest 5-fold,
  adopt on 5/5 (protocol header `docs/plano_acima05.md:6-8`); NB17 P1 was accepted at
  4/5 (`data/nb17_resultados.csv`).
- **Dominant fact:** the relabel track was retracted as a leakage artifact (labels
  regenerated with the held-out fold inside → +0.036…+0.061 inflation; `HANDOFF.md:17-37`,
  `notebooks/Notebook_16.ipynb`). Rule: *anything used inside CV must be regenerable
  without the held-out fold* (`HANDOFF.md:36-37`). Honest ceiling = colleague's sub7 =
  **0.5089 CV → 0.51 public** (`HANDOFF.md:32`, `registo_experiencias.md:346`).
- Noise floor: deltas ≤ ±0.004 are noise (σ 0.008–0.012; `HANDOFF.md:34-35`).
- Cost yardstick: single XGB fit ≈ 41.5 s on an 80/20 split (`data/nb18_resultados.csv`,
  T0). A full sub7 5-fold pass (5 XGB + 4 LGBM + 2 HGB ordinal families × 15 binaries
  × 5 folds ≈ 825 fits) is **≈3–5 h CPU** (my estimate from that yardstick, not measured).
- Environment (verified by execution 2026-11-10): pandas 3.0.6, sklearn 1.9.1,
  xgboost 3.4.1, lightgbm 4.6.0, catboost 1.2.10, **statsmodels 0.14.6**,
  **cleanlab 2.9.0**, **imbalanced-learn 0.14.2**, **torch 2.14.1**, **tabpfn 9.2.0**,
  shap 0.52; no mord/optuna/EBM/tabicl. Conda-only for new deps (`AGENTS.md`).

## 1. Thread A — Ordinal methods beyond Frank & Hall

**What the primary sources say**

| Method | Primary source | What it actually is |
|---|---|---|
| Frank & Hall (zero-one reduction) | Frank & Hall, *A Simple Approach to Ordinal Classification*, ECML 2001, LNCS 2167:145–156 (author copy: researchcommons.waikato.ac.nz, PDF `d4decc50…/download`) | 15 binary classifiers for P(Y>k); already our best recipe (`registo_experiencias.md:262-285`) |
| Threshold/extended-binary reduction | Lin & Li, *Ordinal Regression by Extended Binary Classification*, NIPS 2006 (papers.nips.cc/paper_files/paper/2006/…-Paper.pdf) | Ordered threshold vector θ_k on a shared score g(x); mechanism already covered by F&H + our threshold work M1/M2 |
| CORAL (rank-consistent deep head) | Cao, Mirjalili & Raschka, *Rank consistent ordinal regression for neural networks…*, Pattern Recognition Letters 140:325–331, 2020, arXiv:1901.07884, DOI 10.1016/j.patrec.2020.11.008 | Shared-representation + cumulative heads with rank-consistency guarantees — for **neural nets**; CORN variant arXiv:2111.08851 |
| `mord` | github.com/fabianp/mord; PyPI last release **2023-05-26**; conda-forge last updated **2019-10-10** | sklearn-compatible ordinal logistic/ridge/threshold models — **pip-only and stale → blocked by the conda rule** (repo verdict `docs/plano_acima05.md:28` verified against the registry) |
| Proportional-odds / cumulative link | statsmodels `OrderedModel` docs (statsmodels.org/stable/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html); **installed 0.14.6, import verified** | Latent linear model + ordered cutpoints (logit/probit); docs mark it "experimental"; no intercept allowed; likelihood fit on 311 one-hot features × 16 classes is feasible but linear |
| LightGBM lambdarank as ordinal | LightGBM docs: `objective=lambdarank`, `label_gain` (Parameters.html; Advanced-Topics.html "LambdaRank") | Ranking loss with configurable per-grade gain. **Caveat: it outputs a scalar relevance score, not class probabilities** — using it for 0–15 labels requires a fold-fit score→class mapping. I found **no primary source** showing lambdarank beats F&H on tabular ordinal data (flagged §12) |
| Ordinal forests (OREMP, FIMLR) | — | **No primary source found** for these names (flagged §12); any Python impl would be pip-only anyway → blocked |
| 2024–2026 tabular OC | TOC-UCO benchmark, arXiv:2507.17348 (46 datasets, U. Córdoba); ADABORD, arXiv:2607.21003 (AdaBoost with **ordinal Gini split + ARPS error**, claims significant gains vs 7 OC methods on TOC-UCO, esp. ≥5 classes, code public) | ADABORD's custom split criterion means implementing our own tree/AdaBoost loop — high implementation cost; its ARPS-reweighting idea is close in spirit to our rejected RPS custom loss (+0.0016, 3/5; `docs/plano_acima05.md:92`) |

**Repo status:** F&H = the core of sub7 (i) already tried. CORAL/mord/forests parked or
blocked (M6, `docs/plano_acima05.md:28`). Threshold tuning M1/M2 rejected by honest
nested eval (`docs/plano_acima05.md:23-24`). Custom losses M7/RPS weak (ibid. :92).

**Verdict:** (i)/(iii) mostly. **Novel & cheap here:** (a) statsmodels `OrderedModel` as
a *diverse linear member* — expected **±0.002** (linear models sat at 0.43 in this repo,
`registo_experiencias.md:130`; diversity diagnosis says new views don't help, §G), cost
≈0.5 h, leakage low; (b) lambdarank member — expected **+0.000…+0.004**, cost ≈1–2 h,
leakage low. Both are diversity bets, not mechanism bets.

## 2. Thread B — Optimizing macro-F1 directly

- **Theory:** thresholding rules for F1, incl. a **macro-F1 case study** (plug-in rules
  for micro/macro/per-instance): Lipton, Elkan & Narayanaswamy, *Thresholding Classifiers
  to Maximize F1 Score*, arXiv:1402.1892 (2014). Macro-F1 thresholds are coupled across
  classes (shared denominators) — per-class independent tuning is theoretically shaky.
- **First-party tooling:** sklearn `TunedThresholdClassifierCV` optimizes a **binary
  metric only** and warns that `cv="prefit"` overfits (sklearn 1.9 docs,
  model_selection; source `_classification_threshold.py#L499`) — i.e. sklearn offers no
  macro-F1 threshold tuner; `f1_score(..., sample_weight=…, average='macro')` exists for
  *evaluation* weighting (sklearn.metrics.f1_score docs, docstring verified).
- **Differentiable surrogates:** sigmoidF1, *A Smooth F1 Score Surrogate Loss for
  Multilabel Classification*, arXiv:2108.10566 (2021) — approximates **macro-F1**, but
  for *multilabel* settings; no primary evidence for multiclass tabular GBDT.
  Focal loss (Lin et al., arXiv:1708.02002, ICCV 2017) was designed for dense
  **detection** foreground/background imbalance — no tabular-GBDT evidence found.
  Logit adjustment (Menon et al., *Long-tail learning via logit adjustment*,
  arXiv:2007.07314, ICLR 2021) — training-time bias by log class prior; our post-hoc
  prior-α already tested **−0.0018** (`docs/plano_acima05.md:90`), and prior-dividing
  "chooses always a=0" (`registo_experiencias.md:297`).
- **Cost-sensitive first-party example:** sklearn gallery *Post-tuning the decision
  threshold for cost-sensitive learning* (plot_cost_sensitive_learning).

**Repo status:** (i) tried and honestly rejected: M1 F&H threshold coordinate ascent
(−0.0058 nested), M2 nested per-class offsets (−0.0072), prior-α, E15 thresholds +0.0019,
confusion 16×16 correction rejected, class weights `balanced` already default
(`docs/plano_acima05.md:23-24`, `docs/plano_modelacao.md:145-146`, `:134`).

**Verdict:** (i) mostly closed. **Do not re-propose** threshold tuning, cost matrices,
focal/soft-F1 custom losses (RPS precedent +0.0016 3/5 → M7 verdict stands). Expected
gain from anything in this thread: **≤ +0.003**, cost 0.5–2 h, leakage risk **high**
(every variant tempts same-fold tuning — that's exactly what killed M1/M2).

## 3. Thread C — Label noise & cleanlab, done honestly

- **Primary:** Northcutt, Jiang & Chuang, *Confident Learning: Estimating Uncertainty in
  Dataset Labels*, JAIR 2021, DOI 10.1613/jair.1.12125 (arXiv:1911.00068). cleanlab docs
  are explicit that `pred_probs` **must be out-of-sample** (K-fold OOF), else results are
  invalid (docs.cleanlab.ai/stable/tutorials/pred_probs_cross_val.html).
- **Repo status:** NB11 ran cleanlab — Q1 "41% upper bound" on issue count, Q2
  quarantine drop-worst adopted then shown to be the *selection channel* of the leak
  (`docs/plano_acima05.md:84`, `HANDOFF.md:27-28`). Honest nested relabel is
  neutral-to-negative, all pairs ≤ 0 (`HANDOFF.md:29-31`). GBDT label-noise literature
  exists (Eisenbürger et al., *Training Gradient Boosted Decision Trees on Tabular Data
  Containing Label Noise…*, arXiv:2409.08647 v2 2025 — detection accuracy >99%, but its
  framing confirms detection ≠ score gain).

**Verdict:** (iii)-adjacent — the *mechanism* is fold-safe in principle (OOF-only, per
cleanlab docs), but this repo measured that honest relabel/quarantine gains are ≤0.
Novel variant (cross-fitted cleanlab **sample weights** instead of relabel) has no
primary evidence of beating relabeling and inherits the same neutrality. Expected
**±0.000**, cost ≈1–2 h, leakage risk medium (must regenerate issues inside each fold).
**Recommendation: do not bother** unless for the report's noise-audit section.

## 4. Thread D — Rare classes (172-sample class 1) & specialization

- **Primary:** imbalanced-learn docs — `BalancedRandomForestClassifier` (per-tree
  balanced bootstrap, Chen/Liaw/Breiman 2004), plus `EasyEnsembleClassifier`,
  `RUSBoostClassifier`, `BalancedBaggingClassifier` (imbalanced-learn.org/stable/
  references/…; all installed, imblearn 0.14.2).
- **Repo status:** BRF full run **0.4302** (rapid 0.4053) — rejected as a meta family
  (`docs/plano_acima05.md:71-72`, NB10 row :83). SMOTE banned (`docs/plano_modelacao.md:186`).
  Hierarchy (3 group specialists + meta-logreg) **+0.009** but below the XGB recipe —
  abandoned as a base (`docs/plano_modelacao.md:190-194`). Pairwise gated overrides =
  the per-class-specialist idea in disguise: 6 pairs +0.0047 (rejected), 15 pairs
  screened +0.0162, **P1 honest 5-fold today = 0.4854, +0.0077 vs XGB-311, 4/5 —
  CONFIRMADO** (`data/nb17_resultados.csv`), and C3 (target-mean + pairwise15) screened
  **+0.0239** on NB18 (`data/nb18_resultados.csv`).

**Verdict:** (i) tried in most forms; the **live** levers are the pairwise/C3
integrations (§10). Fold-safe oversampling *inside* the 15 F&H binary heads is novel
here but expected **±0.001** (binary heads already use `balanced` weights), cost ≈1 h,
leakage low → only as a micro-lever. Hierarchical variants on top of sub7: expected
**+0.000…+0.003** (42.9% all-agree-wrong says routing won't fix within-group errors),
cost 3–6 h → low priority.

## 5. Thread E — Survey-specific methodology

- **WGT (official survey weight):** removed as **leakage** because the INE weighting
  calibrated on the **quinquennial age group** (among NUTS II, sex, education,
  household size) — i.e. the weight encodes the target. Source: notebooks/Notebook_01.ipynb
  (markdown "Nota" after the drop cell, lines ~139–160, quoting the INE methodological
  note). ⚠️ The INE note itself was **not re-fetched** in this research (§12) — the claim
  is the repo's quotation of a primary document. Reinstating WGT would be direct target
  leakage → **never**.
- **Routing / paradata:** the survey's "who gets asked what" signal is already modeled
  (17 routing flags, `N_NAO_APLICAVEL`, block counts F2 shelved, `docs/plano_acima05.md:38`,
  T7a −28 cols adopted `:82`); NB14 measured **routing = 36.2% of |SHAP|**
  (`docs/plano_acima05.md:94`). The survey-statistics literature on paradata (Groves &
  Heeringa, *Responsive Design for Household Surveys*, JRSS-A 169(3):439–457, 2006,
  JSTOR 3877429 — abstract seen, full text not fetched) uses paradata for **fieldwork
  cost/response management**, not for per-row target prediction — so it supports the
  *report narrative*, not a new feature idea. Official gates are already extracted from
  the CAPI questionnaire (`docs/plano_acima05.md:52-63`, `docs/mapa_routing.md`).
- **Small-area estimation:** designed for population *estimates* with design-based
  variance, not individual classification — not applicable to per-row macro-F1 (no
  primary source needed for this; it is a scope mismatch).

**Verdict:** (i) done; WGT blocked by leakage; paradata/SAE = report material only.
Expected score gain: **0**.

## 6. Thread F — Test-time / transductive use of the unlabeled test set

- **Rules:** the provided competition text (`docs/info_kaggle.md:101-113`) says the
  training set is for develop/train/optimize and the test set validates generalization;
  it contains **no explicit prohibition** of unsupervised/transductive use — but it
  offers no blessing either (§12: no rule text beyond this doc was available). Course
  protocol still forbids tuning on the public leaderboard (`docs/plano_modelacao.md:88-94`).
- **Repo status:** pseudo-labeling/self-training was simulated honestly (fold j =
  unlabeled) → **0.4770 vs 0.4778, 2/5 — neutral** (`data/nb17_resultados.csv`, row A);
  generative Naive Bayes base + blend destroyed the base (`data/nb17_resultados.csv`,
  row B; `docs/plano_acima05.md:25-26`). Unsupervised representation (E13 MLP) rejected
  (`docs/plano_modelacao.md:15`). Train→test shift already measured ≈ **0.16pp mean**
  (`docs/plano_modelacao.md:164`).
- **Still untried, label-free:** Saerens–Latinne–Decaestecker **EM prior adjustment**
  (Neural Computation 14(1):21–41, 2002, DOI 10.1162/089976602753284446): iterate
  p(y|x) ← p(x|y)p̂(y) on unlabeled rows to re-estimate the label prior, no refit.
  Fold-safe by construction if run on the held-out fold's features only.

**Verdict:** (ii) novel here but **expected +0.000…+0.002** (no shift to correct; prior
scaling already "chooses a=0", `registo_experiencias.md:297`), cost ≈10 min, leakage low
(if implemented fold-safely). Worth only as a fishing experiment; pseudo-labeling and
autoencoders: **do not bother** (measured neutral/negative).

## 7. Thread G — Ensembling beyond what's tried

- **Repo evidence (primary):** soft voting of GB+RF+logreg was *worse* than best
  (`registo_experiencias.md:171`); stacking underperformed simple averaging
  (`registo_experiencias.md:267`); cross-fitted stack +0.0055, metastack E12 0.5037
  frozen, E12b ties (`docs/plano_modelacao.md:12-18`); E8/E15/E16/E14 all ties
  (`docs/plano_modelacao.md:14-17`); seed-average +0.0047 marginal (E14); feature-subset
  diversity (top-150 chosen per fold) = tie (`registo_experiencias.md:309`);
  **42.9% of errors: all 4 bases agree and are wrong** (`docs/plano_acima05.md:12-13`).
- **Implication:** meta-level combination is exhausted; the only ensembling that can
  still move macro-F1 is (a) integrating a *confirmed* post-hoc correction (pairwise-15)
  into the winning recipe, and (b) adding bases whose errors differ at the *base* level
  (lambdarank/ranker member, TabPFN — §9). Row-level bagging (bootstrap rows per seed)
  is untried as such — XGB already uses `subsample=0.8` per tree, so incremental.
- Breiman-style bagging theory (bootstrap-aggregation reduces variance of unstable
  learners) is textbook; no specific primary fetch needed for this claim.

**Verdict:** (i) stacking/combos closed; (ii) row/seed-bagging of the recipe = cheap
finalizer (already half-planned: M4 "seed-bagged metastack … TODO, finalizer",
`docs/plano_acima05.md:26`). Expected **+0.001…+0.004**, cost +3–6 h CPU, leakage low.

## 8. Thread H — Genuinely new (2024–2026)

| Item | Primary source | Feasibility here | Verdict |
|---|---|---|---|
| **TabPFN-3 class limit is obsolete** | PriorLabs docs `changelog/tabpfn-3`: "Classification now supports up to **160 classes** without extensions", 1M rows, weights need **one-time license acceptance**; `models` page (TabPFNv2: 10 classes, 10k rows) | **Installed tabpfn 9.2.0 ships v3/v3.5 checkpoints** (verified locally: `tabpfn/model_loading.py` lists `tabpfn-v3-classifier-v3_default.ckpt`, `tabpfn-v3.5-20260909.safetensors`); torch 2.14.1 ≥ required 2.5 | **Partially unblocked:** the repo's "10-class hard limit" reason (`docs/plano_acima05.md:69`, `HANDOFF.md:87-88`) is **stale for 16 classes**. Remaining gates: license token (user decision), CPU pilot cost. Not pip-blocked for <160 classes |
| TabICL (open, many-class) | Qu et al., *TabICL*, ICML 2025, arXiv:2502.05564; TabICLv2 arXiv:2602.11139; github.com/soda-inria/tabicl (`pip install tabicl`, `support_many_classes=True`, CPU device supported) | **pip-only → blocked by conda rule** (same class as tabpfn-extensions) | (iii) blocked; mention to user only if they grant a pip exception |
| ADABORD (ordinal Gini + ARPS AdaBoost) | arXiv:2607.21003 (2026), code public | Custom tree criterion → substantial in-house code; ARPS≈ our rejected RPS loss | (ii) novel but poor expected gain/cost; park |
| ORCU (ordinal calibration loss) | Kim, Chung & Jang, arXiv:2410.15658 (v4 2026) | Deep-net loss; calibration already shown **not** to move F1 here (ECE 0.105→0.048, F1 flat; `docs/plano_modelacao.md:11-12`) | (i)/(iii) — do not bother |
| Neural ordinal nets on CPU | E13 MLP rejected (`docs/plano_modelacao.md:15`); CORAL/CORN are the neural ordinal family (§A) | torch is installed; 11k rows is MLP-feasible | (i) family already lost — do not bother |
| OrdinalGBT (LightGBM ordinal loss) | ordinalgbt.readthedocs.io — `pip install ordinalgbt`, TODO list still missing all-thresholds/ordistic losses | pip-only, immature | (iii) blocked; its cumulative-threshold loss ≈ F&H mechanism anyway |

## 9. Prioritized shortlist (max 5, ordered by expected gain ÷ cost)

All specs run under the fixed protocol: `StratifiedKFold(5, shuffle, seed=42)` on
`data/train_nb02.csv`, Macro-F1, adoption = honest 5-fold beat the frozen reference
(sub7 = 0.5089, or the experiment's own declared reference) with the bar **+0.010** on
the NB18-style holdout screen first. Every selection step (pairs, thresholds,
encodings) must be **fit inside the training fold** (leak lesson, `HANDOFF.md:36-37`).

1. **Pairwise-15 overrides integrated into the sub7 recipe, nested.** Spec: regenerate
   sub7 3-family OOF on 311, apply the T10 15-pair gated override with pair list +
   override params selected *inside each training fold* (or pre-declared list +
   in-fold params), compare vs sub7 0.5089, 5-fold. — Expected **+0.003…+0.008**
   (P1 measured +0.0077 on flat XGB, 4/5); cost **≈3–5 h** (recipe pass) + minutes;
   leakage **medium→low** (pair list was screened on the NB18 holdout, which sits
   inside the CV data — nesting removes the channel; not nesting = selection channel).
2. **C3 honest confirmation: fold-safe target-mean + pairwise15.** Spec: recompute the
   smoothed target-mean encoding strictly on each fold's training rows, add to the #1
   stack, same protocol; screen on holdout (C3 = **+0.0239**, `data/nb18_resultados.csv`)
   then 5-fold. — Expected **+0.005…+0.015** (holdout screen includes selection
   optimism; individual target-mean was ≤ +0.005, `docs/plano_acima05.md:29`);
   cost **≈3–5 h**; leakage **medium** (encoding must be fold-fit; same pair-list caveat).
3. **Seed/row-bagging finalizer (M4+).** Spec: average the adopted recipe over ≥10
   seeds with per-seed row bootstrap (80% rows) + existing colsample; 5-fold vs adopted
   config. — Expected **+0.001…+0.004** (E14 seed-average +0.0047 precedent);
   cost **+3–6 h**; leakage **low**.
4. **Lambdarank/ranker member for the regressor slot.** Spec: `LGBMRanker(
   objective='lambdarank', label_gain=quadratic 0..15)` (label_gain verified accepted in
   4.6), single model on labels 0–15, score→class via fold-fit quantile mapping,
   average into the recipe as an extra member; 5-fold. — Expected **+0.000…+0.004**
   (no primary evidence it beats F&H; diversity-only bet); cost **≈1–2 h**;
   leakage **low**. (XGB `rank:pairwise` is the same family if LGBM grouping is awkward.)
5. **TabPFN-3 pilot as an extra family — USER-GATED.** Spec: after the user accepts the
   PriorLabs license (TABPFN_TOKEN / gated weights), fit `TabPFNClassifier` from the
   *installed* 9.2.0 (v3 checkpoints, 160-class limit OK, no pip extension needed) on
   each fold's train rows, CPU; check error-overlap vs sub7 before averaging; bar +0.010.
   — Expected **unknown (±0.01)** — genuinely different inductive bias, but unmeasured
   here; cost **1–3 h** pilot; leakage **low**; blocked only on the license decision
   (HANDOFF already lists this as a user decision, `HANDOFF.md:87-88`, but its
   10-class rationale needs updating).

## 10. Do not bother (already tried, rejected, or blocked — do not re-propose in the same form)

Hyperparameter tuning (defaults win, `registo_experiencias.md:232-234`) · SMOTE /
BalancedRandomForest / EasyEnsemble / RUSBoost (BRF 0.4302) · feature selection
(MI/drop-by-gain: T4 −0.0037, T8 −0.0012, `data/nb18_resultados.csv`) · target-mean
encoding *alone* (+0.005 holdout-only) · monotone constraints (+0.0095, under bar) ·
relabel/quarantine channels (leaky or neutral, `HANDOFF.md:25-31`) · threshold tuning
incl. sklearn `TunedThresholdClassifierCV`-style (M1/M2 nested-negative) · confusion-matrix
correction (16×16) · CORAL/mord/ordinal-forest packages (stale, pip-only, or
unverifiable) · LDL/soft-F1/focal/custom losses (M7 + RPS +0.0016 3/5) · self-training /
pseudo-labeling (neutral, `data/nb17_resultados.csv`) · DES/instance gating (negative) ·
symbolic formula search (closed) · external data / PORDATA joins (no keys) · EBM
(no conda pkg) · TabICL & tabpfn-extensions (pip-only) · autoencoder/representation
learning on train+test (E13, NB17-B) · prior scaling / IPF distribution matching (a=0)
· reinstating WGT (target leakage) · per-class cost matrices · manual interactions ·
OvR + calibrated priors (flat multiclass already ~0.02 below F&H) · small-area estimation
(scope mismatch) · ADABORD full reimplementation (poor gain/cost).

## 11. Open items for the user (decisions, not experiments)

1. TabPFN-3 license acceptance → whether to run shortlist #5 (update the stale
   "10-class" rationale in `HANDOFF.md:87-88` when reporting this).
2. Whether a pip exception is ever acceptable (would unlock TabICL — nothing else in
   this research needs pip).

## 12. Could not verify against a primary source (explicit flags)

- **"OREMP / FIMLR ordinal forests"** — no primary paper found under these names in my
  searches; do not cite them without a real source.
- **"lambdarank applied to ordinal targets beats cumulative classification"** — no
  primary source found; shortlist #4 is framed as an empirical diversity shot only.
- **INE note calibration claim (WGT)** — taken from the repo's quotation in
  `Notebook_01.ipynb`; the INE methodological PDF was not re-fetched (search surfaced
  only INS2019 publication pages).
- **Kaggle rule text on transductive/test-time use** — only the course-provided
  `docs/info_kaggle.md` was available; no separate official rules page was located.
- **Groves & Heeringa 2006** — abstract seen via JSTOR/RePEc records, full text not fetched.
- **"Koyejo global-optimality F1"** — a paper I expected to exist was **not** found in
  search; not cited. Macro-F1 optimization claims rest on Lipton et al. arXiv:1402.1892
  and sigmoidF1 arXiv:2108.10566 instead.
- **Cost estimates (CPU hours)** — derived from the NB18 timing yardstick (41.5 s XGB
  80/20), not measured end-to-end; treat as ±50%.

## 13. Log

| Date | Item | Result |
|---|---|---|
| 10 nov | Research pass (docs + primary web sources + env verification by execution); no repo file touched except this one | Document written |
| 10 nov | Env re-check: statsmodels 0.14.6 + `OrderedModel` OK; `LGBMRanker(label_gain=…)` OK; tabpfn 9.2.0 ships v3/v3.5 checkpoints; cleanlab 2.9.0 / imblearn 0.14.2 present | Facts for §1/§8 |
| 10 nov | `data/nb17_resultados.csv` P1 row read: pairwise-15 5-fold **0.4854, +0.0077 vs XGB-311, 4/5 — CONFIRMADO**; `data/nb18_resultados.csv` C3 **+0.0239** (barra +0.015) | Shortlist #1/#2 updated to current state |
| 10 nov | TabPFN blocker re-checked against PriorLabs docs: 160-class limit native in v3 → repo's "10-class" reason stale; license token still gates | §8 + open item 1 |
| 10 nov | **Shortlist executada — vereditos (ecrãs NB18 holdout + 5-fold NB17):** #1 **REJEITADO** (S1 sub7 0.4998 → +pairwise15 0.4948, −0.0050: o mecanismo não empilha na receita ordinal); #2 **CONFIRMADO mas insuficiente** (P2 C3 = 0.4886 ±0.0104, 5/5 vs base-TM, +0.0104 vs XGB-311 — 0.02 abaixo da sub7 0.5089); #3 **REJEITADO** (T15 bagging 0.4650, −0.0050); #4 **REJEITADO** (T16 lambdarank 0.1131/0.1005 — membro fraco: mapeio OOF dá 0.4133 no treino; blend w=0.5 com one-hot é degenerado por construção); #5 pendente de licença TabPFN | §9 shortlist fechada; teto 0.504–0.509 mantém-se |
| 12 nov | **#5 resolvido (licença aceite):** T17 TabPFN-3.5 0.4805 / T17b blend 0.4833 (**+0.0133, passa a barra**) com overlap diverso (ambos erram 39,1%); S2 receita+TabPFN 0.5037 = melhor número do bancada mas **+0.0040 vs S1 < barra +0.010 → arquivado** (§9.5 previa "±0.01 desconhecido" — medido +0.004 na receita, positivo mas abaixo do ruído) | shortlist #5: ecrã cumprido; pendente só 5-fold do T17b se promovido |
