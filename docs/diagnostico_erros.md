# Error diagnosis (honest artifacts only)

Scope: ordered 16-class AGE_RANGE, macro F1, StratifiedKFold(5, seed 42).
Headline scores in Notebook_14 use relabelled labels (leak: NB16 honest-orig
flat med@0.4 = 0.5064 vs relabel-claimed ~0.54; `data/nb16_resultados.csv:22-27`).
All per-class numbers below are recomputed from the honest OOF
`data/nb11_oof311.npz` (`oof`, `proba` keys; plain XGB on current 311-feature
base, macro F1 = 0.4782). NB08 proba files are stale 339-base
(`docs/plano_modelacao.md:9`; new `train_nb02.csv` has 313 cols = 311
features, old file has 411). `data/nb14_relatorio_por_classe.csv` (macro
0.537, `:19`) is the inflated relabel report, shown only for contrast.

## (a) Ranked error pockets (honest XGB-311 OOF, 5662 errors / 10962 rows)

| Rank | Pocket | Size (support) | Honest per-class F1 | Share of total error |
|---|---|---|---|---|
| 1 | Classes 11-14 block (mid-high working ages) | 3513 rows | 0.527 / 0.369 / 0.273 / 0.348 (cl. 11/12/13/14) | 38.1% |
| 2 | Classes 4-7 block (young transitions) | 2817 rows | 0.304 / 0.379 / 0.381 / 0.421 (cl. 4/5/6/7) | 30.8% |
| 3 | Boundary 9<->10 (both classes individually good) | - | 0.668 / 0.645, yet 635 confusions | 11.2% of errors |
| 4 | Class 13 alone (worst single class) | 823 rows | 0.273, recall 0.256 | 10.8% |
| 5 | Class 1 (small, young) | 172 rows | 0.354 | 2.1% |

Worst adjacent pairs (bidirectional, share of 5662 errors): 9<->10: 635
(11.2%, 312+323); 11<->12: 538 (9.5%, 266+272); 12<->13: 464 (8.2%,
196+268). Top three pairs = 28.9% of all errors; class 12 sits in two of
them. Next: 7<->8 (7.1%), 6<->7 (7.0%), 5<->6 (6.2%). Off-by-one = 72.3%
honest (vs 76.0% relabel claim, `data/nb14_resultados.csv:4`); far errors
(|d|>=2) = 1569 (27.7%). Best classes need no help: 0 (F1 0.854, n=207),
9 (0.668), 8 (0.663), 10 (0.645), 2 (0.604).

## (b) Feature-use asymmetries (one experiment each, max 3)

1. Routing-gated features carry SHAP mass but no permutation value.
`PC1A`: SHAP rank 1 (0.657, `data/nb14_shap_global.csv:2`) vs perm rank 11
(0.017, `data/nb14_permutacao.csv:11`); `IN15`: SHAP 0.133 (`:15`) vs perm
0.001, rank ~93 (`data/nb14_permutacao.csv:93`); same pattern `HA1B`
(0.149/0.0087), `PC1C` (0.107/0.004). Routing block = 36.2% of |SHAP|
(`data/nb14_resultados.csv:21`) but SHAP-vs-permutation rank agreement is
only 0.468 (`data/nb14_resultados.csv:16`). Hypothesis: these features
identify answerable-vs-skipped subpopulations (good classes 0/8/9/10) but
do not separate adjacent ages inside the 11-14 and 4-7 blocks. Test: drop
the six tagged `routing_55+` features, retrain 5-fold, and report delta
overall AND per-block F1 (expect: flat overall, worse class 0, unchanged
11-14; any 11-14 gain refutes it).
2. Household/work-status signal saturates before the worst classes.
`HH_INACT` (perm rank 1, 0.068, `data/nb14_permutacao.csv:2`),
`MAINSTAT_Aluno` (perm rank 2, 0.058, `:3`), `HATLEVEL` (perm rank 3, 0.054,
`:4`) dominate both SHAP and permutation globally, and errors>=2 rows show
depleted `HH_ACT`/`HH_INACT`/`MAINSTAT_Aluno` plus raised `N_DIFICULDADES`
and `MAINSTAT_Reformado` (`data/nb14_fatia_erro.csv:4-8,16`, all |z|<=0.28).
Hypothesis: these separate students/retirees (classes 0-2, 15) and wash out
inside 4-7 and 11-14, where finer occupation/health gradients should take
over. Test: add `MAINSTAT x HATLEVEL` and `HH_ACT/HH_INACT x N_DIFICULDADES`
interactions (or a 4-7/11-14-only specialist HGB), 5-fold; success = block
F1 gain on 4-7 or 11-14 with no class-0/2 regression.
3. The 9<->10 boundary is symmetric and high-mass despite good per-class F1.
312 cases of 9->10 and 323 of 10->9 (largest single off-diagonal pair) while
both classes hold F1 ~0.65. Hypothesis: the default argmax threshold sits
inside an overlapping score region that an ordinal rule resolves better
than class weights. Test: apply only the Frank & Hall median/expected-value
decision rule on top of frozen XGB-311 OOF probabilities and measure the
9<->10 confusion change (no retrain); expect the pair to shrink while 8<->9
stays at zero (see non-findings).

## (c) Mechanism coverage (confirmed deltas)

- Ordinality (Frank & Hall): E1 median 0.4979, +0.0243, 5/5 folds
(`data/nb08_resultados.csv:8`); E8 F&H+coarse-fine 0.4982 is a tie
(`:6`), so F&H alone covers the ordinal mechanism. Honest-base echo: NB12
E25 FH 5-seed 0.5195-0.5215 and NB18 S1 F&H+median 0.4998 (+0.0298 vs
single-split BASE, `data/nb18_resultados.csv:2`).
- Adjacent-pair specialists (E2 pairwise +0.0093 stale, `:16`; honest P1
+0.0077, P2 +0.0109, P3 +0.0098, P4 +0.0058, `data/nb17_resultados.csv:7-10`)
cover the same boundaries F&H already fixes: NB18 S1b (S1+pairwise) loses
0.0050 vs S1, C3/C4 stay below S1 (`data/nb18_resultados.csv:3-5`). Pairwise
is redundant once F&H+median is applied; keep at most one gated variant.
- Noise/quarantine: NB11 drop-worst-15% claims +0.0316
(`data/nb11_resultados.csv:2`) but filters on same-model self-confidence
(leaky); the honest quarantine repeat (NB18 T6) gives +0.0069
(`data/nb18_resultados.csv:10`). Treat NB11 Q2 as upper bound, T6 as honest
value. Suspect pocket (`data/nb11_suspects.csv`, 500 rows; `n_issues`=4505,
rate 0.41, `data/nb11_noise.json:2-3`) is 59.8% far-error vs 27.7% overall
and over-represents true classes 15/3/1/4/14 - label-noise candidates, not
a learnable boundary.

## (d) Explicit non-findings

- No tight low-confidence pocket: bottom 5% by OOF confidence holds 7.6% of
errors (err rate 0.78 vs 0.52 overall); bottom 20% holds 27.7%. Errors are
diffuse, not a removable cluster.
- Confident errors exist: top-5% confidence rows still err at 9.5% (52
rows) - calibration alone (ECE 0.0953, `data/nb14_resultados.csv:6`) will
not clear them.
- Boundary 8<->9 has zero confusion in both directions on honest OOF -
nothing to fix there; do not spend boundary budget on it.
- `nb14_fatia_erro.csv` standardized shifts are all weak (|z| <= 0.28,
`IN24` -0.28 strongest, `:2`); no single raw feature separates far errors.
- Relabel-curve and family-ensemble gains (NB13 E26 0.5364, NB15 B1 0.546)
vanish under honest labels (NB16 honest rel-B1 0.4981 vs claimed 0.5460,
inflation +0.048, `data/nb16_resultados.csv:27`) - do not use as evidence.
