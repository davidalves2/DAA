# Breaking the 0.50 ceiling — consolidated plan (models + data)

> Status: active. Sources: NB08 diagnosis (E12 metastack 0.5037 frozen, sub08 pending),
> official INS 2019 CAPI questionnaire (`docs/INS2019_questionario_CAPI.pdf` + extracted
> `docs/INS2019_questionario.txt`), web research (CORAL/mord, threshold tuning, TabPFN limits).
> Protocol for everything below: same `StratifiedKFold(5, shuffle, seed=42)` on
> `train_nb02.csv`, Macro-F1, delta vs frozen base, 5/5 folds to adopt.

## 0. Diagnosis (numbers, all verified by execution)

- 76% of errors off-by-one; 20.6% in 4 cells (9↔10, 11↔12). Worst classes: 13, 4, 6, 5, 12.
- 42.9% of errors: all 4 bases agree and are wrong. 66.7%: no base is right.
  → Diversity of views is exhausted; tuning combinations is done (E8/E12b/E15 all ties).
- Target-encoding/regression-to-mean collapses (−0.16): anything that predicts the group
  mean kills the extreme classes macro-F1 weights equally. Masks cost −0.01 even wide.
- ~19% of LGBM gain is questionnaire routing, not biology (declare in report).
- Realistic upside: +0.005 to +0.02 total. 0.55 needs finer age signal, not more stacking.

## 1. Model-side tracks (no retraining where noted)

| # | Experiment | Expected | Cost | State |
|---|---|---|---|---|
| M1 | Tune the 15 F&H cumulative thresholds (coordinate ascent, cross-fitted, macro-F1, monotonicity constraint + isotonic fix) | +0.002/+0.008 | 0.5 d | **REJECTED 2 nov** — honest nested (tune-4/eval-1) on E25 OOF: 0.5169 vs 0.5227 ref (−0.0058, 1/5) |
| M2 | Proper OvR per-class thresholds, nested (`TunedThresholdClassifierCV`-style) | +0.002/+0.008 | 1 d | **REJECTED 2 nov** — nested per-class logit offsets: −0.0072 (2/5); prior-α −0.0018; median-t re-tune +0.0010 (0.40 already right) |
| M3 | Monotone constraints on `N_DOENCAS` (+0.53), `N_DIFICULDADES` (+0.58 Spearman, verified) in XGB/LGBM retrain | +0.001/+0.005 | 0.5 d | **REJECTED 2 nov** — E26 5-seed: pair −0.0011 (2/5), ext (+IDADE_MINIMA/FAIXA_MINIMA+1, N_NAO_APLICAVEL−1) +0.0003 (3/5); xgboost 3.4 wants tuple/dict, not list |
| M4 | Seed-bagged metastack (whole E12 pipeline over seeds) | +0/+0.005 | 0.5 d | TODO, finalizer |
| M5 | Pair-flip rules on the 4 biggest cells, cross-fitted on metastack | +0/+0.004 | 0.5 d | **REJECTED 2 nov** — coordinate ascent selected 0 flips in all 5 folds; honest delta −0.0003 |
| M6 | CORAL-style rank-consistent head | — | — | PARKED: `mord` stale (2023, sklearn-1.2 era, pip-only). M1 covers the mechanism |
| M7 | LDL / custom XGB losses | — | — | REJECTED: strong in imaging, weak in tabular GBDT (web research) |

## 2. Feature tracks (`Notebook_09` — audit before pipeline)

Rule: build in NB09, full CV loop, adopt only on 5/5-fold delta. On adopt, consolidate
into NB02 (open debt §7.5).

| # | Feature | Construction | Expected |
|---|---|---|---|
| F1 | `N_SEM_RESPOSTA` | Count of `{'Sem resposta','NR'}` per row from `train_nb01.csv` (mirror of `N_NAO_APLICAVEL`; closes §7.3) | ~0, hygiene |
| F2 | Per-block N/A counts | Group the 11 `*_Não aplicável` dummies by domain (work/repro/other) via `dicionario_variaveis.csv`; compresses the 17 routing flags (dilution lesson) | 0/+0.004 |
| F3 | Tobacco constructs | `SMOKER_STATUS` (from `SK2_ESTADO`/`IN45_ESTADO`: never/former/current), intensity (`SK2`, sentinel −1→0+flag), onset (`IN45`); audit per-class means first | +0.002/+0.006 |
| F4 | `IDADE_MAXIMA` + `O_ANSWERED` | O-block answered (`IN24/25/26_ESTADO==Valor`) ⇒ max 55 years (top of class 8 / age-55 edge of class 9); official gate Q-O (see §4). Feature, never veto | +0.003/+0.012 |
| F5 | Block ablation | Drop each: MENTAL(MH*), PAIN(PN*), SUPPORT(SS*), CARE_USE(AM/HO/MD/IC/PA/NT*), FOOD(DH*), EXERCISE(PE*), ALCOHOL(AL*) → keep/discard | finds dead weight |
| F6 | IN15 audit | 4.56% of LGBM gain, values {−2..3} unexplained; per-class curves + SHAP dependence | understanding + report |

## 3. External data — verdict

No row-joinable keys exist (no birth year/date, HHID unique per row — verified), so
PORDATA/INE aggregates add nothing beyond train priors. The one external source that
matters is the survey instrument itself (§4). PORDATA cohort-education stats: report
narrative only.

## 4. Official routing gates (INS 2019 CAPI, verified in extracted text)

| Block | Official rule | Modeling use |
|---|---|---|
| G personal care (PC1A–E) | `Idade ≥ 55 → G1.1; ELSE → I1` (line 1059) | answered G ⇒ class ≥ 9 (55–59; validates FAIXA_MINIMA) |
| H household tasks | downstream of G | 55+ in practice |
| O reproductive | `≤55 → O1/O2; >55 → P1.1` (lines 1907–9) | answered O ⇒ class ≤ 11, ~female (F4) |
| M medicines | men or women>55 → variant A; women≤55 → B (lines 1730–5) | variant ⇒ sex×age bound |
| N preventive | men → P1.1; women → N8.1 mammography (lines 1847–8) | sex routing |
| E absence, L2/L11 occupational | `A0800==1` (employed, MAINSTAT) | answered ⇒ employed |
| A19 pregnancy | women ≤ 55 (line 618) | bound |
| Universe | age ≥ 15 (line 109) | class-0 floor |

## 5. Packages & skills

- Allowed: conda only (`conda install -n daa -c conda-forge … --freeze-installed`), NB08+ first,
  colleague compat keeps NB01–07 on old stack. `shap` 0.52.0 installed (report + NB09 audit).
- TabPFN: REJECTED (10-class hard limit + PriorLabs token gate). Replacement route in
  `Notebook_10`: BalancedRandomForest (imbalanced-learn, per-tree balancing) + LightGBM-DART
  as extra E12 meta families (both native 16-class, CPU-cheap). EBM/InterpretML parked
  (no conda package, pip-only). Rapid: BRF 0.4053 (5s/fold), DART 0.4579 ≈ XGB-rapid — full mode decides.
- Skills (global): scikit-learn, shap, pandas-pro, exploratory-data-analysis,
  jupyter-notebook (patched: conda, no uv), pdf, web-access (`web_search` stack).

## 6. Log

| Date | Item | Result |
|---|---|---|
| 28 out | Questionnaire fetched+mined; gates confirm mapa_routing, add upper bounds | §4 |
| 28 out | HHID joint inference | REJECTED — HHID unique/row, zero overlap train/test |
| 28 out | NB09 full: T7a −28 cols 0.4791 +0.0066 5/5 ADOPTED; T7b +adds 0.4745 2/5 rejected; T4 +0.0019 4/5 and T1 +0.0028 3/5 shelved (do not survive the drop); FOOD stays |
| 28 out | NB02 §4.1 consolidation + 311 regen + XGB-confirm 0.4777 ± 0.0088; NB10 full: DART 0.4782 ≈ XGB, BRF 0.4302, meta96 0.5035 ties E12 → no augment; next: meta rebuild on 311 |
| 28 out | NB11 data-quality: Q0 rules clean (0 routing violations, train=test rates); Q1 cleanlab 41% upper bound; Q2 quarantine ADOPTED — drop-worst 5/10/15% all 5/5 (0.4939/0.5021/0.5093); audit rows incl. employed-master 85+; next: rebuild bases+meta on cleaned train, per-class caps, relabel (RUN_EXTRA) |
| 2 nov | Colleague main merged + verified conflict-free (staged --no-commit); NB02 §4.1 re-applied, 311 regen re-confirmed; NB12 E23–E25 compounding (§7) |
| 2 nov | sub09 (`sub09_fh5_mediana_quar10.csv`) generated locally + format-validated (header, 3655 rows, ints 0–15) | PENDING manual submit; expect ~0.51–0.52 public on historic gaps |
| — | sub08 manual submission | SUPERSEDED by sub09 (never submitted) |
| 2 nov | Merge origin/main verified conflict-free (zero unmerged, no markers, main static); staged --no-commit for review; NB02 §4.1 + 311 regen re-confirmed post-merge |
| 2 nov | **E26 relabel beats drop** (NB13): worst-15% labels ← cross-fit OOF, nothing dropped; F&H-5seed + argmax + FAIXA_MINIMA = **0.5364 mean / 0.5365 agg, 5/5 vs E25**. Dose 5/10/15/20% = 0.5157/0.5262/0.5353/0.5379 (1 seed); random-relabel control 0.5131 → ranking matters; per-class cap 0.5091 and drop-≥2 0.5008 both worse than uncapped drop. `sub10_fh5_rel15.csv` generated, manual submit pending |
| 2 nov | M1/M2/M5 + prior-α rejected by honest nested eval (§1); median-threshold re-tune ≈ 0 |
| 2 nov | Tier 1.1 symbolic search (170 pair-formulas {max,min,\|diff\|} over age/count pool): fast screen → 10 shortlist + top3/top4 — nothing 5/5 on base (best `min(IN15,HHNBPERS_COD)` +0.0067 **4/5**); E26 confirm single −0.0006 (2/5), top3 −0.0024 (0/5) → **CLOSED** (signal absorbed by ordinal+relabel recipe) |
| 2 nov | Tier sweep, all honest-nested/CV: **self-training sim REJECTED** (thr0.8 −0.0001 2/5, thr0.6 +0.0006 3/5, pseudo-acc 65–80% — nothing to gain); **M3 REJECTED** (§1); **RPS custom loss** (obj validated: grad finite-diff checked, train RPS 2.31→0.12) +0.0016 **3/5** → not adopted (confirms M7); **DES/instance gate REJECTED** — E26 beats XGB/reg in every confidence decile (max gap −0.007), cross-fitted logistic gate −0.0018 2/5; oracle headroom +0.158 exists but is not predictable from confidence |
| 2 nov | Relabel dose-20% ×5seed: **0.5407 ± 0.0049** vs 15% 0.5364 → +0.0043 but **3/5** (folds 0–1 slightly lower, 2–4 higher) → tendency, NOT adopted; dose search CLOSED at 15%; sub10 unchanged |
| 3 nov | **NB14 explicabilidade executado** (SHAP0.52 + sklearn + pandas, held-out, additivity ✓ ~1e-5): **relabel no multiclasse =0.5341 ±0.0065 (+0.0564 vs0.4777)** — F&H vantagem sobre XGB plano cai de +0.0245 → +0.0023 sob relabel; estabilidade SHAP orig-vs-relabel ρ=0.977 (mesmo sinal, rótulos limpos explicam o ganho); quota routing36.2% do \|SHAP\|; surrogato depth-4 fidelidade0.4848; ρ(SHAP,permutação)=0.468; ECE0.0953; correção +10/−0. Figuras `docs/fig_nb14_*.png` + `data/nb14_*.csv` |
| 3 nov | **E27 ADOPTADO (NB15):** média igual de4 médias de família sobre relabel-15 {F&H×5, LGBM×4, HGB×2, multiclasse×5} = **0.5460 ±0.0079 / agg 0.5463, 5/5 vs E26 (+0.0096)**. Componentes: mc5 0.5376 (3/5 — empate, opção eficiência 5 modelos vs75), LGBM×4 0.5417 (4/5), HGB×2 0.5355, flat-11 sub7 0.5409 (3/5 → receita do colega só passa o gate com o membro mc5). Params do colega: NB03 `MODELOS_ORDINAIS`. `sub11_4familias_rel15.csv` gerado+validado (91.8% igual a sub10) — **submeter sub11, não sub10**. Caches `data/nb15_*`, resultados `data/nb15_resultados.csv` |
| 4 nov | **🚨 sub11 = 0.48 público (gap −0.066) → RETRAÇÃO dos CVs E24–E27/mc5 (NB16):** canal do relabel medido = **32.4%** dos rótulos sensíveis ao fold de validação (inner-OOF3) → inflação **+0.036…+0.061** (E27: 0.5460 → **0.4981 honesto**; gap→público −0.018 = histórico ✓). **relabel honestamente neutro/negativo** (pares todos ≤0) → via de rótulos morta; E24/E25/E26/mc5 eram artefacto. **Teto honesto 0.504–0.509 = sub7 do colega** (orig-LGB med@0.45 0.5083, orig-flat med@0.40 0.5064; deltas ≤±0.004 = ruído). Teste sante (4 famílias OOF≈TEST) e forensics sub11-vs-sub07 limpos. Registo: `Notebook_16.ipynb`, caches `nb16_*`, `nb16_resultados.csv` |
| 2 nov | TabPFN recheck: core 9.2.0 installed (CPU torch) but `.fit()` blocked by **TABPFN_TOKEN** license gate (ux.priorlabs.ai); `many_class` extension (16>10 classes) is pip/git-only (conda rule); CPU cap 5000 samples (train 10962) → **blocked on user decision** |
| 10 nov | **Shortlist da pesquisa (`docs/pesquisa_quebra_platamar.md`) executada (NB18 ecrãs + NB17 5-fold):** #1 S1 receita sub7 no holdout = **0.4998** (reproduz a sub7) mas +pairwise15 = 0.4948 → **−0.0050, REJEITADO no ecrã** (pairwise-15 não empilha na receita ordinal; ajuda só ao XGB plano); #2 C3 confirmado honesto (P2): **0.4886 ±0.0104, 5/5 vs base-TM 0.4811, +0.0104 vs XGB-311, 4/5 vs P1** — mas fica 0.02 abaixo da sub7 (0.5089), não a substitui; #3 T15 bagging 10 seeds × 80% linhas = 0.4650 (**−0.0050, REJEITADO**); #4 T16 lambdarank 0.1131 (v1, escalas) / 0.1005 (v2, percentis) — diag: corr 0.66 mas mapeio OOF só F1 0.4133 no treino + blend w=0.5 degenerado → **REJEITADO**; #5 TabPFN-3 pendente de licença. Teto honesto mantém-se **0.504–0.509**. Bug latente: ramos de load dos P1–P4 (`int(_z['nov'])`) rebentam com numpy 2.5.3 |
| 12 nov | **TabPFN desbloqueado (licença aceite) + C6 + S2 no NB18:** T17 TabPFN-3.5 sozinho **0.4805** (+0.0105) e T17b blend 0.5 XGB+TabPFN **0.4833 (+0.0133 ≥ barra → promovível a 5-fold)**; overlap: ambos erram só 39,1%, concordam 59,6% = diversidade real (cap CPU 5000 ignorado c/ `ignore_pretraining_limits=True`, regra dos 100% das linhas do NB18; custo 1210 s). **C6 kitchen sink (do utilizador, barra +0.015 pré-registada): 0.4787 (+0.0087) — perde para o single T10 (0.4862) → CLÁUSULA DE MORTE DISPARA: combinações fechadas em definitivo.** Diagnóstico: toda a combo com quarentena falha (C5 0.4791, C6 0.4787 vs C3/C4 0.4939/0.4920 sem ela) — a quarentena (T6) contamina. **S2 receita sub7 + TabPFN (blend 0.5 pré-declarado): 0.5037 = melhor número do bancada, mas +0.0040 vs S1 < barra +0.010 → arquivado** (ruído; sem exceções). Cache `data/nb18_tabpfn_proba.npz`; células T17/S2 em NB18 |

## 7. 0.5230 decomposition (E25, `Notebook_12.ipynb`)

Challenger = colleague's F&H-5seed recipe × 311 base × 10% quarantine + median@0.40 +
`max(pred, FAIXA_MINIMA)`. Aggregate CV (correction is aggregate-only; mean-F1 0.5215).

| Rung | Score | Δ | Origin |
|---|---|---|---|
| Colleague F&H-1seed, 339 | 0.5008 | — | registo §19 (their base+recipe) |
| E23: same recipe, 311 | 0.5022 | +0.0014 | NB09 drop28 |
| E24: + quarantine-10 (train only) | 0.5168 | +0.0146, 5/5 | NB11 Q2 |
| E25: 1 → 5 seeds (mean probas) | 0.5195 | +0.0027 | their recipe |
| E25 + median fallback (conf<0.40) | 0.5215 | +0.0020 | their §20a rule |
| E25 + FAIXA_MINIMA correction | **0.5230** | +0.0015 | their feature+rule |

Reg variant scored 0.5209 (within noise of median; kept as backup). LGBM×4/HGB×2 ordinal
families (their sub7 recipe, +0.003 historically) not yet run on cleaned base — open.
