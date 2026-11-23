# AGENTS.md

Academic ML repo (U. Minho DAA 2026/27). No README, no CI, no build — notebooks + CSVs.

## Start here
- Read `docs/contexto.md` first, then `docs/plano_modelacao.md` (frozen decisions).
- Metric is **Macro F1** over 16 `AGE_RANGE` classes, never accuracy.

## Environment
- Use `conda run -n daa` (pandas 3.0.6 + sklearn 1.9.1 + xgboost 3.4.1 +
  lightgbm 4.6.0 + catboost 1.2.10). No system `pip`; new deps via conda
  (permission-gated bash + manual review). NB01–07 stay on the old stack so a
  colleague without these packages can still run the pipeline; new packages
  go to NB08+ first.
- pandas 3.x: `select_dtypes(include='object')` warns/misbehaves on `str` dtype;
  pass `'str'` explicitly when touching NB01/NB02 code.
- sklearn 1.9: `LogisticRegression` no longer accepts `multi_class` (multinomial
  is default); XGB needs numpy matrix (rejects `[`, `]`, `<` in feature names).

## Pipeline order
- `notebooks/Notebook_01.ipynb` → `train_nb01.csv`/`test_nb01.csv` →
  `notebooks/Notebook_02.ipynb` → `train_nb02.csv`/`test_nb02.csv` →
  `notebooks/Notebook_03.ipynb` → `submissions/*.csv`.
- Derived CSVs are gitignored (`data/*.csv`); only `resumo_variaveis.csv` is committed.
  Notebooks must be re-runnable top-to-bottom to regenerate them.
- New models/features may be implemented; the user reviews diffs and outputs before commit.

## Kaggle discipline
- 3 valid submissions/day; final = 30% public + 70% hidden split.
- Never tune on the public leaderboard; CV (stratified 5-fold, seed 42) decides.
- Submissions are manual by default (3/day budget) — auto-submit only on explicit request.
- Format: header `RowId,AGE_RANGE` (RowId `0..3654` sequential), 3655 rows in original test
  order; see `data/example_submission_10.csv` — the `RowId` column is mandatory.

## Conventions
- Respond in English, concise. Reference code as `path:line`.
- Verify by executing; don't trust notebook prose over outputs.
- Shared repo (`davidalves2/DAA`): keep diffs to notebooks/docs minimal and intentional.
- Git use is allowed (`status`/`diff`/`log` for verification); leave `commit`/`push`
  for the user to review unless explicitly asked (shared repo).
- Generate submission CSVs locally; the user submits manually unless automation is explicitly requested.
