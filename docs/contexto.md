# Contexto do projeto — DAA 2026/27 (ler no início de cada sessão)

## O que é
Trabalho prático de grupo de Dados e Aprendizagem Automática (U. Minho): prever
`AGE_RANGE` (16 faixas etárias, 0–15) a partir do Inquérito Nacional de Saúde 2019.
Duas tarefas: análise/validação + competição Kaggle (métrica **Macro F1**, não accuracy).
Score final: 30% público + 70% split privado. Deadline competição: 1 jan 2027.
Checkpoint: 30 nov / 3 dez 2026. Relatório (20 pp) + apresentação: jan 2027.

## Estado dos dados (`data/`)
- `dataset_train.csv` (10962×237, 236 features + alvo), `dataset_test_X.csv` (3655×236),
  `example_submission_10.csv`.
- Derivados gerados pelos notebooks (gitignored via `data/*.csv`): `train_nb01.csv`
  (10962×262, 260 features + alvo), `test_nb01.csv`, `train_nb02.csv` (10962×313,
  PID + 311 features + alvo), `test_nb02.csv`, `train_nb02_339.csv` (10962×341,
  PID + 339 features + alvo) e `test_nb02_339.csv`.
- `resumo_variaveis.csv`: tipificação das 236 variáveis (commitado).
- Alvo desequilibrado 6.17x: menor classe `1` (18-19, 172 casos), maior `10` (60-64, 1062).

## Estado dos notebooks (`notebooks/`)
- **NB01** (EDA/limpeza): sólido. Remove `WGT` (leakage), trata "Sem resposta"/"Não
  aplicável", `_ESTADO`, HHMM→min, `PA1`, `IN16`. Top sinais: `MAINSTAT`, `MARSTALEGAL`,
  `HATLEVEL`, `IN24`, `HH_ACT`. Atenção: muito sinal é *routing* do questionário.
- **NB02** (encoding+seleção): completo. 409 → 333 → 337 → 339; o §4.1 remove 28
  colunas ruidosas e escreve a base 311 usada pelas experiências. A base 339 da
  entrega regenera-se com o NB02 de `4ae5b6c` (ver `docs/runbook.md`).
- **NB03** (modelo final): Gradient Boosting tunado, XGBoost, ordinal de Frank & Hall
  e a sub07, com 0,5089 de CV observada. Lê a base 339, recusa outra contagem e só
  substitui a entrega quando as previsões coincidem.
- **NB06** (anomalias+retrato+KMeans+PL9): lê a OOF da sub07 quando existe; cai para
  `combined_oof.npz` (0,4704) ou `nb04_oof.npz` (0,4662) num clone sem caches.
- **NB04** (interpretabilidade): congela a OOF da sub06 (0,5051) na base 339. Não
  re-correr na base 311: usa `PE7`, que o §4.1 removeu.
- **NB16** (prova do relabel) e **NB18** (bancada de triagem): ver `docs/runbook.md`.

## Decisões congeladas
Ver `docs/plano_modelacao.md` (fonte de verdade). Resumo: só sklearn no NB04
(`HistGradientBoosting`+`RandomForest`); comparar encodings e formulações do alvo em
vez de apostar; CV 5-fold + holdout decide, Kaggle só confirma; submissões **manuais**;
features em blocos F0–F4 com ablação.

## Ambiente
- Repo git: `github.com:davidalves2/DAA` (partilhado com o colega).
- conda env `daa`: pandas 3.0.6, sklearn 1.9.1, xgboost 3.4.1, lightgbm 4.6.0,
  catboost 1.2.10, shap 0.52 e tabpfn 9.2.0. **Sem** `pip` no sistema; usar
  `conda run -n daa`.
- Os NB01–07 mantêm a stack antiga para o colega que não tem os pacotes novos;
  pacotes novos entram primeiro no NB08+.
- pandas 3.x: warnings `select_dtypes(object)` vs `str` nos notebooks — não partir
  NB01/02.

## Regras de trabalho
- Não implementar modelos novos sem pedido explícito.
- 3 submissões/dia: nunca tunar pelo leaderboard público.
- Responder em inglês no chat; os documentos do relatório ficam em PT-PT.
- Referenciar ficheiros como `path:linha`.
- Verificar outputs executando, não por memória. Não criar ficheiros salvo necessário.
