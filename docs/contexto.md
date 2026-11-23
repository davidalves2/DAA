# Contexto do projeto — DAA 2026/27 (ler no início de cada sessão)

## O que é
Trabalho prático de grupo de Dados e Aprendizagem Automática (U. Minho): prever
`AGE_RANGE` (16 faixas etárias, 0–15) a partir do Inquérito Nacional de Saúde 2019.
Duas tarefas: análise/validação + competição Kaggle (métrica **Macro F1**, não accuracy).
Score final: 30% público + 70% split privado. Deadline competição: 1 jan 2027.
Checkpoint: 30 nov / 3 dez 2026. Relatório (20 pp) + apresentação: jan 2027.

## Estado dos dados (`data/`)
- `dataset_train.csv` (10962×237), `dataset_test_X.csv` (3655×236), `example_submission_10.csv`.
- Derivados gerados pelos notebooks (gitignored via `data/*.csv`): `train_nb01.csv`
  (10962×268), `test_nb01.csv`, `train_nb02.csv` (10962×410), `test_nb02.csv`.
- `resumo_variaveis.csv`: tipificação das 236 variáveis (commitado).
- Alvo desequilibrado 6.17x: menor classe `1` (18-19, 172 casos), maior `10` (60-64, 1062).

## Estado dos notebooks (`notebooks/`)
- **NB01** (EDA/limpeza): sólido. Remove `WGT` (leakage), trata "Sem resposta"/"Não
  aplicável", `_ESTADO`, HHMM→min, `PA1`, `IN16`. Top sinais: `MAINSTAT`, `MARSTALEGAL`,
  `HATLEVEL`, `IN24`, `HH_ACT`. Atenção: muito sinal é *routing* do questionário.
- **NB02** (encoding): incompleto — só codifica (205 ordinais + 22 nominais→171 one-hot
  = 409 features). Sem seleção nem features novas.
- **NB03** (baseline): só `DecisionTree(max_depth=8)`, Macro-F1 CV **0.407** →
  `submissions/sub01_arvore_referencia.csv` (~40% público). Sem confusão nem F1/classe.

## Decisões congeladas
Ver `docs/plano_modelacao.md` (fonte de verdade). Resumo: só sklearn no NB04
(`HistGradientBoosting`+`RandomForest`); comparar encodings e formulações do alvo em
vez de apostar; CV 5-fold + holdout decide, Kaggle só confirma; submissões **manuais**;
features em blocos F0–F4 com ablação.

## Ambiente
- Repo git: `github.com:davidalves2/DAA` (partilhado com o colega).
- conda env `daa`: pandas 3.0.6 + sklearn 1.9.1. **Sem** lightgbm/catboost/xgboost/optuna.
- Sem `pip` no sistema; usar `conda run -n daa`.
- pandas 3.x: warnings `select_dtypes(object)` vs `str` nos notebooks — não partir NB01/02.

## Regras de trabalho
- Não implementar modelos novos sem pedido explícito (pedido atual: só plano/docs).
- 3 submissões/dia: nunca tunar pelo leaderboard público.
- Responder em português (PT-PT), conciso. Referenciar ficheiros como `path:linha`.
- Verificar outputs executando, não por memória. Não criar ficheiros salvo necessário.
