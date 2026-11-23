# Mapa das aulas práticas PL1–PL8 (DAA 2026/27) — leitura integral dos ficheiros

- **Data:** 11 nov 2026
- **Âmbito:** todas as pastas práticas `PL1`…`PL8` em `/home/rui/Documents/MIA-1/DAA/`
  (incl. subpastas `PL3/datasets3`, `PL4/(4)DAA_datasets`, `PL8/daa_pl8_notebooks` e os
  `.zip`), servindo o assignment `AGE_RANGE` (`tp/DAA`). Fora de âmbito: pasta `T/`
  (teóricas), `tp/DAA` (só contexto de leitura).
- **Método:** extração de texto real, não inferência de nomes — PDFs (16) extraídos com
  `pypdf` e lidos na íntegra; notebooks (13) convertidos em dump célula-a-célula
  (markdown + código + outputs executados) e lidos todos; CSVs (19) carregados com pandas
  (shape + colunas); `.zip` (3) listados; `.tex`/`.log` (2) lidos; PNGs (5) abertos e
  vistos. Nenhum ficheiro foi alterado; nenhum experimento de treino foi corrido.
- **Ficheiros inspeccionados: 58 de 58** (16 PDF + 13 ipynb + 19 CSV + 3 zip + 5 PNG +
  1 tex + 1 log).

Contexto do assignment lido antes do mapeamento: `docs/contexto.md`,
`docs/plano_acima05.md` (§0 diagnóstico + §6 log), `registo_experiencias.md`,
`docs/pesquisa_quebra_platamar.md` (§10 "não incomodar").

---

## 1. MAPA — inventário completo

### PL1 — Intro to Data Science & Python/Scikit-learn

| Ficheiro | Tipo | Conteúdo real (com localização) |
|---|---|---|
| `PL1/DAA_Ficha_1(1).pdf` | exercício (1 p.) | T1–T4: instalar Miniconda, criar env Python 3.14 com numpy/pandas/matplotlib/seaborn/scikit-learn/jupyterlab, validar instalação, ler docs. Sem modelação. |
| `PL1/DAA_PL_1(1).pdf` | slides (45 pp.) | Tipos de dados **numérico/categórico/ordinal** (pp.4–8, quiz p.8); média/mediana/moda (pp.10–13, mediana robusta a outliers p.12); variância/desvio-padrão e outliers a >1σ (pp.16–17); PDF/PMF (pp.19–20); percentis e **IQR** (pp.22–23); covariância/correlação + "correlação ≠ causalidade" (pp.25–28); conda/venv/WSL e setup do curso (pp.30–41); hands-on T1–T4 (pp.42–45). |

**Resumo PL1:** só fundamentos (estatística descritiva, tipos de dados, ambiente).
Nada de treino/validação. Dataset nenhum.

### PL2 — Quality Metrics and Model Validation

| Ficheiro | Tipo | Conteúdo real |
|---|---|---|
| `PL2/DAA_Ficha_2(1).pdf` | exercício (1 p.) | Churn telecom: T1 fazer merge de `CallsData`+`ContractData` por `Area Code`/`Phone`; T2 `DecisionTreeClassifier` avaliada por **accuracy e `f1_macro` com 10-fold CV**; T3 matrizes de confusão e análise crítica; T4 variar `criterion`/`max_depth`. |
| `PL2/daa_pl_2(1).ipynb` | nb (44 cél., 23 c/ outputs) | Titanic: hold-out 75/25 `random_state=2021` (cél.8); drop de categóricas (cél.13); métricas cél.18–25 (confusão, accuracy 0.632, precision 0.536, recall 0.511, ROC-AUC 0.611, **F1 0.523**, Fβ=0.5 0.531); "DT regressor" sobre `Fare` (cél.27–35 — cél.30 dá erro `Unknown label type: continuous` por usar o classificador); validação cél.38–43: `cross_val_score` 10-fold (0.63 ±0.08) e **`KFold` manual com matriz de confusão por fold** (cél.43). |
| `PL2/DAA_PL_2(1).pdf` | slides (49 pp.) | DT classificador/regressor (pp.4–13, 28–31); métricas: confusão/accuracy/precision/recall/ROC-AUC/**F1 e Fβ** (pp.19–27), MAE/MSE/RMSE (pp.33–38); validação: hold-out vs CV vs k-fold vs LOO, "regra k=10" (pp.40–45), API `StratifiedKFold`/`GridSearchCV` (p.46), `cross_val_score` e KFold manual (pp.47–48). |
| `PL2/CallsData.csv` | dados | 3333×16 (minutos/chamadas por período, `CustServ Calls`, …) |
| `PL2/ContractData.csv` | dados | 3333×7 (`Churn`, `Int'l Plan`, `VMail Plan`, `State`, `Area Code`, `Phone`) |
| `PL2/titanic_dataset.csv` | dados | 891×12 Titanic padrão |

**Resumo PL2:** origem do nosso protocolo — **macro-F1 + k-fold** são o T2 da ficha;
ensina hold-out vs CV e a crítica de confusão. A ficha de churn (merge + árvore) não tem
notebook associado: o notebook cobre só Titanic.

### PL3 — Data Exploration and Preparation

| Ficheiro | Tipo | Conteúdo real |
|---|---|---|
| `PL3/DAA_Ficha_3.pdf` | exercício (2 pp.) | I) `sentiment_analysis.csv`: T1 tendência central/dispersão/corr.; T2 plots; **T3 tratamentos:** (a) missing, (b) duplicados, (c) **`age_binned` = 3 bins de frequência igual sobre `age`**, (d) extrair ano/mês/dia-da-semana do `birthday`, (e) **remover utilizadores com `WebActivity`<1h E idade>70**, (f) imputar **moda** condicionada em `Sentiment Analysis`, (g) imputar **média** por `Sentiment Analysis`×`MaritalStatus`. II) EPL 2017/18 (T5–T6 perguntas de negócio). |
| `PL3/daa_pl_3.ipynb` | nb (50 cél., 33 c/ outputs) | Wine: inspeção (cél.5–15: `head/shape/dtypes/info/describe/isna`); meta-dados da variável (cél.31); preparação cél.17–21: `drop_duplicates` (cél.19), rename (cél.20), **remoção de linhas por regra `Ash<2 | Alcalinity<=15` → 178→152 linhas (cél.21)**; univariada cél.22–31 (`value_counts`, hist, skew 1.098, kurt 2.105); multivariada cél.32–47: `pairplot`, `groupby(Class).mean()`, **`KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')` sobre `Alcohol` (cél.38)**, boxplots (cél.41–44), scatter, heatmap de correlação (cél.47). |
| `PL3/DAA_PL_3.pdf` | slides (40 pp.) | Pipeline ML de 8 passos c/ train/val/test (p.4); API de preparação pandas (`drop/fillna/interpolate/groupby`, p.16) e sklearn (`MinMaxScaler/StandardScaler/KBinsDiscretizer/LabelEncoder/feature_selection`, p.17); **binning como smoothing** (p.30); univariada/multivariada (pp.20–38). |
| `PL3/datasets3/wine.csv` | dados | 178×14 (target `Class` one/two/three) |
| `PL3/datasets3/sentiment_analysis.csv` | dados | 15167×16 (`CustomerKey`, `WebActivity`, `Sentiment Analysis`, `SentimentRating`, `MaritalStatus`, `Gender`, `EstimatedYearlyIncome`, `NumberOfContracts`, `Age`, `Target`, `Available401K`, `CustomerValueSegment`, …) — usado só na ficha, **não no notebook** |
| `PL3/datasets3/EPL_Player_Stats_2017_2018.csv` | dados | 515×26 (ficha II) |
| `PL3/datasets3.zip` | zip | = os 3 CSVs anteriores |

**Resumo PL3:** EDA + tratamentos. Os passos mais "pesados" para nós: binning de
frequência igual da idade (ficha T3c/cél.38), imputação condicionada por grupo
(ficha T3f/T3g) e **remoção de linhas por regra que envolve a idade** (ficha T3e).

### PL4 — Linear and Logistic Regression

| Ficheiro | Tipo | Conteúdo real |
|---|---|---|
| `PL4/DAA_Ficha_4.pdf` | exercício (2 pp.) | I) `EcommerceCustomers`: T1–T7 regressão linear (`train_test_split` 0.3, coeficientes, MAE/MSE/RMSE); II) `advertising`: T1–T7 logística com **3 solvers**, `confusion_matrix` + `classification_report`, análise do melhor modelo. |
| `PL4/daa_pl_4_I-Linear.ipynb` | nb (58 cél., 22 c/ outputs) | `USA_Housing` (5000×7): EDA cél.5–12 (pairplot, hist, corr heatmap); split 0.4 seed 101 (cél.16); `LinearRegression` + interpretação de coeficientes (cél.20–25, House Age = +164 883 $); resíduos (cél.27–29); MAE 82 288 / RMSE 102 278 (cél.32); **demo LASSO: colunas aleatórias `randomCol1/2`, `Lasso(alpha=2000)` anula `randomCol1` (cél.35–53 — nota: cél.40/45 fazem fit ao X,y inteiros, sem split)**. |
| `PL4/daa_pl_4_II-Logistic.ipynb` | nb (54 cél., 22 c/ outputs) | Titanic: mapa de missing (cél.8); **`Age = fillna(média por Pclass)` via `groupby().transform('median')` (cél.18)**; drop `Cabin` (cél.21) e `dropna` (cél.23); `get_dummies(drop_first)` Sex/Embarked (cél.27–30); split 0.3 seed 101 (cél.33); **3 solvers `newton-cg`/`lbfgs`/`liblinear` com cronometragem (cél.39–46)**; comparação de `classification_report` — macro F1 0.81/0.81/0.80 (cél.49); `ConfusionMatrixDisplay` (cél.50); sugestões de FE: Title do nome, letra da cabine, ticket (cél.51). |
| `PL4/DAA_PL_4(1).pdf` | slides (39 pp.) | Regressão linear + coeficientes (pp.11–15), MAE/MSE/RMSE (p.16), **regularização LASSO vs OLS com colunas random (pp.17–21)**; Titanic EDA + missing (pp.24–29), **imputação mediana por Pclass (pp.30–31)**, drop Cabin/NaN (p.32), dummies (p.33), tabela de solvers/penalidades (p.35), 3 modelos (p.36). |
| `PL4/USA_Housing.csv` | dados | 5000×7 |
| `PL4/titanic_train.csv` / `titanic_test.csv` | dados | 891×12 / 418×11 |
| `PL4/(4)DAA_datasets/advertising.csv` | dados | 1000×10 (`Clicked on Ad`, `Age`, `Timestamp`, …) — só ficha |
| `PL4/(4)DAA_datasets/EcommerceCustomers.csv` | dados | 500×8 — só ficha |
| `PL4/(4)DAA_datasets.zip` | zip | = 5 CSVs (advertising, Ecommerce, USA_Housing, titanic_train/test) |

**Resumo PL4:** regressão e logística; o padrão **imputação por mediana estratificada**
(`groupby(Pclass).transform('median')`) aparece aqui e no PL6. Nota: a ficha pede
`EcommerceCustomers`, o notebook/slides usam `USA_Housing`.

### PL5 — Feature Engineering + Decision Tree Pruning/GridSearch ⭐ (mais relevante)

| Ficheiro | Tipo | Conteúdo real |
|---|---|---|
| `PL5/DAA_Ficha_5.pdf` | exercício (1 p.) | FE (scaling/encoding/novas features) + DT pruning + GridSearch; Titanic T1–T11: GridSearchCV com `criterion/max_depth/min_samples_leaf` (T7), **pré-poda `max_depth` vs pós-poda `ccp_alpha` (T9)**. |
| `PL5/daa_pl_5_I-FE.ipynb` | nb (96 cél., 49 c/ outputs) | `incidents.csv` Guimarães 5000×13. Missing: `affected_roads` 85 NaN, `incidents` 2028 NaN (cél.13); **drop vs `fillna(0)` experimentados em cópias (cél.14–20) → escolhe drop**; **colunas de valor único removidas: `avg_precipitation` (tudo 0) cél.27–31, `city_name` ("Guimaraes") cél.33–36**; `dropna` → 2972 linhas (cél.36); balanço de classes (High 1073/Low 718/Very_High 603/Medium 578, bar+pie, cél.38–41); **6 encoding alternativos do target/categóricas (cél.42–73):** (a) replace com associação ordinal manual None0/Low1/Medium2/High3/VeryHigh4 (cél.45–50), (b) `LabelEncoder` + colunas one-vs-rest `np.where` (cél.55–57), (c) one-hot `LabelBinarizer` (cél.58–60), (d) **`category_encoders.BinaryEncoder`** (cél.61–64), (e) **`BackwardDifferenceEncoder`** (cél.65–66), (f) `pd.factorize` (cél.67–71) → **escolhido factorize**; factorize de `magnitude_of_delay/luminosity/avg_rain` (cél.72–73); **datas: `to_datetime(format=…, errors='coerce')` + `assert` sem nulos (cél.77–78), extracção ano/mês/dia/hora/minuto (cél.80), drop de ano/minuto de valor único (cél.83–84), feature "tempo desde hoje" em dias (cél.89–91)**. |
| `PL5/daa_pl_5_II-DT.ipynb` | nb (82 cél., 37 c/ outputs) | Mesmo dataset com `na_filter=False`; drops + factorize (cél.8–10); corr. heatmap (cél.14) e jointplots/lmplot (cél.17–21); split 0.3 seed 2022 (cél.25); DT base + `plot_tree(max_depth=3)` + `export_text`→`dt_text.log` (cél.31–32); **relatório: macro F1 0.58 / accuracy 0.62 (cél.38)**; **`GridSearchCV` `criterion`×`max_depth` 1–10, 5 folds → melhor `max_depth=6` (cél.46–47)**; avaliação **macro F1 cai para 0.54 com accuracy a subir para 0.64 (cél.51)**; pré-poda `max_depth` 1–24 → 6 (cél.57–61, avaliação cél.64: macro 0.54); **pós-poda `ccp_alpha` via `cost_complexity_pruning_path` → melhor 8.3e-4 (cél.67–75), avaliação cél.78: macro 0.58 (volta ao baseline)**. |
| `PL5/DAA_PL_5.pdf` | slides (55 pp.) | FE completo: missing (drop vs fill, pp.9–13), colunas únicas (pp.14–16), **6 encodings (replace/label/one-hot/binary/backward-difference/factorize, pp.19–29)**, datas (pp.30–35); DT signature com `monotonic_cst` (p.43); GridSearchCV mechanics (pp.45–48); teoria de poda e curva de `ccp_alpha` (pp.49–54). |
| `PL5/incidents.csv` | dados | 5000×13 |
| `PL5/dt_text.log` | texto | 1429 linhas — árvore exportada em texto |
| `PL5/dt_plot.png`, `dt_plot_md.png`, `dt_plot_alpha.png` | imagens | vistos: árvore base (prof.3), árvore do gridsearch (`max_depth=6`), árvore pós-podada |
| `PL5/pp.png` | imagem | visto: pairplot do incidents por classe |
| `PL5/circuit-2q.tex` | tex | **circuito quântico TikZ (`quantikz`, portas R_ZZ/R_YY) — resíduo de outra disciplina, nada a ver com DAA** |

### PL6 — Ensemble Learning

| Ficheiro | Tipo | Conteúdo real |
|---|---|---|
| `PL6/DAA_Ficha_6.pdf` | exercício (1 p.) | Max voting/bagging/boosting/stacking T1–T7; T4 mudar para alvo contínuo `Fare` (DT/LR/RF/ensemble). |
| `PL6/daa_pl_6(1).ipynb` | nb (166 cél., **só 12 com outputs — folha de exercícios com "buracos"**) | Titanic: EDA; **`Age` imputado pela mediana por `Pclass` (cél.13)**; drop `Cabin`+NaN e `Sex/Embarked/Name/Ticket` (cél.14–16, fica só 6 features); split 0.3 seed 2025 (cél.20); DT `max_depth=2` 0.6816 (cél.24–28), LR `liblinear` 0.6779 (cél.36–40), RF (cél.48); **`VotingClassifier` hard (cél.62, 146)**; `StratifiedShuffleSplit` (cél.75); **`BaggingClassifier` + `GridSearchCV` `n_estimators∈{10,40,60,80,100,160}` (cél.77–84)**; bagging+RF (cél.96–105); `GradientBoostingClassifier` (cél.109); **`XGBClassifier` (cél.121 — xgboost é ensinado aqui, `conda install -c conda-forge xgboost` na cél.2)**; **`StackingClassifier` (cél.135)**; comparação de acurácias em dict+barplot (cél.157–163, células por preencher). |
| `PL6/DAA_PL_6(1).pdf` | slides (26 pp.) | Taxonomia de ensembles com assinaturas sklearn: Voting (p.13), `StratifiedShuffleSplit`+Bagging (pp.15–17), GradientBoosting (p.18), **XGBClassifier (p.19)**, Stacking (p.20); comparação em barras (pp.22–25, valores só em imagem). |
| `PL6/titanic_dataset.csv` | dados | 891×12 |

### PL7 — Interpretability and Explainability

| Ficheiro | Tipo | Conteúdo real |
|---|---|---|
| `PL7/DAA_Ficha_7.pdf` | exercício (1 p.) | T1 interpretabilidade local/global; T2 top-3 features com **RF importance, MDI e Permutation Importance**; T3 **SHAP** para RF vs Regressão Linear; T4 mesmo estudo com **XGBoost**. |
| `PL7/daa_pl_7.ipynb` | nb (116 cél., 56 c/ outputs) | Titanic FE: histplots, heatmap de correlação (cél.12), missing (cél.13–15), **mediana da `Age` (cél.17)**, **moda do `Embarked` (cél.29) e da `Cabin` (cél.35)**, `Sex` 1/0 (cél.39), drop `Name/Ticket/Cabin` (cél.42), `LabelEncoder` no `Embarked` (cél.46), **correlação com o target ordenada (cél.48)**, grava `titanic_ds.csv` (cél.50); RF 81.72% (cél.56–59) e `submission.csv` (cél.62); **importâncias cél.63–80: Gini (Sex 0.232, Fare 0.217, PassengerId 0.179 — viés de alta cardinalidade), MDI (idem), `permutation_importance` no hold-out `n_repeats=10, random_state=42` (Sex 0.156, Fare 0.049, Pclass 0.044, PassengerId colapsa p/ 0.016)**; **SHAP cél.81–103: `TreeExplainer`, waterfall local (cél.87), heatmap global (cél.89), scatter por feature Sex/Pclass/Age/Fare/SibSp/Parch (cél.92–101), PDP de SibSp (cél.103)**; **`LinearExplainer` + `summary_plot` + dependence + PDP (cél.104–114)**. |
| `PL7/DAA_PL_7(1).pdf` | slides (48 pp.) | why/how (p.5), tabela **global vs local** (p.26), tabela **model-agnostic vs model-specific** (p.27), RF FI/MDI/PI (pp.28–33), teoria SHAP (p.34), waterfall (p.35), heatmap (p.36), feature-wise SHAP (pp.37–41), PDP (p.42), SHAP lineares (pp.43–47). |
| `PL7/titanic.csv` | dados | 891×12 |
| `PL7/titanic_ds.csv` | dados | 891×9 (dataset limpo gerado pelo nb) |
| `PL7/submission.csv` | dados | 268×2 (`PassengerId`,`Survived` — previsões RF do hold-out) |

### PL8 — Artificial Neural Networks (MLP / PyTorch)

| Ficheiro | Tipo | Conteúdo real |
|---|---|---|
| `PL8/DAA_Ficha_8.pdf` | exercício (1 p.) | T1 redes com PyTorch; T2 variar neurónios/camadas; T2b épocas/LR/loss/optimizer; T3 comparar com DT/RF. |
| `PL8/DAA_PL_8(1).pdf` | slides (62 pp.) | ANN/MLP teoria + frameworks (pp.5–10); split **70/10/20 train/test/val com DataLoader (p.13)**; verificação de balanço de classes (pp.14–17); construção (`nn.Module/Linear/ReLU/Softmax`, init Xavier/Kaiming, p.18); `CrossEntropyLoss` (com nota de classe desequilibrada)/SGD/Adam (p.26); loop de treino (pp.27–31); **tabela final (pp.57–59): M1 (sem init/norm, CEL, SGD) acc 0.730; M2 (Kaiming+Xavier) 0.225; M3 (normalização Age&Fare) 0.674; M4 (Adam) 0.551; M5 (CEL `weights=[1.2,1.2,0.9]`) 0.730** com moral: init padrão já era o melhor, normalização parcial introduz inconsistentes, pesos de classe equilibram desequilíbrio, Adam compensa init/escala. |
| `PL8/daa_pl8_notebooks/daa_pl_8_mlpc_M1.ipynb` | nb (47 cél., 12 c/ outputs) | Alvo **`Pclass`** (3 classes), 8 features incl. `PassengerId` e **`Survived` como input**; `CSVDataset`+`DataLoader` batch=inteiro (cél.16); **split 70/10/20 por `random_split` sem seed (cél.16)**; balanço por split (cél.19–22, grava `data_balance_MLP.png`); MLP 8→24→12→3 (cél.26); 200 épocas SGD momentum 0.9 (cél.32); **output cél.33: accuracy de treino estagnada em 0.204 (bug: `Softmax` aplicado após hidden2 e após hidden3, cél.26)**; avaliação cél.35–40: **acc 0.202, macro F1 0.11, prevê sempre a classe 2**; predict de 1 caso (cél.44). |
| `PL8/daa_pl8_notebooks/daa_pl_8_mlpc_M2.ipynb` | nb (30 cél., **0 outputs — não executado**) | Variante "Kaiming/Xavier": md cita init mas **o código não faz init**; cél.12 `MLP_2()` sem `n_inputs`, cél.16 e cél.24 com parênteses em falta → **não corre tal como está**. |
| `PL8/daa_pl8_notebooks/daa_pl_8_mlpc_M3.ipynb` | nb (53 cél., **0 outputs**) | Normalização min-max de `Age`/`Fare` **em células por preencher** (cél.5–21); lê `titanic_X_scaled.csv`/`titanic_y_scaled.csv` (não incluídos); lógica de treino idêntica com os mesmos erros de sintaxe; **cél.40 chama `train_model(train_dl, test_dl, model)` — treina com o test loader**. |
| `PL8/daa_pl8_notebooks/daa_pl_8_mlpc_M4.ipynb` | nb (30 cél., 0 outputs) | Variante Adam do M2 (diff confirmado: só `optimiser: Adam` + classe `MLP_4`), mesmos erros de sintaxe. |
| `PL8/daa_pl8_notebooks/daa_pl_8_mlpc_M5.ipynb` | nb (33 cél., 0 outputs) | "Melhor config + loss assimétrica" = `CrossEntropyLoss` com class weights + Adam; **`criterion` nunca é definido (cél.19) → `NameError` se executado**. |
| `PL8/daa_pl8_notebooks/data_balance_MLP.png` | imagem | visto: barras de contagem por classe em train/test/val (349/44/98 na classe 3, etc.) |
| `PL8/daa_pl8_notebooks/titanic_ds.csv` (891×9), `titanic_X.csv` (891×8), `titanic_y.csv` (891×1) | dados | artefactos do PL7/PL8 |
| `PL8/daa_pl8_notebooks.zip` | zip | = os 5 notebooks |

---

## 2. APLICAÇÃO ao nosso assignment (ordenado por valor esperado)

Ordenado por valor esperado para fechar/explicar o gap 0.51→0.62. Nada em PL1–PL8
ensina um ganho de +0.11; o material é introdutório (Titanic/wine/incidents) e **a
explicação mais provável do 0.62 alegado é otimismo de avaliação** (ver LEAKAGE-RISK #1).

| # | Item (fonte) | Tag | Aplicação concreta |
|---|---|---|---|
| 1 | **Imputação/estatísticas por estrato** — `fillna(groupby(Pclass).transform('median'))` (`PL4/daa_pl_4_II-Logistic.ipynb` cél.18; `PL6/daa_pl_6(1).ipynb` cél.13) + imputação condicionada por grupo da ficha (`PL3/DAA_Ficha_3.pdf` T3f/T3g: moda por um feature, média por grupo) | **DIRECT** | Experimento fold-safe: dentro de cada fold de treino, para as células `Sem resposta`/`Não aplicável` de numéricos com sentinel, imputar pela **mediana do estrato** (estrato = flags de routing não-alvo, ex. `MAINSTAT`) + flag de missingness; estatísticas só com a parte treino do fold (regra: regenerável sem o fold de validação). Spec: NB09, 5-fold seed 42, adotar só 5/5. |
| 2 | **Binning de frequência igual** — `KBinsDiscretizer(strategy='quantile')` (`PL3/daa_pl_3.ipynb` cél.38; ficha T3c "3 bins iguais sobre age") | **DIRECT** | Companheiros quantílicos (3–5 bins) das top contínuas (`IN15`, contagens, `N_DOENCAS`) como features extras, `KBins` fit dentro do fold. Baixo custo; esperado ~0 para GBDT (já partem em quantis) mas pode dar 0/+0.003 aos membros lineares da família (LR/HGB). |
| 3 | **Encoding alternativo dos nominais: Binary/Backward-Difference** (`PL5/daa_pl_5_I-FE.ipynb` cél.61–66 com `category_encoders`; slides pp.25–27) | **DIRECT** | Substituir as 171 one-hots dos 22 nominais por binary encoding (~3–8 bits/var). Lição NB09 (−28 colunas = +0.0066, "diluição") sugere compressão semântica; esperado 0/+0.004, custo 0.5–1d, sem leakage (encoding é stateless). |
| 4 | **Auditoria MDI vs Permutation vs SHAP para justificar cortes de bloco** (`PL7/daa_pl_7.ipynb` cél.66–79: PassengerId 0.179 Gini → 0.016 permutação; slides pp.27–33) | **DIRECT** (análise, não ganho) | Reproduzir o contraste no nosso F&H/XGB: importâncias MDI inflam cardinalidade alta (one-hots) — usar permutação/SHAP no fold de validação para justificar F5 (ablação de blocos) e o §do relatório. Esperado 0 em CV; custo 0.25d. |
| 5 | **Padrão "tune no mesmo hold-out e reportar"** (`PL5/daa_pl_5_II-DT.ipynb` cél.46–78 tune+avalia no mesmo split; `PL7`/`PL8` 1 split único; `PL2` KFold manual) | **LEAKAGE-RISK** | Este é o **candidato #1 a explicar o 0.62 do colega**: escolher no mesmo split usado para reportar infla (nós vimos +0.01 em `registo_experiencias.md:197` "números otimistas"; a retração NB16 mediu +0.036…+0.061 por fold-sensitivity). Ver Top-5 #5 (diagnóstico, nunca para seleção). |
| 6 | **Remoção de linhas por regra com a idade** — ficha `PL3/DAA_Ficha_3.pdf` T3e "remover utilizadores WebActivity<1h **e idade>70**"; `PL3/daa_pl_3.ipynb` cél.21 (filtro por regra → −26 linhas); `PL5` dropna do alvo (cél.36) | **LEAKAGE-RISK** | Filtro que usa (ou proxima) a idade muda os priors das classes e infla macro-F1 se feito antes do split. É a mesma família do canal de relabel/quarantine já retraído (`HANDOFF.md:25-31`). **Não fazer** como pré-CV; se for só "remover outliers de features", só com limiares definidos dentro do fold. |
| 7 | **`classification_report`/confusão/macro-F1 como métrica de decisão** (`PL2` ficha T2–T3, notebook cél.18–25, slides pp.19–27) | **INDIRECT** | Origem do nosso protocolo (macro-F1 + k-fold) e do costume de olhar por classe. Útil para o relatório; nada de novo. |
| 8 | **Demo de regularização LASSO com colunas aleatórias** (`PL4/daa_pl_4_I-Linear.ipynb` cél.35–53; slides pp.17–21) | **INDIRECT** | Só relatório (porque é que L1 não é a resposta aqui); seleção de features já rejeitada (T4/T8). |
| 9 | **3 solvers da logística / multinomial vs OvR** (`PL4/daa_pl_4_II-Logistic.ipynb` cél.36–49; slides p.35) | **INDIRECT** | Coerente com sklearn 1.9 (`multi_class` removido, multinomial por defeito). Nada a colher para o ganho. |
| 10 | **Peso de classes / loss assimétrica** (`PL8` slides pp.54–59 M5 `weights=[1.2,1.2,0.9]`; `PL5` ficha "check imbalanced data") | **IRRELEVANT** | Já coberto: XGB/LGBM usam `balanced`; "per-class cost matrices" está na lista §10. Desequilíbrio 6.17x não é o gargalo (diagnóstico §0: erros off-by-one). |
| 11 | **Ensembles: voting/bagging/boosting/stacking** (`PL6` notebook cél.62–144, slides pp.13–20) | **IRRELEVANT** | Stacking esgotado, T15 bagging −0.005 rejeitado, seed/row bagging negativo (§10). |
| 12 | **MLP/PyTorch, init, normalização, Adam** (`PL8` 5 notebooks + slides) | **IRRELEVANT** | E13 MLP rejeitado; CORAL/redes ordinais "do not bother" (§8/§10). |
| 13 | **GridSearch/poda de árvore** (`PL5/daa_pl_5_II-DT.ipynb` cél.46–78) | **IRRELEVANT** (ganho) / **INDIRECT** (lição) | Hiperparâmetros rejeitados. **Mas a cél.51 é uma citação perfeita para o relatório: accuracy 0.62→0.64 com macro-F1 0.58→0.54** — otimizar accuracy arruína macro-F1, prova empíica de que métrica ≠ accuracy. |
| 14 | **Decomposição de datas** (`PL5` cél.74–95) | **IRRELEVANT** | INS não tem datas (HHMM→min já tratado no NB01). |
| 15 | **Remoção de colunas de valor único** (`PL5` cél.27–36) | **IRRELEVANT** | Já feito (NB09 T7a, +0.0066). |
| 16 | **Setup de ambiente, EDA descritiva, regressão linear, DT pruning, factorize, binning de wine** (PL1, PL3, PL4-I, PL5-I) | **IRRELEVANT** | Sem aplicação ao ganho; PL1 p/ relatório de ambiente. |

---

## 3. Top-5 ideias concretas para o gap 0.51 → 0.62 (ângulo do tratamento de dados)

N.B.: o gap inteiro (+0.11) **não** é explicável por PL1–PL8 — nenhum passo de curso
vale isso honestamente. As ideias abaixo são o que há de colhível; a explicação mais
provável do 0.62 alegado é a #5 (otimismo de avaliação), coerente com a nossa retração.

**#1 — Imputação estratificada + features de estatística por estrato**
- *O quê:* para `Sem resposta`/`Não aplicável` em numéricos, imputar mediana do estrato
  definido por flags de routing (ex. `MAINSTAT`, bloco respondido), + flag de missingness;
  dentro de cada fold, só com a parte treino.
- *Fonte:* `PL4/daa_pl_4_II-Logistic.ipynb` cél.18 (`groupby(Pclass).transform('median')`),
  `PL6/daa_pl_6(1).ipynb` cél.13, `PL3/DAA_Ficha_3.pdf` T3f/T3g (imputação condicionada).
- *Ganho esperado:* +0.000 a +0.005. *Custo:* 0.5d (NB09, gate 5/5).
- *Risco de leakage:* **baixo se fit dentro do pipeline do fold; alto se pré-computar em
  todo o train** — é exatamente a classe de erro do NB16 (estatística que "vazou" o fold).

**#2 — Companheiros quantílicos das top contínuas (binnning)**
- *O quê:* `KBinsDiscretizer(strategy='quantile', n_bins=3..5)` de `IN15`, contagens,
  `N_DOENCAS`/`N_DIFICULDADES` como features extra, fit por fold.
- *Fonte:* `PL3/daa_pl_3.ipynb` cél.38; `PL3/DAA_Ficha_3.pdf` T3c (bins de frequência
  igual sobre `age`).
- *Ganho esperado:* 0 a +0.003 (ajuda sobretudo aos membros lineares da família).
- *Custo:* 0.5d. *Leakage:* baixo (stateless se fit no fold).

**#3 — Binary encoding dos 22 nominais em vez de 171 one-hots**
- *O quê:* `category_encoders.BinaryEncoder` nos nominais (NB02), mantendo ordinais;
  comparar 5-fold vs 311 atuais.
- *Fonte:* `PL5/daa_pl_5_I-FE.ipynb` cél.61–64 (e BackwardDifference cél.65–66),
  `PL5/DAA_PL_5.pdf` pp.25–27.
- *Ganho esperado:* 0 a +0.004 — justifica-se pela lição de diluição (NB09: −28 colunas
  = +0.0066 adotado). *Custo:* 0.5–1d (`category_encoders` via conda-forge).
- *Leakage:* nenhum (encoding não usa o alvo).

**#4 — Auditoria de importâncias (MDI vs permutação vs SHAP) para decidir ablação de blocos**
- *O quê:* no fold de validação, comparar as três famílias de importância nas features
  do F&H; usar só para justificar F5 (blocos MENTAL/PAIN/SUPPORT/…) e o relatório —
  nunca como filtro automático de features (feature selection já rejeitado).
- *Fonte:* `PL7/daa_pl_7.ipynb` cél.66–79 ( PassengerId: 0.179 Gini vs 0.016 permutação)
  e SHAP cél.84–103; `PL7/DAA_PL_7(1).pdf` pp.27–34.
- *Ganho esperado:* 0 em CV (qualidade de relatório + decisão de ablação).
- *Custo:* 0.25d. *Leakage:* baixo se calculado no fold; **risco médio se servir de
  filtro pré-CV** (é a feature selection já rejeitada: T4 −0.0037).

**#5 — Diagnóstico: quanto do "0.62" é otimismo de protocolo (reprodução do padrão de curso)**
- *O quê:* correr o mesmo pipeline 2 vezes: (a) estilo curso — um único split 70/30,
  tuning + escolhas + relatório no mesmo split; (b) 5-fold honesto seed 42. Medir a
  inflação no nosso dados. Nunca usar (a) para seleção.
- *Fonte:* padrão repetido em `PL5/daa_pl_5_II-DT.ipynb` cél.46–78 (tune e avalia no
  mesmo split; macro 0.58→0.54→0.58), `PL7`/`PL8` (1 split), `PL2` (KFold manual sem
  aninhamento); calibrado pelo que já vimos: `registo_experiencias.md:197` (+0.01
  otimista) e retração NB16 (`plano_acima05.md:96`: +0.036…+0.061 de inflação).
- *Ganho esperado:* não é ganho de modelo — é **explicação mensurável** de parte do gap
  (se a inflação estilo-curso der ~+0.05, o resto do 0.62 é outra coisa).
- *Custo:* 0.5d. *Leakage:* intencional e controlado; resultado nunca entra na seleção.

---

## 4. Could not verify

- **`PL8` M2–M5 não executados e quebrados:** sem outputs; erros de sintaxe (parênteses,
  `MLP_2()` sem args), M3 lê `titanic_X_scaled.csv`/`titanic_y_scaled.csv` que não existem
  na pasta, M5 nunca define `criterion`, M3 cél.40 treina com `test_dl`. As accuracies
  0.730/0.225/0.674/0.551/0.730 vêm **dos slides** (`PL8/DAA_PL_8(1).pdf` pp.57–59), não
  pude confirmá-las por execução dos notebooks.
- **`PL6` é folha de exercícios com buracos:** maioria das células tem placeholders
  (`results['HVotW']= `, `cross_val_score(..., scoring='', cv=)`) — as acurácias reais dos
  ensembles não estão no ficheiro; nos slides (pp.22–25) os valores só existem em imagem,
  não extraíveis em texto.
- **Screenshots dentro dos slides:** todos os 16 PDFs têm camada de texto extraída e lida,
  mas blocos de código/output embutidos como imagem (ex. `PL3/DAA_PL_3.pdf` pp.18–19,
  `PL4` pp.6–13) não foram OCRados.
- **Links externos das fichas** (goo.gl/bit.ly de datasets) não foram acedidos — todos os
  datasets das fichas estão de qualquer forma locais.
- **`PL7/submission.csv`** (268 linhas) não tem ground truth para validar.
- **Pastas `T/` (teóricas Aula 0–8) fora de âmbito** — não inspeccionadas.

---

## 5. Log

| Data | Ação | Resultado |
|---|---|---|
| 11 nov 2026 | Leitura integral de contexto do assignment (`contexto.md`, `plano_acima05.md` §0/§6, `registo_experiencias.md`, `pesquisa_quebra_platamar.md` §10) | Regras e lista "não incomodar" carregadas antes de mapear |
| 11 nov 2026 | Extração e leitura dos 16 PDFs (8 fichas + 8 decks de slides) de PL1–PL8 | 100% do texto lido; páginas/slide citadas no mapa |
| 11 nov 2026 | Dump célula-a-célula (markdown+código+outputs) e leitura dos 13 notebooks | 100% lidos; outputs reais citados por nº de célula |
| 11 nov 2026 | Carga dos 19 CSVs (shape/colunas) + listagem dos 3 zip | Zips = duplicados das pastas; nenhum dataset extra escondido |
| 11 nov 2026 | Abertura dos 5 PNGs + leitura de `dt_text.log` e `circuit-2q.tex` | Árvores/pairplot/balanço confirmados; `circuit-2q.tex` = resíduo quântico irrelevante |
| 11 nov 2026 | Escrita de `docs/mapa_pl.md` (58/58 ficheiros inspeccionados) | Concluído; nenhum outro ficheiro tocado, nenhum treino executado |
