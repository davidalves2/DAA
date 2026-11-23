# Registo de Experiências — DAA 2026/2027 (Previsão de AGE_RANGE)

Registo de todas as experiências, resultados e conclusões, para justificar as decisões no relatório.
**Métrica:** Macro F1. **Validação:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, sempre as mesmas 5 partes.
Os valores "validação" são médias das 5 partes (`cross_val_score`) salvo indicação de "agregado" (`cross_val_predict` + `f1_score` sobre todas as previsões).

---

## 0. Referências mínimas (Notebook 1, secção 3.1)

| Classificador trivial | Macro F1 |
|---|---|
| Prevê sempre a classe maioritária (60-64) | 0,011 |
| Aleatório estratificado | 0,060 |
| Valor teórico do aleatório (1/16) | 0,0625 |

---

## 1. Árvore de decisão — profundidade (dados v1, 409 colunas)

`class_weight='balanced'`

| max_depth | 4 | 6 | **8** | 10 | 12 | 15 | None |
|---|---|---|---|---|---|---|---|
| Macro F1 | 0,3065 | 0,3729 | **0,4073** | 0,4011 | 0,3933 | 0,3800 | 0,3688 |

- Profundidade baixa → *underfitting*; alta → *overfitting*. Ótimo em 8 (desvio padrão 0,006).
- **Repetido com os dados finais (Notebook 3 limpo, secção 2):** 4: 0,3090 | 6: 0,3806 | **8: 0,4044 (± 0,013)** | 10: 0,3996 | 12: 0,3910 | 15: 0,3780 | None: 0,3697. Mesmo padrão; o pré-processamento final não altera a árvore.
- **Submissão 1 → Kaggle 0,40** (validação 0,407: a validação é fiável).

## 2. Seleção de features (Notebook 2)

- Regra: remover colunas cujo valor mais frequente ocorre em ≥ 99% das linhas (70) + colunas duplicadas (6). 409 → 333 colunas.
- Duplicadas: `NUTS2_AG_Algarve/Norte/RA Açores/RA Madeira` (= `REGION`), `JOBISCO_COD_Não aplicável`, `LOCNACE_COD_Não aplicável`.
- Árvore: 0,4073 → 0,4065 (igual, dentro do ruído).
- Gradient Boosting (ver secção 12): 0,4625 → 0,4688 (**ajuda**).

## 3. Novas features — contagens e IMC (Notebook 2)

| Feature | Spearman com a idade | Árvore (só esta feature adicionada) |
|---|---|---|
| (nenhuma) | — | 0,4065 |
| `N_DIFICULDADES` (nº de HA1*/PC1* com dificuldade) | 0,58 | 0,4047 |
| `N_DOENCAS` (nº de CD1* = Sim) | 0,53 | 0,4032 |
| `IMC` | 0,18 | 0,4030 |
| `N_NAO_APLICAVEL` | −0,46 | 0,4070 |
| Todas | — | 0,3993 |

- Fortemente relacionadas com a idade, mas **não ajudam a árvore** (informação já disponível nas colunas originais; árvore única instável).
- **Ajudam a Logística:** 0,4141 (v1) → 0,4260 (v2).
- Gradient Boosting: 0,4688 → 0,4663 (sem ganho; o GB aproxima contagens sozinho).

## 4. Otimização da árvore (GridSearchCV, dados v2)

Grelha: `criterion` {gini, entropy} × `max_depth` {6, 8, 10, 12} × `min_samples_leaf` {1, 5, 10, 20} = 32 combinações.

- Melhor: gini, max_depth=8, min_samples_leaf=10 → 0,4028. Top 10 entre 0,398 e 0,403 (desvios 0,008–0,017).
- Mesmos parâmetros com v1 → 0,4110.
- **Conclusão:** a árvore estabiliza em ~0,40; os hiperparâmetros quase não importam; árvore única instável. **Não adotado.**

## 5. Regressão Logística (dados v2)

`StandardScaler` + `LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=2000)` num pipeline.

| C | 0,01 | 0,1 | 1 |
|---|---|---|---|
| Macro F1 | **0,4260** | 0,4260 | 0,4239 |

- Mesmo modelo com v1: 0,4141 → as contagens ajudam modelos lineares.
- **Submissão 2 → Kaggle 0,43** (validação 0,426).

## 6. Dois modelos especializados (corte dos 55 anos via `HA1A`)

- Treino: 4877 jovens / 6085 seniores; teste: 1626 / 2029.
- Logística: dois modelos 0,4284 ± 0,0099 vs. um modelo 0,4260 → **sem ganho claro. Não adotado.**
- Explicação (confirmada na secção 7): o modelo único só punha 10 pessoas do lado errado dos 55.

## 7. Análise de erros (Logística, dados v2, agregado)

- Acerto exato 43,3%; erro de 1 faixa 37,9%; 2 faixas 12,5%; ≥3 faixas 6,2% → **81% a no máximo uma faixa**.
- Erros que cruzam os 55 anos: 10 pessoas (0,1%).
- F1 por faixa: 15-17 0,723 | 18-19 0,317 | 20-24 0,462 | 25-29 0,371 | 30-34 0,251 | 35-39 0,350 | 40-44 0,339 | 45-49 0,375 | 50-54 0,634 | 55-59 0,631 | 60-64 0,569 | 65-69 0,413 | 70-74 0,318 | 75-79 0,262 | 80-84 0,330 | 85+ 0,481.
- Faixas junto a fronteiras (15-17, 50-54, 55-59) são fáceis; faixas intermédias de cada grupo são difíceis; 65+ com erros mais dispersos.

### 7b. Análise de erros do modelo final (GB otimizado + correção, dados finais, agregado) — Notebook 3 limpo, secção 5

- Macro F1 agregado: **0,4752**.
- Acerto exato 47,9% (Logística: 43,3%); erro de 1 faixa 37,0%; 2 faixas 10,7%; ≥3 faixas 4,4% (Logística: 6,2%) → **84,9% a no máximo uma faixa** (Logística: 81,2%).
- Erros que cruzam os 55 anos: 5 pessoas (Logística: 10).
- F1 por faixa (GB | Logística): 15-17 0,855 | 0,723 · 18-19 0,371 | 0,317 · 20-24 0,593 | 0,462 · 25-29 0,400 | 0,371 · 30-34 0,299 | 0,251 · 35-39 0,369 | 0,350 · 40-44 0,345 | 0,339 · 45-49 0,397 | 0,375 · 50-54 0,658 | 0,634 · 55-59 0,679 | 0,631 · 60-64 0,648 | 0,569 · 65-69 0,533 | 0,413 · 70-74 0,345 | 0,318 · 75-79 0,266 | 0,262 · 80-84 0,355 | 0,330 · 85+ 0,489 | 0,481.
- O GB melhora todas as faixas; maiores ganhos em 15-17, 20-24, 65-69 e 60-64. As mais difíceis continuam a ser 75-79 (0,27) e 30-34 (0,30).
- 18-19 é previsto como 20-24 em 40% dos casos e como 15-17 em 22%.

## 8. Idade mínima declarada (Notebook 2)

Validação: % de casos em que o valor ultrapassa o limite superior da faixa da pessoa.

| Variável | Nº com valor | Acima da idade máxima | Decisão |
|---|---|---|---|
| IN2 | 1433 | 0 (0,0%) | aceite |
| IN3 | 1273 | 0 (0,0%) | aceite |
| IN4 | 746 | 0 (0,0%) | aceite |
| IN25 | 1000 | 28 (2,8%) | **rejeitada** |
| IN26 | 1334 | 1 (0,1%) | aceite |
| IN45 | 2232 | 0 (0,0%) | aceite |
| IN47 | 3273 | 0 (0,0%) | aceite |
| IN24 + 15 | 1476 | 3 (0,2%) | aceite |

- `IDADE_MINIMA` = máximo das aceites (15 se nenhuma); `FAIXA_MINIMA` = faixa correspondente.
- 48,1% das pessoas com idade mínima informativa; 4 casos com faixa mínima acima da real (anomalias → Notebook 4).
- Como feature na Logística: 0,4260 → 0,4266 (residual).
- Como feature no Gradient Boosting: 0,4663 → **0,4742** (o maior ganho de uma feature no GB; o GB não consegue calcular o máximo entre várias colunas).

## 9. Correção pela idade mínima (pós-previsão)

Regra: `previsão = max(previsão, FAIXA_MINIMA)`.

- Logística (agregado): 141 previsões impossíveis (1,3%); 0,4272 → **0,4362** (+0,009).
- Gradient Boosting: +0,0002 (o GB já aprende estes limites).
- Mantida no modelo final por ser inofensiva e garantir previsões coerentes.

## 10. Comparação de modelos (dados v3, agregado)

| Modelo | Macro F1 | Com correção | Tempo (s) |
|---|---|---|---|
| **Gradient Boosting** (HistGradientBoosting, por defeito, balanced) | **0,4668** | **0,4670** | 66,8 |
| Random Forest (300 árvores, min_samples_leaf=2, balanced) | 0,4497 | 0,4516 | 3,9 |
| Regressão Logística (C=0,01) | 0,4272 | 0,4362 | 2,1 |
| SVM (RBF, balanced) | 0,4201 | 0,4272 | 16,5 |
| Árvore de Decisão (depth 8, leaf 10) | 0,4104 | 0,4161 | 0,4 |
| KNN (k=15) | 0,2939 | 0,3062 | 0,8 |
| Naive Bayes (Gaussiano) | 0,2726 | 0,2808 | 0,3 |

- **Repetido no Notebook 3 limpo (dados finais):** valores idênticos (árvore depth 8 sem leaf: 0,4071 / 0,4116). Previsões impossíveis corrigidas por modelo: GB 7 | RF 30 | Logística 141 | SVM 113 | Árvore 74 | KNN 240 | Naive Bayes 165.
- Conjuntos de árvores > árvore única (reduzem instabilidade).
- KNN: todas as 340 variáveis pesam igual na distância. Naive Bayes: pressuposto de independência violado (variáveis redundantes).
- **Submissão 3 (GB por defeito + correção) → Kaggle 0,46** (validação 0,467).

## 11. Otimização do Gradient Boosting

Base: `HistGradientBoostingClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, class_weight='balanced', random_state=42)`.

**Pesquisa 1** (`RandomizedSearchCV`, 12 combinações, 5,6 min) — grelha: learning_rate {0,03; 0,05; 0,1; 0,2} × max_leaf_nodes {15, 31, 63} × min_samples_leaf {10, 20, 50} × l2_regularization {0; 0,1; 1; 10}.

- **Melhor: learning_rate=0,05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1 → 0,4742 ± 0,0073 (156 árvores).**
- Padrão: árvores pequenas + aprendizagem lenta + regularização; árvores de 63 folhas e l2=0 piores.

**Pesquisa 2 — refinada** (15 combinações, 9,2 min) — learning_rate {0,03; 0,05; 0,07} × max_leaf_nodes {7, 10, 15, 20} × min_samples_leaf {20, 40} × l2 {1, 5, 10}.

- Melhor: 0,4747 (lr 0,03, 15 folhas, leaf 40, l2 1) → **igual (dentro do ruído). Não adotado**; mantém-se a configuração da pesquisa 1.

## 12. Contributo de cada etapa dos dados para o Gradient Boosting

Mesmos hiperparâmetros (pesquisa 1).

| Versão dos dados | Nº colunas | Macro F1 | Desvio padrão |
|---|---|---|---|
| Apenas codificação | 409 | 0,4625 | 0,0051 |
| + seleção (99% + duplicadas) | 333 | 0,4688 | 0,0035 |
| + contagens e IMC | 337 | 0,4663 | 0,0085 |
| + idade mínima (**dados finais**) | 339 | **0,4742** | 0,0073 |
| (seleção + idade mínima, sem contagens) | 335 | 0,4705 | — |

- Confirmado na versão limpa do Notebook 2 (secção 7), com os mesmos valores.
- Diferenças da ordem do desvio padrão (0,004–0,009): tendências, não certezas.

**Organização final dos dados:** `train_nb01.csv` → Notebook 2 → `train_nb02.csv` (339 features + alvo). As versões intermédias v2/v3 foram eliminadas; as submissões 1–3 foram feitas com essas versões anteriores.

## 13. Votação suave (GB + Random Forest + Logística, agregado)

- GB otimizado sozinho: 0,4752 | Votação: 0,4717 → **pior. Não adotado** (os modelos mais fracos degradam o melhor).

## 14. Idade mínima alargada (escolaridade e tipo de agregado)

- `HATLEVEL` → idade mínima de conclusão {Secundário 17, Pós-sec. 18, CTeSP 19, Licenciatura 20, Mestrado 22, Doutoramento 25}: 3693 com valor, 4 violações (0,1%).
- `HHTYPE` "filhos ≥ 25 anos" → 25: 1086 com valor, 0 violações.
- Cobertura 48,1% → 67,1%; 8 casos com faixa mínima acima da real.
- GB: 0,4705 → 0,4698 → **sem ganho. Não adotado** (regras de uma só coluna, que o GB já aprende).

---

## 15. Modelo final (Notebook 3 limpo, secção 6)

- Gradient Boosting otimizado treinado com todo o treino: 156 árvores.
- Teste: 5 previsões corrigidas pela idade mínima.
- Distribuição das previsões no teste próxima da do treino (diferenças ≤ 1,1 pontos percentuais em todas as faixas; maior em 55-59: 9,3% → 10,4%).

## 16. Experiências do colega (ramo `rui-1`, Notebooks 4–6 dele) — 27/10/2026

Feitas com o `HistGradientBoostingClassifier` por defeito sobre uma versão **antiga** do `train_nb02.csv` (409 colunas: sem seleção, sem novas features, sem idade mínima). Não usou XGBoost. Mesmas 5 partes (seed 42).

- Modelos de base: árvore 0,407 | Random Forest 200 0,446 | HGB 0,4657. 72% dos erros são de uma faixa ao lado.
- Blocos de features (cumulativo): corte de extremos [p0,5; p99,5] 0,4650 | + contagens "Não aplicável"/"Sem resposta" 0,4628 | + nº doenças CD1 e IMC 0,4657 → **sem ganho**.
- Codificações: one-hot 0,4657 | ordinal pela média da idade dentro de cada parte (`MAINSTAT`, `MARSTALEGAL`) 0,4636 | agrupar categorias < 1% em "RARO" 0,4629 → **sem ganho**.
- Interações `MAINSTAT×HATLEVEL`, `MARSTALEGAL×nº doenças`: 0,4612 → **pior**.
- `HalvingRandomSearchCV`: 0,4637 → **sem ganho**.
- **Classificador + regressor por confiança** (limiar 0,5): 0,4662 → 0,4774. **Escalas por classe** (fator 0,5–2 por faixa): 0,4662 → 0,4715. Em ambos, o limiar/escalas foram escolhidos e avaliados nas mesmas previsões → **números otimistas**. Kaggle reportado ~0,48.
- Ideia aproveitada: a combinação com o regressor (secção 17), agora com o limiar escolhido de forma honesta.

## 17. XGBoost e combinação classificador + regressor (Notebook 3, secção 7) — 27/10/2026

Dados finais (339 features), mesmas 5 partes, tudo com correção pela idade mínima, F1 agregado.

**7.1 Classificadores** (`XGBClassifier(n_estimators=350, learning_rate=0.05, max_depth=4, min_child_weight=3, subsample=0.8, colsample_bytree=0.5)` + `compute_sample_weight('balanced')`):

| Modelo | Macro F1 | Desvio padrão | Acerto exato | ≤1 faixa |
|---|---|---|---|---|
| GB otimizado (sub. 4) | 0,4752 | 0,0077 | 47,9% | 84,9% |
| XGBoost | 0,4796 | 0,0146 | 48,1% | 85,4% |
| Média GB + XGBoost | 0,4768 | 0,0089 | 48,0% | 85,3% |

- GB e XGBoost concordam em 77,4% das pessoas. XGBoost sozinho e média → dentro do ruído, **não adotados isoladamente**.

**7.2 Combinação com regressor** (`HistGradientBoostingRegressor` com `MELHORES_PARAMS_GB` + pesos; sozinho 0,4302). Regra: confiança (prob. máx.) ≥ limiar → classificador; senão → regressor arredondado. Limiar escolhido nas 4 partes de treino de cada parte (avaliação honesta):

| Combinação | Macro F1 | Limiares | Ganho por parte vs sub. 4 |
|---|---|---|---|
| GB + regressor | 0,4772 | 0,3/0,4/0,4/0,4/0,4 | +0,010 +0,006 −0,001 0,000 −0,005 |
| **XGBoost + regressor** | **0,4888** | 0,35 ×5 | +0,044 +0,018 −0,004 +0,012 −0,001 |
| Média GB + XGBoost + regressor | 0,4802 | 0,35/0,3/0,4/0,35/0,35 | +0,027 +0,005 −0,005 +0,006 −0,007 |

- Curva do limiar (XGBoost, todas as pessoas, só ilustrativa): 0,30 → 0,4851 | **0,35 → 0,4888** (24% pelo regressor) | 0,40 → 0,4862 | 0,50 → 0,4783 | 0,55 → 0,4709.
- **Adotado: XGBoost + regressor, limiar 0,35** (+0,014; ganho claro em 3 de 5 partes, nulo nas outras 2).
- Exploração prévia (script fora do notebook) com XGBoost com paragem antecipada: a melhor era média GB + XGBoost + regressor (0,4863), e o XGBoost sozinho dava 0,474. Conclusão robusta: **juntar o regressor ajuda sempre (0,48–0,49)**; qual o classificador melhor está dentro do ruído.

**7.3 Modelo final:** regressor com 308 árvores; no teste, 74,0% decididas pelo XGBoost e 26,0% pelo regressor; 5 corrigidas pela idade mínima; 69,8% iguais à sub. 4 (94,7% a ≤1 faixa). Distribuição: 75-79 sobre-prevista (11,8% vs 7,5% no treino) e 65-69 sub-prevista (6,8% vs 9,5%) → efeito do regressor, que puxa os casos duvidosos dos mais velhos para o "meio". Pista de melhoria.

## 18. Otimização do conjunto XGBoost + regressor (scripts fora do notebook) — 27/10/2026

Mesmas 5 partes, correção sempre aplicada, limiar (e restantes escolhas) escolhidos nas 4 partes de treino de cada parte. Referência nestes scripts: XGBoost base 350 + `reg_hgb` = **0,4886** (sub. 5; 0,4888 no notebook). Ganhos por parte medidos contra a sub. 4.

**a) Hiperparâmetros do XGBoost** (12 configurações × 7 nº de árvores = 84 variantes; uma alteração de cada vez face à base):
- Sozinho: todas entre 0,455 e 0,482 (melhor `colsample_bytree=0.3`, 450 árvores: 0,4824). `max_depth=6` e `learning_rate=0.1` piores.
- + regressor: melhor `min_child_weight=1`, 450 árvores: 0,4914 (+0,003), mas escolhido entre 84 variantes → otimista. **Não adotado** (dentro do ruído).

**b) Média de 5 sementes do XGBoost** (sementes 42, 1, 2, 3, 4): sozinho 0,4824 (600 árv.) | + regressor 0,4901 (600 árv.), com ganho positivo nas 5 partes (+0,038 +0,010 +0,007 +0,014 +0,005) → mais estável.

**c) Regressores** (com XGBoost base): `reg_hgb` 0,4886 | `reg_hgb` 31 folhas sem pesos 0,4909 | `l2=10` sem pesos 0,4901 | XGBRegressor sem pesos 0,4894 | idade esperada das probabilidades (Σ p×faixa) 0,4893 | média HGB/XGB 0,4887 → todos dentro do ruído.

**d) Regras de combinação:**
- Cortes de arredondamento do regressor otimizados (15 cortes, ±0,3 em torno de k+0,5): 0,4913.
- Peso GB/XGBoost escolhido com o limiar: 0,4849 → pior.
- Escalas por classe (honestas): 0,4849 → **pior** (as escalas do Rui eram otimistas).
- Regressor só se as 2 faixas mais prováveis estiverem a ≤2 faixas: 0,4855 → pior.

**e) Junção das melhores:**

| Variante | Macro F1 | Ganhos por parte vs sub. 4 | Acerto | ≤1 faixa |
|---|---|---|---|---|
| **XGB 5 sementes 600 + `reg_hgb` + cortes** | **0,4937** | +0,039 +0,010 +0,018 +0,022 +0,003 | 49,5% | 87,3% |
| XGB 5 sementes 450 + `reg_hgb` + cortes | 0,4925 | +0,041 +0,014 +0,007 +0,018 +0,004 | 49,4% | 87,0% |
| XGB 5 sementes 600 + `reg_hgb` 31 sem pesos + cortes | 0,4922 | +0,036 +0,013 +0,011 +0,016 +0,008 | 49,3% | 87,6% |
| XGB base 350 + `reg_hgb` (sub. 5) | 0,4886 | +0,042 +0,016 −0,008 +0,015 +0,003 | 48,6% | 86,5% |

- A melhor junção ganha +0,005 face à sub. 5 (no limite do ruído; escolhida entre ~12 junções → ligeiramente otimista), mas é **positiva nas 5 partes** face à sub. 4 e corrige a distribuição: 75-79 prevista em 8,4% (sub. 5: 10,7%; real 7,5%).
- Lição: o conjunto está perto do teto (~0,49); hiperparâmetros, escalas e pesos já não rendem. O que ainda ajuda é reduzir a instabilidade (sementes) e ajustar o arredondamento do regressor.

## 19. Modelos novos, stacking e modelo ordinal (scripts fora do notebook) — 27/10/2026

Mesmas 5 partes, correção aplicada, limiar escolhido nas 4 partes de treino.

**a) Modelos novos, sozinhos:** **ordinal Frank & Hall com XGBoost 0,5008** | LightGBM 0,4744 | CatBoost 0,4739 | ExtraTrees 0,4653 | Logística 0,4364.
- Ordinal (Frank & Hall): 15 classificadores binários XGBoost "faixa > k" (k = 0…14; mesmos parâmetros do XGBoost base, 300 árvores, pesos `balanced` das 16 faixas). Probabilidade de cada faixa = P(>k−1) − P(>k), com P(>k) forçada a não subir com k.

**b) Combinações com os modelos da secção 18:**
- Médias de probabilidades (xgb5, ordinal, LGBM, CatBoost, GB): melhor xgb5 + ordinal + regressor 0,4972. Juntar LGBM/CatBoost não ajuda (erram nos mesmos casos).
- Stacking (Regressão Logística sobre log-probabilidades + valores dos regressores, C ∈ {0,003; 0,01; 0,03}): 0,487–0,492 → **não adotado** (não supera a média simples).

**c) Afinação do ordinal** (uma alteração de cada vez; ordinal + `reg_hgb`): 300–450 árvores melhor; `max_depth` 3/5 e `colsample_bytree=0.3` sem ganho. Sementes isoladas variam entre 0,498 e 0,507 → instável, daí a média de 5 sementes.

| Variante | Macro F1 | Ganhos por parte vs sub. 4 | ≤1 faixa |
|---|---|---|---|
| Ordinal 5 sementes 300 + `reg_hgb` + cortes | 0,5070 | +0,049 +0,040 +0,014 +0,038 +0,017 | 88,1% |
| **Ordinal 5 sementes 300 + `reg_hgb`** | **0,5062** | +0,047 +0,043 +0,010 +0,037 +0,017 | 88,1% |
| Ordinal 5 sementes 300 + idade esperada própria | 0,5049 | +0,044 +0,040 +0,012 +0,038 +0,014 | 88,3% |
| 0,7·ordinal + 0,3·xgb5 + `reg_hgb` | 0,5051 | +0,051 +0,042 +0,014 +0,028 +0,013 | 87,8% |
| Ordinal 5 sementes 300 sozinho | 0,5039 | — | 87,9% |
| Ordinal 1 semente 300 + `reg_hgb` | 0,5036 | +0,052 +0,035 +0,008 +0,035 +0,012 | 87,9% |
| XGB 5 sementes + `reg_hgb` + cortes (secção 18) | 0,4937 | | 87,3% |

- **Adotado: ordinal 5 sementes × 300 árvores + `reg_hgb`, limiar 0,35** (escolhido nas 5 partes) → **0,506** (+0,018 face à sub. 5; positivo nas 5 partes). Os cortes (+0,0008) não compensam a complexidade.
- **No notebook (secção 7.5, 28/10/2026): 0,5051** em vez de 0,5062. A diferença vem do regressor: no notebook, os pesos são calculados com o `y` completo (como na 7.2); no script, eram calculados em cada parte. É ruído (< 0,002). Limiar 0,35 nas 5 partes; F1 por parte [0,519; 0,516; 0,497; 0,497; 0,495]; acerto exato 50,0%; ≤1 faixa 88,1%; 75-79 prevista em 9,5% (real 7,5%).
- Regressor decide só 11% das pessoas (o ordinal já usa a ordem). Acerto exato 50,1%. Distribuição OOF próxima da real (75-79: 8,0% vs 7,5%).
- Porque funciona: cada classificador binário aprende uma fronteira "mais novo/mais velho que X" com todas as pessoas (não só as de duas faixas), e a ordem das faixas fica embutida nas probabilidades.
- Teste (`sub06_ordinal_regressor.csv`): 12,9% pelo regressor; 4 corrigidas pela idade mínima; 79,6% iguais à sub. 5 (98,2% a ≤1 faixa). 75-79 prevista em 10,2% (treino 7,5%), 85+ 3,5% (4,5%).

## 20. Tentativas de melhorar o ordinal + regressor (scripts fora do notebook) — 28/10/2026

Referência: ordinal XGB 5 sementes + regressor = **0,5051** (notebook, secção 7.5). Mesmas 5 partes; limiar (e outras escolhas) escolhidos nas 4 partes de treino. Triagem com 2 sementes (referência com 2 sementes: 0,5025).

**a) Regras de decisão sobre as probabilidades do ordinal:**
| Variante | Macro F1 | Nota |
|---|---|---|
| Baixa confiança → **mediana** da distribuição do ordinal (em vez do regressor) | 0,5065 | +0,0014, 4/5 partes; dispensa o regressor |
| Baixa confiança → E[faixa] / mistura reg + E | 0,5045–0,5064 | ruído |
| Suavizar probabilidades das vizinhas antes do argmax | 0,5053 | ruído |
| Dividir pela frequência das faixas (P / prior^a) | 0,5051 | escolhe sempre a = 0 |
| Máscara pelo corte dos 55 (`HA1A`) | 0,5045 | o modelo só põe 0,2% da probabilidade do lado errado |

**b) Variantes do modelo ordinal** (2 sementes; sozinho / + regressor):
| Variante | Sozinho | + reg |
|---|---|---|
| Referência XGB (2 sementes) | 0,5009 | 0,5025 |
| Pesos `balanced`^0,5 | 0,4987 | 0,4982 |
| Sem pesos | 0,4981 | 0,4963 |
| LightGBM como base | 0,5021 | 0,5056 |
| HGB como base (9× mais lento) | 0,4974 | 0,4997 |
| XGB `learning_rate=0.03`, 500 árvores | 0,5022 | 0,5050 |
| 150 features mais importantes (escolhidas em cada parte) | 0,5027 | 0,5010 |
| Features novas (adultos, prop. ativos/inativos, nº "Sem resposta", anos a fumar) | 0,5020 | 0,5043 |

→ nenhuma variante isolada supera a referência.

**c) Médias de modelos ordinais de algoritmos diferentes:**
| Variante | + reg | + mediana | Partes positivas (melhor) |
|---|---|---|---|
| XGB ×5 (atual) | 0,5051 | 0,5065 | 4/5 |
| XGB ×5 + LGBM ×4 | 0,5074 | 0,5049 | 2/5 |
| XGB ×5 + LGBM ×4 + HGB ×2 | 0,5060 | **0,5089** | 4/5 |
| Todas as variantes ordinais (~25 modelos) | 0,5083 | 0,5082 | 3–4/5 |

- Melhor: **+0,0038** (XGB + LGBM + HGB, mediana), escolhido entre ~15 combinações → otimista. Abaixo do limiar de +0,005 → **não adotado; não gastar submissão**. O modelo final continua a ser a sub. 6.
- Lição: o modelo ordinal + regressor está num patamar (~0,505–0,509); features, pesos, hiperparâmetros e algoritmos de base já não rendem acima do ruído.

## 21. Notebook 4 — descobertas da interpretabilidade (28/10/2026)

- SHAP calculado com `get_booster().predict(DMatrix(...), pred_contribs=True)`: o `shap` 0.49 dá `ValueError` com o `base_score` vetorial dos modelos multiclasse do XGBoost 3.x.
- Importância global (XGBoost 16 faixas, amostra estratificada de 2000): MAINSTAT 6,9%, IN15 5,5%, PC1A 5,3%, MARSTALEGAL 4,8%... Bloco do corte dos 55 (IN15, PC1A–E, HA1B, HA1G) ≈ 28% no total. `IN15` e `PC1A`–`E` seguem o corte dos 55 sem exceções, tal como a `HA1A`.
- **Segundo corte, aos 65 anos:** `HH_ACT + HH_INACT == 0` ⇒ 65+ (3202 pessoas; só 2 exceções, ambas de 15-17). As variáveis contam apenas os membros do agregado com menos de 65 anos (provavelmente 16/18–64), por isso uma pessoa com 65+ que vive só com pessoas de 65+ tem 0. O contrário não é verdade: 807 pessoas com 65+ vivem com pessoas mais novas. O modelo já respeita esta regra (só 5 violações nas previsões OOF); aplicá-la como correção: 0,5051 → 0,5052 (ruído, não adotada).
- Uma pessoa de 15-17 vive sozinha e tem `HH_ACT + HH_INACT = 0` → candidata a anomalia (secção 4).
- **Anomalias (NB4, secção 4):** 16 incoerências lógicas (4 impossíveis, 9 muito improváveis, 3 raras); 36 anomalias segundo o modelo (`P_DECLARADA` = P(faixa real ± 1) < 0,05); só 2 em comum.
- **"Sem resposta" no bloco dos 55** (`HA1*`, `PC1*`, `IN15` = −2) ⇒ 55+, em todas as 43 pessoas (a pergunta foi feita). O modelo não aprende isto (é raro) e prevê menos de 55 em 4 delas. Estas 4 (PID 10646, 13435, 1265, 13291) são **artefactos do modelo**, não perfis anómalos. Usada como correção: 0,50512 → 0,50514 (ruído; 12 casos no teste) → não adotada.
- **Features derivadas do NB4** (ordinal XGB, 2 sementes; 28/10): `MEMBROS_65_MAIS` = HHNBPERS − HH_ACT − HH_INACT − HHNBPERS_0_13 (0 ⇒ menos de 65, só 3 exceções: a própria pessoa com 65+ conta como membro 65+), `PERGUNTADO_55` (bloco dos 55 ≠ "Não aplicável"), `SEM_MEMBROS_ATIVOS`. Resultados: sozinho 0,5007 (referência 0,5009), + reg 0,5011 (0,5025), + mediana 0,5038 (0,5042); em média com a referência 0,5053–0,5061 → **sem ganho**. O modelo já combina estas variáveis sozinho.
- Menopausa declarada antes dos 40: 3 pessoas (PID 874, 905, 8743). O PID 874 (20-24, previsto 50-54, confiança 0,92) é o caso mais forte de provável erro na idade.

## Registo de submissões

| # | Data | Modelo | Validação | Kaggle |
|---|---|---|---|---|
| 1 | 25/10/2026 | Árvore de decisão (depth 8), dados v1 | 0,407 | 0,40 |
| 2 | 26/10/2026 | Regressão Logística (C=0,01), dados v2 | 0,426 | 0,43 |
| 3 | 26/10/2026 | Gradient Boosting (por defeito) + correção, dados v3 | 0,467 | 0,46 |
| 4 | 26/10/2026 | Gradient Boosting otimizado + correção, dados finais | 0,475 | 0,45 |
| 5 | 27/10/2026 | XGBoost + regressor HGB (limiar 0,35) + correção | 0,489 | 0,48 |
| 6 | 27/10/2026 | Ordinal Frank & Hall (XGBoost, 5 sementes × 300) + regressor HGB (limiar 0,35) + correção | 0,506 | 0,49 |
| 7 | 28/10/2026 | Média de 3 ordinais (XGB ×5 + LGBM ×4 + HGB ×2) + mediana se confiança < 0,40 + correção (script fora do notebook) | 0,509 | **0,51** |

**Nota sobre a submissão 4:** validação 0,475 (melhor que a sub. 3, 0,467) mas Kaggle 0,45 (sub. 3: 0,46). A diferença de 0,01 está dentro da incerteza da tabela pública do Kaggle, calculada só sobre parte das 3655 pessoas do teste (o desvio padrão entre partes da nossa validação, com ~2190 pessoas cada, já é ~0,007). A escolha do modelo final baseia-se na validação cruzada (10 962 pessoas), mais fiável do que a tabela pública; escolher pela tabela pública arrisca *overfitting* a essa amostra.

**Nota sobre a submissão 6 (28/10/2026):** Kaggle 0,49 contra 0,506 na validação (diferença de 0,016, um pouco acima do habitual ≤ 0,01). Mesmo assim, é a melhor submissão até agora (+0,01 face à sub. 5, a mesma subida que na validação, 0,489 → 0,506). A ordem entre os modelos mantém-se; a diferença absoluta é ruído da tabela pública. **Sub. 7 (28/10/2026):** Kaggle **0,51**, validação 0,509. A melhor nos dois critérios; a subida face à sub. 6 é coerente (+0,004 na validação, +0,02 no Kaggle, este último sobretudo ruído). **Modelo final: sub. 7.** Passada para o Notebook 3 (7.7: validação 0,5089 com o CSV `oof_ordinal_lgbm_hgb.csv`; 7.8: modelo final, 3655/3655 previsões iguais ao ficheiro submetido).

## Lições principais

1. A validação cruzada estratificada reproduz o Kaggle com diferenças ≤ 0,01 → decisões tomadas localmente.
2. O desenho do questionário ("Não aplicável") é a principal fonte de informação sobre a idade (corte perfeito aos 55 anos na `HA1A`).
3. Modelos diferentes reagem de forma diferente às mesmas features: contagens ajudam modelos lineares, não árvores; limites que combinam várias colunas (idade mínima) ajudam o GB.
4. Regras lógicas (idade mínima) renderam mais do que a otimização de hiperparâmetros na Logística.
5. A partir de ~0,474 as diferenças entre variantes ficam abaixo do ruído da validação (±0,006).
