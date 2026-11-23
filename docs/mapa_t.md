# Mapa da pasta de teoria `DAA/T` — o que ela (não) explica sobre 0.51 → 0.62

> **Data:** 2026-11-11 (sessão de mapeamento).
> **Âmbito:** pasta `/home/rui/Documents/MIA-1/DAA/T` (9 ficheiros, todos PDF, 231 páginas
> no total; sem subpastas). Nenhum outro ficheiro do repo foi tocado; nenhum treino executado.
> **Método:** todos os PDFs abertos e extraídos com `pypdf` (env `conda daa`); páginas com
> pouco texto rasterizadas com Ghostscript e lidas visualmente (10 renders); metadados
> internos (data de criação/autor) lidos via `pypdf` metadata; diff programático entre os
> dois decks ANN. Contexto lido antes: `docs/contexto.md`, `docs/plano_acima05.md` (§0, §6),
> `registo_experiencias.md`, `docs/pesquisa_quebra_platamar.md` (§9, §10), `docs/plano_modelacao.md`.
> **Datas:** mtimes de todos os PDFs = **2026-11-11 23:02–23:04** (copiados para a pasta hoje);
> criação interna dos PDFs vai de **2026-09-21** (Aula 0) a **2026-11-10** (Aula6-ANN-1 e Aula8
> — material mesmo fresco, semana 9 do plano `T/Aula 0, p.9`).

**Veredito de topo (1 linha):** a pasta é material de teoria de licenciatura (Alpaydin/Tan-style),
sem uma única folha sobre métrica macro-F1, desequilíbrio de classes, regressão ordinal ou
pesos de inquérito — **não contém nenhuma receita de "data treatment" que justifique +0.11
honesto**; o achado mais valioso é que o deck de Ensembles **ensina o stacking ingénuo (o
padrão que infla CV)**, o que aponta para a explicação mais barata da reivindicação do colega.

---

## 1. MAPA — inventário completo

### 1.1 Tabela por ficheiro

| Ficheiro | Pág. | Criação PDF | Autor | Tópico | Conteúdo concreto (fórmulas/métodos/protocolos) |
|---|---|---|---|---|---|
| `T/Aula 0 - DAA 2627.pdf` | 14 | 2026-09-21 | — | Introdução, logística | Definições de ML (Mitchell 1997, AAAI, p.7); programa da cadeira (p.8: intro/DS/KDD, supervisão, NN, evolucionário); **calendário completo** (p.9: fase TP 30/nov+2/dez, prova escrita 16/dez/2026, TP final 11–13/jan/2027, avaliação por pares 15/jan, recurso 20/jan); avaliação 10% quizzes + 40% teoria + 50% prática, nota mín. 8 na teoria (p.12); referências Mitchell/Alpaydin/Domingos (p.10) |
| `T/Aula2a - DAA 2627.pdf` | 20 | 2026-09-28 | — | Conceitos e metodologias | ML vs Data Science (p.4); aprendizagem simbólica vs não-simbólica (p.9); **supervisionada** y=f(x), classificação vs regressão (p.10); não-supervisionada: clustering/redução/associação (p.11); RL: Q-Learning vs SARSA (p.12); quiz de taxonomia de 20 tarefas (p.13–14); **SEMMA** Sample-Explore-Modify-Model-Assess (p.17); **CRISP-DM** (p.18, crisps-dm.eu); inquérito KDnuggets 2016/17 de metodologias (p.16, verificado por render: CRISP-DM 43%/42% = 66 menções, "My own" 27.5%, SEMMA 8.5%/3%); diagrama "Data Science Pipeline" (p.15, render) e fluxo de pipeline ML (p.19) |
| `T/Aula2b - DAA 2627.pdf` | 28 | 2026-09-28 | Cesar Analide | **Data Exploration and Preparation** (o único deck de tratamento de dados) | Sumário (p.2: qualidade, preparação básica, scaling, outliers, feature selection, missing values, discretização, binning, feature engineering); caso de Wald (aviões WWII, survivorship bias, p.3); problemas de qualidade (p.5: missing values incl. **"features that are not applicable in all cases"** = o nosso routing; duplicados; p.6: noise/outliers); exploração (p.8: tendência central, dispersão, distribuições, correlação; p.10 tabelas de contingência/qui-quadrado de associação; p.11 "devemos manter features altamente correlacionadas?"); screenshot RapidMiner Data Explorer (p.9, render: min/max/mean/std/var/skew/kurtosis/missing + histograma); preparação básica (p.14–15: joins, concat, filtros, agregações); **scaling** (p.17–18: justificação p/ classificadores de distância; standardização z=(x−μ)/σ e min-max z=(b−a)(x−min)/(max−min)+a — *os rótulos das duas fórmulas estão trocados no slide*); **outliers** (p.19: estatístico z-score/boxplot, knowledge-based, model-based one-class SVM/isolation forest/clustering; dilema **drop vs cap**); **feature selection** (p.20–22: remover por % missing, qui-quadrado vs alvo, skew, baixo desvio, correlação entre features; PCA; wrapper; embedded Lasso/regularização; SFS/SBS); **missing values** (p.23: remover / média / interpolação / **mask**); **encoding nominal** (p.24: one-hot vs label vs binary; label = inteiros ordenados; caveat "huge number of categories"); **binning** (p.25: robustez vs perda de informação); **feature engineering** (p.26–27: decomposição de data 2021-10-29 → ano/mês/dia/hora/dia-da-semana/weekend/holiday) |
| `T/Aula3 - DAA 2627_RL.pdf` | 23 | 2026-10-05 | Victor Alves | Linear & Logistic Regression | Axiomas da linearidade (p.3); regressão múltipla ŷ=θ₀+Σθⱼxⱼ (p.5–6); custo MSE J=(1/2m)Σ(h−y)² (p.8, com código numpy); **sigmoid** 1/(1+e^(−θᵀx)) e limiar 0.5 (p.9–10); **multiclasse por OvR** — um binário por classe, escolhe-se a maior probabilidade (p.11); log-loss (p.12); mínimos quadrados θ=(XᵀX)⁻¹Xᵀy (p.14); gradiente descendente θⱼ:=θⱼ−α(1/m)Σ(h⁽ⁱ⁾−y⁽ⁱ⁾)xⱼ⁽ⁱ⁾, α = learning rate (p.15–17); tabela batch/estocástico/mini-batch × complexidade m/n (p.19); overfitting com **termos de interação** apresentados como exagero de complexidade (p.20); **regularização** L1 Lasso (zeros → seleção de features) vs L2 Ridge (encolhe, mantém) com λ (p.21); standardização/normalização para GD (p.22) |
| `T/Aula4 - DAA 2627_DT.pdf` | 20 | 2026-10-12 | Victor Alves | Decision Trees | **Gini** G=1−Σp² com exemplo (0.395/0.336 → total ponderado 0.364, p.5); escolha do split = menor impureza (p.6–7); atributos contínuos: ordenar, médias adjacentes, testar limiares (p.8); regression trees (p.9–10, render: StatQuest dosagem→eficácia, folhas 4.2%/2.5%/52.8%/100% com limiares 14.5/29/23.5; método p.11–14: mín. SSE por limiar, folha = média, min. ~20 amostras); **poda** (p.15–16: pre-pruning `min_samples_split`/`min_sample_leaf`, post-pruning profundidade máx.; p.17 **cost-complexity Rα(T)=R(T)+α|T|**, melhor sub-árvore **via cross-validation**, `cost_complexity_pruning_path`); algoritmos ID3/C4.5/J48/CART/CHAID (p.18); prós/contras (p.19: não precisa de normalização/scaling, missing values não afetam, instável a pequenas mudanças, **fraco com muitas interações**); referências Quinlan/Breiman/Tan/StatQuest (p.20) |
| `T/Aula5 - DAA 2627_Ensembles.pdf` | 15 | 2026-10-20 | Victor Alves | Ensemble Learning | Sumário (p.2: voting, bagging, boosting, stacking, **blending**); max voting hard/soft com exemplo 65.67% (p.4); média e **média ponderada** por desempenho (p.5); bagging bootstrap com analogia dos 25 alunos (p.6); **Random Forest** (p.7: bootstrap + subset aleatório de features ≈ √n, ~1/3 OOB, erro OOB, afinação do nº de features pelo OOB); boosting = reponderar erros (p.8); **AdaBoost vs XGBoost** (p.9: gradient boosting = gradiente descendente sobre os parâmetros dos novos estimadores, crescimento sequencial, paralelismo, regularização); **stacking** (p.11: meta-features = previsões dos modelos base **no próprio train**; **p.12 (render): diagrama Level-0 Models → "Predictions" sobre o MESMO Training Data → concatenado → Level 1 Model — sem cross-fitting**) ; **blending** (p.13: meta treinado nas previsões de um **hold-out validation set**); paralelo vs sequencial + "combinar learners semelhantes/bons não ajuda; heterogéneos/fracos ajudam" (p.14); referências (p.15: Bonab & Can *Less Is More* arXiv:1709.02925; **Kuncheva & Whitaker, measures of diversity**, ML 51:181–207; Opitz & Maclin JAIR 11:169–198) |
| `T/Aula6 -DAA 2627_ANN.pdf` | 37 | 2026-10-27 | Victor Alves | Artificial Neural Networks | **p.2: diagrama de protocolo Kaggle** — X_train/y_train → val, "Test → Kaggle", leaderboard **público 30% / privado 70%**, números 0.88 (train) / 0.79 (val) / 0.85 (público) / 0.785 (privado) rotulados **"Overfitting"**; neurónio/artifícios (p.8); funções de ativação (p.9; **p.10 (render): tabela identity/binary step/logistic/tanh/arctan/ReLU/PReLU/ELU/SoftPlus com equações e derivadas** + taxonomia saturadas vs não-saturadas); MLP feedforward (p.12–13); forward pass vetorizado em numpy (p.14); exemplo XNOR (p.15–16); **saídas de classificação** (p.17: 1 neurónio com limiar vs 1-of-C, argmax, softmax para probabilidades); treino/backprop (p.18–21: gradiente, épocas, batches, **critérios de paragem com subset de validação**, regra da cadeia, exemplo mattmazur); topologia/hiperparâmetros (p.23–25: 1 camada oculta → áreas lineares, 2 → convexas, 3 → arbitrárias; hidden nodes testar ½×–2× o input; "usar os poucos nós suficientes no conjunto de validação"); overfitting/regularização (p.26: early stopping, decay); **learning curves — 7 diagnósticos (p.29–35):** (1) underfit flat → mais dados/features/capacidade, menos regularização; (2) underfit ainda a descer → mais épocas + early stop + lr; (3) overfit no ponto de inflexão → reduzir lr/capacidade/regularização/dropout; (4) bom ajuste; (5) train pequeno vs val → mais dados, amostragem aleatória, **cross-validation**; (6) val pequeno → mais dados de validação/CV; (7) **val mais fácil que train → verificar duplicados entre splits, verificar *information leakage*, amostragem aleatória, CV**); prós/contras (p.36: requer pré-processamento específico, caixa negra); frameworks (p.37: tensor, autodiff, BLAS/cuDNN) |
| `T/Aula6 -DAA 2627_ANN-1.pdf` | 39 | **2026-11-10** | Victor Alves | ANN (atualizado) + **Tabular Foundation Models** | **Superset exato do anterior** (diff programático: só acrescenta p.38–39 + espaços em branco): **p.38 "Tabular Foundational Models"** com link ao **TabArena leaderboard** (huggingface.co/spaces/TabArena/leaderboard); **p.39**: "Tabular foundation models are based on **Prior-Fitted Networks (PFNs)**… pré-treinado em milhões de datasets sintéticos gerados de **structural causal models**", referência **TabFM** (research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data) |
| `T/Aula8-DAA 2627_Clustering.pdf` | 35 | **2026-11-10** | Victor Alves | Unsupervised: redução de dimensionalidade, anomalias, clustering | "Unsupervised é difícil porque não há métrica de erro" (p.2); distâncias Euclidean/Manhattan/Jaccard (p.5); inventário de algoritmos (p.6); **redução dimensional** (p.7: visualização + **"preparatory tool for other ML algorithms"**; feature selection vs projeção); **PCA** (p.8: 5 passos (standardizar → matriz correlação → eigenvetores → ordenar → elbow), 80/95% da variância, incremental/randomized/kernel); manifold LLE/Isomap/t-SNE (p.9); **autoencoders** (p.10: bottleneck; denoising/variacional/conv/recorrente); "!!! make sure you scaled the data" (p.11); **detecção de anomalias** (p.12: uso em *data cleaning* antes de treinar; p.13: IQR/Tukey, isolation forest, one-class SVM, LOF, MCD; **p.14 (render): fluxograma "Data Cleaning? → sim: Tukey → Fast-MCD; não: LOF → one-class SVM/Isolation Forest"**); clustering exige preprocessing + standardização (p.16); **K-Means** (p.17–19: algoritmo, elbow, prós/contras, "não lida bem com muitas features", MiniBatch, K-Medoids/K-Modes); **p.20/23/25/27/29/31/33 = screenshots de demonstração do sklearn** (6 formas sintéticas × 10 algoritmos MiniBatchKMeans/AffinityProp/MeanShift/Spectral/Ward/Agglomerative/DBSCAN/OPTICS/BIRCH/GMM com tempos; render verificado em 2: seta vermelha a destacar MiniBatchKMeans e Agglomerative); hierarchical + ligações single/complete/average/centroid/Ward (p.21–22); spectral (p.24: autovalores da matriz P×P de similaridade); DBSCAN (p.26: core sample com `min_samples` dentro de `eps`); affinity propagation (p.28); mean shift (p.30); BIRCH (p.32); **seleção de algoritmo** (p.34: K≈√n, correr primeiro DBSCAN/Affinity Propagation); referências Han & Kamber, Witten & Frank, Tan et al., sklearn 0.15 (p.35) |

### 1.2 Sumário por tema (narrativo)

**A. Metodologia de projeto (Aula 0, Aula2a).** CRISP-DM e SEMMA, pipeline Data Science,
motivos para usar metodologia (replicabilidade/gestão/boas prâcticas — Aula2a p.16–18).
Nada operacional para o score; matéria-prima para a secção de metodologia do relatório de 20 pp.

**B. Tratamento de dados (Aula2b) — o deck central para a nossa pergunta.** Cobre exatamente
a classe de coisas que o colega pode ter chamado "different data treatment": missing values
(remove/mean/interp/**mask**), scaling, outliers (drop vs cap), feature selection (qui-quadrado,
correlação, PCA, wrapper, Lasso), encoding (one-hot/label/binary), binning, feature engineering.
**Mas quase tudo já está feito ou rejeitado no repo:** masks = 17 flags de routing + `N_NAO_APLICAVEL`
(feat F1/F2 do `plano_acima05.md:36-39`); encoding comparison = decisão D2 (`plano_modelacao.md:81-90`,
empate técnico `registo_experiencias.md:194`); feature selection = T4/T8 rejeitados
(`pesquisa_quebra_platamar.md:262-264`); clipping de outliers = bloco F1 sem ganho
(`plano_modelacao.md:138`); binning/PCA nunca correram mas são quase certamente neutros para
GBDT one-hot. O único conceito ainda *em aberto* é o **"mask"** enquanto categoria e as
contagens por bloco (F2).

**C. Modelos (Aula3, Aula4, Aula5, Aula6×2).** Regressão linear/logística (incl. OvR p.11 e
regularização L1/L2), árvores (Gini, poda cost-complexity com CV), ensembles (bagging/RF OOB,
boosting, AdaBoost vs XGBoost, stacking vs blending, diversidade Kuncheva & Whitaker), ANN
(backprop, topologia, learning curves). A nossa pipeline já está para além destes (XGB/LGBM/HGB
+ ordinal Frank & Hall, `registo_experiencias.md:262-285`). Dois pontos aproveitáveis:
a **diagnóstica de learning curves** (Aula6 p.29–35, incl. o caso 7 de leakage) e a definição
**blending = meta em hold-out vs stacking = meta em previsões in-sample** (Aula5 p.11–13).

**D. Não-supervisionado (Aula8).** PCA/manifold/autoencoders/anomalias/clustering. Autoencoder
já rejeitado (E13, NB17-B); detecção de anomalias é a via de *row-quarantine* que foi
retractada como canal de seleção do leak (`plano_acima05.md:96`); **KMeans como features =
experimento E22 ainda "na fila"** (`plano_modelacao.md:23`).

**E. Validação e leakage — espalhado, nunca como aula própria.** Não existe nesta pasta
nenhuma folha sobre k-fold estratificado, macro-F1, desequilíbrio de classes ou calibração:
o calendário (`T/Aula 0, p.9`) coloca "Qualidade e validação de modelos" na **coluna PL
(laboratório)**, que não está na pasta. O que existe: o diagrama público/privado 30/70 com
exemplo de overfitting ao público (Aula6 p.2), a checklist de leakage das learning curves
(Aula6 p.35: duplicados entre splits, information leakage, amostragem aleatória, CV),
cross-validation na poda de árvores (Aula4 p.17) e a distinção stacking/blending (Aula5 p.11–13).

**F. Novidade (Aula6-ANN-1 p.38–39, criado 2026-11-10).** TabArena leaderboard, PFNs e TabFM —
a cadeira está a apontar explicitamente para **tabular foundation models**, o que dá suporte
institucional ao piloto TabPFN-3 já desbloqueado em teoria (`pesquisa_quebra_platamar.md:212`,
pendente de licença `TABPFN_TOKEN`).

---

## 2. APLICAÇÃO ao nosso assignment (ordenado por valor esperado)

| # | Item (fonte) | Tag | Especificação / razão |
|---|---|---|---|
| 1 | **Auditar a reivindicação "0.62 com data treatment" do colega** contra o padrão de stacking ingénuo do curso: meta-modelo treinado nas previsões **in-sample** dos base models (`T/Aula5 p.11–12`, diagrama verificado) e a checklist de leakage (`T/Aula6 p.35`) | **DIRECT** (diagnóstico, não experimento) | Ler o código/notebooks do colega e verificar 4 coisas: (a) meta/stack fitado em previsões dos base **no próprio train** (infla CV, é o padrão do slide); (b) qualquer transformação fitada **antes** do CV (encoding alvo, scaler, seleção, remoção de linhas); (c) colunas-alvo derivadas no dataset (ex. `WGT`, que calibra por faixa etária — `Notebook_01` via `pesquisa_quebra_platamar.md:143-147`); (d) regeneração de labels com o fold de validação dentro (o nosso caso relabel, `plano_acima05.md:96`). Custo ~1–2 h; ganho = explicar potencialmente **todo o gap de −0.11**; risco de leak n/a. Se o 0.62 sobreviver ao audit, a pasta /T não o explica. |
| 2 | **Piloto TabPFN-3 como família extra** — a cadeira endereça PFNs/TabFM/TabArena (`T/Aula6-ANN-1 p.38–39`) + pesquisa §8 (v3 com 160 classes, instalação já presente) | **DIRECT** | Especificação já escrita em `pesquisa_quebra_platamar.md:250-257` (fit por fold no train, CPU, barra +0.010, verificar overlap de erros antes de média). Esperado **±0.01 desconhecido** (única viés indutivo genuinamente novo); custo 1–3 h; leak **baixo**; **bloqueado** no aceitar da licença (decisão de utilizador). |
| 3 | **F2: contagens de N/A por bloco** — teoria do "mask"/"features not applicable in all cases" (`T/Aula2b p.5, p.23`) | **DIRECT** | Construir em NB09 a partir de `train_nb01.csv`: agrupar os dummies `*_Não aplicável` por domínio (trabalho/reprodução/outros) via `dicionario_variaveis.csv` (spec em `plano_acima05.md:39`). Fold-safe por construção (contagens por linha, sem rótulos). Esperado **0/+0.004**; custo ~2 h; leak **baixo**. Já está na fila F2 — esta pasta só reforça a justificação teórica. |
| 4 | **E22: features de clustering fold-safe** — redução dimensional como "preparatory tool" (`T/Aula8 p.7`) + K-Means (`T/Aula8 p.17`, heurística K≈√n p.34) | **DIRECT** | Especificação: `KMeans(n_clusters=√n_tr, random_state=42, n_init=10)` fit **só nas linhas de treino de cada fold** → transform do fold de validação → adicionar `CLUSTER_ID` (one-hot) + distância ao centróide mais próximo; 5-fold seed 42, barra +0.010. Esperado **0/+0.003** (o próprio slide avisa: K-Means "does not cope well with a very large number of features", `T/Aula8 p.19` — 311 one-hots são hostis); custo ~0.5 h; leak **baixo** se fitado in-fold (fitar no dataset todo = leak). Experimento E22 já listado como pendente (`plano_modelacao.md:23`). |
| 5 | **Scores de anomalia como features** (LOF / Isolation Forest / Tukey outlier score) — `T/Aula8 p.13–14`, `T/Aula2b p.19` | **DIRECT (com salvaguarda)** | Fold-fit do detector só no train do fold → score como feature num único membro. Esperado **0/+0.002**; custo ~1 h; leak **médio** se o score for calculado no dataset completo. ⚠️ A versão *remover linhas* é **LEAKAGE-RISK** (ver #9). |
| 6 | Diagrama público/privado 30/70 com exemplo de overfitting (`T/Aula6 p.2`) + checklist de leakage das learning curves (`T/Aula6 p.35`) + distinção blending vs stacking (`T/Aula5 p.13`) | **INDIRECT** (relatório/metodologia) | Material de citação para justificar D4 ("CV decide, Kaggle confirma", `plano_modelacao.md:100-106`) e a regra fold-regenerable (`HANDOFF.md:36-37`). Zero ganho de score; obrigação dos 20 pp. |
| 7 | CRISP-DM/SEMMA (`T/Aula2a p.16–18`), viés/tendência do Wald (`T/Aula2b p.3`), bagging/boosting bias-variance e diversidade (`T/Aula5 p.14–15`), Gini/poda (`T/Aula4 p.5, p.17`), L1/L2 (`T/Aula3 p.21`), porquê trees não precisam de scaling (`T/Aula4 p.19` vs `T/Aula3 p.22`) | **INDIRECT** (relatório) | Enquadramento teórico das escolhas já tomadas (encoding, `class_weight='balanced'`, árvore depth 8, porquê não há ganho em scaling para GBDT). |
| 8 | **Stacking "como no slide"** (`T/Aula5 p.11–12`) | **LEAKAGE-RISK** | Meta-modelo sobre previsões in-sample dos base = superestima CV. O repo já usa OOF cross-fitted (E12 `plano_modelacao.md:17-18`) e mesmo assim stacking esgotou-se. **Não implementar a variante do slide**; usá-la como explicação do 0.62 do colega (item 1). |
| 9 | **Drop/cap de outliers e "data cleaning" com deteção de anomalias antes do treino** (`T/Aula2b p.19`, `T/Aula8 p.12–14`) | **LEAKAGE-RISK** | Exatamente o canal de seleção que retractou E24–E27 (relabel/quarantine: inflação +0.036…+0.061, `plano_acima05.md:96`; `pesquisa_quebra_platamar.md:103-105`). Qualquer remoção de linhas tem de ser regenerável sem o fold de validação. Bloqueado. |
| 10 | **Feature selection por qui-quadrado/correlação com o alvo** (`T/Aula2b p.21`) | **LEAKAGE-RISK** (+ já rejeitado) | Calcular no dataset todo = canal de seleção no CV; e de qualquer forma T4/T8 já rejeitados (`pesquisa_quebra_platamar.md:262-264`). Não re-propor. |
| 11 | Escolha de modelo pelo leaderboard público (números públicos no slide `T/Aula6 p.2`) | **LEAKAGE-RISK** | O próprio slide rotula 0.85 público vs 0.785 privado de "Overfitting"; regra da casa: nunca tunar pelo público (`contexto.md:40`, `plano_modelacao.md:104-106`). Cumprida — manter. |
| 12 | Fórmulas de regressão linear/logística e OvR (`T/Aula3`), detalhes de árvores (`T/Aula4`), mecânica de ANN/backprop (`T/Aula6`), algoritmos de clustering um a um (`T/Aula8`), logística/calendário (`T/Aula 0`), quiz de taxonomia (`T/Aula2a p.13`) | **IRRELEVANT** | Famílias já testadas ou inadequadas: lineares ~0.43 (`registo_experiencias.md:61-70`), MLP/E13 rejeitado, árvore = baseline 0.407, clustering supervisionado nunca foi gargalo. |

---

## 3. Top-5 ideias concretas para fechar 0.51 → 0.62 (ângulo data-treatment/teoria)

> Honestidade primeiro: **nada nesta pasta vale +0.11 honesto**. As 5 abaixo são o melhor que
> ela oferece; a #1 é de longe a de maior valor esperado porque ataca a própria premissa do gap.

| # | Ideia | Fonte | Ganho esperado | Custo | Risco de leak |
|---|---|---|---|---|---|
| 1 | **Audit forense do "data treatment" do colega** (notebooks/PR dele) contra a checklist: stacking in-sample, transformações fitadas antes do CV, colunas-alvo (`WGT`), labels regeneradas com o fold dentro | `T/Aula5 p.11–13`; `T/Aula6 p.35` | **Explica o gap inteiro** se for artefacto (o caso relabel inflou +0.036…+0.061, `plano_acima05.md:96`; stacking ingénuo tipicamente +0.05–0.15 em CV) | 1–2 h de leitura | n/a (só leitura) |
| 2 | **TabPFN-3 piloto** como família extra (único viés indutivo novo apontado pela própria cadeira) | `T/Aula6-ANN-1 p.38–39`; `pesquisa_quebra_platamar.md:212, 250-257` | **±0.01 desconhecido** — maior probabilidade isolada de mover o teto, mas não chega a 0.62 sozinho | 1–3 h CPU + decisão de licença | baixo |
| 3 | **F2 contagens de routing por bloco** (mask theory) — spec já escrita | `T/Aula2b p.5, p.23`; `plano_acima05.md:39` | 0 / +0.004 | ~2 h (NB09) | baixo (contagens por linha) |
| 4 | **E22 KMeans fold-safe** (cluster id + distância ao centróide) | `T/Aula8 p.7, p.17, p.19, p.34`; `plano_modelacao.md:23` | 0 / +0.003 (311 one-hots são hostis a K-Means) | ~0.5 h | baixo se in-fold; **alto se fitado no dataset todo** |
| 5 | **Scores de anomalia como features** (LOF/IsolationForest/Tukey, fit por fold) — a via "limpar dados" segura | `T/Aula8 p.13–14`; `T/Aula2b p.19` | 0 / +0.002 | ~1 h | **médio** — a versão de *remover linhas* está banida (canal do leak) |

**Leitura estratégica:** se a #1 mostrar que o 0.62 é artefacto (muito provável dado o histórico
relabel + a forma como o curso ensina stacking), o caminho até lá é fechar as portas #3–#5
(ganhos somados plausíveis +0.005…+0.015) e decidir a licença TabPFN (#2) — **não** repetir
qualquer coisa da lista "do not bother" (`pesquisa_quebra_platamar.md:259-275`), que esta
pasta inteira reconfirma como teoria de manual sem novidade aplicável.

---

## 4. Não foi possível verificar

- **Notas de orador / slides omitidos:** os PDFs são exports de PPTX sem notas; não dá para
  confirmar se os slides têm texto adicional falado em aula (onde poderia estar "a receita").
- **Páginas só com imagem não amostradas** (caracterizadas por título + padrão, sem render):
  `T/Aula 0 p.5` (imagem decorativa "Overload"), `T/Aula2a p.2, p.5` (divisores de secção),
  `T/Aula2b p.13` (imagem de data viz), `T/Aula6 p.3` (divisor "Contents") e
  `T/Aula8 p.25/27/29/31/33` (mesmos grids de demonstração do sklearn já vistos em p.20/p.23).
  Renders efetuados e lidos: `T/Aula2a p.15–16`, `T/Aula2b p.9`, `T/Aula4 p.9–10`,
  `T/Aula5 p.12`, `T/Aula6 p.10, p.28`, `T/Aula8 p.14, p.20, p.23` — todos consistentes com o texto.
- **Ausências na pasta:** não existe "Aula 1" nem "Aula 7" (numeração com lacuna) e **não há
  nenhum deck de validação/métricas** — o calendário (`T/Aula 0 p.9`) coloca "Qualidade e
  validação de modelos" na coluna PL; os materiais de laboratório não estão nesta pasta e não
  puderam ser inspecionados.
- **Conteúdo externo à pasta:** a reivindicação de 0.62 do colega não pode ser verificada a
  partir de /T (depende do código dele — ver §2 #1).
- **Ferramentas:** sem `pdftotext`/`pdfinfo`/tesseract no sistema; texto via `pypdf`, páginas
  via Ghostscript + leitura visual; sem OCR (irrelevante — os renders foram lidos como imagem).

---

## 5. Log

| Data | Item | Resultado |
|---|---|---|
| 11 nov 2026 | Pasta `DAA/T` mapeada por completo: 9 PDFs / 231 páginas, todos extraídos e lidos; mtimes 23:02–23:04 (copiados hoje), criação interna 21/09 → 10/11/2026 | Este documento |
| 11 nov | `Aula6-ANN-1` confirmado por diff como superset do `Aula6-ANN` — **2 slides novos p.38–39: TabArena / PFN / TabFM**; `Aula8` também criado 10/11 | §MAPA F; candidato Top-5 #2 |
| 11 nov | Diagrama de stacking (`T/Aula5 p.12`) verificado por render: meta-features geradas no **mesmo** Training Data (sem cross-fitting) → **stacking ingénuo = padrão de inflação de CV** | Top-5 #1 (audit ao colega) |
| 11 nov | Confirmado que **não há** deck de validação/métricas na pasta (está na coluna PL, `T/Aula 0 p.9`); nenhum conteúdo sobre macro-F1, desequilíbrio ou ordinalidade | §4 |
| 11 nov | Inventário APLICAÇÃO taggeado: 6 DIRECT, 3 INDIRECT, 4 LEAKAGE-RISK, restantes IRRELEVANT; **nenhuma ideia de +0.11 encontrada** — veredito do cabeçalho mantido | §2, §3 |
