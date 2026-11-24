# Previsão da faixa etária a partir do Inquérito Nacional de Saúde 2019

Trabalho de grupo de Dados e Aprendizagem Automática, 2026/27. Versão de 24 de novembro de 2026.

## Resumo

O objetivo foi prever a faixa etária de 16 classes a partir do perfil de saúde e socioeconómico do INS 2019, e usar as previsões para encontrar perfis incoerentes e caracterizar as faixas. A métrica é o Macro-F1. O modelo final é um conjunto de três famílias ordinais de Frank e Hall, com 165 classificadores binários, média de probabilidades e decisão por argmax ou mediana. A CV observada é 0,5089 e o resultado público é 0,51. O valor 0,5089 foi escolhido entre variantes e limiares avaliados nas mesmas folds, por isso é um máximo observado e não uma estimativa sem seleção. A medição honesta do mesmo tipo de receita, sem o canal do relabel, dá 0,504 a 0,509. A linha de relabel que chegou a 0,52 a 0,546 está retratada, com a submissão 11 a confirmar o vazamento no leaderboard público. A análise de erros mostra que o erro é de sinal: 72,3% dos erros do XGB-311 honesto são de uma faixa e os blocos 4 a 7 e 11 a 14 concentram 68,9% do erro. O clustering não recupera as faixas, o que sustenta a leitura de que o sinal etário está difuso. O relatório separa a base 339, usada pela entrega, da base 311, usada pela análise e pelas experiências do ramo.

## 1. Âmbito e objetivos

O projeto tem duas componentes. A primeira é a competição Kaggle ML@DAA 2026/27, com 10 962 pessoas de treino e 3 655 de teste, 236 variáveis explicativas e o alvo `AGE_RANGE` em 16 faixas, de 0 a 15. A segunda é a análise pedida no enunciado: identificar perfis anómalos, ou seja, pessoas cujo perfil biológico e social corresponde a outra faixa, e caracterizar cada faixa com os indicadores disponíveis.

A métrica oficial é o Macro-F1. Cada uma das 16 classes pesa o mesmo, o que torna a métrica sensível às classes pequenas. A classe 1, de 18 a 19 anos, tem 172 casos e a classe 10, de 60 a 64 anos, tem 1 062. O rácio é 6,17. As referências triviais são 0,011 para a classe maioritária, 0,060 para um classificador aleatório estratificado e 0,0625 para 1 em 16. O resultado final de 0,51 fica muito acima destas referências, mas longe de uma classificação fiável para uso individual.

A competição permite três submissões por dia. O resultado final é 30% da tabela pública e 70% de um split privado. A regra do grupo foi nunca escolher um modelo pela tabela pública. A CV decide e o público confirma, como está em `docs/plano_modelacao.md` D4.

## 2. Dados e processamento

### 2.1 Fonte e limpeza

Os dados vêm do Inquérito Nacional de Saúde de 2019, realizado pelo INE. O ficheiro de treino tem 10 962 linhas e 237 colunas, incluindo o alvo. O de teste tem 3 655 linhas e 236 colunas. As variáveis cobrem demografia, trabalho, escolaridade, estado de saúde, doenças crónicas, limitações, autonomia, cuidados de saúde, hábitos e características do agregado.

O Notebook 1 faz a limpeza. A coluna `WGT` foi excluída porque o peso oficial é calibrado por grupo etário quinquenal e codifica o alvo. Os códigos `Sem resposta`, `NR` e `Não aplicável` foram preservados como valores distintos, porque a diferença entre eles transporta informação sobre o routing do questionário. As conversões de tempo, escolaridade e idade declarada estão documentadas no notebook. O resultado é `train_nb01.csv`, com 260 features e o alvo.

O routing é uma parte central dos dados e uma limitação. O questionário tem 133 perguntas universais, 31 com gate de 55 anos ou mais, 51 parciais, 10 raras e 2 com gate de sexo (`docs/mapa_routing.md`). Uma pergunta não feita produz `Não aplicável`, e isso revela a idade. Os dois cortes oficiais são aos 55 anos, no bloco G e na `PC1A`, e aos 65 anos, quando `HH_ACT + HH_INACT == 0`. O segundo corte vale para 3 202 pessoas e tem duas exceções, ambas na faixa 15 a 17.

### 2.2 Codificação, seleção e features

O Notebook 2 codifica as variáveis categóricas. As 205 ordinais e binárias ficam numa coluna cada, as 22 nominais passam a 171 colunas one-hot e as 33 numéricas ficam como estão. O total é 409 features.

A seleção remove 70 colunas quase constantes e 6 duplicadas. O conjunto passa a 333. A criação de features acrescenta `N_DIFICULDADES`, `N_DOENCAS`, `IMC` e `N_NAO_APLICAVEL`, e o total passa a 337. A idade mínima declarada acrescenta `IDADE_MINIMA` e `FAIXA_MINIMA`, e o total passa a 339. A validação dos limites usa `IN2`, `IN3`, `IN4`, `IN25`, `IN26`, `IN45`, `IN47` e `IN24 + 15`. A `IN25` foi rejeitada por ter 2,8% de valores acima do limite superior da faixa. A tabela seguinte resume o contributo de cada etapa, medido com o mesmo HistGradientBoosting e as mesmas folds.

| Etapa | Features | Macro-F1 | Desvio padrão |
|---|---|---|---|
| Apenas codificação | 409 | 0,4625 | 0,0051 |
| + seleção | 333 | 0,4688 | 0,0035 |
| + contagens e IMC | 337 | 0,4663 | 0,0085 |
| + idade mínima | 339 | 0,4742 | 0,0073 |

A idade mínima é a alteração com maior efeito. O HistGradientBoosting não consegue calcular o máximo entre várias colunas, e a feature entrega-lhe essa combinação. As contagens e o IMC não melhoram o modelo sozinhas, mas ficam por serem baratas e por ajudarem modelos lineares.

O ramo de análise usa uma segunda base. O Notebook 2 §4.1 remove 28 colunas dos blocos `PE`, `PN`, `SS` e `MH`, que o Notebook 9 identificou como ruído. O conjunto passa a 311 features. Esta remoção deu +0,0066 em 5 de 5 folds no ecrã do Notebook 9 e foi adotada para as experiências dos Notebooks 16 a 18. A base 311 não é a base 339 menos 28 colunas. O `N_NAO_APLICAVEL` é recalculado depois do corte e muda em 10 702 linhas. Para reproduzir a base 339 é preciso correr o Notebook 2 anterior ao §4.1, não concatenar colunas.

O relatório usa duas bases ao longo do texto. A base 339 é a da entrega, do Notebook 3 e do Notebook 4. A base 311 é a das experiências controladas dos Notebooks 16 a 18. Cada tabela de resultados indica a base. Esta separação é uma limitação de reprodutibilidade e é discutida no §9.

### 2.3 Verificações de coerência

O treino e o teste não têm shift relevante. A diferença média das distribuições é de 0,16 pontos percentuais. A auditoria de valores impossíveis encontrou sete linhas no treino, entre elas uma pessoa de 75 a 79 anos que declara ser estudante, uma pessoa reformada com menos de 35 anos e uma com `FAIXA_MINIMA` acima da faixa real. A remoção destas linhas foi testada e deu 0,4608 contra 0,4650, uma diferença dentro do ruído. As linhas ficaram no treino e a lista vive no relatório, não no pipeline.

## 3. Metodologia

### 3.1 Protocolo de validação

Todas as comparações usam a mesma validação cruzada estratificada de 5 partes, com shuffle e semente 42. A métrica é o Macro-F1. Antes de gastar um 5-fold, cada ideia passa por um ecrã num holdout único de 80/20, com uma barra de +0,010. A adoção exige 5 de 5 folds. O ruído de uma fold é de ±0,004 e o desvio padrão entre folds fica entre 0,008 e 0,012. As exceções de 4 de 5 folds estão marcadas nos documentos de origem.

O ecrã de holdout tem 2 193 linhas e foi usado em cerca de 30 variantes. Sob a hipótese nula, o máximo esperado de 30 variantes ronda +0,02. Por isso, os valores de +0,0162 do T10 e +0,0210 do T14 são ecrãs e não ganhos. Os valores que contam são os do 5-fold: +0,0077 para o T10/P1 e +0,0058 para o T14/P4, ambos abaixo da barra. Esta distinção está no §5.3.

### 3.2 Regra fold-safe

A regra adotada é que tudo o que é usado dentro da CV tem de ser regenerável sem o fold de validação. A regra nasceu da retração do relabel. Quando os rótulos de treino são substituídos por previsões de um modelo que viu o fold de validação, a CV deixa de medir generalização. O mesmo vale para a quarentena que escolhe as linhas piores por confiança calculada no dataset inteiro. O Notebook 16 mede o canal e repete as receitas com rótulos regenerados dentro de cada fold. O resultado está no §5.3.

A regra também se aplica ao clustering e ao autoencoder do §7. O scaler e o modelo não supervisionado ajustam-se ao treino de cada fold. A validação é atribuída depois. Os rótulos não entram no ajuste.

### 3.3 Submissões

As submissões são manuais. O notebook gera o CSV e o utilizador submete, para não gastar as três submissões diárias em execuções automáticas. A tabela do §4.3 tem as sete submissões. A sub08 foi gerada e validada, mas nunca foi submetida. A sub09 e a sub10 têm registos contraditórios nos documentos e não são usadas como evidência. A sub11 foi submetida e retratada.

## 4. Modelos

### 4.1 Modelos de referência

A primeira comparação usou sete famílias com os mesmos dados e as mesmas folds. A tabela seguinte tem o resultado agregado da CV e o resultado depois da correção `max(previsão, FAIXA_MINIMA)`.

| Modelo | Macro-F1 | Com correção | Previsões corrigidas |
|---|---|---|---|
| Gradient Boosting | 0,4668 | 0,4670 | 7 |
| Random Forest | 0,4497 | 0,4516 | 30 |
| Regressão Logística | 0,4272 | 0,4362 | 141 |
| SVM RBF | 0,4201 | 0,4272 | 113 |
| Árvore de decisão | 0,4071 | 0,4116 | 74 |
| KNN, k=15 | 0,2939 | 0,3062 | 240 |
| Naive Bayes | 0,2726 | 0,2808 | 165 |

A árvore de decisão foi otimizada em profundidade. A melhor é `max_depth=8`, com 0,4072 ± 0,0057. Profundidades maiores overfitam. O Gradient Boosting é o melhor modelo de referência. A pesquisa de hiperparâmetros escolheu `learning_rate=0,05`, `max_leaf_nodes=15`, `min_samples_leaf=20` e `l2_regularization=1`, com 156 árvores por paragem antecipada. A pesquisa refinada não melhorou o resultado e não foi adotada.

### 4.2 Formulação ordinal

O alvo é ordenado e os erros de uma faixa dominam. A formulação ordinal de Frank e Hall transforma o problema em 15 perguntas binárias, uma por cada fronteira: a faixa é maior do que k? As probabilidades acumuladas são forçadas a ser monótonas e depois convertidas nas 16 probabilidades de classe. A formulação melhora o Macro-F1 porque aproveita a ordem e porque cada classificador binário tem um problema mais fácil do que a classificação de 16 classes.

O XGBoost ordinal com 5 sementes dá 0,504 com argmax e correção. O XGBoost multiclasse de 16 classes dá 0,480. A diferença de +0,024 é o maior ganho isolado do projeto e aparece em 5 de 5 folds. A combinação com o regressor HGB, com limiar 0,35 escolhido nas folds de treino, dá 0,5051. Esta é a submissão 6.

A submissão 7 acrescenta duas ideias. A primeira é repetir as 15 perguntas com LightGBM e HistGradientBoosting, com parâmetros equivalentes. As três famílias têm 5, 4 e 2 sementes, o que dá 165 classificadores binários. As probabilidades das três famílias são somadas e divididas por três. A segunda ideia é a decisão. Quando a probabilidade máxima é igual ou superior a 0,40, usa-se a faixa mais provável. Abaixo disso, usa-se a mediana das probabilidades, ou seja, a faixa onde a probabilidade acumulada passa 50%. No fim aplica-se `max(previsão, FAIXA_MINIMA)`.

No teste, 80,5% das decisões usam o argmax, 19,5% usam a mediana e 4 previsões são corrigidas pela idade mínima. Na execução original, as 3 655 previsões são iguais às do ficheiro submetido. A média das três famílias com mediana dá 0,5089. As famílias isoladas com mediana dão 0,5065 para o XGBoost, 0,5044 para o LightGBM e 0,5006 para o HistGradientBoosting. A concordância na faixa mais provável entre as três é 77,7%. A média compensa parte dos erros de cada família. Estes valores são os da execução que gerou o ficheiro entregue; a re-execução difere em menos de 0,002 (ver §9).

### 4.3 Submissões

| # | Data | Modelo | Base | CV | Público | Ficheiro |
|---|---|---|---|---|---|---|
| 1 | 25/10 | Árvore de decisão, profundidade 8 | 409, versão anterior | 0,407 | 0,40 | `sub01_arvore_referencia.csv` |
| 2 | 26/10 | Regressão Logística, C=0,01 | 409, versão anterior | 0,426 | 0,43 | `sub02_logistica.csv` |
| 3 | 26/10 | Gradient Boosting por defeito, com correção | 409, versão anterior | 0,467 | 0,46 | `sub03_gradient_boosting.csv` |
| 4 | 26/10 | Gradient Boosting otimizado, com correção | 339 | 0,475 | 0,45 | `sub04_gradient_boosting_otimizado.csv` |
| 5 | 27/10 | XGBoost com regressor HGB, limiar 0,35 | 339 | 0,489 | 0,48 | `sub05_xgb_regressor.csv` |
| 6 | 27/10 | Ordinal XGBoost com regressor, limiar 0,35 | 339 | 0,5051 | 0,49 | `sub06_ordinal_regressor.csv` |
| 7 | 28/10 | Três ordinais com mediana, limiar 0,40 | 339 | 0,5089 | 0,51 | `sub07_ordinal_3modelos_mediana.csv` |

A CV da sub02 aparece como 0,426 no registo de submissão e 0,4272 no quadro agregado do Notebook 3. A diferença é de arredondamento e não muda a ordem. A sub04 mostra o ruído da tabela pública. A CV deu 0,475 e o público deu 0,45. A decisão fica com a CV. A sub06 e a sub07 confirmam a ordenação. A sub08 do metastack E12, no ficheiro `sub08_metastack.csv`, foi gerada e validada, mas nunca foi submetida. O valor 0,5037 que aparece em alguns documentos é o E12 e não o melhor modelo do projeto. O melhor modelo é a sub07.

A ressalva de seleção aplica-se a esta tabela. O Notebook 3 escolhe os hiperparâmetros do Gradient Boosting com `RandomizedSearchCV` nas mesmas folds que reporta, escolhe a melhor de seis variantes ordinais nas mesmas folds e escolhe o limiar 0,40 nas mesmas folds. A célula 22 chama "aninhado" ao limiar, mas as probabilidades OOF usadas para o escolher foram geradas por modelos que viram o fold de avaliação. O valor 0,5089 é um máximo observado. O relatório escreve "CV observada" e não "estimativa sem enviesamento". O público de 0,51 é a confirmação externa.

### 4.4 Hiperparâmetros

| Modelo | Parâmetros |
|---|---|
| Árvore de decisão | `max_depth=8`, `min_samples_leaf=10` |
| Regressão Logística | `C=0,01`, `class_weight='balanced'`, `max_iter` por defeito |
| Gradient Boosting | `learning_rate=0,05`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `l2_regularization=1`, `max_iter=500`, paragem antecipada, 156 árvores |
| XGBoost multiclasse | `n_estimators=350`, `learning_rate=0,05`, `max_depth=4`, `min_child_weight=3`, `subsample=0,8`, `colsample_bytree=0,5` |
| XGBoost ordinal | 15 binários por semente, `n_estimators=300`, `learning_rate=0,05`, `max_depth=4`, `min_child_weight=3`, `subsample=0,8`, `colsample_bytree=0,5`, sementes 42, 1, 2, 3, 4 |
| LightGBM ordinal | 15 binários por semente, `n_estimators=300`, `learning_rate=0,05`, `num_leaves=15`, `min_child_samples=20`, `subsample=0,8`, `subsample_freq=1`, `colsample_bytree=0,5`, `reg_lambda=1`, sementes 42, 1, 2, 3 |
| HistGradientBoosting ordinal | 15 binários por semente, `max_iter=300`, `learning_rate=0,05`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `l2_regularization=1`, sem paragem antecipada, sementes 42 e 1 |
| Regressor da sub06 | HistGradientBoostingRegressor, `max_iter=500`, paragem antecipada, limiar 0,35 |
| Decisão final | argmax se `max(p) ≥ 0,40`, senão mediana; depois `max(previsão, FAIXA_MINIMA)` |

Todos os classificadores usam pesos `balanced`, calculados com `compute_sample_weight('balanced', y)`. O peso é importante para o Macro-F1. Sem pesos, o modelo ordinal perde cerca de 0,006.

### 4.5 Ablação

A tabela seguinte separa o contributo dos dados do contributo da formulação. A primeira parte vem do Notebook 2 §7 e usa o mesmo HistGradientBoosting. A segunda vem do Notebook 3 §7.4 a §7.7 e usa o mesmo XGBoost e as mesmas folds.

| Alteração | Macro-F1 | Delta |
|---|---|---|
| Codificação, 409 features | 0,4625 | referência |
| + seleção, 333 | 0,4688 | +0,0063 |
| + contagens e IMC, 337 | 0,4663 | −0,0025 |
| + idade mínima, 339 | 0,4742 | +0,0079 |
| XGBoost multiclasse, 16 classes | 0,480 | referência |
| Ordinal XGBoost, 5 sementes | 0,504 | +0,024 |
| Ordinal + regressor, sub06 | 0,5051 | +0,0011 |
| Média de 3 famílias + mediana, sub07 | 0,5089 | +0,0038 |

O ganho grande vem da formulação ordinal. A idade mínima é o segundo maior ganho. As contagens e o IMC não ajudam o Gradient Boosting, mas ficam porque são baratas e porque ajudam a análise descritiva.

### 4.6 Alternativas testadas e rejeitadas

A tabela condensa as experiências com número. A lista completa está em `docs/plano_acima05.md` §6 e `docs/pesquisa_quebra_platamar.md` §13.

| Família | Experiência | Resultado | Veredito |
|---|---|---|---|
| Decision-time | M1, thresholds de Frank e Hall com validação aninhada | 0,0058 abaixo em 1 de 5 folds | rejeitado |
| Decision-time | M2, offsets por classe | 0,0072 abaixo em 2 de 5 | rejeitado |
| Decision-time | M5, pair-flips | 0,0003 abaixo, zero flips | rejeitado |
| Restrições | M3, restrições monotónicas | 0,0011 abaixo e 0,0003 acima | rejeitado |
| Ordinal | Pairwise-15, T10 e P1 | +0,0162 no holdout, 0,4854 em 5-fold | não empilha na sub07, S1b 0,0050 abaixo |
| Ordinal | C3, target-mean com pairwise-15 | 0,4886, +0,0109 em 4 de 5 | confirmado, cerca de 0,02 abaixo da sub07 |
| Labels | Relabel E26 e E27 | 0,5364 e 0,5460 com relabel, 0,498 a 0,508 honestos | retratado |
| Labels | Quarentena honesta T6 | +0,0069 | abaixo da barra |
| Ensemble | E12 e E12b, metastack | 0,5037 e 0,5027 | congelado, superado pela sub07 |
| Ensemble | C6, kitchen sink | 0,4787 | combinações fechadas |
| Bagging | T15, 10 sementes com 80% das linhas | 0,4650, 0,0050 abaixo | rejeitado |
| Ranker | T16, lambdarank | 0,10 a 0,11 | rejeitado |
| Clustering | T18, distâncias ao KMeans fold-safe | 0,4691, −0,0009 | rejeitado, E22 fechada |
| Clustering | T18b, erro de reconstrução PCA | 0,4736, +0,0036 | abaixo da barra, autoencoder não treinado |
| Foundation | T17, TabPFN-3.5 | 0,4805, +0,0105 | passou o ecrã |
| Foundation | T17b, mistura XGBoost com TabPFN | 0,4833, +0,0133 | passou o ecrã, 5-fold não corrido |
| Foundation | S2, receita sub07 com TabPFN | 0,5037, +0,0040 contra S1 | abaixo da barra |
| Outros | Self-training, Naive Bayes, BRF, DART e MLP | neutros ou negativos | rejeitados |
| Hierarquia | Três especialistas com meta-logística | +0,009 contra o global | abaixo da receita XGBoost |

As duas famílias com margem teórica, pairwise-15 e TabPFN, foram testadas e não passam a barra. A única linha que batia o teto era vazamento. O T17b tem um ecrã de +0,0133, mas é um de cerca de 30 ecrãs no mesmo holdout. O 5-fold não foi corrido. O relatório apresenta-o como ecrã e não como ganho.

## 5. Resultados e análise crítica

### 5.1 Modelo final

A sub07 tem 0,5089 de CV observada e 0,51 no público. O ficheiro `submissions/sub07_ordinal_3modelos_mediana.csv` tem 3 656 linhas, cabeçalho `RowId,AGE_RANGE`, RowId de 0 a 3654 e 16 classes. O SHA-256 do ficheiro é `2782b8823463e9562b7f92942caa5fc007a56f69a5f4b5fe3931fb0e3f59f88d`.

A OOF guardada da sub07, com a regra final de decisão (limiar 0,40, mediana e piso), tem **0,5087** de Macro-F1, 50,2% de acerto exato e 88,9% a uma faixa ou menos. O erro de uma faixa é 77,8% dos erros. Há 235 pessoas com |gap| ≥ 3, 87 envelhecidas e 148 rejuvenescidas, e 61 com |gap| ≥ 4. A entrega original tem 0,5089 de CV observada; esta OOF é a da re-execução, com 0,5087.

A tabela seguinte tem o F1 por classe na OOF da sub07 e na OOF da sub06, com o suporte de cada classe no treino.

| Faixa | n | F1 sub06 | F1 sub07 | Delta |
|---|---|---|---|---|
| 15-17 | 207 | 0,860 | 0,869 | +0,009 |
| 18-19 | 172 | 0,470 | 0,467 | −0,003 |
| 20-24 | 399 | 0,630 | 0,636 | +0,006 |
| 25-29 | 377 | 0,430 | 0,432 | +0,002 |
| 30-34 | 449 | 0,330 | 0,339 | +0,009 |
| 35-39 | 647 | 0,380 | 0,385 | +0,005 |
| 40-44 | 859 | 0,390 | 0,400 | +0,010 |
| 45-49 | 862 | 0,420 | 0,431 | +0,011 |
| 50-54 | 905 | 0,670 | 0,655 | −0,015 |
| 55-59 | 1014 | 0,690 | 0,676 | −0,014 |
| 60-64 | 1062 | 0,680 | 0,680 | 0,000 |
| 65-69 | 1040 | 0,560 | 0,555 | −0,005 |
| 70-74 | 1000 | 0,370 | 0,390 | +0,020 |
| 75-79 | 823 | 0,330 | 0,352 | +0,022 |
| 80-84 | 650 | 0,370 | 0,389 | +0,019 |
| 85+ | 496 | 0,500 | 0,483 | −0,017 |
| **Macro** | **10962** | **0,5051** | **0,5087** | **+0,0036** |

A coluna da sub06 vem do quadro do Notebook 4, que usa a OOF de 0,5051. A coluna da sub07 vem da OOF guardada na re-execução, com 0,5087. A entrega original tem 0,5089 de CV; a diferença está no §9.

![F1 por classe na OOF da sub07](fig_sub7_f1.png)

A figura ordena as classes pela dificuldade. As classes 4 a 7 e 12 a 14 ficam abaixo de 0,45. A classe 11 fica em 0,56 e a 0 em 0,87.

A matriz de confusão da sub07 está na figura seguinte. As linhas são a faixa real e as colunas a faixa prevista. A diagonal é o acerto.

![Matriz de confusão da sub07](fig_sub7_confusao.png)

As seis maiores confusões fora da diagonal são 55-59 para 60-64: 330; 65-69 para 70-74: 289; 70-74 para 75-79: 287; 60-64 para 55-59: 281; 50-54 para 45-49: 248; 45-49 para 40-44: 235.

### 5.2 Estrutura do erro

A análise honesta usa o XGB-311 e está em `docs/diagnostico_erros.md`. Há 5 662 erros em 10 962 linhas, com Macro-F1 0,4782. Dos 5 662 erros, 4 093 são de uma faixa, ou 72,3%. Os outros 1 569, ou 27,7%, são de duas ou mais. Na sub06 a proporção de erros de uma faixa é próxima de 76%, porque a OOF da sub06 tem 50,0% de acerto exato e 38,0% de erro de uma faixa. Na OOF da sub07 é 77,8%, com 50,2% de acerto exato e 38,7% de erro de uma faixa. O valor de 72,3% é o do XGB-311 e não se aplica aos modelos ordinais.

No XGB-311, os blocos 11 a 14 concentram 38,1% do erro, com 3 513 linhas e F1 de 0,527, 0,369, 0,273 e 0,348. Os blocos 4 a 7 concentram 30,8%, com 2 817 linhas e F1 de 0,304, 0,379, 0,381 e 0,421. Juntos valem 68,9% do erro. A confusão 9 e 10 vale 11,2%, com 635 casos, apesar de as duas classes terem F1 de 0,668 e 0,645. A classe 13 é a pior, com F1 0,273 e recall 0,256. A classe 1 vale 2,1% do erro, com F1 0,354. As melhores classes são a 0, com 0,854, a 9, com 0,668, a 8, com 0,663, a 10, com 0,645 e a 2, com 0,604.

As três maiores confusões adjacentes são 9 e 10, com 635 casos, 11 e 12, com 538, e 12 e 13, com 464. Juntas somam 28,9% dos erros. No XGB-311, a fronteira 8 e 9 tem zero confusão nos dois sentidos. Não vale gastar orçamento de fronteira aí. A matriz da sub07, no §5.1, desloca as maiores confusões para 55-59 e 60-64 e para os pares 65-74 e 70-79. A estrutura de erro não é a mesma do XGB-311, o que é esperado depois da formulação ordinal e da média de três famílias.

![Distribuição do erro absoluto na OOF da sub07](fig_sub7_gap.png)

A figura mostra a concentração do erro em zero, uma e duas faixas. A cauda acima de três faixas é pequena e é a que alimenta a análise de anomalias.

Não há um bolsão removível de baixa confiança. Os 5% menos confiantes concentram 7,6% dos erros, com taxa de 0,78 contra 0,52 no global. Os 20% menos confiantes concentram 27,7%. Os 5% mais confiantes ainda erram 9,5%, ou 52 linhas. A calibração não resolve estes casos. O diagnóstico do Notebook 8 mostra que 42,9% dos erros são casos em que as quatro bases concordam e erram e que em 66,7% nenhuma base acerta. A diversidade de vistas está esgotada.

### 5.3 Seleção, teto honesto e retração

O valor 0,5089 é um máximo observado. O Notebook 16 repete as receitas com os rótulos regenerados dentro de cada fold. A tabela seguinte compara o valor antigo com o par honesto da mesma receita. A coluna "com relabel" é o valor que circulou nos Notebooks 13 e 15. A coluna "par honesto" usa a mesma receita, regenera os rótulos sem o fold de validação e avalia com os rótulos originais.

| Receita | Com relabel | Par honesto | Inflação |
|---|---|---|---|
| `flat`, mediana a 0,40 | 0,5409 | 0,5007 | +0,0402 |
| `LGB`, mediana a 0,45 | 0,5417 | 0,5040 | +0,0377 |
| `FH`, mediana a 0,35 | 0,5364 | 0,5003 | +0,0361 |
| `HGB` | 0,5355 | 0,4909 | +0,0446 |
| `B1`, quatro famílias | 0,5460 | 0,4981 | +0,0479 |
| `mc5` | 0,5376 | 0,4766 | +0,0610 |

O melhor fallback honesto de cada receita original é 0,5083 para o `orig-LGB` com mediana a 0,45, 0,5067 para o `orig-B1` com mediana a 0,40, 0,5064 para o `orig-flat` com mediana a 0,40, 0,5035 para o `orig-FH` com mediana a 0,35, 0,4979 para o `orig-HGB` e 0,4849 para o `orig-mc5`. O intervalo 0,504 a 0,509 resume estes valores. A sub07, com 0,5089, e o público, com 0,51, são consistentes com ele.

A retração tem duas evidências. A primeira é a sub11. A sub11 foi submetida e deu 0,48 público contra 0,546 estimados. O gap de −0,066 é muito acima do histórico de ±0,016 e mostra que a CV estava inflacionada. A segunda é o Notebook 16, que mede o canal em 32,4%. O canal tem uma ressalva. O relabel original escolhe as linhas pior classificadas pela probabilidade do rótulo observado. O CV honesto do Notebook 16 escolhe os 15% pior por confiança máxima. Os dois conjuntos só se sobrepõem em 269 de 1 644 linhas, ou 16,4%. O mecanismo ainda precisa de uma repetição com a mesma regra. A retração mantém-se pela sub11. O Notebook 16 não prova sozinho o mecanismo.

A leitura para o relatório é que o teto honesto é 0,504 a 0,509. O valor 0,5089 está nesse intervalo. A diferença para o público é pequena. Não há evidência de que exista um caminho de +0,11 escondido nos dados.

### 5.4 Interpretabilidade

O Notebook 4 usa a OOF da sub06, com 0,5051 e base 339. O SHAP usa um XGBoost plano de 16 classes, com 0,480, e não a sub07. A importância global em percentagem do total é 6,9 para `MAINSTAT`, 5,5 para `IN15`, 5,3 para `PC1A`, 4,8 para `MARSTALEGAL`, 4,0 para `PC1C`, 3,5 para `PC1D`, 3,4 para `HH_INACT`, 3,0 para `PC1E`, 2,9 para `PC1B`, 2,9 para `HATLEVEL`, 2,5 para `IDADE_MINIMA` e 2,4 para `HH_ACT`. As 20 primeiras explicam 60% do total.

O bloco do corte dos 55, com `IN15`, `PC1A` a `PC1E`, `HA1B` e `HA1G`, vale cerca de 28% da importância. O bloco de routing mais largo, com `PC1*`, `HA1*`, `IN15` e `IN11`, vale cerca de 36,2% da massa absoluta de SHAP. Este segundo número foi medido no modelo relabel do Notebook 14 e é uma estimativa. O ranking SHAP do modelo original e do relabel tem rho 0,977, por isso a estimativa é estável, mas não é uma medição no modelo final.

![Importância global em SHAP do XGBoost de 16 classes, base 339, OOF da sub06](fig_sub6_shap_global.png)

A figura mostra a média do valor absoluto de SHAP em percentagem do total. O modelo é o XGBoost plano de 16 classes do Notebook 4, com 0,480, e não a sub07. Serve para ler o sinal, não para medir o modelo final.

A concordância entre SHAP e permutação é rho 0,468 no ranking de 311 features do modelo relabel do Notebook 14. O `PC1A` é rank 1 em SHAP e 11 em permutação. O `IN15` tem 0,133 em SHAP e 0,001 em permutação. O `HA1B` tem 0,149 e 0,0087. O `PC1C` tem 0,107 e 0,004. O routing identifica quem respondeu o quê, mas não separa idades adjacentes dentro dos blocos 4 a 7 e 11 a 14. A importância deve ser lida como descritiva e não causal.

O segundo corte, aos 65 anos, aparece quando `HH_ACT + HH_INACT == 0`. Vale para 3 202 pessoas, com duas exceções na faixa 15 a 17. Como correção pós-previsão, o resultado passa de 0,5051 para 0,5052, o que é ruído. Não foi adotada. As 43 pessoas com `Sem resposta` no bloco dos 55 ou mais são todas de 55 ou mais. O modelo prevê menos de 55 em quatro delas. São artefactos do modelo e não perfis anómalos.

![O que separa menos de 65 de 65 ou mais, valores SHAP](fig_sub6_fronteiras.png)

A figura mostra as variáveis que separam as duas metades no modelo do Notebook 4. O `HH_INACT` e o `HH_ACT` dominam. O corte é binário e não ajuda a separar idades dentro de cada metade, o que explica por que as classes 11 a 14 continuam difíceis.

## 6. Anomalias e caracterização

### 6.1 Anomalias da sub06

O Notebook 4 combina regras lógicas com um critério de probabilidade. As regras base encontram 16 incoerências, quatro impossíveis, nove muito improváveis e três raras. A regra da menopausa declarada antes dos 40 acrescenta três pessoas, duas delas novas. As regras ficam com 18 pessoas. O critério do modelo encontra 36 pessoas com `P_DECLARADA < 0,05`. As duas listas sobrepõem-se em três pessoas. O total é 51 pessoas, ou 0,47% do treino. Destas, sete são prováveis erros de registo, 40 são perfis excecionais e quatro são artefactos do modelo. As 43 pessoas com `Sem resposta` no bloco dos 55 ou mais são uma população separada. Só quatro estão entre as 51.

Esta análise usa a sub06, com 0,5051 e base 339. É a análise de anomalias da entrega do colega e está congelada no Notebook 4. O relatório declara o modelo.

### 6.2 Anomalias da sub07

Na OOF da sub07 (re-execução, 0,5087), 235 pessoas têm |gap| ≥ 3, das quais 87 foram previstas mais velhas e 148 mais novas. Destas, 61 têm |gap| ≥ 4. As rejuvenescidas têm em média 1,39 comorbilidades, contra 2,58 no grupo normal; as envelhecidas ficam perto do normal, com 2,62. O valor é próximo das 260 anomalias da OOF do modelo combinado, com 0,4704. As 51 da sub06 usam outra definição (`P_DECLARADA < 0,05`) e não se comparam diretamente. O relatório usa o número da sub07 quando fala do modelo final e declara o modelo quando cita os outros dois.

![Anomalias de perfil na OOF da sub07](fig_sub7_anomalias.png)

A figura mostra os 10 727 casos normais, 148 rejuvenescidos e 87 envelhecidos, e as comorbilidades médias de cada grupo. O eixo vertical do painel da esquerda é logarítmico para os três grupos serem visíveis.

### 6.3 Retrato das 16 faixas

O retrato usa os rótulos reais e 16 indicadores sociais e de saúde. A faixa 15-17 tem 99% de estudantes, 91% de saúde boa ou muito boa e 77% de prática de desporto. A faixa 18-24 é a transição: 76% ainda estudam aos 18-19 e 46% já trabalham aos 20-24. Entre os 25 e os 49 anos, 75% a 87% trabalham, o casamento passa a maioritário a partir dos 35 anos, as crianças pequenas em casa chegam a 63% aos 35-39 e o ensino superior fica perto de 30%. A faixa 50-54 ainda tem 77% de empregados e 2,2 doenças crónicas em média. Entre os 55 e os 64, o emprego cai de 68% para 41% e a reforma sobe de 7% para 31%. A partir dos 65, a reforma vai de 72% a 90%, a viuvez sobe de 27% aos 70-74 para 64% aos 85 ou mais e viver sozinho chega a 58% aos 85 ou mais. A saúde boa cai de 91% aos 15-17 para 12% aos 85 ou mais e as doenças crónicas sobem de 0,7 para 4,3. O IMC atinge o máximo de 27,5 aos 65-69 e desce ligeiramente depois.

Este retrato explica a dificuldade do modelo. As faixas de 25 a 49 anos têm perfis parecidos e o modelo confunde-as com as vizinhas. As faixas acima dos 65 distinguem-se pela reforma e pela viuvez, mas não por sinais finos de idade. As variáveis do inquérito marcam as etapas da vida, não os anos dentro de cada etapa.

![Retrato das 16 faixas por indicadores sociais e de saúde](fig_retrato_faixas.png)

A figura usa os rótulos reais e não o modelo, por isso não muda com a versão da OOF. Cada linha é um indicador e cada coluna uma faixa. A cor é relativa ao mínimo e ao máximo da linha. As transições de estudo, trabalho, reforma e viuvez são visíveis. As faixas de 25 a 49 anos têm perfis parecidos, e é aí que o modelo erra mais.

## 7. Clustering e autoencoders

### 7.1 O que a ficha 9 pede

A ficha 9, tarefa T5, pede clustering e autoencoders no dataset do trabalho prático. A ficha não fixa a arquitetura, as métricas nem os artefactos. As tarefas T1 a T4 usam o dataset College e pedem KMeans, KMedoids e DBSCAN, com matriz de confusão e `classification_report` depois de alinhar os rótulos. A T5 transfere os métodos para o nosso dataset. O Notebook 6 já tinha um KMeans global com k=16. Esta secção acrescenta o bloco fold-safe, o KMedoids e o DBSCAN, mede a concordância entre partições e decide se vale a pena treinar um autoencoder.

### 7.2 Protocolo

O bloco usa as mesmas 5 folds estratificadas, com semente 42, sobre a base 311. Em cada fold, o `StandardScaler` e o clustering ajustam-se só ao treino. A validação é atribuída depois. O KMeans e o KMedoids usam k=16. O KMedoids corre numa amostra de 2 000 linhas do treino com 30 componentes SVD, por ser quadrático no número de linhas. O DBSCAN não tem `predict`: cada linha de validação recebe o cluster do ponto core mais próximo dentro de `eps`, ou −1 se não houver nenhum. O `eps` é o quantil 90% da distância ao quinto vizinho no treino. O ruído não é removido nem forçado numa classe. Fica como flag.

As métricas são ARI, homogeneidade, completude, V-measure, AMI, silhouette, pureza, número de clusters e taxa de ruído. O ARI é a métrica principal porque corrige o acaso e é invariante a permutações de rótulos. A pureza não corrige o acaso e serve só para descrever.

### 7.3 Resultados

| Método | ARI | Homog. | Compl. | V | AMI | Silhouette | Pureza | Clusters | Ruído |
|---|---|---|---|---|---|---|---|---|---|
| KMeans | 0,097 ± 0,006 | 0,260 ± 0,015 | 0,268 ± 0,015 | 0,264 ± 0,015 | 0,248 ± 0,016 | 0,020 ± 0,002 | 0,234 ± 0,012 | 16 ± 0 | 0,000 ± 0,000 |
| KMedoids | 0,079 ± 0,012 | 0,237 ± 0,021 | 0,247 ± 0,020 | 0,242 ± 0,021 | 0,226 ± 0,021 | 0,049 ± 0,007 | 0,218 ± 0,013 | 16 ± 0 | 0,000 ± 0,000 |
| DBSCAN | 0,000 ± 0,000 | 0,000 ± 0,000 | 1,000 ± 0,000 | 0,000 ± 0,000 | 0,000 ± 0,000 | n/a | 0,099 ± 0,002 | 1 ± 0 | 0,067 ± 0,009 |

A completude do DBSCAN é 1 por construção, porque há uma só cluster e todas as classes ficam dentro dela. Não é sinal de qualidade. O ARI de 0 e a pureza perto da classe maioritária dizem o contrário. É o mesmo tipo de armadilha que a silhouette da ficha 9, que era alta com uma cluster e 15 pontos de ruído.

O resultado principal é que nenhum clustering recupera as faixas. O KMeans tem ARI 0,0966 ± 0,0064, o KMedoids 0,0785 ± 0,0123 e o DBSCAN colapsa numa só cluster com 6,7% de ruído. A homogeneidade fica entre 0,24 e 0,26, longe de 1. A silhouette é 0,020 para o KMeans, o que mostra clusters sobrepostos. O KMedoids tem silhouette 0,049, mas um ARI menor. A pureza é 23,4% no KMeans e 21,8% no KMedoids, perto da classe maioritária.

A concordância entre partições é baixa. O ARI entre KMeans e KMedoids é 0,351. O ARI entre KMeans e DBSCAN é 0,009, mas o DBSCAN tem uma só cluster, por isso o valor não mede concordância real. O ARI entre sementes do KMeans é 0,665 e 0,694, o que mostra que a partição é razoavelmente estável dentro do mesmo algoritmo. A instabilidade está entre algoritmos, não entre sementes.

![Clusters do KMeans por faixa real, base 311](fig_km_idade.png)

A figura cruza os clusters com a faixa real. Não há blocos limpos na diagonal. Os clusters ordenam-se por idade média, mas cada um cobre várias faixas. É a imagem do sinal difuso.

As distâncias ao KMeans como features (T18) também não ajudam. O T18 deu 0,4691 contra 0,4700, um delta de −0,0009. A E22 fecha sem adoção.

O DBSCAN reproduz a lição da ficha 9. Com os dados padronizados e o `eps` escolhido pela distância ao quinto vizinho, o resultado é uma só cluster e uma cauda de ruído. A ficha usou `eps=8000` nos dados sem escala e chegou a uma cluster e 15 pontos de ruído. Nos dois casos, a silhouette pode ser alta e não significa qualidade. No nosso caso, a leitura é que os dados formam uma nuvem contínua, sem grupos separados. Isto é consistente com a tese do sinal difuso.

### 7.4 Autoencoder

O autoencoder do PL9 usa uma rede 7 → 16 → 8 → 2, com MSE, Adam e 150 épocas. Usa o conjunto de teste como validação. Esse protocolo não se copia, porque o teste é o holdout do Kaggle e a curva de validação escolhe a arquitetura e o número de épocas. O nosso protocolo usa validação interna ou as folds.

Antes de treinar a rede, fizemos um proxy com PCA. Em cada fold, o PCA explica 90% da variância com cerca de 156 componentes e reconstrói a validação com RMSE médio de 0,312. A correlação entre o erro de reconstrução e a idade é 0,004. A correlação com o erro do modelo é 0,027. O erro de reconstrução não separa idades nem erros. A barra para treinar a rede era +0,010.

O erro de reconstrução também entrou como feature no ecrã do Notebook 18. O T18b deu 0,4736 contra 0,4700, um delta de +0,0036. Fica abaixo da barra de +0,010. O autoencoder MLP não foi treinado.

A comparação com a ficha é instrutiva. O KMedoids da ficha tem 0,26 de accuracy sem alinhamento e 0,736 de accuracy e 0,723 de macro-F1 depois de alinhar os rótulos. O valor de 0,736 é um alinhamento post-hoc nas mesmas 777 linhas, não uma estimativa em holdout. A lição é que a accuracy sem alinhamento engana e que o alinhamento tem de ser aprendido no treino. No nosso caso, o ARI evita o problema porque não depende da permutação.

## 8. Limitações

A primeira limitação é a seleção. O valor 0,5089 foi escolhido entre variantes e limiares nas mesmas folds. O público de 0,51 confirma a ordem de grandeza, mas não transforma o valor numa estimativa sem enviesamento. A medição honesta do mesmo tipo de receita dá 0,504 a 0,509.

A segunda é a base. A entrega usa 339 features. A análise e as experiências usam 311. As duas bases não são uma cópia uma da outra, porque o `N_NAO_APLICAVEL` é recalculado depois do corte. O relatório indica a base em cada tabela. Uma comparação direta entre a sub07 e as experiências da base 311 tem de ter isto em conta.

A terceira é o routing. Cerca de 28% da importância do SHAP vem do bloco dos 55 e a quota mais larga do routing chega a 36%. O modelo aprende quem respondeu o quê, e isso pode inflacionar a CV e não corresponder a biologia. O mesmo vale para a idade mínima, que codifica a idade declarada noutras perguntas. Estas features são legítimas para o Kaggle, mas devem ser declaradas como limitação numa aplicação real.

A quarta é o sinal. O clustering não recupera as faixas, o erro é sobretudo de uma faixa e os blocos 4 a 7 e 11 a 14 concentram 68,9% do erro. As variáveis do inquérito separam as grandes etapas da vida, estudar, trabalhar, reformar, enviuvar, mas não separam idades adjacentes dentro dessas etapas. Faltam perguntas-ponte para os 20 a 59 anos.

A quinta é a avaliação das anomalias. As 51 pessoas da sub06 e as anomalias da sub07 são candidatas, não verdades. Não há rótulo de anomalia. As regras lógicas apanham contradições evidentes e o modelo apanha perfis improváveis. A lista serve para auditoria, não para decidir sobre pessoas.

A sexta é o PL10. O material local só tem o arquivo de dados de GTZAN, sem slides nem notebook. A lição é metodológica. Um split aleatório de chunks de 3 segundos da mesma faixa pode inflar a métrica, porque pedaços da mesma faixa caem em folds diferentes. No nosso caso, `PID` e `HHID` são únicos por linha e não há grupos, por isso o split estratificado por pessoa é correto. Não há número local que suporte o +0,064 que circulou na sessão anterior, e o relatório não o cita.

A sétima é o público. A tabela pública é 30% do resultado e tem ruído. A diferença entre CV e público chega a 0,025 na sub04. As decisões foram tomadas na CV, como a D4 exige. A tabela privada de 70% só é conhecida no fim.

A oitava é o 0,62. O valor é um score público de um terceiro colega, sem código disponível. Não é reproduzível e está cerca de 10 sigma acima do nosso teto. A explicação mais provável é otimismo de protocolo. O próprio repo tem um precedente, o commit `0b83faa "0.59"`, obtido com limpeza global antes da CV e seguido de 0,40. O 0,62 não é meta nem evidência.

## 9. Reprodutibilidade

O ambiente é conda, com Python 3.14.5, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.6.0, catboost 1.2.10 e shap 0.52. Os Notebooks 1 a 7 mantêm a stack antiga para o colega que não tem os pacotes novos. Não há `pip` no sistema.

A ordem do pipeline está em `docs/runbook.md`. O Notebook 1 gera `train_nb01.csv`, com 260 features. O Notebook 2 gera `train_nb02.csv`, com 311 features. O Notebook 3 gera a sub07, com a base 339. O Notebook 6 gera as anomalias e o retrato. O Notebook 4 e o Notebook 5 são evidência histórica. O Notebook 16 é prova congelada. O Notebook 18 é a bancada. Os Notebooks 7 a 15, 17 e 19 estão em `notebooks/arquivo/` e não se re-correm.

A entrega tem três problemas resolvidos nesta versão. A base 339 não estava preservada. O ficheiro `train_nb02_339.csv` foi regenerado com o Notebook 2 do commit `4ae5b6c` e guardado em `data/`. O treino tem SHA-256 `66522e9e732b1b9dc9e22e9e1bdd5fa5c98bdbcb54cc9909e601dc512f82b1fe` e o teste tem `cccf9fc5cbfae520dfb04fd9ea71327dc88bb52bc0e1530482ca88bba2a9cc9e`. O manifesto com os 339 nomes está em `data/nb02_339_manifest.json` na máquina de entrega e é regenerável com o mesmo notebook. O Notebook 3 passou a ler `train_nb02_339.csv` por omissão, a verificar que tem 339 features e a recusar escrever a submissão noutra base. A OOF da sub07 passou a ser guardada em `data/sub07_oof.npz`, com SHA-256 `aa9aca24f245c5665413aa4ebbf7f07b50ac3aafc870a9c75265d5b9dc49cdbd`. O ficheiro submetido foi copiado para fora do repositório antes da re-execução e o SHA-256 está no §5.1. A re-execução do Notebook 3 na base 339 deu 0,5087 de CV e 3507 de 3655 previsões iguais ao ficheiro entregue, ou 96,0%. As 148 previsões restantes diferem em decisões de fronteira. A entrega é o ficheiro original. O Notebook 3 não substitui a entrega quando as previsões diferem: escreve `sub07_ordinal_3modelos_mediana_reproduzido.csv` e só substitui com `NB03_OVERWRITE=1`.

Restam três ressalvas. O Notebook 4 lê a base 339 e a célula 31 usa `PE7`, que o Notebook 2 §4.1 removeu. Re-correr o Notebook 4 na base 311 rebenta com `KeyError`. Os outputs guardados são o artefacto congelado. O Notebook 6 lê a OOF da sub07 quando o ficheiro existe e cai para a OOF do modelo combinado, com 0,4704, ou para a `nb04_oof.npz`, com 0,4662, quando não existe. Num clone limpo, sem caches, a contagem de anomalias muda. O relatório declara o modelo. Os dados brutos, os CSV derivados e o `example_submission_10.csv` estão no `.gitignore`. Um clone limpo não corre o Notebook 1 sem os dados. A diferença entre o `.gitignore` e o `AGENTS.md` sobre os `.npz` versionados ficou resolvida com exceções explícitas para os três ficheiros de evidência.

Para verificar a submissão:

```bash
wc -l submissions/sub07_ordinal_3modelos_mediana.csv   # 3656 linhas
head -1 submissions/sub07_ordinal_3modelos_mediana.csv # RowId,AGE_RANGE
# RowId de 0 a 3654 e 16 classes de 0 a 15
```

## 10. Conclusões

O projeto entrega um modelo ordinal com 0,5089 de CV observada e 0,51 público, a partir de 339 features e de 165 classificadores binários. O resultado é consistente com o teto honesto de 0,504 a 0,509 e está muito acima das referências triviais. O maior ganho veio da formulação ordinal, com +0,024 sobre o XGBoost multiclasse. A idade mínima deu +0,008. A média de três famílias com mediana deu +0,004.

A linha de relabel foi retratada. A sub11 deu 0,48 público contra 0,546 estimados e o Notebook 16 mede um canal de 32,4%. A medição honesta da mesma receita fica em 0,498 a 0,508. A lição metodológica é que os rótulos e as seleções usados dentro da CV têm de ser regeneráveis sem o fold de validação.

O erro é de sinal e não de modelo. No XGB-311, 72,3% dos erros são de uma faixa; na sub07, são 77,8%. Os blocos 4 a 7 e 11 a 14 concentram 68,9% do erro do XGB-311. O clustering não recupera as faixas e o autoencoder não passa o proxy. As variáveis do inquérito separam as grandes etapas da vida, mas não separam idades adjacentes. Para subir de forma material seria preciso recolher perguntas-ponte para os 20 a 59 anos e validar os registos à entrada.

As recomendações são três. Primeira, manter a sub07 como entrega e não gastar submissões em variantes dentro do ruído. Segunda, declarar a base e o modelo em cada análise e não citar a CV do relabel. Terceira, usar o modelo como rastreio de perfis desviantes com controlo de routing, e não como classificação individual.

## Referências

1. INE, Inquérito Nacional de Saúde 2019, Instituto Nacional de Estatística, 2019.
2. ML@DAA, competição Kaggle 2026/27, enunciado e regras, 2026.
3. E. Frank e M. Hall, A simple approach to ordinal classification, ECML 2001.
4. T. Chen e C. Guestrin, XGBoost: a scalable tree boosting system, KDD 2016.
5. G. Ke et al., LightGBM: a highly efficient gradient boosting decision tree, NeurIPS 2017.
6. F. Pedregosa et al., Scikit-learn: machine learning in Python, JMLR 12, 2011.
7. S. Lundberg e S.-I. Lee, A unified approach to interpreting model predictions, NeurIPS 2017.
8. DAA, PL9, Clustering and Autoencoders, slides, ficha e notebooks, 2026/27.
9. DAA, PL10, dados de GTZAN em formato de chunks, 2026/27.
10. Documentação interna do repositório: `docs/plano_modelacao.md`, `docs/plano_acima05.md`, `docs/diagnostico_erros.md`, `docs/mapa_routing.md`, `docs/runbook.md` e `registo_experiencias.md`.
