# Contributos do ramo `rui-1` para o relatório

Para os colegas do grupo de DAA 2026/27 que vão escrever o relatório de 20 páginas e preparar a apresentação.

Este documento explica o que o ramo acrescenta ao `main`, que números pode citar, onde estão os artefactos e o que não deve citar. Não substitui as fontes. Cada número tem a fonte ao lado. Em caso de conflito, mandam `docs/plano_modelacao.md` para as decisões congeladas e `docs/plano_acima05.md` §6 para o registo de experiências.

Estado verificado em 23 de novembro de 2026. `rui-1` e `origin/rui-1` apontam para `0cab462`. `origin/main` aponta para `4ae5b6c` e está integrado no ramo. A árvore versionada está limpa; este documento ainda não está versionado.

## Como usar este documento

O §1 resume o essencial para a reunião de grupo. Os §2 e §3 dão material para as secções 2 a 6 do relatório. O §4 liga cada secção do esqueleto `docs/relatorio.md` aos artefactos. O §5 lista o que não deve ser citado. Os §6 e §7 tratam da reprodutibilidade e das pendências.

Uso CV para a validação cruzada estratificada de 5 partes com semente 42 sobre `data/train_nb02.csv`, com Macro-F1. Uso público para o leaderboard do Kaggle. Nenhum modelo foi escolhido pelo público.

## 1. Sumário executivo

1. O valor observado é 0,504 a 0,509 de Macro-F1 em CV, com 0,51 público. O melhor resultado é a submissão 7, ficheiro `submissions/sub07_ordinal_3modelos_mediana.csv`, com 0,5089 em CV. Há duas ressalvas. A sub7 foi escolhida entre seis variantes e limiares avaliados nas mesmas folds, por isso 0,5089 é um máximo observado, não uma estimativa não enviesada. A re-medição do NB16 de receitas fixas dá 0,504 a 0,508, também com o limiar escolhido nas mesmas folds. O relatório deve apresentar a sub7 como modelo final e o intervalo como observado, não como um teto sem seleção.
2. A linha de relabel e quarentena que chegou a 0,52 a 0,55 em CV está retratada. A evidência direta é a sub11: 0,48 público contra 0,546 estimados. O NB16 mede o canal em 32,4%, mas a experiência honesta usa uma regra de seleção diferente da do relabel original. O canal mede a probabilidade do rótulo observado; o CV honesto seleciona os 15% pior por confiança máxima, e os dois conjuntos só se sobrepõem em 16,4%. A retração mantém-se pela sub11, mas o mecanismo ainda precisa de uma repetição com a mesma regra. A regra adotada é que tudo o que é usado dentro da CV tem de ser regenerável sem o fold de validação.
3. As alternativas foram testadas e fechadas com número. Thresholds, restrições monotónicas, pairwise-15, target-mean, quarentena, bagging, lambdarank, TabPFN, metastacking e combos estão em `docs/plano_acima05.md` §6 e `docs/pesquisa_quebra_platamar.md` §13, com delta e veredito. Isto serve para a análise crítica do relatório.
4. O erro é de sinal, não de modelo. No XGB-311 honesto, 72,3% dos erros são de uma faixa e 27,7% de duas ou mais. Os blocos 11 a 14 e 4 a 7 concentram 68,9% dos erros. A classe 13 é a pior, com F1 0,273. O diagnóstico está em `docs/diagnostico_erros.md`.
5. O sinal etário está difuso. O KMeans sem alvo com k=16 não recupera as faixas. Na base 339, o ARI é 0,11 e a pureza vai de 14% a 48%; na base 311 atual, o ARI é 0,0907 e a pureza vai de 13,3% a 35,4%. Na análise fold-safe do NB06, o KMeans tem ARI 0,0966 e o KMedoids 0,0785; o DBSCAN colapsa numa só cluster com 6,7% de ruído. O proxy PCA não mostra sinal (rho 0,004 com a idade) e o autoencoder MLP não foi treinado. Cerca de 36% da massa absoluta de SHAP vem do routing do questionário. Este número é uma estimativa e tem a ressalva do §3.5. Isto sustenta a tese do teto e deve ser declarado como limitação.
6. O material do curso não contém a solução para +0,11. Os mapas PL1 a PL8, com 58 de 58 ficheiros, e T, com 9 PDFs e 231 páginas, convergem. Não há método que valha +0,11 com honestidade. Há ganhos pequenos que se podem somar, de 0,005 a 0,015, e que levariam a cerca de 0,51 a 0,525. Ver `docs/mapa_aplicavel.md`.
7. O 0,62 de um terceiro colega é um score público, sem código disponível, e não é reproduzível. Está cerca de 10 sigma acima do nosso teto, mas não é falsificável até à entrega. A explicação mais provável é otimismo de protocolo. O próprio repo tem um precedente. O commit `0b83faa "0.59"` foi obtido com limpeza global antes da CV e seguido de `0.40`. Não usar o 0,62 como meta nem como evidência.
8. O ramo entrega ao relatório a análise de erros, a interpretabilidade, as anomalias, os mapas de aplicabilidade e uma organização que separa o pipeline das experiências fechadas. Também deixa identificados os problemas de reprodutibilidade e de seleção que o relatório deve declarar.

## 2. O que o ramo acrescenta ao main

O main, no commit `4ae5b6c` de 29 de outubro, contém o pipeline 01 a 04 e o modelo ordinal final do colega. O ramo `rui-1` acrescenta o seguinte.

| Área | Artefacto | Uso no relatório |
|---|---|---|
| Prova de honestidade | `notebooks/Notebook_16.ipynb` | Mede o canal do relabel, quantifica a inflação e fixa o teto 0,504 a 0,509. É o documento principal da secção de metodologia e validação. |
| Bancada de triagem | `notebooks/Notebook_18.ipynb` | Ecrã holdout 80/20 com barra +0,010. Testa T0 a T17, C1 a C6 e S1 a S2. Fornece a tabela do que foi tentado e rejeitado. |
| Análise de erros | `docs/diagnostico_erros.md` | 5 662 erros, bolsos, pares adjacentes, ausência de bolsão de baixa confiança e não-findings. |
| Anomalias e retrato | `notebooks/Notebook_06.ipynb` | Anomalias de perfil com diferença absoluta de 3 ou mais faixas, direção do erro, controlo de routing, retrato das 16 faixas, KMeans e o bloco PL9 de clustering fold-safe. Usa a OOF da sub7 quando `sub07_oof.npz` existe; num clone sem caches cai para a combined 0,4704. |
| Interpretabilidade | `notebooks/Notebook_04.ipynb`, do main e mantido | SHAP de um XGBoost plano de 16 classes (0,480) e OOF/anomalias da sub6 (0,5051), com importância por faixa, fronteiras do ordinal e casos individuais. |
| Pesquisa primária | `docs/pesquisa_quebra_platamar.md`, `docs/pesquisa_ganhos.md` | 9 threads contra fontes primárias, shortlist executada e lista do que não vale a pena repetir. |
| Mapas do curso | `docs/mapa_pl.md`, `docs/mapa_t.md`, `docs/mapa_aplicavel.md` | Cobertura de 58 de 58 ficheiros e de 9 PDFs, com síntese priorizada e especificações fold-safe. |
| Base limpa 311 | `notebooks/Notebook_02.ipynb` §4.1 | Remove 28 colunas dos blocos PE, PN, SS e MH. O T7a deu +0,0066 em 5 de 5 folds. É a base dos NB16 a NB18. A sub7 usa a base 339. |
| Reorganização | `notebooks/arquivo/`, `docs/INDICE.md`, `docs/runbook.md`, `.gitignore` | Separa o pipeline, que fica na raiz, das experiências fechadas, que ficam no arquivo. Apagou 29 MB de caches retratados e passou a versionar os documentos markdown. |

O merge com o main está completo dentro de `0cab462`. `HEAD..origin/main` está vazio. Os NB01 a NB04 ficaram com a nossa versão, que é um superconjunto. O NB02 tem as células §4.1 que o main não tem. Há uma diferença anterior de uma linha em `submissions/sub01_arvore_referencia.csv`, no RowId 2601, vinda de `726d635`, sem impacto.

No ramo, só a `sub11` tem registo de submissão, com 0,48 público, e está retratada. `sub05_combined`, `sub06_xgb`, `sub07_frank_hall`, `sub07b` e `sub08` são candidatos locais sem submissão. Para `sub09` e `sub10`, os documentos dizem "pendente" e "confirmar com o utilizador", por isso não afirmar que nunca foram submetidos. Não confundir `sub07_frank_hall.csv`, que é um candidato do ramo, com `sub07_ordinal_3modelos_mediana.csv`, que é a sub7 submetida.

## 3. Números para o relatório

### 3.1 Dados e protocolo

O treino tem 10 962 pessoas e o teste tem 3 655. O alvo `AGE_RANGE` tem 16 faixas, de 0 a 15. O desequilíbrio é de 6,17 vezes. A menor classe é a 1, de 18 a 19 anos, com 172 casos. A maior é a 10, de 60 a 64 anos, com 1 062 casos.

As referências triviais são 0,011 para a classe maioritária, 0,060 para o aleatório estratificado e 0,0625 para o valor teórico de 1 em 16. Não há shift entre treino e teste. A diferença média é de 0,16 pontos percentuais, como está em `docs/plano_modelacao.md` §7.

A coluna `WGT` foi excluída porque o peso oficial é calibrado por grupo etário quinquenal e codifica o alvo. O routing tem 133 perguntas universais, 31 com gate de 55 anos ou mais, 51 parciais, 10 raras e 2 com gate de sexo, como está em `docs/mapa_routing.md`. Os cortes oficiais são aos 55 anos, no bloco G e na `PC1A`, e aos 65 anos, quando `HH_ACT + HH_INACT == 0`. O segundo corte vale para 3 202 pessoas e tem 2 exceções, ambas na faixa 15 a 17.

O protocolo é `StratifiedKFold` com 5 partes, shuffle e semente 42, com Macro-F1 e as mesmas folds para todas as comparações. Antes de qualquer 5-fold há um ecrã holdout 80/20 com barra de +0,010. O ruído é de ±0,004, com sigma entre 0,008 e 0,012. A adoção exige 5 de 5 folds. As exceções de 4 de 5 estão marcadas.

O Kaggle permite 3 submissões por dia. O resultado final é 30% público e 70% privado. Nunca se tuna pelo público.

Há duas bases em circulação. É importante dizer qual delas suporta cada resultado.

| Versão | Conteúdo | Features | Usada por |
|---|---|---|---|
| Raw | INS 2019 após a limpeza do NB01 | 236 variáveis de origem; 260 features no `train_nb01.csv` | NB01 |
| One-hot | 205 ordinais/binárias, 171 one-hot e 33 numéricas | 409 | NB02 §3 |
| Seleção | Remove 70 quase constantes e 6 duplicadas | 333 | NB02 §4 |
| Novas features | `N_DIFICULDADES`, `N_DOENCAS`, `IMC`, `N_NAO_APLICAVEL` | 337 | NB02 §5 |
| Idade mínima | `IDADE_MINIMA` e `FAIXA_MINIMA` | 339 | sub7, NB03, NB04 |
| Blocos ruidosos | Remove 28 colunas de PE, PN, SS e MH | 311 | NB16, NB17, NB18 e o `train_nb02.csv` atual |

O `train_nb02.csv` atual tem 313 colunas, que são `PID`, 311 features e `AGE_RANGE`. O teste tem 312, que são `PID` e 311 features. O NB02 §4.1 remove as 28 colunas e exporta 311, mas a conclusão do notebook ainda diz 339 e a saída do §8 ainda diz 340/339. Os dois textos estão antigos. O relatório deve indicar a base usada em cada resultado.

### 3.2 Modelo final e submissões

A receita da sub7 usa três famílias ordinais de Frank e Hall, com 15 classificadores binários por família. Cada classificador responde se a faixa é maior do que k. As probabilidades das famílias são somadas e divididas por três. As famílias são XGBoost com 5 sementes, LightGBM com 4 e HistGradientBoosting com 2, o que dá 165 classificadores binários. A decisão usa o argmax quando a confiança é de 0,40 ou mais. Abaixo disso, usa a mediana das probabilidades. No fim aplica `max(previsão, FAIXA_MINIMA)`.

No teste, 80,5% das decisões usam o argmax, 19,5% usam a mediana e 4 previsões são corrigidas pela idade mínima. As 3 655 previsões do ficheiro submetido foram confirmadas na execução original. A re-execução de 24 de novembro, na base 339 regenerada, deu 0,5087 e 3507 de 3655 previsões iguais; ver §6.

| Número | Data | Modelo | CV | Público |
|---|---|---|---|---|
| 1 | 25/10 | Árvore de decisão com profundidade 8 | 0,407 | 0,40 |
| 2 | 26/10 | Regressão logística com C=0,01 | 0,426 | 0,43 |
| 3 | 26/10 | Gradient Boosting default com correção | 0,467 | 0,46 |
| 4 | 26/10 | Gradient Boosting otimizado com correção | 0,475 | 0,45 |
| 5 | 27/10 | XGBoost com regressor HGB e limiar 0,35 | 0,489 | 0,48 |
| 6 | 27/10 | Ordinal XGBoost com regressor e limiar 0,35 | 0,5051 | 0,49 |
| 7 | 28/10 | Três ordinais com mediana, limiar 0,40 e correção | 0,509 | 0,51 |

A fonte é `registo_experiencias.md` §8.1 e §21, e o NB03 §7.7 e §7.8. A sub4 mostra que o público tem ruído. A CV deu 0,475 e o público deu 0,45. A decisão fica com a CV.

A ressalva de seleção aplica-se à tabela. O NB03 escolhe os hiperparâmetros do Gradient Boosting com `RandomizedSearchCV` nas mesmas folds que reporta, escolhe a melhor de seis variantes ordinais nas mesmas folds e escolhe o limiar 0,40 nas mesmas folds. A célula 22 do NB03 chama "aninhado" ao limiar, mas as probabilidades OOF usadas para o escolher foram geradas por modelos que viram o fold de avaliação. O valor 0,5089 é, por isso, um máximo observado. O relatório deve dizer "CV observada" e não "estimativa não enviesada"; o público 0,51 é a confirmação externa.

O NB16 voltou a medir o teto na base 311, sem o canal do relabel.

| Receita | CV honesta par | CV com relabel | Inflação | Melhor fallback honesto |
|---|---|---|---|---|
| `flat`, mediana a 0,40 | 0,5007 | 0,5409 | +0,0402 | 0,5064 |
| `LGB`, mediana a 0,45 | 0,5040 | 0,5417 | +0,0377 | 0,5083 |
| `FH`, mediana a 0,35 | 0,5003 | 0,5364 | +0,0361 | 0,5035 |
| `HGB` | 0,4909 | 0,5355 | +0,0446 | 0,4979 |
| `B1`, quatro famílias | 0,4981 | 0,5460 | +0,0479 | 0,5067 |
| `mc5` | 0,4766 | 0,5376 | +0,0610 | 0,4849 |

A coluna "CV honesta par" é o par honesto da receita com relabel: a mesma receita, com os rótulos regenerados sem o fold de validação e avaliada com os rótulos originais. A coluna "melhor fallback honesto" é a melhor variante da receita original, sem relabel, e é a base do intervalo 0,504 a 0,509. As duas colunas não devem ser misturadas: a inflação compara pares, o teto usa o melhor fallback. A sub7, com 0,5089, e o público, com 0,51, são consistentes com esse intervalo.

### 3.3 O que foi testado e rejeitado

A tabela seguinte condensa as experiências. A lista completa, com data e delta, está em `docs/plano_acima05.md` §6.

| Família | Experiência | Resultado | Veredito |
|---|---|---|---|
| Decision-time | M1, thresholds de Frank e Hall com validação aninhada | 0,0058 abaixo em 1 de 5 folds | rejeitado |
| Decision-time | M2, offsets por classe com validação aninhada | 0,0072 abaixo em 2 de 5 folds | rejeitado |
| Decision-time | M5, pair-flips | 0,0003 abaixo, com zero flips | rejeitado |
| Restrições | M3, restrições monotónicas | 0,0011 abaixo e 0,0003 acima | rejeitado |
| Ordinal | Pairwise-15, T10 e P1 | +0,0162 no holdout, 0,4854 em 5-fold, +0,0077 em 4 de 5 | não empilha na sub7, com S1b 0,0050 abaixo |
| Ordinal | C3, target-mean com pairwise-15, P2 | 0,4886, +0,0109 em 4 de 5 | confirmado, mas cerca de 0,02 abaixo da sub7, arquivado |
| Labels | Relabel E26 e E27, quarentena NB11 | 0,5364 e 0,5460 com relabel, 0,498 a 0,508 honestos | retratado por vazamento |
| Labels | Quarentena honesta T6 | +0,0069 | abaixo da barra |
| Ensemble | E12 e E12b, metastack | 0,5037 e 0,5027 | congelado, superado pela sub7 |
| Ensemble | C6, kitchen sink | 0,4787, abaixo do T10 com 0,4862 | combinações fechadas |
| Bagging | T15, 10 sementes com 80% das linhas | 0,4650, 0,0050 abaixo | rejeitado |
| Ranker | T16, lambdarank | 0,10 a 0,11 | rejeitado |
| Foundation | T17, TabPFN-3.5 | 0,4805, +0,0105 | passou o ecrã |
| Foundation | T17b, mistura de XGBoost com TabPFN | 0,4833, +0,0133 | passou a barra, 5-fold não corrido |
| Foundation | S2, receita sub7 com TabPFN | 0,5037, +0,0040 contra S1 | abaixo da barra de +0,010, arquivado |
| Clustering | T18, distâncias ao KMeans fold-safe | 0,4691, −0,0009 | rejeitado, E22 fechada |
| Clustering | T18b, erro de reconstrução PCA | 0,4736, +0,0036 | abaixo da barra, autoencoder não treinado |
| Outros | Self-training, Naive Bayes, BRF, DART e MLP | neutros ou negativos | rejeitados |
| Hierarquia | Três especialistas com meta-logística | +0,009 contra o global | abaixo da receita XGBoost |

A mensagem para o relatório é que o teto não resulta de falta de tentativa. As duas famílias com margem teórica, pairwise-15 e TabPFN, foram testadas e não passam a barra. A única linha que batia o teto era vazamento.

Os números do NB18 vêm de um único holdout de 2 193 linhas, com cerca de 30 variantes testadas. O ruído é de ±0,01 a 0,02 e o máximo esperado de 30 variantes sob a hipótese nula ronda +0,02. Por isso, T10 +0,0162 e T14 +0,0210 são ecrãs, não ganhos de relatório. Só contam depois de um 5-fold. O T10/P1 ficou em +0,0077 e o T14/P4 em +0,0058, ambos abaixo da barra. A frase "as duas famílias não passam a barra" deve ser lida com esta ressalva.

### 3.4 Análise de erros

A análise honesta usa o XGB-311 e está em `docs/diagnostico_erros.md`.

- Há 5 662 erros em 10 962 linhas. O Macro-F1 é 0,4782.
- 72,3% dos erros são de uma faixa. Os outros 27,7%, ou 1 569 erros, são de duas ou mais.
- As classes 11 a 14 concentram 38,1% do erro, com 3 513 linhas e F1 de 0,527, 0,369, 0,273 e 0,348.
- As classes 4 a 7 concentram 30,8%, com 2 817 linhas e F1 de 0,304, 0,379, 0,381 e 0,421.
- A confusão 9 e 10 vale 11,2%, com 635 casos e F1 de 0,668 e 0,645.
- A classe 13 vale 10,8% do erro, com F1 0,273 e recall 0,256.
- A classe 1 vale 2,1%, com F1 0,354.
- Os três maiores pares adjacentes são 9 e 10 com 635 casos, 11 e 12 com 538 e 12 e 13 com 464. Juntos somam 28,9% dos erros.
- As melhores classes são a 0 com F1 0,854, a 9 com 0,668, a 8 com 0,663, a 10 com 0,645 e a 2 com 0,604.
- Não há um bolsão removível de baixa confiança. Os 5% menos confiantes concentram 7,6% dos erros, com taxa de 0,78 contra 0,52 no global. Os 20% menos confiantes concentram 27,7%. Os 5% mais confiantes ainda erram 9,5%, ou 52 linhas.
- A fronteira 8 e 9 tem zero confusão nos dois sentidos. Não vale gastar orçamento de fronteira aí.
- No diagnóstico do NB08, com a base 339, 42,9% dos erros são casos em que as quatro bases concordam e erram. Em 66,7%, nenhuma base acerta. A diversidade de vistas está esgotada.

### 3.5 Interpretabilidade

O NB04 usa a OOF da sub6, com 0,5051 e base 339. O SHAP do NB04 usa um XGBoost plano de 16 classes, com 0,480, não a sub7.

- A OOF da sub6 tem 0,5051, acerto exato de 50,0%, 88,1% a uma faixa ou menos e 11,2% das decisões pelo regressor.
- 4 178 pessoas, ou 38%, ficam a uma faixa. 64 pessoas têm erro de 4 ou mais. Dessas, 48 parecem mais novas e 16 mais velhas.
- A importância global em SHAP, em percentagem do total, é 6,9 para `MAINSTAT`, 5,5 para `IN15`, 5,3 para `PC1A`, 4,8 para `MARSTALEGAL`, 4,0 para `PC1C`, 3,5 para `PC1D`, 3,4 para `HH_INACT`, 3,0 para `PC1E`, 2,9 para `PC1B`, 2,9 para `HATLEVEL`, 2,5 para `IDADE_MINIMA` e 2,4 para `HH_ACT`. As 20 primeiras explicam 60% do total.
- O bloco de routing, com `PC1*`, `HA1*`, `IN15` e `IN11`, representa cerca de 36,2% da massa absoluta de SHAP. Este número foi medido no modelo relabel do NB14. O ranking SHAP do modelo original e do relabel tem rho 0,977, por isso é uma estimativa estável, mas não é uma medição no modelo final. Se o relatório precisar do número exato, recalcular a quota sobre o SHAP do NB04, que é honesto, ou declarar a ressalva.
- A concordância entre SHAP e permutação é rho 0,468 no ranking de 311 features. `PC1A` é rank 1 em SHAP e 11 em permutação. `IN15` tem 0,133 em SHAP e 0,001 em permutação. `HA1B` tem 0,149 e 0,0087. `PC1C` tem 0,107 e 0,004. O routing identifica quem respondeu o quê, mas não separa idades adjacentes dentro dos blocos 4 a 7 e 11 a 14.
- O segundo corte, aos 65 anos, aparece quando `HH_ACT + HH_INACT == 0`. Vale para 3 202 pessoas, com 2 exceções, ambas na faixa 15 a 17. Como correção pós-previsão, o resultado passa de 0,5051 para 0,5052, o que é ruído. Não foi adotada.
- As 43 pessoas com "Sem resposta" no bloco dos 55 anos ou mais são todas de 55 ou mais. O modelo prevê menos de 55 em 4 delas. São artefactos do modelo, não perfis anómalos.
- O KMeans com k=16 e sem alvo foi executado na base 339 e está congelado nos outputs do NB06. Nessa base, o ARI é 0,11 e a pureza vai de 14% a 48%; o cluster mais velho concentra 4,27% das anomalias. Na base 311 atual, o ARI cai para 0,0907 e a pureza para 13,3% a 35,4%. Isto sustenta a tese do sinal difuso, não a de clusters etários.

### 3.6 Anomalias, duas análises que não se misturam

| Análise | OOF | Definição | Resultado | Uso |
|---|---|---|---|---|
| NB04 §4 | Sub6, 0,5051, base 339 | Regras lógicas e `P_DECLARADA < 0,05` | 16 incoerências lógicas (4 impossíveis, 9 muito improváveis, 3 raras), mais a regra da menopausa (3 pessoas, 2 novas), e 36 anomalias do modelo, com 3 sobreposições. O total é 51 pessoas, ou 0,47%, sendo 40 perfis excecionais, 7 prováveis erros de registo e 4 artefactos do modelo. As 43 pessoas com "Sem resposta" no bloco dos 55 ou mais são uma população separada; só 4 delas estão entre as 51. | Secção de anomalias, com o modelo sub6 declarado |
| NB06 | `combined_oof.npz`, 0,4704, base 339 | Diferença absoluta de 3 ou mais faixas | 260 anomalias, sendo 119 envelhecidas e 141 rejuvenescidas. 53 têm diferença de 4 ou mais. As comorbilidades médias são 2,05 e 1,75, contra 2,58 no grupo normal e 2,56 na população completa. 55 das 141 rejuvenescidas são reformadas. | Perfil, retrato das 16 faixas e KMeans, com o modelo declarado |

`docs/relatorio.md` §6 chamava às 260 "OOF final". Não é. A OOF dessa análise é a do modelo combinado de 0,4704, não a sub7 de 0,5089. O NB06 atual usa a OOF da sub7 quando `data/sub07_oof.npz` existe. A linha de 260 descreve a análise antiga sobre a combined e deve ser citada com o modelo declarado.

### 3.7 Pesquisa, mapas do curso e o 0,62

- Os mapas PL1 a PL8 têm 58 de 58 ficheiros lidos. O mapa T tem 9 PDFs e 231 páginas. Os dois convergem. Nada no material vale +0,11 com honestidade. O que há são ganhos pequenos, os A3 a A8, de 0,005 a 0,015, que levariam a cerca de 0,51 a 0,525.
- A shortlist de pesquisa foi executada. Pairwise-15 foi rejeitado na receita. C3 foi confirmado, mas fica insuficiente. Bagging e lambdarank foram rejeitados. O TabPFN passou o ecrã no T17b, mas o S2 ficou abaixo da barra. Tudo está em `docs/pesquisa_quebra_platamar.md` §9 e §13.
- O 0,62 é um score público de um terceiro colega, sem acesso ao código. Está cerca de 10 sigma acima de nós, com sigma de 0,01 na tabela pública, mas é inverificável até janeiro. Há duas hipóteses. A primeira é um tratamento de dados genuinamente melhor. A segunda é overfitting ao público com probing, com 3 submissões por dia. O repo tem um precedente. O commit `0b83faa "0.59"` usou um `train_clean.csv` limpo globalmente antes da CV, com LGBM em 0,5823, e foi seguido de `0.40 F1`. Não usar o 0,62 como meta nem como prova de que existe um caminho de +0,11.

## 4. Mapa entre secções do relatório e artefactos

| Secção de `docs/relatorio.md` | Fonte principal | O que copiar |
|---|---|---|
| 1. Áreas e objetivos | `docs/contexto.md`, `docs/plano_modelacao.md` §1 | Tarefa, métrica, 16 classes, 30/70 e 3 submissões por dia |
| 2. Metodologia | `docs/plano_modelacao.md` D1 a D5 e §3, `docs/plano_acima05.md` §0 e §6, `docs/runbook.md` | Protocolo de CV, holdout, barras, regra fold-safe, decisões e rejeições |
| 3. Dados e processamento | NB01, NB02, §3.1, `docs/mapa_routing.md` | 236 variáveis de origem, 260 features no NB01, 409 para 333, 339 e 311, `WGT`, routing e idade mínima |
| 4. Modelos | `registo_experiencias.md` §8.1, NB03 §7.7 e §7.8, §3.2 | Tabela das 7 submissões, receita da sub7 e 165 binários |
| 5. Resultados e análise crítica | `docs/diagnostico_erros.md`, NB04, NB16, §3.3 a §3.5 | 72,3% off-by-one, bolsos, teto, retração e SHAP |
| 6. Anomalias | NB04 §4, NB06 | 51 anomalias na sub6 ou 260 na OOF 0,4704, com o modelo declarado |
| 7. Recomendações | `docs/plano_modelacao.md` §7, `docs/mapa_aplicavel.md` §3, `docs/pesquisa_quebra_platamar.md` §10 | Perguntas-ponte para 20 a 59 anos, validação à entrada e limites do survey |
| Limitações | §3.5, §3.7 e §5 | Routing com 36%, base 339 contra 311, público 30/70 e o 0,62 não verificável |

Já existem figuras e tabelas. As 29 figuras `docs/fig_nb14_*.png` têm cabeçalhos do modelo relabel e servem apenas como contraste. Para o relatório, preferir as figuras do NB04, que são do modelo sub6, e as do NB06, que são o retrato por faixa e o KMeans. O `docs/fig_routing.png` e o `docs/mapa_routing.md` mostram o routing por pergunta, idade e sexo. As tabelas de F1 por classe e de comparação de modelos estão em `registo_experiencias.md` §7b, §10 e §21. Os ficheiros `data/nb14_*.csv` e `data/nb18_resultados.csv` são locais e não estão versionados. Os números estão repetidos nos documentos. Não prometer ficheiros que o leitor não tem.

## 5. O que não citar

| Item | Porquê | O que usar |
|---|---|---|
| CV de 0,52 a 0,546 de E24 a E27, mc5 e B1, nos NB12, NB13 e NB15 | Vazamento. Os rótulos foram regenerados com o fold de validação dentro e a inflação é de 0,036 a 0,061. | NB16 com receitas fixas, 0,498 a 0,508, com a ressalva da regra de seleção. Sub7 com 0,5089. |
| 0,5089 como CV não enviesada | O NB03 escolhe hiperparâmetros, a melhor de seis variantes e o limiar nas mesmas folds. É um máximo observado. | Dizer "CV observada 0,5089"; o público 0,51 é a confirmação externa. |
| NB16 como prova do mecanismo do relabel | O canal mede a probabilidade do rótulo observado, mas o CV honesto seleciona os 15% pior por confiança máxima. Os conjuntos só se sobrepõem em 16,4%. | A sub11 deu 0,48 público contra 0,546 estimados; essa é a evidência direta. O canal 32,4% fica como medição com a ressalva da regra. |
| T14 +0,0210 e T10 +0,0162 como ganhos | São ecrãs de um único holdout com cerca de 30 variantes; o máximo esperado sob a hipótese nula ronda +0,02. | T10/P1 +0,0077 e T14/P4 +0,0058, ambos abaixo da barra. Não citar como ganho. |
| `sub10` e `sub11` como bons candidatos | Retratados. A `sub11` deu 0,48 público contra 0,546 estimados. | Sub7 como final. |
| NB14, com SHAP, figuras, ECE 0,0953 e 76% off-by-one, como evidência honesta | O modelo usa relabel e os cabeçalhos estão contaminados. | NB04, que usa a sub6, e `docs/diagnostico_erros.md`, com 72,3%. |
| Quarentena NB11 Q2, com +0,0316 e 0,5093 | É o canal de seleção do vazamento. | T6 honesto, com +0,0069, abaixo da barra. |
| NB07 e `sub05_combined`, com 0,4704 | Fica abaixo da sub04 e está arquivado. | Sub7. |
| NB08 E12, com 0,5037, como modelo final | Está congelado, mas foi superado pela sub7, e usa a base 339. | Sub7, com 0,5089. |
| `sub07_frank_hall.csv` como a sub7 | É um candidato do ramo com Frank e Hall e mistura, não o ficheiro submetido. | `sub07_ordinal_3modelos_mediana.csv`. |
| 76% off-by-one | É o número do relabel. | 72,3% honesto no XGB-311. |
| 260 anomalias na OOF final | A OOF é a do modelo combinado de 0,4704. A sub7 tem o seu próprio número. | Dizer o modelo; o NB06 usa a OOF da sub7 quando o ficheiro existe. |
| 339 features como base atual | O `train_nb02.csv` atual tem 311. A base 339 é da sub7, do NB03 e do NB04. | Indicar a base em cada resultado. |
| KMeans com ARI 0,11 como base 311 | 0,11 e pureza 14% a 48% são da base 339; na 311, o ARI é 0,0907 e a pureza 13,3% a 35,4%. | Declarar a base ao lado do ARI. |
| 0,62 como facto ou meta | É um score público de terceiro, sem código e inverificável. | Mencionar como alegação externa, se for necessário, com ressalva. |
| Re-correr NB13, NB14, NB15, NB17 e NB19 | Estão congelados e dependem de caches e labels antigos. | Ler os vereditos em `notebooks/arquivo/README.md`. |

Alguns nomes confundem. O NB04 é a interpretabilidade da entrega do colega, com a sub6 e a base 339, não um modelo novo. O NB06 é anomalias e retrato, não treino. O NB18 é a bancada, não o pipeline. `sub06_xgb.csv`, do ramo, não é o `sub06_ordinal_regressor.csv`, do colega. `sub07_frank_hall.csv`, do ramo, não é o `sub07_ordinal_3modelos_mediana.csv`, que foi submetido.

## 6. Como verificar a reprodutibilidade

O ambiente é `conda run -n daa`, com Python 3.14.5, pandas 3.0.6, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.6.0, catboost 1.2.10, shap 0.52 e tabpfn 9.2.0. Só se usa conda. Os NB01 a NB07 mantêm a stack antiga para o colega que não tem estes pacotes. Não há `pip` no sistema.

A ordem do pipeline está em `docs/runbook.md`. O NB01 demora cerca de 5 minutos e gera `train_nb01.csv`. O NB02 demora cerca de 5 minutos e gera `train_nb02.csv` com 311 features. A base 339 regenera-se com o NB02 de `4ae5b6c`. O NB03 demora 30 a 60 minutos e gera a sub7 e `sub07_oof.npz`. O NB06 demora cerca de 6 minutos. O NB16 é prova congelada. O NB18 é a bancada ativa. O NB04 e o NB05 são evidência histórica. Os NB07 a NB15, NB17 e NB19 estão em `notebooks/arquivo/` e não se re-correm.

Para verificar a submissão final:

```bash
wc -l submissions/sub07_ordinal_3modelos_mediana.csv   # 3656 linhas, header e 3655 casos
head -1 submissions/sub07_ordinal_3modelos_mediana.csv # RowId,AGE_RANGE
# RowId de 0 a 3654, 16 classes de 0 a 15
```

Há três ressalvas de reprodutibilidade.

1. O NB03 e a base 339. A base foi regenerada com o NB02 de `4ae5b6c` e ficou em `data/train_nb02_339.csv` e `data/test_nb02_339.csv`. O treino tem SHA-256 `66522e9e…` e o teste `cccf9fc5…`; o manifesto com os 339 nomes está em `data/nb02_339_manifest.json`. O NB03 lê a base 339 por omissão, verifica a contagem e recusa escrever a submissão noutra base. O CSV submetido foi copiado para fora do repositório antes da re-execução. A re-execução deu 0,5087 e 3507 de 3655 previsões iguais (96,0%); as restantes diferem em decisões de fronteira. A entrega ficou intacta e o NB03 escreve um candidato quando as previsões diferem.
2. O NB04. Lê `train_nb02.csv` e a cache `previsoes_oof_nb04.csv`, que é local e não está versionada. Foi executado sobre 339. Re-correr com 311 não muda só os números: a célula 31 usa `PE7`, que o NB02 §4.1 removeu, e rebenta com `KeyError`. Os outputs guardados são o artefacto congelado.
3. O NB06. Lê `data/sub07_oof.npz` quando existe e cai para `combined_oof.npz`, com 0,4704, e depois para `nb04_oof.npz`, com 0,4662, num clone sem caches. A contagem de anomalias muda com a OOF. O relatório declara o modelo.

No git, `rui-1` e `origin/rui-1` apontam para `0cab462`. O `origin/main` aponta para `4ae5b6c` e está integrado. O `main` local pode estar atrás do `origin/main`, o que não afeta o relatório.

O TabPFN é opcional. A licença está aceite. O `TABPFN_TOKEN` está em `~/.bashrc` e também tem de ser passado como variável de ambiente ao `conda run`, que não lê o bashrc. Os pesos estão em `~/.cache/tabpfn`. O limite de 5 000 amostras em CPU foi contornado com `ignore_pretraining_limits=True`, pela regra do NB18 de usar 100% das linhas. A cache é `data/nb18_tabpfn_proba.npz`.

## 7. Pendências e decisões

1. T17b em 5-fold honesto, cerca de 1,7 horas. O ecrã holdout dá +0,0133, mas é um de cerca de 30 ecrãs no mesmo holdout; sob a hipótese nula, o máximo esperado ronda +0,02. O S2, com a receita sub7 e TabPFN, dá +0,0040, abaixo da barra. Recomendação: não gastar 1,7 h só para citar. Se o relatório quiser fechar a linha, correr uma vez e reportar com a ressalva de múltiplos testes. Caso contrário, citar como ecrã holdout e dizer que o 5-fold não foi corrido; não apresentar como ganho.
2. O lote A3 a A8 de `docs/mapa_aplicavel.md` §1 e §2. Inclui contagens de N/A por bloco, imputação estratificada, binary encoding, quantis, KMeans e scores de anomalia. A soma esperada é de 0,005 a 0,015 e é uma estimativa, não uma medição. O problema maior não é o ganho: antes de acrescentar features, é preciso fixar a base 339/311 e repetir o NB16 com a mesma regra do relabel. Só vale a pena depois disso.
3. Uma pergunta casual ao terceiro colega sobre a validação do 0,62. Não é bloqueante. O protocolo fica congelado de qualquer forma.
4. O relatório. O `docs/relatorio.md` foi reescrito em 24 de novembro com a sub7, a base de cada resultado, a retração, o F1 por classe, a matriz de confusão, a ablação, os hiperparâmetros, o bloco PL9 e a reprodutibilidade. Falta a revisão do grupo e a exportação para PDF. As figuras continuam locais (`docs/fig_*.png`). O maior bloqueador passou a ser a revisão e o corte para caber em 20 páginas.
5. O bug dos P1 a P4 do NB17, com `int(_z['nov'])`, foi corrigido em 10 de novembro com `int(_z['nov'].item())` nas quatro células; os notebooks afetados estão arquivados. Não há bug de runtime conhecido nessa via. Os problemas abertos são de reprodutibilidade: a base 339 não está preservada, o NB03 reescreve a sub7 se for re-corrido, o NB04 rebenta na base 311 e o NB06 cai para a OOF 0,4704 ou para 466 anomalias num clone limpo.

## 8. Frases prontas

Estas frases podem ser adaptadas. A fonte está ao lado.

- O valor observado é 0,504 a 0,509 de Macro-F1 em validação cruzada estratificada de 5 partes com semente 42. A submissão 7, com 0,5089, obteve 0,51 no leaderboard público. O 0,5089 é um máximo sobre variantes e limiares avaliados nas mesmas folds; o público 0,51 confirma-o externamente, mas não o torna uma estimativa não enviesada. Fonte: NB16 e `registo_experiencias.md` §8.1.
- A linha de relabel está retratada. A submissão 11 deu 0,48 público contra 0,546 estimados, o que mostra que a CV estava inflacionada. O NB16 mede o canal em 32,4%, mas a experiência honesta usa uma regra de seleção diferente da do relabel original; o mecanismo ainda precisa de uma repetição com a mesma regra. Fonte: sub11, NB16 e `docs/plano_acima05.md` §6.
- Dos 5 662 erros do XGB-311 honesto, 72,3% são de uma faixa e 27,7% de duas ou mais. Os blocos 11 a 14 e 4 a 7 concentram 68,9% dos erros. Fonte: `docs/diagnostico_erros.md` §a.
- As três maiores confusões adjacentes, 9 e 10, 11 e 12, e 12 e 13, somam 28,9% de todos os erros. A classe 13 é a pior, com F1 0,273. Fonte: `docs/diagnostico_erros.md` §a.
- O bloco de routing do questionário representa cerca de 36% da massa absoluta de SHAP. A concordância entre SHAP e permutação é rho 0,468, por isso a importância deve ser lida como descritiva, não causal. Fonte: NB14, com a ressalva do modelo relabel, e `docs/diagnostico_erros.md` §b.
- A análise não supervisionada com KMeans e k=16 não recupera as faixas. Na base 339, o ARI é 0,11; na base 311, é 0,0907. Isto sustenta a tese de que o sinal de idade está difuso. Fonte: NB06, com a base declarada.
- Nenhuma das alternativas testadas, thresholds, monotonia, pairwise, target-mean, quarentena, bagging, lambdarank, TabPFN e combos, superou a submissão 7 com honestidade. A lista completa está em `docs/plano_acima05.md` §6. Fonte: `docs/pesquisa_quebra_platamar.md` §13.
- O valor de 0,62 atribuído a um terceiro colega é um score público, sem código disponível. Não é reproduzível e não deve ser usado como meta do projeto. Fonte: `docs/mapa_aplicavel.md` §4.

## 9. Glossário mínimo

- Macro-F1. Média não ponderada do F1 das 16 classes. Uma classe rara pesa tanto como a maioritária.
- Off-by-one. Erro de uma faixa, com cerca de 5 anos. É o erro dominante no projeto.
- Routing. O percurso do questionário, ou quem responde o quê. Codifica idade e sexo e pode inflacionar a CV.
- Relabel. Substituir rótulos de treino por previsões de um modelo. Se o modelo gerador viu o fold de validação, há vazamento.
- Quarentena. Remover ou relabelar as linhas piores por confiança. Se a confiança é calculada no dataset todo, é canal de seleção.
- Frank e Hall. 15 classificadores binários cumulativos, um para cada P(y>k). É a base da sub7.
- Sub7. O ficheiro `submissions/sub07_ordinal_3modelos_mediana.csv`, com três famílias ordinais, mediana e correção. Tem 0,5089 em CV e 0,51 público.
- Base 339 e base 311. A base 339 é a da sub7, do NB03 e do NB04. A base 311 é a dos NB16 a NB18 e resulta de remover 28 colunas dos blocos PE, PN, SS e MH, mas não é uma cópia da 339 menos 28: o `N_NAO_APLICAVEL` é recalculado depois do corte e muda em 10 702 linhas. Para reproduzir a 339, é preciso correr o NB02 anterior ao §4.1, não concatenar colunas.
- Teto honesto. O melhor Macro-F1 que sobrevive a validação sem vazamento. É 0,504 a 0,509.
- Ecrã holdout. Avaliação rápida num único split 80/20 com barra de +0,010, antes de gastar um 5-fold.

## 10. Índice de fontes

| Pergunta | Documento |
|---|---|
| O que é o projeto, os dados e as regras | `docs/contexto.md`, `docs/info_kaggle.md` |
| Que decisões estão congeladas e porquê | `docs/plano_modelacao.md`, D1 a D5 e §7 |
| O que foi experimentado e qual o veredito | `docs/plano_acima05.md` §6, `docs/pesquisa_quebra_platamar.md` §13 |
| Onde falha o modelo | `docs/diagnostico_erros.md` |
| O que se aproveita do curso | `docs/mapa_pl.md`, `docs/mapa_t.md`, `docs/mapa_aplicavel.md` |
| Como correr o pipeline | `docs/runbook.md` |
| O que o colega fez nas submissões 1 a 7 | `registo_experiencias.md` |
| Esqueleto do relatório | `docs/relatorio.md` |
| Que notebook está fechado e porquê | `notebooks/arquivo/README.md`, `docs/INDICE.md` |
