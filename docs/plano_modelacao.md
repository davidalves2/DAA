# Plano de Modelação — Notebook_04 e seguintes

> Documento de decisão partilhado do grupo. > Documento de decisão partilhado do grupo. Origem: síntese das discussões do projeto
> (espírito do `to-spec`, adaptado: sem user stories e sem detalhes de código) após
> entrevista de grill ao roadmap inicial. Estado atual: árvore de decisão `max_depth=8`,
> Macro-F1 CV 0.407, `sub01_arvore_referencia.csv` submetida (~40% público).
> Última atualização: 2026-10-26.

**NB08 (27 out, base 339):** Frank & Hall cumulativo **0.4979** (+0.0243, 5/5 folds;
argmax 0.4935, blend 0.5 com XGB 0.4973). Coarse→fine 0.4842 (+0.0106, 5/5);
pairwise 0.4829 (+0.0093); soft labels τ=0.5 0.4816 (+0.0080). Calibração corrige
ECE 0.105→0.048 mas **não muda F1**; stack cross-fitted vale +0.0055 (não +0.01).
Correção de confusão 16×16 e pesos alternativos rejeitados. **E8 (combinações
F&H+coarse→fine+pairwise) não bate o F&H** (0.4982 vs 0.4979, empate técnico).
E9 (pesos fronteira), E10 (pairwise suave), E11 (recusadores), E13 (MLP),
E17 (CatBoost) rejeitados; E15 (thresholds) +0.0019 e E16 (LightGBM) +0.0007
empates técnicos; E14 (seed-average) +0.0047 marginal. **E12 metastack
(logreg C=0.1 sobre OOFs XGB+HGB+F&H+E4): 0.5037, 5/5 — CONGELADO.**
E12b (5 bases: +seedavg+LGBM) 0.5027, sem ganho. `sub08_metastack.csv` GERADA
e validada (3655 linhas, formato ok; submissão manual pendente); sub07 fica como
fallback. E18 (LGBM tunado) 0.4654, E19 (XGB tunado) 0.4723, E20 (CatBoost
tunado) 0.4457: tuning não bate defaults (4 rejeições); E21 (MI top-150) e
E22 (KMeans como features) na fila. **NB17 (vias radicais, base 311): base
generativa Naive Bayes 0.35, blend com XGB-311 0.40 (rejeitado: destrói a base
0.478); self-training aninhado honesto 0.4778 → 0.4770 (rejeitado: pseudo-labels
confiantes não acrescentam informação).** NB18 (bancada de triagem: holdout único
80/20, só XGB, baseline 0.4700) fecha o lote T0–T8+C1, tudo abaixo da barra de
+0.010: T6 quarentena honesta +0.0069 (Q2 era canal), T5 target-mean +0.0050,
T7 pairwise +0.0047 (C2 saltado pelo gate), resto ≤ +0.001 ou negativo.
Nada vai a completo. NB18 lote 2 (T9–T12): T10 pairwise-15 +0.0162 no holdout
→ P1 confirmacao 5-fold na NB17 a correr; T12 monotonicas +0.0095 arquivado
(abaixo da barra por 0.0005); T9 smoothing +0.0059; T11 auto-rejeitado pela
regra (>1% violacoes, como IN25). **P1–P4 (NB17) confirmam em 5-fold
(base xgb311 0.4777): P2 target-mean+pairwise 0.4886 (+0.0109, 4/5);
P3 mono+pairwise 0.4875 (+0.0098, 4/5); P1 pairwise 0.4854 (+0.0077, 4/5);
P4 gated 0.4835 (+0.0058, 4/5). Maior ganho honesto de tratamento desde T7a.**
Transferencia para o ensemble ordinal (sub7) parqueada por decisao. **NB19
(regras de decisao sobre OOFs congeladas, guia de `docs/pesquisa_ganhos.md`):
bias aninhado +0.0005 (E15 era otimismo) e -0.0038 no F&H; suavizacao adjacente
+0.0023/+0.0005; mediana +0.0047 (abaixo da barra; 9<->10 mexe 635->624, sem
resolver). Alavanca de decision-time tambem seca; B (re-decodificacao
rank-consistente) parqueado (exige 75 refits).** **Combinacoes C1–C6: C3 +0.0239
e C4 +0.0220 no holdout, mas P2/P3 confirmam so +0.0109/+0.0098 (sub-aditivas);
C6 kitchen sink +0.0087 < melhor single (T14 +0.0210) → sinergia encerrada.
Transfer T1/T2/T6–T14 para o ensemble ordinal morto por S1b (-0.005).**

**Rerun combinada na base nova (339) — FECHADO 27 out:** sanity HGB 0.4674;
stack t*=0.55 0.4735; escalas 0.4717; FINAL stack+escalas+correção **0.4704** <
sub04 (0.475). `sub05_combined.csv` gerada mas **NÃO submeter** — não bate o
candidato do colega. NB06 refrescado: 260 anomalias (era 466). Competição congela
aqui; nada mais a correr do lado modelação.
> (espírito do `to-spec`, adaptado: sem user stories e sem detalhes de código) após
> entrevista de grill ao roadmap inicial. Estado atual: árvore de decisão `max_depth=8`,
> Macro-F1 CV 0.407, `sub01_arvore_referencia.csv` submetida (~40% público).
> Última atualização: 2026-10-26.

## 1. Problema

Prever `AGE_RANGE` (16 faixas, 0–15) a partir do perfil biológico/comportamental/
socioeconómico do INS 2019. Métrica oficial: **Macro F1-Score** (peso igual às 16
classes), não accuracy. Constrangimentos: 3 submissões válidas/dia no Kaggle; score
final = 30% leaderboard público + 70% split privado oculto; 10962 linhas treino /
3655 teste; classes desequilibradas (rácio 6.17x; `18-19` só 172 casos).

## 2. Decisões tomadas (e porquê — resultado do grill)

### D1. Só sklearn no Notebook_04, sem dependências novas
**Decisão:** modelo principal com `HistGradientBoosting` (+ `RandomForest` como
segundo ponto de comparação). Nada de LightGBM/CatBoost/XGBoost para já.
**Porquê:** ambiente `daa` idêntico para todo o grupo, zero fricção de instalação,
reprodutibilidade garantida no checkpoint de 30 nov / 3 dez. O salto árvore simples →
boosting é o maior ganho esperado mesmo sem sair do sklearn.
**Revisitada 27 out (XGBoost 3.4.1 instalado no env):** XGB default 0.4736 ± 0.0062
nas folds fixas — melhor default isolado. **Promovido a base 27 out: XGB FINAL
stack(t*=0.45)+escalas+correção 0.4841 > sub04 (0.475) com margem → ADOTADO.**
`sub06_xgb.csv` pronto (expectativa pública ~0.47). Ensemble HGB+XGB testado e
descartado (0.472–0.473: erros correlacionados, média dilui).
Notas: precisa matriz numpy (one-hots com `<` nos nomes) + `sample_weight`
(sem `class_weight` multiclasse).

### D2. Encoding: comparar, não apostar
**Decisão:** testar em paralelo, com o mesmo protocolo CV: (a) baseline atual
(one-hot, 409 features); (b) `MAINSTAT`/`MARSTALEGAL`/`HATLEVEL` recodificadas como
**ordinais manuais** com ordem documentada (ex. estudante → empregado → reformado);
(c) restantes nominais raras com frequency encoding ou agrupamento de categorias raras.
**Porquê:** o grill mostrou que a aposta única é o maior risco — a ordem "correta" de
`MARSTALEGAL` (solteiro → … → viúvo) não é óbvia e `MAINSTAT` mistura inatividade
eterogénea. Comparar em CV resolve empiricamente e dá material para o relatório.
**Regra:** cada ordem ordinal adotada tem de ser justificada por escrito (tabela no
notebook), porque o relatório é avaliado nessa justificação.

### D3. Ordinalidade do alvo: comparar três formulações
**Decisão:** (a) multiclasse puro como referência; (b) regressor 0–15 contínuo +
arredondamento; (c) stacking simples classificador + regressor. Frank & Hall /
XGBOrdinal ficam como trabalho futuro explícito.
**Porquê:** o alvo é ordenado e os erros off-by-one dominam certamente a matriz de
confusão, mas a métrica continua a ser Macro-F1 nominal — nenhuma formulação ganha
por decreto. Comparar as três no mesmo CV e só depois decidir.

### D4. CV decide, Kaggle confirma; submissões sempre manuais
**Decisão:** 5-fold estratificada com Macro-F1 + 1 holdout estratificado interno
decidem tudo. Kaggle só para confirmação, submissões escassas e **sempre manuais**
(pelo próprio, nunca automatizadas pelo notebook).
**Porquê:** com final 30/70 público/privado, escolher pelo público é overfitting ao
split visível. Desvio público→privado > ~2–3pp = sinal de alarme, não de sucesso.
As 3 submissões/dia gastam-se em confirmação, não em tuning.

### D5. Features: leque comparável, em blocos
**Decisão:** construir blocos de features ligáveis/desligáveis para ablação:
F0 base atual; F1 clipping/log de outliers (`IN59` max 8000, `IN55` max 7000,
`HO34_COD`, `IN57`); F2 contagens (`n_NãoAplicável`, `n_SemResposta` por linha);
F3 agregados clínicos (soma limitações `PL*/PC*/HA*`, contagem comorbilidades `CD1*`,
IMC de `BM1/BM2`); F4 interações (`MAINSTAT×HATLEVEL`, `MARSTALEGAL×HHTYPE`).
**Porquê:** permite tabela de ablação F0 → F0+F1 → … no relatório, que é exatamente
o que a avaliação pede ("como e em que hiperparâmetros", "análise crítica").
**Nota de honestidade para o relatório:** F2/F3 exploram em parte o *routing* do
questionário (ex. autonomia só perguntada a 55+), não pura biologia — assinalar como
limitação na parte de anomalias (ver §5).

## 3. Protocolo de validação (fixo para todas as comparações)

1. Mesma `StratifiedKFold(n_splits=5, shuffle, seed=42)` e mesmo holdout para todas
   as variantes — sem isto as comparações são inválidas.
2. Métrica primária: Macro-F1; secundárias: F1 por classe + matriz de confusão
   (obrigatórias — hoje não existem e escondem onde falhamos: classes 0/1 jovens?).
3. `class_weight='balanced'` como default nos modelos que suportam; registar sempre.
4. Cada experiência: 1 linha numa tabela de resultados (variante, Macro-F1 médio ± dp,
   F1 nas 3 piores classes). Sem tabela, a experiência não conta.

## 4. Próximos passos (ordem)

1. [x] Notebook_04, parte 1: `HistGradientBoosting` + `RandomForest` sobre as 409
       features atuais, com o protocolo do §3. Resultado: HGB **0.4657** (RF 0.4463,
       árvore 0.4072 reproduzida). Abaixo dos ~0.50 esperados.
2. [x] Matriz de confusão + F1 por classe do melhor modelo. Piores: classes 1, 13, 4;
       72% dos erros off-by-one → formulações ordinais no NB05.
3. [x] Blocos F1–F3 + tabela de ablação. Nenhum ganho fora do dp
       (F0 0.4657 / +F1 0.4650 / +F2 0.4628 / +F3 0.4657). Segue conjunto completo
       por desempate técnico.
4. [x] Comparação de encodings (D2) e de formulações do alvo (D3). Encodings em
       empate técnico (one-hot 0.4657 / ordinal 0.4636 / raro 0.4629). Stacking
       t*=0.5 adotado (~+1pp, OOF ~0.476 combinado com escalas).
5. [x] 1ª submissão de confirmação no Kaggle (manual): `sub02_stack.csv` →
       **0.46 público** (vs ~0.476 OOF; gap −0.016 dentro do tolerado: otimismo das
       escalas sintonizadas na OOF + wobble run-to-run). Sem alarme (>2–3pp seria).
6. [x] Bloco F4 + thresholds por classe em OOF. F4 rejeitado (0.4612 < 0.4657).
       Escalas: +0.5pp sozinhas, ~+0 sobre stacking (fixes sobrepostos). Tuning
       rejeitado (full-CV 0.4637 < default).
7. [ ] Congelar modelo final por CV; escrever secção de metodologia do relatório a
       partir deste documento.
8. [x] NB06 anomalias + retrato por faixa: 466 anomalias (207 envelhecidas / 259
       rejuvenescidas), casos erro-vs-biologia identificados, tabela-mãe 16 faixas.
       Falta ainda o mapa routing pergunta×idade×sexo (§7 item 4).
9. [x] NB08 fronteiras/ordinalidade/calibração (base 339, 5-fold): F&H 0.4979
       (+0.024, 5/5); coarse→fine 0.4842 (+0.011, 5/5); pairwise 0.4829 (+0.009);
       soft labels 0.4816 (+0.008); calibração ECE 0.105→0.048 sem ganho de F1;
       confusão 16×16 e pesos alternativos rejeitados. E9–E11, E13, E17 rejeitados;
       E14/E15/E16 marginais ou empates. **E12 metastack 0.5037 (5/5) CONGELADO**;
       E12b (5 bases) 0.5027 sem ganho. Falta sub08 manual (fiel à OOF).

## 5. Fora de âmbito (para já)

- LightGBM / CatBoost / XGBoost / Optuna (rever por D1).
- Frank & Hall / XGBOrdinal / losses ordinais (CORAL/ORDER).
- Seleção automática de features e interpretabilidade SHAP (entra na fase de relatório,
  após modelo congelado).
- Publicar specs em issue tracker / aplicar labels de triage (o `to-spec` original
  pressupunha `setup-matt-pocock-skills`, que não foi corrido neste repo).

## 7. Melhorias identificadas (revisão de dados 27 out — NB01–03 + sondagens)

Origem: heatmaps/boxplots dos NB01–03 e 6 sondagens novas. Diagnóstico central: o
problema é **separação intra-grupo** (reformado→97% nos 60+, empregado→96% nas 2–10;
o jogo está dentro dos empregados e dos 55+), e vários top preditores são routing
escondido (`IN45` só ex-fumadores, `IN2/3/4` só migrantes). Sem shift treino→teste
(0.16pp médio). Por valor esperado:
1. [x] F4 v2 intra-grupo: `SEX×MARSTALEGAL` (viúvo 70+ é 4:1 mulheres), `EMP×filhos`,
       `EMP×IN45`. Resultado: 0.4621 vs base 0.4662 (delta −0.004, dentro do dp) —
       REJEITADO. Lição: o HGB já captura interações de baixa ordem via splits; produtos
       manuais são redundantes. Fecha-se a porta das interações com número.
2. [x] Filtro de variância/quase-mortas. 47 binárias raras (<0.5%) + 4 constantes
       droparam-se (413→366). Resultado: 0.4666 vs 0.4657 (delta +0.0009, ruído).
       ADOTADO como higiene (treino ~11% mais leve, SHAP mais limpo), sem pretensão de score.
3. [ ] Segmento recusadores (168 pessoas com >20 "Sem resposta").
7. [x] Frank & Hall ordinal, 28 out (15 binários HGB, mesmas folds): 0.4755,
       0.4764 com correção — empata o stacking, abaixo da receita XGB (0.4841).
       Formulação NÃO é o gargalo; teto é pobreza de sinal intra-grupo.
4. [x] Mapa routing pergunta×idade×sexo: `docs/mapa_routing.md` + `mapa_routing.csv` +
       `fig_routing.png`. 133 universais / 31 gate-55+ / 51 parciais / 10 raras /
       2 gate-sexo. Achados: `IN11` gate de velhice com gradiente, reprodutivo com
       corte ~60, trabalho gate-emprego. Valida FAIXA_MINIMA; define anomalia.
5. [ ] Consolidação no NB02 após freeze (dívida técnica: NB04/05 duplicam lógica).
6. [x] Limpeza de impossíveis auditada e REJEITADA para treino: 7 linhas
       (aluno 70+, reformado <35, faixa_min>real; +1 casado 15-17 só auditoria).
       Controlo limpo mesma base: 0.4608 sem vs 0.4650 com (delta −0.004, ruído).
       A lista (`data/impossiveis.json`) vive no relatório, não no treino.

Não fazer: re-testar espaçamentos ordinais, SMOTE, dropar por baixa correlação-alvo.
8. [x] E1 severidade ADL (N_SEV_ADL + ponderada) para a cadeia velha, 28 out:
       0.4623 vs sanity 0.4674 — NEGATIVO. Fecha-se a via "features para pares":
       nem o gradiente de severidade ajuda o HGB (redundante + ruído).
9. [x] Hierárquico (3 especialistas HGB jovem/meia/sénior + meta-logística
       sobre 64 probas OOF), 28 out: **0.4747 vs global 0.4657 (delta +0.009)**.
       Confirma o diagnóstico intra-grupo, mas abaixo da receita XGB (0.4841).
       Variante XGB dispensada: empilhar mais pilotos no mesmo CV acumula
       otimismo de seleção. **Modelação congela aqui.**

## 6. Riscos assumidos

- O sinal de *routing* (quem responde o quê) pode inflacionar a CV e não corresponder
  a biologia — aceitar no Kaggle, declarar no relatório.
- `IN29` (skew −3.1, quase constante) e similares podem ser ruído — a ablação dirá.
- pandas 3.x no env `daa`: atenção a `select_dtypes(object)` vs `str` (já há warnings
  nos notebooks) — não partir o pipeline do Notebook_01/02 ao mexer.
