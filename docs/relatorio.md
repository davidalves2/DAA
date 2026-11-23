# Relatório TPG — esqueleto (20 pp máx.)

> Preenchível a partir dos notebooks + docs. Estado: números e figuras prontos;
> texto corrido por escrever. Convenção: [PRONTO] = copiar dos artefactos;
> [FALTA] = trabalho em aberto.

## 1. Áreas, objetivos e como lá chegar [PRONTO — resumir docs/plano_modelacao.md §1]

Prever `AGE_RANGE` (16 faixas) a partir do INS 2019; usar previsões para anomalias
de perfil; caracterizar faixas. Duas pistas exploradas em paralelo (árvores/boosting
+ stacking vs GB tunado + idade mínima), convergentes (~0.47).

## 2. Metodologia [PRONTO]

- Protocolo fixo: `StratifiedKFold` 5×seed 42, Macro-F1, `class_weight='balanced'`,
  1 holdout interno; CV decide, Kaggle confirma (30/70 público/privado, 3/dia).
- Decisões D1–D5 + §7 (docs/plano_modelacao.md): só sklearn; comparar, não apostar;
  tuning por último; rejeições com número (encodings, F4/F4v2, tuning, votação,
  oversampling descartado por análise).

## 3. Dados e processamento [PRONTO]

- NB01: 10962/3655, 236 vars, códigos de falta, `_ESTADO`, conversões (HHMM, PA1, IN24).
- NB02 (+ colega): 409 → 333 (seleção, +0.006) → 339 (+idade mínima, +0.008);
  novas features sem ganho no GB; `resumo_variaveis.csv`; `docs/mapa_routing.md` +
  `fig_routing.png` (133 universais / 31 gate-55+ / 51 parciais / 10 raras / 2 gate-sexo).
- Sem shift treino→teste (0.16pp). WGT excluída (leakage). [FALTA: texto corrido + EDA
  descritiva do NB01 em prosa.]

## 4. Modelos [PRONTO — tabela]

| Modelo | CV | Kaggle |
|---|---|---|
| Árvore depth=8 | 0.407 | 0.40 (sub01) |
| LogReg | 0.427 | 0.43 (sub02 colega) |
| RF 200 | 0.446–0.450 | — |
| HGB/GB default | 0.466–0.467 | 0.46 (sub03; sub02_stack) |
| GB tunado + correção | 0.475 | ... (sub04, pendente) |
| HGB + stack + escalas | ~0.476 | — (não submetido: empate) |
| Combinado base nova | 0.4704 | — (não submetido: abaixo sub04) |
| XGB multiclasse (base NB08) | 0.4736 | — |
| Frank & Hall (mediana) | 0.4979 | — (sub07 fallback, não submetida) |
| Metastack XGB+HGB+F&H+E4 (congelado) | **0.5037** | — (sub08 em geração) |

Hiperparâmetros: defaults + balanced, salvo tuning documentado (halving rejeitado,
refined search sem ganho). [FALTA: curvas de validação como figuras.]

## 5. Resultados e análise crítica [PRONTO — números; FALTA texto]

- Teto intra-grupo: regras baratas resolvem o grosso (reformado→97% 60+...); o jogo é
  dentro dos empregados (44%, sinal restante: IN45, filhos, altura) e 55+ (autonomia).
- 72% dos erros off-by-one → ordinalidade; stacking +1pp, escalas +0.5pp sozinhas,
  ~+0 combinadas; F4/F4v2/interações redundantes (HGB já captura).
- Wobble run-to-run ±0.01; escalas com otimismo OOF; distorção de prior (classe 13).
- Figuras: confusão NB04/NB05, F1-por-classe, ablação, curvas tuning.

## 6. Anomalias [PRONTO — NB06]

- 260 anomalias (|gap|≥3) na OOF final (466 na fraca): 119 envelhecidas / 141
  rejuvenescidas; menos comorbilidades que o normal (perfil, não doença).
- Casos erro-vs-biologia (12012, 14056 vs 13301, 1265, 7191...).
- Retrato das 16 faixas (tabela-mãe). [FALTA: controlar routing por caso (mapa §7.4)
  e redação final.]

## 7. Recomendações [FALTA — rascunhar]

- Recolha: perguntas-ponte 20–59 (o vazio de sinal), validar registos impossíveis à
  entrada (estudante 75–79), rever caudas IN59/IN55.
- Modelação futura: LightGBM (revisitar D1) só com orçamento; poda de raras por
  defeito; `sample_weight` no regressor em vez de oversampling.
- Uso: rastreio de perfis desviantes com controlo de routing, não classificação crua.
