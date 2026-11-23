# notebooks/arquivo/ — experiências fechadas (evidência congelada)

> Criado 2026-11-23 na reorganização: a raiz ficou com o pipeline + prova
> honesta; tudo o que foi rejeitado/retraído/parked veio para aqui.
> **Estes notebooks estão congelados** — não re-correr, não editar. Os vereditos
> vivem em `docs/plano_acima05.md` §6 e `docs/plano_modelacao.md`.

## Porque foi cada um arquivado

| Notebook | Veredito (onde está registado) |
|---|---|
| 05 codificação/formulação/afinação | Histórico base antiga; runbook congela |
| 07 receita final base 339 | Fechado negativo 0.4704 < sub04, NÃO submeter |
| 08 fronteiras/ordinalidade | E12 frozen mas superseded; E8/E15/E16 empates, resto rejeitado |
| 09 auditoria features | T7a adotado (consolidado no NB02 §4.1); resto rejeitado/shelved |
| 10 BRF + DART | Rejeitado: sem augment sobre E12 |
| 11 data quality/quarentena | Q2 era canal de seleção do leak (NB16) — retraído |
| 12 compounding E23–E25 | Intermédio, superseded por E26+ |
| 13 relabel E26 | **Retraído (NB16): inflação +0.036…+0.061** |
| 14 explicabilidade E26 | Tainted: headlines sobre labels relabelled; figuras extraídas p/ `docs/` |
| 15 E27 4 famílias | **Retraído (NB16)** — artefacto do mesmo canal |
| 17 vias radicais | Tudo rejeitado (NB 0.35, self-train neutro) |
| 19 regras decision-time | Parked: bias +0.0005, mediana +0.0047 < barra |

## O que fica na raiz (e porquê)

01 (EDA/limpeza) → 02 (encoding 311) → 03 (modelo final sub7) → 06 (anomalias) =
pipeline + entrega; 04 = interpretabilidade da entrega do colega (main);
16 = prova do leak + teto honesto (metodologia); 18 = bancada ativa.

## Nota de paths

Os resolvers de `DATA_DIR` destes ficheiros aceitam as duas profundidades
(`notebooks/` e `notebooks/arquivo/`): `../../data` primeiro, depois `../data`,
depois `data`.
