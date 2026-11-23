# Runbook — ordem de execução dos notebooks

> Layout desde 23 nov 2026: pipeline na raiz de `notebooks/`; experiências
> fechadas em `notebooks/arquivo/` (congeladas, ver o README lá).
> NB04/05 são evidência histórica (base antiga 409; CSVs em `*_old.csv`). O pipeline
> final é NB01 → NB02 → NB03 → NB06 (NB07 arquivado: 0.4704 < sub04, não submeter).
> O NB08 está arquivado (experimental, base 339). Correr sempre de cima a baixo, kernel `daa`.

| # | Notebook | Base | Tempo | Esperado | Re-correr? |
|---|---|---|---|---|---|
| 01 | EDA e limpeza → `train_nb01.csv` | raw | ~5 min | sem NaN em texto; asserts passam | Só se o raw mudar |
| 02 | Codificação+seleção+idade+drop ruidoso → `train_nb02.csv` (311+alvo) | nb01 | ~5 min | 10962×313, 0 NaN, `FAIXA_MINIMA` presente | Só se NB01 mudar |
| 03 | Comparação + GB tunado + correção (colega) | nb02 nova | ~30 min | tuned ~0.474–0.475; sub04 | Congelado salvo erro |
| 06 | Anomalias + retrato (lê `combined_oof.npz`, fallback `nb04_oof.npz`) | OOF | <1 min | 260 anomalias; tabela 16 faixas | Sempre que a OOF mudar |
| 16 | Vazamento do relabel + CV honesto (prova do teto 0.504–0.509) | OOF | — | medida +0.036…+0.061, veredito | Congelado (evidência) |
| 18 | Bancada de triagem (holdout único, só XGB) | nb02 | min–h | S1/S2/T17 = únicos ≥ barra | Ativa |
| 04/05 | Baselines, ablação, encodings, stacking, tuning | base antiga | ~2h | Congelados com outputs (0.466→~0.476) | NÃO — só com `*_old.csv` restaurados |
| arquivo/ | NB05, NB07–NB15, NB17, NB19 (rejeitados/retraídos/parked) | — | — | Congelados, ver README lá | NÃO re-correr |

> NB07 (validação combinada → `sub05_combined.csv`) foi arquivado: final 0.4704 < sub04,
> morte registada em `plano_acima05.md` §6 (12 nov). NB08 arquivado idem (E12 superseded).

## Regras

- `PID` é coluna na base nova (era índice na antiga): nunca entra nas features.
- Ficheiros `*_old.csv` = backup da base antiga; não apagar até à entrega.
- Submissões: manuais, poucas, só por decisão registada no plano (passo 5).
- Checkpoints em `data/nb05_*.npz|json`: apagar o ficheiro força reexecução da célula.
