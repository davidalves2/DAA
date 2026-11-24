# Runbook — ordem de execução dos notebooks

> Layout desde 23 nov 2026: pipeline na raiz de `notebooks/`; experiências
> fechadas em `notebooks/arquivo/` (congeladas, ver o README lá).
> NB04/05 são evidência histórica (base antiga 409; CSVs em `*_old.csv`).
> O pipeline final é NB01 → NB02 → NB03 → NB06.
> O NB08 está arquivado (experimental, base 339). Correr sempre de cima a baixo, kernel `daa`.

| # | Notebook | Base | Tempo | Esperado | Re-correr? |
|---|---|---|---|---|---|
| 01 | EDA e limpeza → `train_nb01.csv` | raw | ~5 min | 10962×262, 260 features + alvo, sem NaN em texto | Só se o raw mudar |
| 02 | Codificação+seleção+idade+drop ruidoso → `train_nb02.csv` (311+alvo) | nb01 | ~5 min | 10962×313, 0 NaN, `FAIXA_MINIMA` presente | Só se NB01 mudar |
| 02-339 | NB02 de `4ae5b6c` → `train_nb02_339.csv` / `test_nb02_339.csv` | nb01 | ~5 min | 10962×341 e 3655×340, 339 features | Só se a entrega for refeita |
| 03 | Comparação + GB tunado + ordinal + sub07 | nb02-339 | 30 a 60 min | tuned ~0.474–0.475; CV 0.5089; re-execução 0.5087 e 3507/3655 iguais à sub07; `sub07_oof.npz` | Congelado; o assert recusa a base 311 |
| 06 | Anomalias + retrato + KMeans + PL9 T5 | OOF sub07, senão combined/nb04 | ~6 min | anomalias da sub07, tabela 16 faixas, bloco PL9 | Sempre que a OOF mudar |
| 16 | Vazamento do relabel + CV honesto (prova do teto 0.504–0.509) | OOF | — | medida +0.036…+0.061, veredito | Congelado (evidência) |
| 18 | Bancada de triagem (holdout único, só XGB) | nb02 (311) | min–h | S1/S2/T17/T18 = únicos ≥ barra | Ativa |
| 04/05 | Baselines, ablação, encodings, stacking, tuning | base antiga | ~2h | Congelados com outputs (0.466→~0.476) | NÃO — só com `*_old.csv` restaurados |
| arquivo/ | NB05, NB07–NB15, NB17, NB19 (rejeitados/retratados/parked) | — | — | Congelados, ver o README lá | NÃO re-correr |

## Base 339 da entrega

O NB02 atual aplica o §4.1 (remove 28 colunas de `PE`, `PN`, `SS`, `MH`) e escreve
311 features. A sub07 usa a base anterior ao §4.1, com 339 features. As duas bases
não são uma cópia uma da outra: o `N_NAO_APLICAVEL` é recalculado depois do corte e
muda em 10 702 linhas. Para regenerar a 339:

1. Extrair o NB02 de `4ae5b6c` para `notebooks/.tmp_nb02_339.ipynb`.
2. Trocar as duas linhas de `to_csv` para `train_nb02_339.csv` e `test_nb02_339.csv`.
3. Executar de cima a baixo. O resultado tem 10962×341 e 3655×340 no CSV, com `PID`.
4. Confirmar o SHA-256 contra `data/nb02_339_manifest.json` (manifesto local).

O NB03 lê `train_nb02_339.csv` por omissão. Se o ficheiro faltar, para com uma
mensagem que aponta para aqui. A variável `NB02_BASE` permite escolher outro nome,
mas o assert de 339 features recusa qualquer base com outra contagem.

## Regras

- `PID` é coluna na base nova (era índice na antiga): nunca entra nas features.
- Ficheiros `*_old.csv` = backup da base antiga; não apagar até à entrega.
- Submissões: manuais, poucas, só por decisão registada no plano (passo 5).
- Checkpoints em `data/nb05_*.npz|json`: apagar o ficheiro força reexecução da célula.
- O NB03 só substitui a sub07 quando as previsões coincidem; caso contrário escreve `sub07_ordinal_3modelos_mediana_reproduzido.csv`. Fazer backup do CSV antes de qualquer re-execução.
