# Índice de `docs/` — mapa dos relatórios e notas (13 documentos)

> Registo criado 2026-11-23 na reorganização. `docs/*.md` é versionado; figuras
> (`fig_*.png`), PDFs, textos extraídos e CSVs ficam locais (`.gitignore`).
> Em caso de dúvida sobre decisões, o plano congelado manda: `plano_modelacao.md`.

## Decisões e estado (ler por esta ordem)
| Doc | O que é |
|---|---|
| `contexto.md` | Ponto de entrada: tarefa, dados, estado dos notebooks, regras de trabalho |
| `plano_modelacao.md` | **Fonte de verdade** das decisões congeladas (D1–D5) + log de experiências |
| `plano_acima05.md` | Plano para partir o teto 0.50: diagnóstico §0, tracks, **log §6 (a morada de todas as experiências)** |
| `runbook.md` | Ordem de execução dos notebooks e o que cada um regenera |

## Resultados e experiências
| Doc | O que é |
|---|---|
| `registo_experiencias.md` | (na raiz, versionado) registo do colega — submissões 1–7 e secções 1–21 || `diagnostico_erros.md` | Estrutura de erros honesta: 72% off-by-one, bolsos por classe |
| `pesquisa_ganhos.md` | Pesquisa externa: ganhos além de 0.51 (losses ordinais, etc.) |
| `pesquisa_quebra_platamar.md` | Pesquisa externa (10 nov): 9 threads contra fontes primárias + shortlist executada; §10 "não incomodar" |
| `mapa_pl.md` | Mapa PL1–PL8 (58/58 ficheiros do curso prático) com tags de aplicação |
| `mapa_t.md` | Mapa da pasta de teoria T (9 PDFs / 231 pp) com tags de aplicação |
| `mapa_aplicavel.md` | **Síntese dos dois mapas**: itens priorizados com specs fold-safe + ordem de execução |

## Notebooks (`notebooks/` — raiz = pipeline+prova; `arquivo/` = fechados)
| Onde | O quê |
|---|---|
| raiz | 01 EDA → 02 encoding 311 → 03 modelo final · 04 interpretabilidade · 06 anomalias · 16 prova do leak · 18 bancada ativa |
| `arquivo/` | 05, 07–15, 17, 19 congelados (rejeitados/retraídos/parked; ver README lá) |

## Referência
| Doc | O que é |
|---|---|
| `info_kaggle.md` | Regras da competição (3 submissões/dia, formato, 30/70) |
| `relatorio.md` | Estrutura/esboço do relatório de jan 2027 |
| `mapa_routing.md` | Gates de routing do questionário (questão × idade × sexo) |

## Não versionados (locais)
- Figuras `fig_nb14_*.png`, `fig_routing.png` (NB14/NB01) · `INS2019_questionario_CAPI.pdf` +
  `.txt` (extraído) · `dicionario_variaveis.csv`, `mapa_routing.csv` (dados).
