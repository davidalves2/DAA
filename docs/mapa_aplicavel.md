# Mapa aplicável — o que colher de PL1–PL8 + T para o assignment (síntese)

> **Data:** 11 nov 2026. **Fontes:** `docs/mapa_pl.md` (58/58 ficheiros PL1–PL8,
> leitura integral) + `docs/mapa_t.md` (9 PDFs / 231 pp da pasta T). Este documento
> é a segunda entrega pedida: o mapa combinado do que **aplicamos** ao assignment,
> com specs de experimento fold-safe e ordem de execução. Contexto: teto honesto
> 0.504–0.509; regra fold-regenerable (`HANDOFF.md:36-37`); lista "do not bother"
> em `docs/pesquisa_quebra_platamar.md` §10. Nenhum treino executado por este doc.

## 0. Veredito de topo (as duas pastas convergem)

1. **Nada no material de curso vale +0.11 honesto.** Os dois mapeamentos, feitos
   por agentes independentes, chegaram à mesma conclusão: PL1–PL8 é introdutório
   (Titanic/wine/incidents) e T é teoria de licenciatura sem uma folha sobre
   macro-F1, desequilíbrio, ordinalidade ou pesos de inquérito.
2. **A explicação mais provável do "0.62 com diferente data treatment" é otimismo
   de protocolo**, e ambas as pastas *ensinam exatamente o padrão*:
   - `T/Aula5 p.11–12` (verificado por render): stacking ingénuo — meta-modelo
     treinado nas previsões dos base **no próprio train**, sem cross-fitting;
   - `PL5/daa_pl_5_II-DT.ipynb` cél.46–78: tuning + escolha + relatório no
     **mesmo split**; `PL7`/`PL8`: um split único sem seed;
   - Histórico próprio: +0.01 de otimismo medido (`registo_experiencias.md:197`)
     e retração NB16 com inflação **+0.036…+0.061** (`plano_acima05.md:96`).
   → **Auditar o protocolo do colega é a ação de maior valor esperado do mapa
   inteiro** (§2, item A1) — antes de qualquer experimento nosso.
3. O que há de aplicável são **micro-alavancas honestas somadas: +0.005 a +0.015**
   (→ ~0.51–0.52), mais o piloto TabPFN-3 (viés indutivo novo, ±0.01 desconhecido,
   bloqueado na licença). **Isso não fecha sozinho para 0.62** — se o 0.62
   sobreviver a um audit honesto, a resposta está fora destas duas pastas.

## 1. MAPA APLICÁVEL — itens combinados, ordenados por valor esperado

Tags: DIRECT = aplicável agora com spec; DIAG = diagnóstico (não é ganho de score);
INDIRECT = relatório/metodologia; BLOCKED = não fazer (leakage/já rejeitado).

| # | Item | Fonte | Tag | Spec (protocolo fixo: 5-fold seed 42; ecrã NB18 barra +0.010) | Ganho esp. | Custo | Leak |
|---|---|---|---|---|---|---|---|
| **A1** | **Audit forense ao "0.62" do colega** — ler o código dele e checar 4 coisas: (a) meta/stack fitado em previsões **in-sample**; (b) transformações fitadas **antes** do CV (scaler/encoding/seleção/remoção de linhas); (c) colunas-alvo derivadas (`WGT` calibra por faixa etária); (d) labels regeneradas com o fold dentro (o nosso caso relabel) | `T/Aula5 p.11–13`, `T/Aula6 p.35`; padrão `PL5` cél.46–78 | **DIAG** | Só leitura do PR/notebooks dele; checklist acima; perguntar: CV 5-fold honesto? holdout? público? | **Explica o gap inteiro** se artefacto | 1–2 h | n/a |
| **A2** | **TabPFN-3 como família extra** — a própria cadeira passou a apontar para PFNs/TabFM/TabArena (`T/Aula6-ANN-1` p.38–39, criado 10/11) | `T/Aula6-ANN-1`; spec `pesquisa_quebra_platamar.md` §9.5 | **DIRECT** | fit por fold no train, CPU, verificar overlap de erros vs sub7 antes de média; barra +0.010 | **±0.01 desconhecido** (único viés novo) | 1–3 h | baixo — **bloqueado na licença** (`TABPFN_TOKEN`, decisão do utilizador) |
| **A3** | **F2: contagens de N/A por bloco** ("mask" / *features not applicable in all cases*) | `T/Aula2b p.5, p.23`; spec `plano_acima05.md:39` | **DIRECT** | Em NB09: agrupar dummies `*_Não aplicável` por domínio via `dicionario_variaveis.csv`; contagens por linha (fold-safe por construção) | 0 / +0.004 | ~2 h | baixo |
| **A4** | **Imputação estratificada + flag de missingness** — mediana do estrato (routing flags, ex. `MAINSTAT`) para sentinels em numéricos, estatísticas **só na parte treino do fold** | `PL4-II` cél.18, `PL6` cél.13, `PL3` ficha T3f/T3g | **DIRECT** | NB18 ecrã → 5-fold; nunca pré-computar no train todo (é a classe de erro NB16) | +0.000 / +0.005 | 0.5 d | baixo in-fold / **alto se pré-CV** |
| **A5** | **Binary encoding dos 22 nominais** (~3–8 bits/var) em vez de 171 one-hots —compressão semântica alinhada com a lição de diluição (NB09: −28 col = +0.0066) | `PL5-I` cél.61–66, `T/Aula2b p.24` | **DIRECT** | NB18 ecrã vs base 311; `category_encoders` via conda-forge (regra conda) | 0 / +0.004 | 0.5–1 d | nenhum (stateless) |
| **A6** | **Companheiros quantílicos** (`KBinsDiscretizer(strategy='quantile')`) das top contínuas (`IN15`, contagens, `N_DOENCAS`) | `PL3` cél.38, ficha T3c | **DIRECT** | fit por fold; sobretudo para membros lineares da família | 0 / +0.003 | 0.5 d | baixo |
| **A7** | **E22: features de KMeans fold-safe** — `CLUSTER_ID` + distância ao centróide, `K≈√n`, fit **só no train do fold** | `T/Aula8 p.7, p.17, p.19, p.34`; na fila `plano_modelacao.md:23` | **DIRECT** | NB18 ecrã; o slide avisa que K-Means "não lida bem com muitas features" (311 one-hots são hostis) | 0 / +0.003 | ~0.5 h | baixo in-fold / **alto se fitar no dataset todo** |
| **A8** | **Scores de anomalia como features** (LOF/IsolationForest/Tukey, fit por fold) — a versão segura de "limpar dados" | `T/Aula8 p.13–14`, `T/Aula2b p.19` | **DIRECT (c/ salvaguarda)** | score como feature num membro; **nunca remover linhas** (canal já retraído) | 0 / +0.002 | ~1 h | médio |
| **A9** | **Auditoria MDI vs permutação vs SHAP** para justificar ablação de blocos (F5) e o relatório — MDI infla cardinalidade alta (one-hots) | `PL7` cél.66–79 (PassengerId 0.179 → 0.016) | **INDIRECT** | no fold de validação; **nunca** como filtro automático (feature selection rejeitada: T4 −0.0037) | 0 (relatório) | 0.25 d | baixo / médio se virar filtro |
| **A10** | **Diagnóstico de inflação de protocolo** — correr o mesmo pipeline (a) estilo curso: 1 split, tune+escolha+relatório no mesmo split; (b) 5-fold honesto; medir a inflação nos nossos dados | padrão `PL5` cél.46–78; calibrado por `registo:197` e NB16 | **DIAG** | nunca usar (a) para seleção; resultado entra no relatório como explicação do gap | **explicação mensurável** do 0.62 | 0.5 d | intencional/ controlado |
| **A11** | Citações de relatório: 30/70 público/privado com overfitting (`T/Aula6 p.2`), checklist de leakage (`T/Aula6 p.35`), stacking vs blending (`T/Aula5 p.13`), CRISP-DM (`T/Aula2a`), **accuracy 0.62→0.64 com macro-F1 0.58→0.54** (`PL5-II` cél.51 — prova empíica métrica≠accuracy) | ambos os mapas | **INDIRECT** | matéria-prima dos 20 pp. | 0 | — | — |

### Bloqueado / não aplicar (junção das duas listas de LEAKAGE-RISK — não re-propor)

- **Stacking "como no slide"** (meta em previsões in-sample) — infla CV; já usamos OOF cross-fitted e mesmo assim esgotou.
- **Remoção de linhas por outlier/anomalia/filtro com a idade** (`PL3` ficha T3e "idade>70", `T/Aula8` data cleaning) = o canal de seleção que retractou E24–E27.
- **Estatísticas fitadas em todo o dataset antes do CV** (qui-quadrado/alvo, `PL3`/`T/Aula2b p.21`) — seleção pré-CV = leak + T4/T8 já rejeitados.
- **Escolher/tunar pelo leaderboard público** (o próprio `T/Aula6 p.2` rotula 0.85 público vs 0.785 privado de "Overfitting").
- Reconfirmado como irrelevante por ambas as pastas: ensembles de votação/bagging (T15 −0.005), MLP/ANN (E13), L1/L2/feature-selection, class-weight asymmetric (já `balanced`), datas (INS não tem), escala/normalização para GBDT.

## 2. Ordem de execução recomendada

1. **A1 primeiro (hoje, 1–2 h):** pedir ao colega o protocolo por trás do 0.62
   (notebooks/PR + como validou). Se for stacking in-sample / transformações
   pré-CV / holdout otimista → **o gap é protocolo, não sinal** — fica registado
   no relatório e a pressão para reinventar tratamento de dados desaparece.
2. **Decisão da licença TabPFN (A2)** — única aposta com viés indutivo novo; a
   cadeira endereça PFNs esta semana. Precisa do `TABPFN_TOKEN` (utilizador).
3. **Lote barato de micro-alavancas honestas (A3+A4+A6+A7)** — todos cabem no
   ecrã NB18 (barra +0.010) antes de qualquer 5-fold; ~1 dia no total.
4. **A5 (binary encoding)** se o lote 3 mostrar sinal de compressão; senão fica
   na fila.
5. **A10 (diagnóstico de inflação)** para o relatório, quando houver folga.
6. **Não fazer** nada da lista bloqueada, por mais que o 0.62 alegado aponte
   para lá — a nossa retração já pagou essa lição.

## 3. Reality check (honestidade numérica)

| Caminho | Contribuição plausível |
|---|---|
| Soma dos DIRECT honestos (A3–A8) | +0.005 … +0.015 |
| TabPFN-3 (A2, se licença + sinal) | ±0.01 desconhecido |
| **Total honesto alcançável** | **~0.51–0.525 — não 0.62** |
| A1 (se o 0.62 for artefacto) | explica 100% do gap sem mexer no modelo |

Ou seja: ou o 0.62 sobrevive a um audit e então está fora destas pastas
(perguntar ao colega *o que* fez é mais barato que qualquer experimento), ou não
sobrevive e o teto 0.504–0.509 mantém-se com as micro-alavancas acima.

## 4. Perguntas em aberto

1. ~~**0.62 = CV 5-fold, holdout, ou público?**~~ **RESPONDIDO 12 nov:** é um
   **score público do Kaggle**, de **outro colega** (não o do repo partilhado) —
   sem acesso ao código, A1 (audit forense) fica **inviável**; só pergunta casual
   se ele algum dia partilhar o método. O número não é ruído (~10σ acima de nós:
   a tabela pública tem σ≈0.01) mas é **inverificável até à entrega** — duas
   hipóteses vivas: (a) tratamento de dados genuinamente melhor (~0.62 honesto);
   (b) overfitting ao público (probing com ~40 submissões possíveis; risco de
   queda no privado 70%). **Espelho histórico:** o próprio repo tem um commit
   descartado `"0.59"` (0b83faa, 22 out) — LGBM CV 0.5823 sobre `train_clean.csv`
   limpo **globalmente antes do CV** (`01_EDA.ipynb` cél.9), logo seguido do
   reinício honesto (`"0.40 F1"`) — gap +0.10 da mesma magnitude do alegado 0.62.
2. Licença TabPFN aceite? (desbloqueia A2.)
3. Materiais da coluna PL "Qualidade e validação de modelos" (`T/Aula 0 p.9`)
   — a pasta T não tem o deck de validação; se existir noutra pasta, vale mapear.

## 5. Log

| Data | Item | Resultado |
|---|---|---|
| 11 nov | Síntese dos dois mapas (`mapa_pl.md`, `mapa_t.md`) numa lista única priorizada com specs fold-safe | Este documento |
| 11 nov | Convergência verificada: ambos os agentes, independentemente, apontam otimismo de protocolo/stacking ingénuo como explicação provável do 0.62 | §0, A1 |
| 11 nov | 11 itens DIRECT/DIAG + 4 famílias bloqueadas consolidadas; soma honesta estimada +0.005…+0.015 | §1, §3 |
