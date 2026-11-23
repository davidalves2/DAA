# Mapa routing pergunta × idade × sexo

> Como ler: cada pergunta do questionário só é feita a um subgrupo (idade, sexo, situação).
> "Não aplicável" não é falta — é a sombra do guião do inquérito, e anda com a idade.
> Fonte: `docs/mapa_routing.csv` (taxa de NA por coluna × 16 faixas + sexo) e
> `docs/fig_routing.png`. Método: % de "Não aplicável" por faixa no `train_nb01`
> (sem shift treino→teste, pelo que o mapa vale para o teste).

## Perfis (227 colunas de texto)

- **universal (133):** perguntadas a todas (NA <5%). O grosso do questionário.
- **gate-55+ (31):** ~100% NA até aos 50–54, perguntadas depois. Inclui as 17 de
  autonomia (`PC1*`, `HA1*`, `PC2/3`, `IN15`), o bloco trabalho
  (`FT_PT`, `JOBSTAT/JOBISCO/LOCNACE_COD`, `AW1/AW2`), deslocações (`IN10/11/12`),
  `IN60`, `NT3` e os `_ESTADO` gémeos. Abaixo dos 55, zero informação nestas.
- **gate-sexo (2):** `PA7_NAC`, `PA8_NAC` — só mulheres, todas as idades.
- **rara/quase-morta (10):** NA >90% global (`IN13`, `IN62`, `AC2/AC_IN*`, `IN49`,
  `PE5`, `IN4/IN25/IN59_ESTADO`). Alvos do filtro de variância; várias já caíram.
- **parcial (51):** routing condicional sem corte único — bloco reprodutivo
  (`IN21/22/27`, `NT1/2`, `IN25/26`: perguntadas em idade fértil, corte ~60),
  tabaco/álcool por estatuto de consumidor (`IN45` só ex-fumadores, `IN2–4` só
  migrantes), seguros e saúde por utilização.

## Achados novos

1. **`IN11` é gate de velhice com gradiente** (NA 0.99 → 0.74 dos 70+ aos 85+), não
   corte binário como a autonomia. Nunca assinalada antes.
2. **Bloco reprodutivo tem duplo gate** (idade fértil × mulher) com corte etário
   brusco aos ~60 (`NT1`: 0.43 → 0.90/1.0). `IN22` chega a 0.52 NA nos 15–17.
3. **Bloco trabalho é gate-emprego**, não gate-idade — correlaciona com idade via
   `MAINSTAT`, o que explica parte do sinal de `HATLEVEL/JOB*`.
4. O mapa **valida a `FAIXA_MINIMA`** do colega (idades candidatas vêm de colunas
   `parciais` com valor) e **define anomalia corretamente**: divergir do routing
   esperado para a idade, não apenas do modelo.

## Uso

- Features: justifica F2/F3 e o aviso de "sinal de routing" no relatório.
- Anomalias (NB06): controlar o routing antes de declarar biologia vs erro.
- Relatório: `fig_routing.png` como figura da secção de dados.
