# Flusso DHW in `new_building.py`

Questo documento descrive il nuovo flusso DHW introdotto in `new_building.py`.

## Sequenza di calcolo

1. Il fabbisogno orario annuo di acqua calda sanitaria viene calcolato con `DHW.py` secondo EN 12831-3.
2. Se l'utente abilita l'accumulo, il profilo passa prima nel calcolo di storage secondo EN 15316-5.
3. Il carico passa poi nella distribuzione secondo EN 15316-3.
4. Il risultato finale della distribuzione rappresenta la richiesta lato generatore DHW.

## Scelta tra generatore unico e generatore separato

In `INPUT_SYSTEM_HVAC` è disponibile il parametro:

- `same_generator_for_heating_and_dhw`

### Se `True`

Heating e DHW sono forniti dallo stesso generatore.
In questo caso il calcolo DHW non richiede un generatore separato:

- il fabbisogno di heating viene già calcolato dal ramo HVAC principale;
- il fabbisogno DHW viene calcolato con `DHW.py`;
- il valore DHW dopo storage/distribuzione viene sommato al fabbisogno heating per ottenere il carico complessivo del generatore.

### Se `False`

Heating e DHW sono forniti da generatori distinti.
In questo caso l'utente deve definire i parametri del generatore DHW in:

- `dhw_generator_config`

Nel ramo separato il file non somma il DHW al carico heating, ma lascia il fabbisogno DHW come richiesta autonoma lato generatore.

## Accumulo opzionale

Il parametro:

- `dhw_storage_enabled`

attiva o disattiva il passaggio in EN 15316-5.

### Se `True`

- il fabbisogno orario DHW passa nello storage;
- lo storage restituisce:
  - perdite di accumulo
  - ausiliari
  - energia lato generatore dello storage
- il valore lato generatore dello storage viene poi passato alla distribuzione.

### Se `False`

- il fabbisogno DHW passa direttamente alla distribuzione;
- non vengono calcolate le perdite di accumulo.

## File di output

Il flusso produce:

- `hvac_stage_results.csv`
- `dhw_storage_15316_5_hourly_results.csv` se lo storage è attivo
- `dhw_storage_15316_5_summary.csv` se lo storage è attivo
- `dhw_distribution_15316_3_hourly_results.csv`
- `dhw_distribution_15316_3_summary.csv`

## Parametri principali

- `same_generator_for_heating_and_dhw`
- `dhw_storage_enabled`
- `dhw_storage_config`
- `dhw_generator_config`
- `distribution_15316_3_config.dhw`

