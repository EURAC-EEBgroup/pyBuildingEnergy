# Dichiarazione del campo di applicazione: multizona, facciate adattive, emissione

Questo documento definisce il campo di applicazione dichiarato e gli input da fornire. È
una dichiarazione di tracciabilità, non una dichiarazione di conformità già dimostrata. La
conformità richiede anche le verifiche delle procedure e dei risultati previste dalle norme
(sezione 6).

Lo script `examples/multizone_declaration_example.py` contiene un esempio completo e
commentato di tutti i campi elencati qui. Ogni campo da dichiarare è marcato `[DECLARE]`.

## 1. Campo di applicazione

Calcolo orario di edifici multizona secondo EN ISO 52016-1:2017. Il calcolo comprende:

- zone termiche accoppiate, climatizzate o a temperatura libera;
- zone adiacenti non climatizzate;
- scambio d'aria interzonale assegnato;
- elementi trasparenti adattivi semplificati secondo EN ISO 52016-3, con le proprietà
  degli stati fornite in input e il controllo dichiarato;
- sistemi di emissione per zona, per riscaldamento e raffrescamento, secondo EN 15316-2;
- un circuito idraulico di riscaldamento opzionale con nodo comune (EN 15316-1, Annex C).

Il calcolo si esegue con `simulate_multizone_emission(config, emission_by_zone, ...)`.
Senza emissione si usa `ISO52016.Temperature_and_Energy_needs_calculation_multizone`.

## 2. Input da dichiarare

### 2.0 Località e clima

Coordinate dell'edificio, fonte climatica (file EPW o PVGIS alle stesse coordinate) e
calendario d'uso devono riferirsi allo stesso sito. Un edificio simulato in un clima
diverso dal proprio va dichiarato come tale e non rappresenta un caso reale.

### 2.1 Zone (`building.zones[]`)

| Campo | Contenuto |
|---|---|
| `name`, `building_type_class` | identificativo e destinazione d'uso |
| `net_floor_area`, `volume` | superficie utile [m²] e volume [m³] |
| `heating_setpoint`, `heating_setback`, `cooling_setpoint`, `cooling_setback` | °C |
| `heating_profile`, `cooling_profile` | disponibilità oraria (feriale e festivo) |
| `heating_capacity`, `cooling_capacity` | W, usate dal solver termico |
| `internal_gains[]` | `full_load` e profili orari per persone, apparecchi e illuminazione |
| ventilazione e infiltrazioni | portate, profili e condizioni dell'aria immessa |
| `free_floating` | `true`: la zona non riceve né riscaldamento né raffrescamento |

I criteri di zonizzazione del §6.4.2 (5 %/1 %, 4 K/3 h, 0,003 m²/m², 3 dm³/(s·m²)) sono
controllati automaticamente. Le segnalazioni vanno riportate oppure giustificate.

### 2.2 Partizioni tra zone (`building_surface[]`, `boundary: "INTERNAL"`)

| Campo | Contenuto |
|---|---|
| `zone`, `adjacent_zone` | zone sui due lati |
| `area`, `orientation` | m², azimut e inclinazione |
| `u_value`, `thermal_capacity`, `construction_class` | trasmittanza, capacità areica, distribuzione della massa |
| coefficienti superficiali interni | sulle due facce |

Ogni parete comune va inserita una sola volta. `validate_multizone_input` segnala i nomi di
zona inesistenti e le partizioni incoerenti.

### 2.3 Scambio d'aria interzonale (`building.interzone_airflows[]`)

| Campo | Contenuto |
|---|---|
| `from`, `to` | zona sorgente e zona ricevente |
| `q_v_m3_s` / `q_v_m3_h` / `ach_from_zone` | portata; `ach_from_zone` richiede il `volume` della zona sorgente |
| `makeup_from_outdoor` | l'aria uscente dalla zona sorgente è compensata con aria esterna |
| `follow_ventilation_profile` | la portata segue il profilo di ventilazione |

Per ogni coppia di zone è ammesso un solo verso: il flusso nei due versi viene rifiutato.
Lo scambio riguarda solo il calore sensibile.

### 2.4 Zone adiacenti non climatizzate (`adjacent_zones[]`)

`type` (`external` / `internal`), `b_ztu` (opzionale; se fornito azzera gli apporti),
`c_ztu_ve` (default 0,5), `c_ztu_h_max`, `g_value` e la geometria degli elementi; inoltre
`building.adj_zones_present: true`.

Nel motore multizona la superficie verso la zona non climatizzata si dichiara con
`boundary: "UNCONDITIONED"` e `name_adj_zone`, il nome della zona adiacente.
- θ_ztu usa la temperatura operativa all'ora precedente delle zone climatizzate confinanti,
  pesata con U·A delle superfici in comune.
- Gli apporti del tipo interno (Formule 65, 68) sono ripartiti con lo stesso peso.
- Il metodo ibrido non supporta questo contorno.
- La ventilazione proveniente dalla zona non climatizzata (6.5.10.2) è collegata solo nei
  motori monozona.

### 2.5 Finestre e facciate adattive

Per ogni finestra:

- `zone`, `area`, `width`, `height`, `orientation`. Il calcolo delle schermature usa `width` e
  `height`, quindi `width × height` deve corrispondere all'`area` della finestra
  dichiarata. Se i dati descrivono una fascia vetrata aggregata, dichiarare la convenzione
  adottata (per esempio altezza equivalente = area / larghezza);
- `u_value`, `g_value`, `frame_area_fraction`, `solar_transmittance_correction_factor`;
- gli ombreggiamenti esterni (Annex F).

Per una finestra adattiva (`adaptive_envelope`):

| Campo | Contenuto |
|---|---|
| `type`, `position` | dispositivo e posizione (esterna, interna, intercapedine) |
| `control_type` | manuale, motorizzato, automatico, passivo |
| `space_category`, `system` | scenario di controllo, coerente con la destinazione d'uso della zona (uno scenario diverso va motivato), e servizi serviti |
| `states{}` | per ogni stato: `u_value`, `g_value`, `tau_v` |
| opzionali | `hysteresis`, `environment_variable`, `daytime_hours`, `sleeping_hours`, `daylight`, `glare_policy`, `glare_limit_lx`, `annex_c_edition` |

Per ogni stato va dichiarato se U, g e τᵥ si riferiscono al **sistema combinato** vetro +
dispositivo in quello stato. Va dichiarata anche l'origine dei dati: produttore, prova,
calcolo o valore assunto.

### 2.6 Emissione per zona (`emission_by_zone[zona]`, blocchi `heating` e `cooling`)

| Campo | Contenuto |
|---|---|
| `stratification_K`, `control_K`, `radiation_K`, `hydraulic_balancing_K`, `room_automation_K`, `embedded_K` | componenti di Δθ (EN 15316-2) |
| `nominal_power_kW` | potenza alle condizioni nominali dichiarate; `0` = nessun emettitore per il servizio; se omessa la potenza è illimitata |
| `fan_power_W`, `fan_count` | ventilatori (sempre conteggiati) |
| `control_power_W`, `control_count`, `electric_control_aux` | regolazione elettrica: conteggiata **solo** con `electric_control_aux: true` |
| `convective_fraction`, `backup_available` | quota convettiva e presenza di backup |

Opzioni della chiamata da dichiarare:

| Opzione | Default | Significato |
|---|---|---|
| `emission_inc` | `"recalculate"` | seconda simulazione multizona al setpoint equivalente (M2-2); `"approximate"` ignora l'accoppiamento tra le zone |
| `couple_emitter_capacity` | `True` | `nominal_power_kW` limita `heating_capacity` e `cooling_capacity` nel solver: un emettitore insufficiente modifica la temperatura della zona e il controllo della facciata |
| `unlisted_zones` | `"free_floating"` | una zona senza emettitore è a temperatura libera; `"ideal"` la climatizza con l'impianto ideale |
| `hydraulic_systems` | `None` | circuito per zona con nodo comune; per le zone con emettitore riusa il risultato di emissione |

## 3. Scelte di metodo

| Aspetto | Scelta | Riferimento |
|---|---|---|
| Accoppiamento tra zone climatizzate | soluzione simultanea nella stessa ora, senza confine all'ora precedente | scostamento dalla lettera dell'Annex D.2, ammesso da ISO/TR 52016-2 §6.3.6 |
| Zone adiacenti non climatizzate | confine all'ora precedente (θ_op(t−1)), Formule 58–59 | EN ISO 52016-1 §6.5.9 |
| Opzione B | Formule 7–8 | valori coincidenti con ISO/TR 52016-2, Tabella 2 |
| c_ztu,h;max | segue la Formula 59 | il testo della TR la contraddice: vedi `docs/standards_errata.md` |
| Aggregazione delle finestre | g equivalente che conserva Σ g·A·(1−F_fr); ombreggiamento propria di ogni finestra | Formula 69 |
| Edizione DIN EN ISO 52016-1:2018 | testo identico; l'appendice nazionale esclude il metodo orario per la verifica in Germania | DIN NA |

Per la scelta sulle zone adiacenti è stato fatto un confronto con il calcolo all'istante t,
su un caso annuale con b = 0,5. Il fabbisogno di riscaldamento cambia di +0,08 %; θ_ztu
differisce al massimo di 0,55 K in singole ore.

## 4. Esclusioni

- trasporto di umidità tra zone (il bilancio latente è calcolato per zona);
- radiazione solare trasmessa attraverso vetrate interne tra zone (serre e verande
  accoppiate);
- rete dei flussi d'aria con pressioni e flusso bidirezionale tra le stesse zone;
- modello dettagliato della batteria del fan coil;
- cavità ventilate adattive dettagliate;
- lo stato della facciata non è collegato al consumo dell'illuminazione artificiale;
- con `emission_inc="approximate"` l'avviso di emettitore sottodimensionato può comparire
  anche quando il solver è già limitato.

## 5. Risultati da riportare

- fabbisogni orari e annuali di riscaldamento e raffrescamento per zona: `Q_HVAC_<zona>`,
  tabella annuale;
- temperature delle zone e flussi attraverso le partizioni e l'aria interzonale
  (`Q_ve_interzone_<zona>`);
- stati orari delle facciate e valori U, g e τᵥ applicati (log del contesto adattivo per
  zona) e fattori di ombreggiamento `F_sh_<elemento>`;
- energia in ingresso all'emettitore, ausiliari, backup ed energia non soddisfatta;
- capacità impostate nell'input e capacità applicate nel solver
  (`result.applied_capacities`), etichettate separatamente; zone a temperatura libera
  (`result.free_floating_zones`);
- comfort secondo EN 16798-1, Annex B (`result.comfort(method="adaptive" | "static")`): ore per
  categoria I–IV e K·h fuori dalla categoria II per zona. Va dichiarato il metodo scelto
  (adattivo B.2.2 per gli edifici senza raffrescamento meccanico; statico secondo la
  Tabella B.1) e il profilo di occupazione usato;
- potenza insufficiente: ore con il limite di capacità, K·h fuori dal setpoint ed energia
  non fornita (`Q_HVAC_req − Q_HVAC`). Con `couple_emitter_capacity=True` una carenza
  compare come temperatura fuori setpoint, non come energia non soddisfatta del modulo di
  emissione: un valore nullo di quest'ultima, da solo, non dimostra il rispetto del
  comfort;
- confronto tra facciata adattiva e riferimento per zona
  (`compare_adaptive_with_reference_multizone`): indicatori della Tabella 6, statistiche
  degli stati, rapporto del §8, con gli stessi edificio, clima, profili e capacità
  applicate. Le percentuali sulle K·h non sono calcolate se il valore di riferimento è
  sotto 1 K·h;
- `multizone_report` e `coupling_report()`, avvisi emessi, versione delle norme e
  dell'applicazione, origine degli input.

## 6. Verifiche disponibili

| Verifica | Esito |
|---|---|
| ASHRAE 140, Case 960 (zona soleggiata accoppiata) | passa: riscaldamento Δ 1,044 MWh (0,96–1,12); raffrescamento Δ −1,804 MWh (−1,81…−1,27), a 0,006 MWh dal limite |
| Zone adiacenti: Formule 54, 58–59, 65, 68, E.5–E.6, confine all'ora precedente | test analitici (`tests/test_unconditioned_zone_52016_1.py`); nel multizona `tests/test_multizone_unconditioned_zone.py` |
| Comfort EN 16798-1 e potenza insufficiente | `tests/test_comfort_16798_1.py`, `tests/test_multizone_emission.py` |
| Accoppiamento e flussi interzonali, zonizzazione, Tabella 2 della TR | `tests/test_multizone_52016_1.py` |
| Facciate adattive multizona e ombreggiamento per finestra | `tests/test_adaptive_envelope_52016_3_multizone.py` |
| Emissione multizona, capacità, nodo idraulico, zone senza emettitore | `tests/test_multizone_emission.py`, `tests/test_emission_capacity_15316_2.py` |

Prima di una dichiarazione formale vanno ancora fatti:

- rieseguire la suite completa sulla versione dichiarata;
- confronto multizona con un riferimento esterno: i file EnergyPlus presenti nel
  repository non sono utilizzabili (vedi `CHANGELOG`);
- sostituire i valori illustrativi (stati della schermatura, potenze dei fan coil) con dati
  di cui siano dichiarate origine e condizioni di riferimento;
- confrontare con risultati esterni di riferimento su casi multizona con facciate adattive,
  oltre alla verifica di integrazione su 168 ore già eseguita.
