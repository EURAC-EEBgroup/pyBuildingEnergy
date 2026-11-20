# Luce naturale e artificiale nel multizona

Il modello opzionale `zone.daylight_lighting` valuta un punto di riferimento, non una mappa spaziale e non costituisce una verifica EN 17037. Disattivato per default. La luce artificiale copre prima la differenza tra il target e la luce naturale, entro la prestazione installata. Il suo calore sostituisce gli apporti di illuminazione preesistenti della zona: non configurarli due volte. L'illuminazione è accesa solo quando i profili di occupazione e illuminazione sono entrambi positivi.

```python
zone["daylight_lighting"] = {
    "method": "simplified", "target_lux": 300, "design_lux": 400,
    "installed_power_w": 200, "heat_fraction": 0.9, "mode": "continuous",
    "windows": {
        "Nome finestra": {"a_dayl": 0.1, "b_dayl": 0.0, "H_gl_1m": 1.2},
        # Per una finestra fissa aggiungere "tau_v": 0.7.
    },
}
```

Il metodo usa 115 lm/W, la formula semplificata 52016-3 con coefficienti forniti dall'utente, la trasmittanza dell'effettivo stato adattivo e il fattore di ombreggiamento. La somma dei contributi di più finestre è un'approssimazione e richiede coefficienti calibrati per la stanza e per quel punto. I parametri vanno riferiti ai nomi degli elementi dopo l'aggregazione del solver. `mode` può essere `continuous` o `on_off`; `design_lux` è la luce artificiale prodotta a piena potenza sul punto e può essere inferiore al target. La luce artificiale calcolata è una stima lineare senza dimming minimo, standby o ballast.

Le serie restituite dal solver sono `daylight_lux_<zona>`, `electric_lux_<zona>`, `total_lux_<zona>`, `lighting_power_W_<zona>` e `lighting_heat_W_<zona>`. Per l'energia integrare la potenza sul tempo. Gli stati sono nei campi `adaptive_<finestra>_*` allo stesso timestamp.

## Risultati esterni (ad esempio Radiance)

Con `method: "external"`, `illuminance_lux` è un dizionario `{timestamp: lux}` con un valore per ogni passo del solver, incluso l'eventuale warm-up. I valori rappresentano l'illuminamento naturale effettivo del punto, già calcolato con clima, geometria, ombre e **stati effettivi**; non viene applicato nuovamente `tau_v`. Mancanze e dati non validi producono errore. Il formato timestamp accetta `YYYY-MM-DD HH:MM:SS` o ISO 8601 come quello dell'indice della simulazione.

Radiance non è lanciato dalla libreria: occorre modellare geometria, superfici, BSDF e clima, calcolare le serie per gli stati e comporle usando il log delle finestre. Poiché la luce artificiale cambia gli apporti termici e può cambiare i successivi stati adattivi, una semplice importazione a due passaggi non garantisce la coerenza: occorre iterare simulazioni termiche e illuminotecniche fino a convergenza degli stati, oppure realizzare una co-simulazione. Per più finestre è necessario considerare le loro combinazioni e le interazioni ottiche. Non si devono interpretare i risultati esterni con calendario, meteo o stati diversi come risultati accoppiati.
