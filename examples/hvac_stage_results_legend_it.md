# Legenda parametri output HVAC

Questo file spiega i nomi delle colonne esportate da `hvac_stage_results.csv`.

## Colonne lato edificio

- `building_Q_HC`: fabbisogno termico netto della zona, positivo in riscaldamento e negativo in raffrescamento.
- `building_Q_H`: quota di carico di riscaldamento dell'edificio.
- `building_Q_C`: quota di carico di raffrescamento dell'edificio.
- `building_T_op0`: temperatura operativa iniziale usata dal calcolo dell'edificio.
- `building_T_air`: temperatura dell'aria interna.
- `building_T_op`: temperatura operativa interna.
- `building_T_ext`: temperatura esterna.

## Colonne lato sistema HVAC

- `hvac_Q_h(kWh)`: fabbisogno termico in ingresso al sistema HVAC per il passo corrente.
- `hvac_QH_em_i_in(kWh)`: energia termica richiesta al sottosistema di emissione.
- `hvac_QH_dis_i_req(kWh)`: energia termica richiesta dal sottosistema di distribuzione.
- `hvac_QH_dis_i_in(kWh)`: energia termica in ingresso al sottosistema di distribuzione dopo perdite e recuperi.
- `hvac_QH_gen_out(kWh)`: energia termica utile erogata dal generatore.
- `hvac_EHW_gen_in(kWh)`: energia termica in ingresso al generatore.
- `hvac_EHW_gen_aux(kWh)`: energia elettrica ausiliaria del generatore.
- `hvac_QW_gen_i_ls_rbl_H(kWh)`: quota di perdite del generatore recuperabile come calore.
- `hvac_Q_w_dis_i_ls(kWh)`: perdite termiche di distribuzione.
- `hvac_Q_w_dis_i_aux(kWh)`: energia ausiliaria della distribuzione.
- `hvac_Q_w_dis_i_ls_rbl_H(kWh)`: quota recuperabile delle perdite di distribuzione.
- `hvac_ΦH_em_eff(kW)`: potenza termica efficace dell'emissione.
- `hvac_θH_em_flow(°C)`: temperatura di mandata lato emissione.
- `hvac_θH_em_ret(°C)`: temperatura di ritorno lato emissione.
- `hvac_θH_dis_flw(°C)`: temperatura di mandata lato distribuzione.
- `hvac_θH_dis_ret(°C)`: temperatura di ritorno lato distribuzione.
- `hvac_θX_gen_cr_flw(°C)`: temperatura di mandata del circuito primario/generatore.
- `hvac_θX_gen_cr_ret(°C)`: temperatura di ritorno del circuito primario/generatore.
- `hvac_V_H_em_eff(m3/h)`: portata volumetrica efficace lato emissione.
- `hvac_V_H_dis(m3/h)`: portata volumetrica lato distribuzione.
- `hvac_V_H_gen(m3/h)`: portata volumetrica lato generatore.
- `hvac_efficiency_gen(%)`: efficienza del generatore.
- `hvac_emission_calculation_mode`: modalità di calcolo usata per il blocco di emissione.

## Nota

Le colonne con prefisso `building_` provengono dalla simulazione dell'edificio.
Le colonne con prefisso `hvac_` provengono dal flusso di emissione, distribuzione e generazione.

