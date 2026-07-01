import pandas as pd
import numpy as np
from pybuildingenergy.global_inputs import TB14 as TB14_backup
from pybuildingenergy.source.emission_15316_2 import EmissionSystemCalculator
from pybuildingenergy.source.distribution_15316_3 import DistributionSystemCalculator
from pybuildingenergy.source.generation_15316_4_1 import BoilerGeneratorCalculator

class HeatingSystemCalculator:
    """
    Heating system calculator following ISO 15316-1.
    Supports 4 emitter circuit types (C.2, C.3, C.4, C.5).
    """

    def __init__(self, input_data):
        """Initialize the calculator with a dictionary of input parameters."""
        self.input_data = input_data
        self._load_constants()
        self._load_input_parameters()

    def _load_constants(self):
        """
        Load constants and reference tables.

        - TB14: ISO 15316-1 table with nominal deltas and exponents per emitter type.
        - c_w: Specific heat capacity of water in Wh/(kg·K).
        """
        # Use user-provided TB14 if available, otherwise fallback to global one
        if "TB14" in self.input_data and isinstance(self.input_data["TB14"], pd.DataFrame):
            self.TB14 = self.input_data["TB14"]
            print("⚙️  Custom TB14 table loaded from input_data.")
        else:
            self.TB14 = TB14_backup
            print("⚙️  Default TB14 table loaded from global_inputs.py.")

        # Specific heat capacity of water (Wh/kgK)
        self.c_w = 1.16

    def _load_input_parameters(self):
        """
        Load input parameters from the provided dictionary and set defaults.
        Splits into three blocks: emission, distribution, and generation (primary).
        """
        inp = self.input_data

        # --- Emission (secondary) ---
        self.EMitter_type = inp.get('emitter_type', 'Floor heating')
        self.ΦH_em_n = inp.get('nominal_power', 8)  # kW
        self.emission_efficiency = inp.get('emission_efficiency', 90)  # %
        self.MIX_EM = inp.get('mixing_valve', True)
        self.flow_temp_control_type = inp.get('flow_temp_control_type',
                                              'Type 1 - Based on demand')
        self.auxiliars_power = inp.get('auxiliars_power', 0)  # W
        self.alpha_em_i_aux = inp.get('auxiliary_recovery_factor', 100)  # %

        # Operating hours per step (h)
        self.tH_em_i_ON = inp.get('emission_operation_time', 1)

        # Mixing valve presence and offset (°C)
        self.ΔθH_em_mix_sahz_i = inp.get('mixing_valve_delta', 2)

        # Emitter circuit type selector (0=C.2, 1=C.3, 2=C.4, 3=C.5)
        self.selected_emm_cont_circuit = inp.get('selected_emm_cont_circuit', 0)

        # Data tables for emission and outdoor curve (secondary)
        self.df_heat_emm_data = inp.get('heat_emission_data', self._default_emission_data())
        self.df_out_temp = inp.get('outdoor_temp_data', self._default_outdoor_data())
        self.θem_flw_sahz_i = inp.get('constant_flow_temp', [42])  # constant secondary supply (if used)

        # Emission calculation mode:
        # - 'simplified': keep the internal ISO 15316-1 emission approximation
        # - 'en15316-2': use EmissionSystemCalculator step-by-step per hour
        self.emission_calculation_mode = str(inp.get('emission_calculation_mode', 'simplified')).lower()
        if self.emission_calculation_mode in ('iso_15316_2', '15316_2', 'en15316-2', 'en15316_2'):
            self.emission_calculation_mode = 'en15316-2'
        else:
            self.emission_calculation_mode = 'simplified'
        self.emission_15316_2_config = inp.get('emission_15316_2_config', inp.get('emission_config', {}))
        self._emission_calc_cache = None

        # --- Distribution (between emitter and manifold) ---
        self.distribution_calculation_mode = str(inp.get('distribution_calculation_mode', 'simplified')).lower()
        if self.distribution_calculation_mode in ('analitico', 'analitica', 'analytical'):
            self.distribution_calculation_mode = 'analytical'
        elif self.distribution_calculation_mode in ('semplice', 'simplificato'):
            self.distribution_calculation_mode = 'simplified'
        assert self.distribution_calculation_mode in ('simplified', 'analytical')

        self.analytical_distribution_config = inp.get('distribution_15316_3_config', inp.get('distribution_config'))
        self.Heat_losses_recovered = inp.get('heat_losses_recovered', True)
        self.f_h_dist_i_ls = inp.get('distribution_loss_recovery', 90)  # %
        self.simplified_or_holistic_approach = inp.get('simplified_approach', 80)  # %
        self.f_h_dist_i_aux = inp.get('distribution_aux_recovery', 80)  # %
        self.W_H_dist_i_aux = inp.get('distribution_aux_power', 30)  # W
        self.Heat_loss_coefficent_dist = inp.get('distribution_loss_coeff', 48)  # W/K
        self.tH_dis_i_ON = inp.get('distribution_operation_time', 1)  # h

        # System recoverable losses (kWh)
        self.QW_sys_ls_rbl_tz_i = inp.get('recoverable_losses', 0.0)
        # --- Generation (primary) ---
        self.full_load_power = inp.get('full_load_power', 24.0)                # kW
        self.max_monthly_load_factor = inp.get('max_monthly_load_factor', 100) # %
        self.tH_gen_i_ON = inp.get('tH_gen_i_ON', 1.0)                         # h

        # Generator auxiliaries and recoverable fraction
        self.auxiliary_power_generator = inp.get('auxiliary_power_generator', 0.0)  # % of EHW_gen_in
        self.fraction_of_auxiliary_power_generator = inp.get('fraction_of_auxiliary_power_generator', 40.0)  # %

        # Generator hydraulic circuit type
        self.generator_circuit = str(inp.get('generator_circuit', 'independent')).lower()
        if 'indipendet' in self.generator_circuit or 'indipendent' in self.generator_circuit:
            self.generator_circuit = 'independent'

        # Generator pump control and nominal ΔT (°C)
        self.speed_control_generator_pump = inp.get('speed_control_generator_pump', 'variable')  # 'deltaT_constant' | 'variable'
        self.generator_nominal_deltaT = inp.get('generator_nominal_deltaT', 20.0)

        # Optional primary setpoints (used if circuit is independent)
        self.θHW_gen_flw_set = inp.get('θHW_gen_flw_set')  # °C or None
        self.θHW_gen_ret_set = inp.get('θHW_gen_ret_set')  # °C or None

        # Calculation policy vs. Q_H (building needs)
        self.calc_when_QH_positive_only = inp.get('calc_when_QH_positive_only', True)
        self.fill_zeros_when_skipped = inp.get('fill_zeros_when_skipped', False)

        # Off-cycle behavior when calc_when_QH_positive_only is True
        # 'idle'  : empty row (NaN/0)
        # 'temps' : temps only, powers/flows = 0
        # 'full'  : full computation even if Q_H <= 0
        self.off_compute_mode = str(inp.get('off_compute_mode', 'idle')).lower()
        assert self.off_compute_mode in ('idle', 'temps', 'full')

        # Primary hydraulic separator behavior
        self.allow_dilution = True
        self.primary_line_loss = 1.0  # °C temperature drop in primary supply before the node

        # --- Generator efficiency model ---
        # 'simple'       : linear heuristic vs. return temperature
        # 'parametric'   : piecewise/threshold model
        # 'manufacturer' : 1D (return) or 2D (flow/return) map
        self.efficiency_model = str(inp.get('efficiency_model', 'simple')).lower()

        # Parameters for 'parametric' model
        self.eta_max     = float(inp.get('eta_max', 110.0))  # %
        self.eta_no_cond = float(inp.get('eta_no_cond', 95.0))
        self.T_ret_min   = float(inp.get('T_ret_min', 20.0)) # °C
        self.T_ret_thr   = float(inp.get('T_ret_thr', 50.0)) # °C

        # Manufacturer curves (optional)
        self.manuf_curve_1d = inp.get('manuf_curve_1d')  # dict or None
        self.manuf_curve_2d = inp.get('manuf_curve_2d')  # dict or None

        # --- Primary flow temperature control (independent from secondary) ---
        self.gen_flow_temp_control_type = inp.get(
            'gen_flow_temp_control_type',
            'Type A - Based on outdoor temperature'   # 'Type A' | 'Type B' | 'Type C'
        )

        # Primary outdoor curve data
        self.df_gen_out_temp = inp.get('gen_outdoor_temp_data', pd.DataFrame({
            "θext_min_gen": [-7],
            "θext_max_gen": [15],
            "θflw_gen_max": [60],   # °C, max primary supply
            "θflw_gen_min": [35],   # °C, min primary supply
        }, index=["Generator curve"]))

        # Constant primary flow setpoint (used by Type C)
        self.θHW_gen_flw_const = float(inp.get('θHW_gen_flw_const', 50.0))
        self.generation_calculation_mode = str(inp.get('generation_calculation_mode', 'legacy')).lower()
        if self.generation_calculation_mode in ('boiler_15316_4_1', 'boiler', 'normative'):
            self.generation_calculation_mode = 'boiler_15316_4_1'
        else:
            self.generation_calculation_mode = 'legacy'

        self.boiler_generation_config = inp.get('boiler_generation_config')
        self._boiler_calc_cache = None

        # Caches for the analytical distribution branch (performance):
        # - ``_distribution_calc_cache`` holds a single DistributionSystemCalculator
        #   instance, instead of rebuilding it for every hour.
        # - ``_distribution_fast`` stores the static parameters (sections,
        #   design flow, head loss, pump efficiency factor, control constants)
        #   used by the scalar fast path in ``calculate_distribution_analytical``.
        self._distribution_calc_cache = None
        self._distribution_fast = None
        # Cache for the precomputed EN 15316-2 emission timeseries
        self._emission_ts_cache = None

    def _get_emission_calculator(self):
        if self._emission_calc_cache is None:
            self._emission_calc_cache = EmissionSystemCalculator(self.emission_15316_2_config)
        return self._emission_calc_cache

    def _precompute_emission_timeseries(self, df):
        """
        Run EN 15316-2 emission calculation ONCE for the whole input time series
        and cache the per-row outputs as numpy arrays. This avoids the very
        expensive pattern of calling ``EmissionSystemCalculator.run_timeseries``
        with a single-row DataFrame for each of the 8760 hours.
        """
        if self.emission_calculation_mode != 'en15316-2':
            self._emission_ts_cache = None
            return

        emission_calc = self._get_emission_calculator()
        time_step_hours = float(self.input_data.get('emission_time_step_hours', self.tH_em_i_ON))

        df_in = df.copy()
        if 'time_step_hours' not in df_in.columns:
            df_in['time_step_hours'] = time_step_hours

        ts = emission_calc.run_timeseries(df_in).timeseries
        # constants (per-service config, do not depend on the row)
        theta_inc_K = float(emission_calc.services["H"]["temperature_increase_K"])
        f_conv = float(emission_calc.services["H"]["convective_fraction"])

        self._emission_ts_cache = {
            'q_h_em_out_kWh':     ts['Q_H_em_out_kWh'].to_numpy(dtype=float),
            'theta_H_int_inc_C':  ts['theta_H_int_inc_C'].to_numpy(dtype=float),
            'QH_em_i_in':         ts['Q_H_em_in_kWh'].to_numpy(dtype=float),
            'W_H_em_aux_kWh':     ts['W_H_em_aux_kWh'].to_numpy(dtype=float),
            'QH_em_ls_kWh':       ts['Q_H_em_ls_kWh'].to_numpy(dtype=float),
            'q_h_em_out_inc_kWh': ts['Q_H_em_out_inc_kWh'].to_numpy(dtype=float),
            'theta_H_int_inc_K':  theta_inc_K,
            'fH_em_conv':         f_conv,
            'time_step_hours':    time_step_hours,
            'n':                  len(ts),
        }
        # position counter consumed by _calculate_emission_step during the loop
        self._current_emission_pos = 0

    def _calculate_emission_step(self, q_h_kWh, θint, θext):
        """
        Return the emission-side quantities for a single hourly step.

        - ``simplified`` mode: historical internal approximation.
        - ``en15316-2``  mode: read from the precomputed cache populated by
          :meth:`_precompute_emission_timeseries`. If no cache is available
          (e.g. ad-hoc call outside the time loop) a one-row fallback is used.
        """
        if self.emission_calculation_mode != 'en15316-2':
            common = self.calculate_common_emission_parameters(q_h_kWh, θint)
            qh_in = float(common['QH_em_i_in'])
            return {
                'q_h_em_out_kWh': float(q_h_kWh),
                'θint_eff': float(θint),
                'QH_em_i_in': qh_in,
                'ΦH_em_eff': float(common['ΦH_em_eff']),
                'W_H_em_aux_kWh': 0.0,
                'QH_em_ls_kWh': max(qh_in - float(q_h_kWh), 0.0),
                'q_h_em_out_inc_kWh': float(q_h_kWh),
                'theta_H_int_inc_K': 0.0,
                'fH_em_conv': np.nan,
            }

        cache = getattr(self, '_emission_ts_cache', None)
        if cache is not None:
            pos = getattr(self, '_current_emission_pos', None)
            if pos is not None and 0 <= pos < cache['n']:
                ts_h = cache['time_step_hours']
                qh_in = float(cache['QH_em_i_in'][pos])
                return {
                    'q_h_em_out_kWh':     float(cache['q_h_em_out_kWh'][pos]),
                    'θint_eff':           float(cache['theta_H_int_inc_C'][pos]),
                    'QH_em_i_in':         qh_in,
                    'ΦH_em_eff':          qh_in / max(ts_h, 1e-9),
                    'W_H_em_aux_kWh':     float(cache['W_H_em_aux_kWh'][pos]),
                    'QH_em_ls_kWh':       float(cache['QH_em_ls_kWh'][pos]),
                    'q_h_em_out_inc_kWh': float(cache['q_h_em_out_inc_kWh'][pos]),
                    'theta_H_int_inc_K':  cache['theta_H_int_inc_K'],
                    'fH_em_conv':         cache['fH_em_conv'],
                }

        # Fallback: ad-hoc single-row call (kept for backward compatibility).
        emission_calc = self._get_emission_calculator()
        time_step_hours = float(self.input_data.get('emission_time_step_hours', self.tH_em_i_ON))
        emission_input = pd.DataFrame(
            [{
                'Q_H_kWh': float(q_h_kWh),
                'T_op': float(θint),
                'T_ext': float(θext),
                'time_step_hours': time_step_hours,
            }]
        )
        result = emission_calc.run_timeseries(emission_input)
        row = result.timeseries.iloc[0]
        summary = result.summary
        return {
            'q_h_em_out_kWh': float(row.get('Q_H_em_out_kWh', q_h_kWh)),
            'θint_eff': float(row.get('theta_H_int_inc_C', θint + summary.get('theta_H_int_inc_K', 0.0))),
            'QH_em_i_in': float(summary.get('QH_em_in_kWh', q_h_kWh)),
            'ΦH_em_eff': float(summary.get('QH_em_in_kWh', q_h_kWh) / max(time_step_hours, 1e-9)),
            'W_H_em_aux_kWh': float(summary.get('WH_em_aux_kWh', 0.0)),
            'QH_em_ls_kWh': float(summary.get('QH_em_ls_kWh', 0.0)),
            'q_h_em_out_inc_kWh': float(summary.get('QH_em_out_inc_kWh', q_h_kWh)),
            'theta_H_int_inc_K': float(summary.get('theta_H_int_inc_K', 0.0)),
            'fH_em_conv': float(summary.get('fH_em_conv', np.nan)),
        }

    def _idle_row(self, q_h_kWh, θint, θext):
        """
        Build an 'idle' row when Q_H <= 0 and the policy is to skip calculations.
        Returns a structure with NaN (or zeros) and calculated=False.
        """
        import numpy as _np
        _nan = 0.0 if self.fill_zeros_when_skipped else _np.nan
        return {
            'Q_h(kWh)': q_h_kWh,
            'T_op(°C)': θint,
            'T_ext(°C)': θext,

            'ΦH_em_eff(kW)': _nan,
            'θH_em_flow(°C)': _nan,
            'θH_em_ret(°C)': _nan,
            'V_H_em_eff(m3/h)': _nan,
            'θH_em_flw_min_req(°C)': _nan,

            'θH_nod_out(°C)': _nan,
            'θH_cr_flw(°C)': _nan,
            'θH_cr_ret(°C)': _nan,
            'V_H_cr(m3/h)': _nan,
            'θH_dis_flw(°C)': _nan,
            'θH_dis_ret(°C)': _nan,
            'βH_em(-)': _nan,

            'Q_w_dis_i_ls(kWh)': _nan,
            'Q_w_dis_i_aux(kWh)': _nan,
            'Q_w_dis_i_ls_rbl_H(kWh)': _nan,
            'QH_dis_i_req(kWh)': _nan,
            'QH_dis_i_in(kWh)': _nan,
            'V_H_dis(m3/h)': _nan,

            'max_output_g1(kWh)': _nan,
            'QH_gen_out(kWh)': _nan,
            'θX_gen_cr_flw(°C)': _nan,
            'θX_gen_cr_ret(°C)': _nan,
            'V_H_gen(m3/h)': _nan,
            'efficiency_gen(%)': _nan,
            'EHW_gen_in(kWh)': _nan,
            'EHW_gen_aux(kWh)': _nan,
            'QW_gen_i_ls_rbl_H(kWh)': _nan,
            'QW_gen_out(kWh)': _nan,
            'QHW_gen_out(kWh)': _nan,
            'EH_gen_in(kWh)': _nan,
            'EWH_gen_in(kWh)': _nan,

            'calculated': False,
        }

    def _default_emission_data(self):
        """
        Default table for emission circuit limits/setpoints on the secondary side.
        """
        return pd.DataFrame({
            "θH_em_flw_max_sahz_i": [45],
            "ΔθH_em_w_max_sahz_i": [8],
            "θH_em_ret_req_sahz_i": [20],
            "βH_em_req_sahz_i": [80],
            "θH_em_flw_min_tz_i": [28],
        }, index=[
            "Max flow temperature HZ1",
            "Max Δθ flow / return HZ1",
            "Desired return temperature HZ1",
            "Desired load factor with ON-OFF for HZ1",
            "Minimum flow temperature for HZ1"
        ])

    def _default_outdoor_data(self):
        """
        Default outdoor curve (secondary) defining min/max supply vs. min/max outdoor T.
        """
        return pd.DataFrame({
            "θext_min_sahz_i": [-10],
            "θext_max_sahz_i": [16],
            "θem_flw_max_sahz_i": [45],
            "θem_flw_min_sahz_i": [28],
        }, index=[
            "Minimum outdoor temperature",
            "Maximum outdoor temperature",
            "Maximum flow temperature",
            "Minimum flow temperature"
        ])

    def load_csv_data(self, csv_path_or_df):
        """
        Load time-series inputs from CSV or DataFrame.
        Expected columns (aliases supported later): Q_H, T_op, T_ext.

        Converts Q_H (if in Wh) to kWh into a new 'Q_H_kWh' column.
        """
        if isinstance(csv_path_or_df, str):
            df = pd.read_csv(csv_path_or_df, index_col=0, parse_dates=True)
        else:
            df = csv_path_or_df.copy()
        
        # Transform in kWh 
        if 'Q_H' in df.columns:
            df['Q_H_kWh'] = df['Q_H'] / 1000.0
            

        self.timeseries_data = df
        return df

    def calculate_common_emission_parameters(self, heating_needs_kWh, θint):
        """
        Compute common parameters for all emission circuit types (C.2–C.5):
        - Effective emission power over the step
        - Nominal flow rate
        - Emitter-air temperature delta and exponent (from TB14)
        """
        # Net system demand after recoverable system losses
        QH_sys_out_tz_i = max(heating_needs_kWh - self.QW_sys_ls_rbl_tz_i, 0.0)
        QH_sys_out_hz_i = QH_sys_out_tz_i
        QH_em_i_out = QH_sys_out_hz_i

        # Initial average emission power (kW)
        if self.tH_em_i_ON == 0:
            ΦH_em_eff_initial = 0.0
        else:
            ΦH_em_eff_initial = QH_em_i_out / self.tH_em_i_ON

        # Approximate load factor and ON time
        BetaH_em = min(ΦH_em_eff_initial / self.ΦH_em_n, 1.0) if self.ΦH_em_n > 0 else 0.0
        tH_cr_i_ON = self.tH_em_i_ON * BetaH_em
        WH_em_i_aux = tH_cr_i_ON * self.auxiliars_power / 1000.0

        # Emission losses and input (kWh)
        QH_em_i_ls = QH_em_i_out * (100 - self.emission_efficiency) / self.emission_efficiency
        QH_em_i_in = QH_em_i_out + QH_em_i_ls - WH_em_i_aux * self.alpha_em_i_aux / 100.0

        # Final average effective emission power (kW)
        if self.tH_em_i_ON == 0:
            ΦH_em_eff = 0.0
        else:
            ΦH_em_eff = QH_em_i_in / self.tH_em_i_ON

        # Nominal hydraulic parameters from TB14
        ΔθH_em_n = float(self.TB14.loc[self.EMitter_type, "Emitters_nominal_deltaTeta_Water_C"])
        V_H_em_nom = self.ΦH_em_n / (ΔθH_em_n * self.c_w) if ΔθH_em_n > 0 else 0.0
        ΔθH_em_air = float(self.TB14.loc[self.EMitter_type, "Emitters_nominale_deltaTeta_air_C"])
        nH_em = float(self.TB14.loc[self.EMitter_type, "Emitters_exponent_n"])

        return {
            'QH_sys_out_hz_i': QH_sys_out_hz_i,
            'QH_em_i_in': QH_em_i_in,
            'ΦH_em_eff': ΦH_em_eff,
            'V_H_em_nom': V_H_em_nom,
            'ΔθH_em_n': ΔθH_em_n,
            'ΔθH_em_air': ΔθH_em_air,
            'nH_em': nH_em
        }

    def calculate_type_C2(self, common_params, θint):
        """
        C.2 — CONSTANT mass flow rate and CONSTANT water temperature on the emitter.
        Secondary flow is nominal; supply/return derived from power balance.
        """
        ΦH_em_eff = common_params['ΦH_em_eff']
        V_H_em_nom = common_params['V_H_em_nom']
        ΔθH_em_air = common_params['ΔθH_em_air']
        nH_em = common_params['nH_em']

        # Effective mass flow = nominal
        V_H_em_eff = V_H_em_nom

        # Emitter-air temperature difference (nonlinear exponent)
        if ΦH_em_eff > 0 and self.ΦH_em_n > 0:
            ΔθH_em_air_eff = ΔθH_em_air * ((ΦH_em_eff / self.ΦH_em_n) ** (1 / nH_em))
        else:
            ΔθH_em_air_eff = 0.0

        # Average emitter temperature
        θH_em_avg = ΔθH_em_air_eff + θint

        # Water ΔT across emitter
        if V_H_em_eff > 0:
            ΔθH_em_w_eff = ΦH_em_eff / (self.c_w * V_H_em_eff)
        else:
            ΔθH_em_w_eff = 0.0

        # Supply/return emitter temperatures
        θH_em_flow = θH_em_avg + ΔθH_em_w_eff / 2
        θH_em_ret = θH_em_avg - ΔθH_em_w_eff / 2

        # Minimum supply required (mixing valve adds margin)
        θH_em_flw_min = θH_em_flow + self.ΔθH_em_mix_sahz_i if self.MIX_EM else θH_em_flow

        return {
            'θH_em_flow': θH_em_flow,
            'θH_em_ret': θH_em_ret,
            'V_H_em_eff': V_H_em_eff,
            'θH_em_avg': θH_em_avg,
            'θH_em_flw_min': θH_em_flw_min,
            'ΔθH_em_air_eff': ΔθH_em_air_eff
        }

    def calculate_type_C3(self, common_params, θint):
        """
        C.3 — VARYING mass flow rate and CONSTANT emitter temperature.
        Return temperature is limited; flow adapts to meet power.
        """
        ΦH_em_eff = common_params['ΦH_em_eff']
        ΔθH_em_air = common_params['ΔθH_em_air']
        nH_em = common_params['nH_em']
        ΔθH_em_n = common_params['ΔθH_em_n']

        # Requested return temperature (at least ambient)
        θH_em_ret_req = float(self.df_heat_emm_data.loc["Desired return temperature HZ1",
                                                        "θH_em_ret_req_sahz_i"])
        θH_em_ret_set = max(θH_em_ret_req, θint)

        # Limits
        ΔθH_em_w_n = ΔθH_em_n
        θH_em_flw_max = float(self.df_heat_emm_data.loc["Max flow temperature HZ1",
                                                        "θH_em_flw_max_sahz_i"])
        ΔθH_em_w_max = float(self.df_heat_emm_data.loc["Max Δθ flow / return HZ1",
                                                       "ΔθH_em_w_max_sahz_i"])

        # Nonlinear emitter-air delta
        if ΦH_em_eff > 0 and self.ΦH_em_n > 0:
            ΔθH_em_air_eff = ΔθH_em_air * ((ΦH_em_eff / self.ΦH_em_n) ** (1 / nH_em))
        else:
            ΔθH_em_air_eff = 0.0

        # Average emitter temperature and supply limit
        θH_em_avg = ΔθH_em_air_eff + θint
        θH_em_flw_lim = min(θH_em_flw_max, θH_em_avg + ΔθH_em_w_max / 2)

        # Supply/return constrained by return setpoint
        θH_em_flw = min(θH_em_flw_lim, 2 * θH_em_avg - θH_em_ret_set)
        θH_em_ret = 2 * θH_em_avg - θH_em_flw

        # Effective ΔT and mass flow
        ΔθH_em_w_eff = θH_em_flw - θH_em_ret
        V_H_em_act = ΦH_em_eff / (ΔθH_em_w_eff * self.c_w) if ΔθH_em_w_eff > 0 else 0.0

        # Minimum distribution supply
        θH_em_flw_min = θH_em_flw + self.ΔθH_em_mix_sahz_i if self.MIX_EM else θH_em_flw

        return {
            'θH_em_flow': θH_em_flw,
            'θH_em_ret': θH_em_ret,
            'V_H_em_eff': V_H_em_act,
            'θH_em_avg': θH_em_avg,
            'θH_em_flw_min': θH_em_flw_min,
            'ΔθH_em_air_eff': ΔθH_em_air_eff
        }

    def calculate_type_C4(self, common_params, θint):
        """
        C.4 — INTERMITTENT flow rate (ON-OFF control).
        Uses an ON-power evaluation and duty cycle to meet hourly energy.
        """
        ΦH_em_eff = common_params['ΦH_em_eff']
        V_H_em_nom = common_params['V_H_em_nom']
        ΔθH_em_air = common_params['ΔθH_em_air']
        nH_em = common_params['nH_em']
        ΔθH_em_n = common_params['ΔθH_em_n']

        ΔθH_em_w_n = ΔθH_em_n
        βH_em_req = float(self.df_heat_emm_data.loc["Desired load factor with ON-OFF for HZ1",
                                                    "βH_em_req_sahz_i"])

        # Nominal mass flow
        V_H_em_eff = V_H_em_nom

        # Nonlinear emitter-air delta
        if ΦH_em_eff > 0 and self.ΦH_em_n > 0:
            ΔθH_em_air_eff = ΔθH_em_air * ((ΦH_em_eff / self.ΦH_em_n) ** (1 / nH_em))
        else:
            ΔθH_em_air_eff = 0.0

        # Theoretical supply during ON
        if ΦH_em_eff > 0 and self.ΦH_em_n > 0 and βH_em_req > 0:
            θH_em_flw_calc = (
                θint + (ΔθH_em_air + ΔθH_em_w_n / 2) *
                (((ΦH_em_eff / self.ΦH_em_n) * (100 / βH_em_req)) ** (1 / nH_em))
            )
        else:
            θH_em_flw_calc = θint

        # Minimum distribution supply
        θH_em_flw_min = θH_em_flw_calc + self.ΔθH_em_mix_sahz_i if self.MIX_EM else θH_em_flw_calc

        return {
            'θH_em_flow': θH_em_flw_calc,
            'θH_em_ret': θint,  # refined later in operating conditions
            'V_H_em_eff': V_H_em_eff,
            'θH_em_avg': θint,
            'θH_em_flw_min': θH_em_flw_min,
            'θH_em_flw_calc': θH_em_flw_calc,
            'ΔθH_em_air_eff': ΔθH_em_air_eff,
            'ΔθH_em_w_n': ΔθH_em_w_n,
            'βH_em_req': βH_em_req
        }

    def calculate_type_C5(self, common_params, θint):
        """
        C.5 — CONSTANT mass flow and VARIABLE heat exchange (bypass behavior).
        Keeps nominal flow; actual exchange varies with selected minimum supply.
        """
        ΦH_em_eff = common_params['ΦH_em_eff']
        V_H_em_nom = common_params['V_H_em_nom']
        ΔθH_em_air = common_params['ΔθH_em_air']
        nH_em = common_params['nH_em']
        ΔθH_em_n = common_params['ΔθH_em_n']

        θH_em_flw_min_set = float(self.df_heat_emm_data.loc["Minimum flow temperature for HZ1",
                                                            "θH_em_flw_min_tz_i"])
        βH_em_req = float(self.df_heat_emm_data.loc["Desired load factor with ON-OFF for HZ1",
                                                    "βH_em_req_sahz_i"])
        ΔθH_em_w_n = ΔθH_em_n

        # Nominal mass flow
        V_H_em_eff = V_H_em_nom

        # Supply calculation (similar to C.4)
        if ΦH_em_eff > 0 and self.ΦH_em_n > 0 and βH_em_req > 0:
            θH_em_flw_calc = (
                θint + (ΔθH_em_air + ΔθH_em_w_n / 2) *
                (((ΦH_em_eff / self.ΦH_em_n) * (100 / βH_em_req)) ** (1 / nH_em))
            )
        else:
            θH_em_flw_calc = θint

        # Selected minimum supply
        θH_em_flw_set = 0  # user-defined hook
        θH_em_flw_min = max(θH_em_flw_calc, θH_em_flw_min_set, θH_em_flw_set)

        # Minimum distribution supply with mixing margin
        θH_em_flw_min_final = θH_em_flw_calc + self.ΔθH_em_mix_sahz_i if self.MIX_EM else θH_em_flw_calc

        return {
            'θH_em_flow': θH_em_flw_min,
            'θH_em_ret': θint,  # refined later
            'V_H_em_eff': V_H_em_eff,
            'θH_em_avg': θint,
            'θH_em_flw_min': θH_em_flw_min_final,
            'θH_em_flw_calc': θH_em_flw_calc,
            'ΔθH_em_air_eff': 0
        }

    def calculate_circuit_node_temperature(self, θext, emission_results):
        """
        Compute the secondary node (circuit) supply temperature θH_nod_out according to the
        selected secondary control policy: based on demand, outdoor, or constant.
        """
        # Outdoor-based (secondary climatic curve)
        θem_flw_min = float(self.df_out_temp.loc["Minimum flow temperature", "θem_flw_min_sahz_i"])
        θem_flw_max = float(self.df_out_temp.loc["Maximum flow temperature", "θem_flw_max_sahz_i"])
        θext_max = float(self.df_out_temp.loc["Maximum outdoor temperature", "θext_max_sahz_i"])
        θext_min = float(self.df_out_temp.loc["Minimum outdoor temperature", "θext_min_sahz_i"])

        if θext_max != θext_min:
            θH_cr_flw_ext = min(
                max(θem_flw_min + (θem_flw_max - θem_flw_min) * (θext_max - θext) / (θext_max - θext_min),
                    θem_flw_min),
                θem_flw_max
            )
        else:
            θH_cr_flw_ext = θem_flw_min

        # Constant secondary supply
        θH_cr_flw_const = self.θem_flw_sahz_i[0]

        # Demand-based (from emission block)
        θH_cr_flw_demand = emission_results['θH_em_flw_min']

        # Select policy
        if self.flow_temp_control_type == 'Type 1 - Based on demand':
            θH_nod_out = θH_cr_flw_demand
        elif self.flow_temp_control_type == 'Type 2 - Based on outdoor temperature':
            θH_nod_out = θH_cr_flw_ext
        elif self.flow_temp_control_type == 'Type 3 - Constant temperature':
            θH_nod_out = θH_cr_flw_const
        else:
            θH_nod_out = θH_cr_flw_ext

        return θH_nod_out

    def calculate_operating_conditions(self, emission_results, common_params, θint, θH_nod_out):
        """
        Compute full operating conditions depending on the selected circuit type:
        returns 7+ key outputs (circuit temps/flow, distribution temps, load factor, etc.).
        """
        circuit_type = self.selected_emm_cont_circuit
        ΦH_em_eff = common_params['ΦH_em_eff']
        QH_sys_out_hz_i = common_params['QH_sys_out_hz_i']

        if circuit_type == 0:  # C.2 Constant flow
            # Circuit supply is the node supply
            θH_cr_flw = θH_nod_out
            # Circuit return equals emitter return
            θH_cr_ret = emission_results['θH_em_ret']

            # Circuit mass flow from power and ΔT
            V_H_cr = ΦH_em_eff / ((θH_cr_flw - θH_cr_ret) * self.c_w) if (θH_cr_flw - θH_cr_ret) > 0 else 0.0

            # Distribution supply: with mixing valve use emitter supply; otherwise follow circuit
            θH_dis_flw = emission_results['θH_em_flow'] if self.MIX_EM else θH_cr_flw

            # Distribution return equals emitter return
            θH_dis_ret = emission_results['θH_em_ret']
            θH_em_ret_eff = emission_results['θH_em_ret']

            # Load factor
            βH_em = ΦH_em_eff / self.ΦH_em_n if self.ΦH_em_n > 0 else 0.0
            θH_em_flw_min = emission_results['θH_em_flw_min']

        elif circuit_type == 1:  # C.3 Variable flow
            θH_cr_flw = θH_nod_out

            if self.MIX_EM:
                θH_cr_ret = emission_results['θH_em_ret']
                θH_dis_flw = emission_results['θH_em_flow']
            else:
                θH_cr_ret = max(θint, 2 * emission_results['θH_em_avg'] - θH_cr_flw)
                θH_dis_flw = θH_cr_flw

            V_H_cr = ΦH_em_eff / ((θH_cr_flw - θH_cr_ret) * self.c_w) if (θH_cr_flw - θH_cr_ret) > 0 else 0.0
            θH_dis_ret = emission_results['θH_em_ret']
            θH_em_ret_eff = emission_results['θH_em_ret']
            βH_em = ΦH_em_eff / self.ΦH_em_n if self.ΦH_em_n > 0 else 0.0
            θH_em_flw_min = emission_results['θH_em_flw_min']

        elif circuit_type == 2:  # C.4 ON-OFF
            # Effective emitter supply during ON
            θH_em_flw_eff = emission_results['θH_em_flw_calc'] if self.MIX_EM else θH_nod_out

            ΔθH_em_air = common_params['ΔθH_em_air']
            nH_em = common_params['nH_em']
            ΔθH_em_w_n = emission_results.get('ΔθH_em_w_n', 5)

            # Relative output when ON
            if (ΔθH_em_air + ΔθH_em_w_n / 2) > 0:
                βH_em_ON = max((θH_em_flw_eff - θint) / (ΔθH_em_air + ΔθH_em_w_n / 2), 0) ** nH_em
            else:
                βH_em_ON = 0.0

            ΦH_em_ON = self.ΦH_em_n * βH_em_ON

            # ON time to meet hourly energy
            t_H_cr_ON = QH_sys_out_hz_i / ΦH_em_ON if ΦH_em_ON > 0 else 0.0

            # Emitter return during ON
            V_H_em_eff = emission_results['V_H_em_eff']
            if (self.c_w * V_H_em_eff) > 0:
                θH_em_ret = θH_em_flw_eff - (ΦH_em_ON / (self.c_w * V_H_em_eff))
            else:
                θH_em_ret = θint
            θH_em_ret_eff = θH_em_ret

            # Circuit temps/flow
            θH_cr_flw = θH_nod_out
            θH_cr_ret = θH_em_ret
            V_H_cr = ΦH_em_eff / ((θH_cr_flw - θH_cr_ret) * self.c_w) if (θH_cr_flw - θH_cr_ret) > 0 else 0.0

            # Distribution temps
            θH_dis_flw = θH_em_flw_eff
            θH_dis_ret = θH_em_ret

            # Duty-cycle load factor
            βH_em = (t_H_cr_ON / self.tH_em_i_ON) if self.tH_em_i_ON > 0 else 0.0
            θH_em_flw_min = emission_results['θH_em_flw_min']

        elif circuit_type == 3:  # C.5 Constant flow & variable exchange
            # Effective emitter supply
            θH_em_flw_eff = emission_results['θH_em_flw_min'] if self.MIX_EM else θH_nod_out

            # Emitter return from power and nominal flow
            V_H_em_eff = emission_results['V_H_em_eff']
            if (self.c_w * V_H_em_eff) > 0:
                θH_em_ret = θH_em_flw_eff - (ΦH_em_eff / (self.c_w * V_H_em_eff))
            else:
                θH_em_ret = θint
            θH_em_ret_eff = θH_em_ret

            # Average emitter temp and ON power estimate
            θH_em_avg = (θH_em_flw_eff + θH_em_ret) / 2
            try:
                ΔθH_em_air = common_params['ΔθH_em_air']
                nH_em = common_params['nH_em']
                ΦH_em_ON = self.ΦH_em_n * (((θH_em_avg - θint) / ΔθH_em_air) ** nH_em) if ΔθH_em_air > 0 else 0.0
            except Exception:
                ΦH_em_ON = 0.0

            # Circuit mass flow
            V_H_cr = ΦH_em_eff / ((θH_nod_out - θH_em_ret) * self.c_w) if (θH_nod_out - θH_em_ret) * self.c_w > 0 else 0.0

            # Distribution temps
            θH_dis_flw = θH_em_flw_eff
            θH_dis_ret = θH_em_ret

            # Load factor
            βH_em = (ΦH_em_eff / ΦH_em_ON) if ΦH_em_ON > 0 else 0.0

            θH_cr_flw = θH_nod_out
            θH_cr_ret = θH_em_ret
            θH_em_flw_min = emission_results['θH_em_flw_min']

        else:
            raise ValueError("selected_emm_cont_circuit must be in {0,1,2,3}")

        return {
            'θH_cr_flw': θH_cr_flw,
            'θH_cr_ret': θH_cr_ret,
            'V_H_cr': V_H_cr,
            'θH_dis_flw': θH_dis_flw,
            'θH_dis_ret': θH_dis_ret,
            'βH_em': βH_em,
            'θH_em_flw_min': θH_em_flw_min,
            'θH_em_ret_eff': θH_em_ret_eff,
        }

    # -------------------- DISTRIBUTION --------------------
    def calculate_distribution(self, θH_dis_flw, θH_dis_ret, θint, QH_em_i_in):
        """
        Compute distribution-side quantities (from terminal to manifold):
        - Thermal losses, auxiliaries, recoverable part
        - Required inlet energy and hydraulic flow on distribution
        """
        if self.distribution_calculation_mode == 'analytical':
            return self.calculate_distribution_analytical(θH_dis_flw, θH_dis_ret, θint, QH_em_i_in)

        return self.calculate_distribution_simplified(θH_dis_flw, θH_dis_ret, θint, QH_em_i_in)

    def calculate_distribution_simplified(self, θH_dis_flw, θH_dis_ret, θint, QH_em_i_in):
        """
        Simplified distribution model kept for backward compatibility.
        """
        # Losses [kWh] ~ ( (T_tubes_avg - T_int) * U [W/K] / 1000 ) * t[h]
        T_media = (θH_dis_flw + θH_dis_ret) / 2
        Q_w_dis_i_ls = (T_media - θint) * self.Heat_loss_coefficent_dist / 1000.0 * self.tH_dis_i_ON

        # Auxiliaries
        Q_w_dis_i_aux = self.tH_dis_i_ON * self.W_H_dist_i_aux / 1000.0 * self.f_h_dist_i_aux / 100.0

        # Recoverable fraction of losses
        if (not self.Heat_losses_recovered) or (self.simplified_or_holistic_approach == 0):
            Q_w_dis_i_ls_rbl_H = 0.0
        else:
            Q_w_dis_i_ls_rbl_H = Q_w_dis_i_ls * self.f_h_dist_i_ls / 100.0

        # Required inlet and actual inlet to distribution
        QH_dis_i_req = QH_em_i_in
        QH_dis_i_in = QH_dis_i_req + Q_w_dis_i_ls - Q_w_dis_i_aux - Q_w_dis_i_ls_rbl_H

        # Distribution mass flow from energy balance
        ΔT_dis = max(θH_dis_flw - θH_dis_ret, 0.0)
        if self.c_w * ΔT_dis * self.tH_dis_i_ON > 0:
            V_H_dis = QH_dis_i_in / (self.c_w * ΔT_dis * self.tH_dis_i_ON)
        else:
            V_H_dis = 0.0

        return {
            'Q_w_dis_i_ls': Q_w_dis_i_ls,
            'Q_w_dis_i_aux': Q_w_dis_i_aux,
            'Q_w_dis_i_ls_rbl_H': Q_w_dis_i_ls_rbl_H,
            'QH_dis_i_req': QH_dis_i_req,
            'QH_dis_i_in': QH_dis_i_in,
            'V_H_dis': V_H_dis
        }

    def _distribution_analytical_config(self, θH_dis_flw, θH_dis_ret):
        """
        Build a DistributionSystemCalculator configuration from either a detailed
        user-provided config or the simplified legacy parameters.
        """
        if isinstance(self.analytical_distribution_config, dict):
            return self.analytical_distribution_config

        pipe_section = {
            'length_m': float(self.input_data.get('distribution_length_m', 1.0)),
            'equivalent_length_m': float(self.input_data.get('distribution_equivalent_length_m', 0.0)),
            'linear_thermal_transmittance_W_mK': float(
                self.input_data.get('distribution_loss_coeff', self.Heat_loss_coefficent_dist)
            ),
            'ambient_temperature_C': float(self.input_data.get('distribution_ambient_temperature_C', 20.0)),
            'recoverable': bool(self.Heat_losses_recovered),
        }

        return {
            'time_step_hours': float(self.tH_dis_i_ON),
            'demand_unit': 'kWh',
            'heating': {
                'operation_mode': 'demand',
                'nominal_power_kW': float(self.ΦH_em_n),
                'design_flow_m3_h': float(self.input_data.get('distribution_design_flow_m3_h', 0.0)),
                'design_deltaT_K': float(self.input_data.get('distribution_design_deltaT_K', 10.0)),
                'supply_temperature_C': float(θH_dis_flw if np.isfinite(θH_dis_flw) else 45.0),
                'return_temperature_C': float(θH_dis_ret if np.isfinite(θH_dis_ret) else 35.0),
                'pipe_sections': [pipe_section],
                'pump_control_code': int(self.input_data.get('distribution_pump_control_code', 4)),
                'eei': float(self.input_data.get('distribution_eei', 0.23)),
                'hydraulic_correction_factor': float(self.input_data.get('distribution_hydraulic_correction_factor', 1.0)),
                'recoverable_aux_fraction': float(
                    self.input_data.get('distribution_aux_recovery_fraction', self.f_h_dist_i_aux / 100.0)
                ),
                'design_flow_m3_h': float(self.input_data.get('distribution_design_flow_m3_h', 0.0)),
                'max_length_m': float(self.input_data.get('distribution_length_m', 1.0)),
                'pressure_loss_per_m_kPa': float(self.input_data.get('distribution_pressure_loss_per_m_kPa', 0.10)),
                'additional_pressure_kPa': float(self.input_data.get('distribution_additional_pressure_kPa', 0.0)),
                'resistance_ratio': float(self.input_data.get('distribution_resistance_ratio', 0.30)),
                'design_delta_pressure_kPa': float(self.input_data.get('distribution_design_delta_pressure_kPa', 0.0)),
                'pump_selection_factor': float(self.input_data.get('distribution_pump_selection_factor', 1.0)),
                'pump_label_power_kW': float(self.input_data.get('distribution_pump_label_power_kW', 0.0)),
                'part_load_mode': str(self.input_data.get('distribution_part_load_mode', 'load')),
            },
        }

    def _init_distribution_fast(self, θH_dis_flw, θH_dis_ret):
        """
        Precompute the static parameters of the analytical distribution model so
        that the per-hour call can be done in pure-scalar math without going
        through pandas. Stored on ``self._distribution_fast``.
        """
        from pybuildingenergy.source.distribution_15316_3 import (
            _pump_control_constants, _KWH_EPS,
        )
        cfg_full = self._distribution_analytical_config(θH_dis_flw, θH_dis_ret)
        # Reuse a DistributionSystemCalculator just to normalize the options
        # (handles aliases, defaults, _section_options, etc.).
        if self._distribution_calc_cache is None:
            self._distribution_calc_cache = DistributionSystemCalculator(cfg_full)
        opts = self._distribution_calc_cache.services["H"]

        # Static aggregates over pipe sections
        sections = []
        rec_mask = []
        for s in opts["pipe_sections"]:
            length = float(s["length_m"]) + float(s["equivalent_length_m"])
            psi    = float(s["linear_thermal_transmittance_W_mK"])
            ambient = float(s["ambient_temperature_C"])
            recoverable = bool(s["recoverable"])
            sections.append((psi * length, ambient, recoverable))
            rec_mask.append(recoverable)

        # Design flow
        design_flow = float(opts["design_flow_m3_h"])
        nominal_power = float(opts["nominal_power_kW"])
        if design_flow <= _KWH_EPS and nominal_power > _KWH_EPS:
            # _WATER_HEAT_CAPACITY_DENSITY_KWH_M3K = 1.163 by EN 15316 (kWh/(m3·K))
            design_flow = nominal_power / (1.163 * float(opts["design_deltaT_K"]))

        # Design head loss
        delta_p = float(opts["design_delta_pressure_kPa"])
        if delta_p <= _KWH_EPS:
            delta_p = (
                (1.0 + float(opts["resistance_ratio"]))
                * float(opts["pressure_loss_per_m_kPa"])
                * float(opts["max_length_m"])
                + float(opts["additional_pressure_kPa"])
            )

        p_hydr = delta_p * design_flow / 3600.0 if design_flow > _KWH_EPS else 0.0

        # Pump efficiency factor (constant per design point — depends only on p_hydr)
        label_power = float(opts["pump_label_power_kW"])
        if p_hydr <= _KWH_EPS:
            f_e = 0.0
        elif label_power > _KWH_EPS:
            f_e = label_power / p_hydr
        else:
            p_hydr_W = p_hydr * 1000.0
            if 1.0 < p_hydr_W < 2500.0:
                f_e = (1.7 * p_hydr_W + 17.0 * (1.0 - np.exp(-0.3 * p_hydr_W))) / p_hydr_W
            else:
                f_e = max(float(opts["pump_selection_factor"]), 1.0)

        cp1, cp2 = _pump_control_constants("H", int(opts["pump_control_code"]))

        self._distribution_fast = {
            'sections':            sections,      # list of (psi*L, ambient, recoverable)
            'design_flow':         design_flow,
            'delta_p':             delta_p,
            'p_hydr':              p_hydr,
            'f_e':                 f_e,
            'cp1':                 cp1,
            'cp2':                 cp2,
            'eei':                 float(opts["eei"]),
            'hyd_corr':            float(opts["hydraulic_correction_factor"]),
            'rec_aux_fraction':    float(opts["recoverable_aux_fraction"]),
            'operation_mode':      opts["operation_mode"],
            'part_load_mode':      opts["part_load_mode"],
            'nominal_power_kW':    nominal_power,
            'time_step_hours':     float(self.tH_dis_i_ON),
            'has_pump':            p_hydr > _KWH_EPS and design_flow > _KWH_EPS and delta_p > _KWH_EPS,
        }
        self._dist_eps = _KWH_EPS

    def calculate_distribution_analytical(self, θH_dis_flw, θH_dis_ret, θint, QH_em_i_in):
        """
        Analytical distribution model — fast scalar implementation that
        replicates DistributionSystemCalculator (EN 15316-3) for service H,
        a single section list and a single time step. The math is identical
        to ``_simulate_service('H')`` with all pandas overhead removed.
        """
        # Lazy init of the static part on first call (after secondary T is known)
        if getattr(self, '_distribution_fast', None) is None:
            self._init_distribution_fast(θH_dis_flw, θH_dis_ret)

        S   = self._distribution_fast
        EPS = self._dist_eps

        q_out  = max(float(QH_em_i_in), 0.0)
        hours  = S['time_step_hours']

        # operation hours
        mode = S['operation_mode']
        if mode == 'continuous':
            op_hours = hours
        elif mode.startswith('fixed_fraction:'):
            try:
                fr = float(mode.split(':', 1)[1])
            except Exception:
                fr = 0.0
            op_hours = hours * max(0.0, min(1.0, fr))
        else:  # 'demand'
            op_hours = hours if q_out > EPS else 0.0

        mean_temp = 0.5 * (float(θH_dis_flw) + float(θH_dis_ret))

        # Thermal losses
        q_loss = 0.0
        q_loss_rbl = 0.0
        for psi_L, ambient, recoverable in S['sections']:
            delta = mean_temp - ambient
            if delta < 0.0:
                delta = 0.0
            loss = psi_L * delta * op_hours / 1000.0
            q_loss += loss
            if recoverable:
                q_loss_rbl += loss

        # Part-load fraction beta
        if S['part_load_mode'] == 'constant_when_on':
            beta = 1.0 if op_hours > EPS else 0.0
        else:  # 'load'
            nominal = S['nominal_power_kW']
            power = q_out / hours if hours > 0.0 else 0.0
            if nominal <= EPS:
                nominal = max(power, EPS)
            beta = (power / nominal) if (nominal > EPS and op_hours > EPS) else 0.0
            if beta < 0.0:   beta = 0.0
            elif beta > 1.0: beta = 1.0

        # Pump auxiliary
        if not S['has_pump'] or beta <= EPS:
            w_aux = 0.0
            w_hydr = 0.0
            epsilon = 0.0
        else:
            epsilon = S['f_e'] * (S['cp1'] + S['cp2'] / beta) * S['eei'] / 0.25
            w_hydr  = S['p_hydr'] * beta * op_hours * S['hyd_corr']
            w_aux   = w_hydr * epsilon

        aux_rbl = w_aux * S['rec_aux_fraction']
        aux_rvd = w_aux - aux_rbl

        thermal_rbl = q_loss_rbl  # service H sign
        q_in = q_out + q_loss - aux_rvd
        if q_in < 0.0:
            q_in = 0.0

        return {
            'Q_w_dis_i_ls':      q_loss,
            'Q_w_dis_i_aux':     w_aux,
            'Q_w_dis_i_ls_rbl_H': thermal_rbl + aux_rbl,
            'QH_dis_i_req':      q_out,
            'QH_dis_i_in':       q_in,
            'V_H_dis':           S['design_flow'],
        }

    def _efficiency_from_model(self, θflw, θret, load_frac=1.0):
        """
        Return generator efficiency (%) according to the selected efficiency model.
        Arguments are on the PRIMARY side (flow/return).
        """
        import numpy as _np
        model = self.efficiency_model

        # 1) Simple heuristic vs. return temperature
        if model == 'simple':
            if θret < 50.0:
                eff = 110.0 - 0.5 * (θret - 20.0)
            else:
                eff = 95.0
            return max(min(eff, 110.0), 1.0)

        # 2) Parametric piecewise model
        if model == 'parametric':
            eta_max   = float(self.eta_max)
            eta_nc    = float(self.eta_no_cond)
            T_min     = float(self.T_ret_min)
            T_thr     = float(self.T_ret_thr)

            Tret_clip = max(min(θret, T_thr), T_min)
            slope = (eta_max - eta_nc) / (T_min - T_thr)
            if θret <= T_thr:
                eff = eta_max + slope * (Tret_clip - T_min)
            else:
                eff = eta_nc
            return max(min(eff, 110.0), 1.0)

        # 3) Manufacturer curves
        if model == 'manufacturer':
            # 3a) 1D curve: eff = f(T_return)
            if isinstance(self.manuf_curve_1d, dict):
                Tret = _np.array(self.manuf_curve_1d.get('Tret', []), dtype=float)
                eta  = _np.array(self.manuf_curve_1d.get('eta',  []), dtype=float)
                if Tret.size >= 2 and eta.size == Tret.size:
                    eff = _np.interp(θret, Tret, eta)
                    return max(min(float(eff), 110.0), 1.0)

            # 3b) 2D surface: eff = f(T_flow, T_return) via bilinear interpolation
            if isinstance(self.manuf_curve_2d, dict):
                Tflw = _np.array(self.manuf_curve_2d.get('Tflw', []), dtype=float)
                Tret = _np.array(self.manuf_curve_2d.get('Tret', []), dtype=float)
                ETA  = _np.array(self.manuf_curve_2d.get('eta',  []), dtype=float)

                if (Tflw.size >= 2 and Tret.size >= 2 and ETA.shape == (Tflw.size, Tret.size)):
                    i = _np.searchsorted(Tflw, θflw) - 1
                    j = _np.searchsorted(Tret, θret) - 1
                    i = min(max(i, 0), Tflw.size - 2)
                    j = min(max(j, 0), Tret.size - 2)

                    x1, x2 = Tflw[i], Tflw[i+1]
                    y1, y2 = Tret[j], Tret[j+1]
                    Q11 = ETA[i, j]
                    Q12 = ETA[i, j+1]
                    Q21 = ETA[i+1, j]
                    Q22 = ETA[i+1, j+1]

                    tx = 0.0 if x2 == x1 else (θflw - x1) / (x2 - x1)
                    ty = 0.0 if y2 == y1 else (θret - y1) / (y2 - y1)

                    eff = (
                        Q11 * (1 - tx) * (1 - ty) +
                        Q21 * tx       * (1 - ty) +
                        Q12 * (1 - tx) * ty       +
                        Q22 * tx       * ty
                    )
                    return max(min(float(eff), 110.0), 1.0)

            # Fallback to simple model if maps are invalid
            if θret < 50.0:
                eff = 110.0 - 0.5 * (θret - 20.0)
            else:
                eff = 95.0
            return max(min(eff, 110.0), 1.0)

        # Global fallback
        if θret < 50.0:
            eff = 110.0 - 0.5 * (θret - 20.0)
        else:
            eff = 95.0
        return max(min(eff, 110.0), 1.0)

    def _generator_flow_setpoint(self, θext, θH_dis_flw_demand):
        """
        Compute the PRIMARY (generator-side) supply setpoint Tp_sup_set according to
        the chosen primary control:

        - 'Type A - Based on outdoor temperature': uses primary's own weather curve
        - 'Type B - Based on demand'            : follows the secondary demand (θH_dis_flw)
        - 'Type C - Constant temperature'       : uses self.θHW_gen_flw_const
        """
        row = self.df_gen_out_temp.iloc[0] if len(self.df_gen_out_temp) else {}
        θext_min = float(row.get("θext_min_gen", -7))
        θext_max = float(row.get("θext_max_gen", 15))
        θflw_max = float(row.get("θflw_gen_max", 60))
        θflw_min = float(row.get("θflw_gen_min", 35))

        ctrl = str(self.gen_flow_temp_control_type)

        # Type A: primary climatic curve
        if ctrl.startswith('Type A'):
            if θext_max != θext_min:
                Tp = θflw_min + (θflw_max - θflw_min) * (θext_max - θext) / (θext_max - θext_min)
                return max(min(Tp, θflw_max), θflw_min)
            return θflw_min

        # Type B: follow secondary demand (node/secondary supply)
        if ctrl.startswith('Type B'):
            return float(θH_dis_flw_demand)

        # Type C: constant
        return float(self.θHW_gen_flw_const)

    def calculate_generation(self, θH_dis_flw, θH_dis_ret, V_H_dis, QH_dis_i_in):
        if self.generation_calculation_mode == 'boiler_15316_4_1':
            return self.calculate_generation_boiler(θH_dis_flw, θH_dis_ret, V_H_dis, QH_dis_i_in)

        EPS = 1e-9
        rho = getattr(self, "rho_w", 1000.0)        # kg/m3
        cw  = self.c_w                               # Wh/(kg·K), e.g. 1.163

        # 1) Energy cap (kWh)
        max_output_g1 = self.full_load_power * (self.max_monthly_load_factor / 100.0) * self.tH_gen_i_ON
        QX_gen_j_out_req = float(QH_dis_i_in)
        QH_gen_out = min(max(QX_gen_j_out_req, 0.0), max_output_g1)

        # 2) Convert distribution flow to MASS FLOW (kg/h) for balances
        V_dis = max(float(V_H_dis), 0.0)             # m3/h
        ms    = rho * V_dis                          # kg/h

        if QH_gen_out <= EPS or ms <= EPS:
            return {
                'max_output_g1(kWh)': max_output_g1,
                'QH_gen_out(kWh)': 0.0,
                'θX_gen_cr_flw(°C)': float(θH_dis_flw),
                'θX_gen_cr_ret(°C)': float(θH_dis_ret),
                'V_H_gen(m3/h)': 0.0,                # zero volumetric flow
                'efficiency_gen(%)': float('nan'),
                'EHW_gen_in(kWh)': 0.0,
                'EHW_gen_aux(kWh)': 0.0,
                'QW_gen_i_ls_rbl_H(kWh)': 0.0,
                'QW_gen_out(kWh)': 0.0,
                'QHW_gen_out(kWh)': 0.0,
                'EH_gen_in(kWh)': 0.0,
                'EWH_gen_in(kWh)': 0.0,
            }

        # Initial efficiency guess from model (can be updated later with actual temps)
        θHW_gen_ret_eff = self.θHW_gen_ret_set if self.θHW_gen_ret_set is not None else float(θH_dis_ret)
        θHW_gen_flw_eff = self.θHW_gen_flw_set if self.θHW_gen_flw_set is not None else float(θH_dis_flw)
        efficiency = self._efficiency_from_model(θHW_gen_flw_eff, θHW_gen_ret_eff, load_frac=1.0)

        if self.generator_circuit == "direct":
            # Primary mass flow equals secondary mass flow
            mp = ms                                   # kg/h
            Vp = mp / rho                             # m3/h
            θX_gen_cr_flw = float(θH_dis_flw)
            θX_gen_cr_ret = float(θH_dis_ret)

        else:
            # Independent circuit
            Ts_sup_req = float(θH_dis_flw)
            Ts_ret     = float(θH_dis_ret)

            # ΔT on primary
            if str(self.speed_control_generator_pump) == 'deltaT_constant':
                dTp = float(self.generator_nominal_deltaT)
            else:
                dTp = float(self.generator_nominal_deltaT) * (QH_gen_out / max(self.full_load_power, 1e-6))
                dTp = max(dTp, 1.0)

            # Primary supply setpoint
            if self.θHW_gen_flw_set is not None:
                Tp_sup_set = float(self.θHW_gen_flw_set)
            else:
                ext = getattr(self, "current_theta_ext", None)
                Tp_sup_set = float(self._generator_flow_setpoint(θext=ext, θH_dis_flw_demand=Ts_sup_req))

            Tp_sup = max(Tp_sup_set, Ts_sup_req)

            # Primary line loss in °C
            self.primary_line_loss = getattr(self, 'primary_line_loss', 1.0)
            Ts_sup_eff = max(Tp_sup - self.primary_line_loss, Ts_ret)

            # Primary MASS flow (kg/h) from requested energy
            mp = QH_gen_out * 1000.0 / (cw * dTp)     # kWh → Wh (*1000) / (Wh/kgK * K) = kg

            # Anti-dilution: compare mass flows (same unit now)
            if not getattr(self, 'allow_dilution', False):
                TOL = 0.1; MAX_ITER = 20
                for _ in range(MAX_ITER):
                    mix_sup = Tp_sup if mp >= ms else (mp * Tp_sup + (ms - mp) * Ts_ret) / max(ms, 1e-9)
                    if (Tp_sup - mix_sup) > TOL and mp < ms:
                        mp *= 1.10
                    else:
                        Ts_sup_eff = mix_sup
                        break
            else:
                Ts_sup_eff = Tp_sup if mp >= ms else (mp * Tp_sup + (ms - mp) * Ts_ret) / max(ms, 1e-9)

            # Primary return using mass flows
            Tp_ret = Tp_sup - (ms / mp) * (Ts_sup_eff - Ts_ret) if mp > EPS else Tp_sup

            θX_gen_cr_flw = Tp_sup
            θX_gen_cr_ret = Tp_ret
            Vp = mp / rho                              # m3/h

            # Optional: re-evaluate efficiency with actual temps via the chosen model
            efficiency = self._efficiency_from_model(θX_gen_cr_flw, θX_gen_cr_ret, load_frac=QH_gen_out/max(self.full_load_power, EPS))

        # Energy accounting
        efficiency = max(min(efficiency, 110.0), 1.0)
        EHW_gen_in  = (QH_gen_out * 100.0) / efficiency if efficiency > 0 else 0.0
        EHW_gen_aux = EHW_gen_in * (self.auxiliary_power_generator / 100.0)

        # If this is the “recoverable fraction of generator thermal losses”, rename accordingly
        gen_losses = max(EHW_gen_in - QH_gen_out, 0.0)
        QW_gen_i_ls_rbl_H = gen_losses * (self.fraction_of_auxiliary_power_generator / 100.0)

        QW_gen_out  = 0.0
        QHW_gen_out = QH_gen_out + QW_gen_out

        EH_gen_in  = (EHW_gen_in * QH_gen_out / QHW_gen_out) if QHW_gen_out > EPS else 0.0
        EWH_gen_in = (EHW_gen_in * QW_gen_out / QHW_gen_out) if QHW_gen_out > EPS else 0.0

        return {
            'max_output_g1(kWh)': max_output_g1,
            'QH_gen_out(kWh)': QH_gen_out,
            'θX_gen_cr_flw(°C)': θX_gen_cr_flw,
            'θX_gen_cr_ret(°C)': θX_gen_cr_ret,
            'V_H_gen(m3/h)': Vp,                     # volumetric primary flow (coherent unit)
            'efficiency_gen(%)': efficiency,
            'EHW_gen_in(kWh)': EHW_gen_in,
            'EHW_gen_aux(kWh)': EHW_gen_aux,
            'QW_gen_i_ls_rbl_H(kWh)': QW_gen_i_ls_rbl_H,
            'QW_gen_out(kWh)': QW_gen_out,
            'QHW_gen_out(kWh)': QHW_gen_out,
            'EH_gen_in(kWh)': EH_gen_in,
            'EWH_gen_in(kWh)': EWH_gen_in,
        }

    def _build_boiler_generation_config(self):
        if isinstance(self.boiler_generation_config, dict):
            return self.boiler_generation_config
        raise ValueError(
            "generation_calculation_mode='boiler_15316_4_1' requires boiler_generation_config "
            "with EN 15316-4-1 case tables."
        )

    def _get_boiler_calculator(self):
        if self._boiler_calc_cache is None:
            self._boiler_calc_cache = BoilerGeneratorCalculator(self._build_boiler_generation_config())
        return self._boiler_calc_cache

    def _solve_primary_hydraulics(self, θH_dis_flw, θH_dis_ret, V_H_dis, QH_gen_out):
        """
        Compute primary-side flow temperatures (θX_gen_cr_flw/ret) and the
        primary volumetric flow Vp using the same logic as the legacy generation
        branch (direct / independent circuit, climatic curve A/B/C, anti-dilution).

        Returns dict with keys: θX_gen_cr_flw, θX_gen_cr_ret, Vp (m3/h).
        """
        EPS = 1e-9
        rho = getattr(self, "rho_w", 1000.0)
        cw  = self.c_w

        ms = rho * max(float(V_H_dis), 0.0)              # kg/h secondary mass flow

        if QH_gen_out <= EPS or ms <= EPS:
            return {
                'θX_gen_cr_flw': float(θH_dis_flw),
                'θX_gen_cr_ret': float(θH_dis_ret),
                'Vp':            0.0,
            }

        if self.generator_circuit == "direct":
            return {
                'θX_gen_cr_flw': float(θH_dis_flw),
                'θX_gen_cr_ret': float(θH_dis_ret),
                'Vp':            ms / rho,
            }

        # Independent circuit
        Ts_sup_req = float(θH_dis_flw)
        Ts_ret     = float(θH_dis_ret)

        # ΔT on primary
        if str(self.speed_control_generator_pump) == 'deltaT_constant':
            dTp = float(self.generator_nominal_deltaT)
        else:
            dTp = float(self.generator_nominal_deltaT) * (QH_gen_out / max(self.full_load_power, 1e-6))
            dTp = max(dTp, 1.0)

        # Primary supply setpoint (Type A/B/C climatic curve)
        if self.θHW_gen_flw_set is not None:
            Tp_sup_set = float(self.θHW_gen_flw_set)
        else:
            ext = getattr(self, "current_theta_ext", None)
            Tp_sup_set = float(self._generator_flow_setpoint(θext=ext, θH_dis_flw_demand=Ts_sup_req))

        Tp_sup = max(Tp_sup_set, Ts_sup_req)

        # Primary line loss
        self.primary_line_loss = getattr(self, 'primary_line_loss', 1.0)
        Ts_sup_eff = max(Tp_sup - self.primary_line_loss, Ts_ret)

        # Primary mass flow from requested energy
        mp = QH_gen_out * 1000.0 / (cw * dTp)

        # Anti-dilution iteration on the mixing valve
        if not getattr(self, 'allow_dilution', False):
            TOL = 0.1
            for _ in range(20):
                mix_sup = Tp_sup if mp >= ms else (mp * Tp_sup + (ms - mp) * Ts_ret) / max(ms, 1e-9)
                if (Tp_sup - mix_sup) > TOL and mp < ms:
                    mp *= 1.10
                else:
                    Ts_sup_eff = mix_sup
                    break
        else:
            Ts_sup_eff = Tp_sup if mp >= ms else (mp * Tp_sup + (ms - mp) * Ts_ret) / max(ms, 1e-9)

        Tp_ret = Tp_sup - (ms / mp) * (Ts_sup_eff - Ts_ret) if mp > EPS else Tp_sup
        return {
            'θX_gen_cr_flw': Tp_sup,
            'θX_gen_cr_ret': Tp_ret,
            'Vp':            mp / rho,
        }

    def calculate_generation_boiler(self, θH_dis_flw, θH_dis_ret, V_H_dis, QH_dis_i_in):
        """
        EN 15316-4-1 boiler generation branch. Computes the primary-side
        hydraulics (θX_gen_cr_flw/ret, Vp) consistently with the legacy branch,
        then feeds θ_avg / θ_return to the BoilerGeneratorCalculator.
        """
        # 1) Energy cap (kWh) — same convention as legacy branch
        max_output_g1 = self.full_load_power * (self.max_monthly_load_factor / 100.0) * self.tH_gen_i_ON
        QH_gen_out = min(max(float(QH_dis_i_in), 0.0), max_output_g1)

        # 2) Primary-side hydraulics (climatic curve, anti-dilution, ΔT)
        hyd = self._solve_primary_hydraulics(θH_dis_flw, θH_dis_ret, V_H_dis, QH_gen_out)
        Tp_sup = hyd['θX_gen_cr_flw']
        Tp_ret = hyd['θX_gen_cr_ret']
        Vp     = hyd['Vp']

        # 3) Delegate to EN 15316-4-1 boiler model using primary-side temperatures
        calc = self._get_boiler_calculator()
        boiler_out = calc.compute_step(
            Q_gen_out_kWh=float(QH_gen_out),
            t_use_h=float(self.tH_gen_i_ON),
            theta_avg_C=(float(Tp_sup) + float(Tp_ret)) / 2.0,
            theta_return_C=float(Tp_ret),
            Q_H_gen_out_kWh=float(QH_gen_out),
            Q_W_gen_out_kWh=0.0,
        )

        # 4) Merge hydraulic outputs into the boiler dict (overrides defaults)
        boiler_out['max_output_g1(kWh)'] = max_output_g1
        boiler_out['θX_gen_cr_flw(°C)']  = Tp_sup
        boiler_out['θX_gen_cr_ret(°C)']  = Tp_ret
        boiler_out['V_H_gen(m3/h)']      = Vp
        return boiler_out

    # -------------------- ONE STEP EXECUTION --------------------
    def compute_step(self, q_h_kWh, θint, θext):
        """
        Execute the full computation for a single time-step:
        1) emission (one of C.2/C.3/C.4/C.5)
        2) choose node supply temperature (secondary control)
        3) operating conditions
        4) distribution
        5) generation (primary)
        Returns a dict with main outputs (temps, flows, energies).
        """
        # 1) Emission-side step, either simplified ISO 15316-1 internal logic
        #    or delegated EN 15316-2 per hourly row.
        em_step = self._calculate_emission_step(q_h_kWh, θint, θext)
        q_h_em_in = float(em_step['QH_em_i_in'])
        θint_eff = float(em_step['θint_eff'])
        q_h_em_out = float(em_step['q_h_em_out_kWh'])

        # 2) Common emission params for the internal ISO 15316-1 path.
        #    Keep them even in EN 15316-2 mode because downstream HVAC blocks
        #    still expect the historical emission-side power and load factor.
        common = self.calculate_common_emission_parameters(q_h_em_out, θint_eff)
        common['QH_em_i_in'] = q_h_em_in
        common['ΦH_em_eff'] = float(em_step['ΦH_em_eff'])
        self.current_theta_ext = float(θext)  # stored for primary control (Type A)

        # 3) Emission block per selected type
        if self.selected_emm_cont_circuit == 0:
            em = self.calculate_type_C2(common, θint_eff)
        elif self.selected_emm_cont_circuit == 1:
            em = self.calculate_type_C3(common, θint_eff)
        elif self.selected_emm_cont_circuit == 2:
            em = self.calculate_type_C4(common, θint_eff)
        elif self.selected_emm_cont_circuit == 3:
            em = self.calculate_type_C5(common, θint_eff)
        else:
            raise ValueError("selected_emm_cont_circuit must be in {0,1,2,3}")

        # 4) Node supply temp (secondary control)
        θH_nod_out = self.calculate_circuit_node_temperature(θext, em)

        # 5) Operating conditions
        op = self.calculate_operating_conditions(em, common, θint_eff, θH_nod_out)

        # 6) Distribution block
        dist = self.calculate_distribution(op['θH_dis_flw'], op['θH_dis_ret'], θint_eff, q_h_em_in)

        # 7) Generation (primary) block
        gen = self.calculate_generation(
            θH_dis_flw=op['θH_dis_flw'],
            θH_dis_ret=op['θH_dis_ret'],
            V_H_dis=dist['V_H_dis'],
            QH_dis_i_in=dist['QH_dis_i_in'],
        )

        def _gen_value(*keys, default=0.0):
            for key in keys:
                if key in gen:
                    return gen[key]
            return default

        # 7) Collect outputs
        out = {
            # per-step inputs
            'Q_h(kWh)': q_h_kWh,
            'T_op(°C)': θint_eff,
            'T_ext(°C)': θext,

            # emission (selected)
            'ΦH_em_eff(kW)': common['ΦH_em_eff'],
            # effective distribution/emission supply shown as emitter flow
            'θH_em_flow(°C)': op['θH_dis_flw'],
            'θH_em_ret(°C)': op['θH_em_ret_eff'],
            'V_H_em_eff(m3/h)': em['V_H_em_eff'],
            'θH_em_flw_min_req(°C)': em['θH_em_flw_min'],

            # node / operating
            'θH_nod_out(°C)': θH_nod_out,
            'θH_cr_flw(°C)': op['θH_cr_flw'],
            'θH_cr_ret(°C)': op['θH_cr_ret'],
            'V_H_cr(m3/h)': op['V_H_cr'],
            'θH_dis_flw(°C)': op['θH_dis_flw'],
            'θH_dis_ret(°C)': op['θH_dis_ret'],
            'βH_em(-)': op['βH_em'],

            # distribution
            'Q_w_dis_i_ls(kWh)': dist['Q_w_dis_i_ls'],
            'Q_w_dis_i_aux(kWh)': dist['Q_w_dis_i_aux'],
            'Q_w_dis_i_ls_rbl_H(kWh)': dist['Q_w_dis_i_ls_rbl_H'],
            'QH_dis_i_req(kWh)': dist['QH_dis_i_req'],
            'QH_dis_i_in(kWh)': dist['QH_dis_i_in'],
            'V_H_dis(m3/h)': dist['V_H_dis'],
            'QH_em_i_in(kWh)': q_h_em_in,
            'QH_em_ls(kWh)': em_step['QH_em_ls_kWh'],
            'W_H_em_aux(kWh)': em_step['W_H_em_aux_kWh'],
            'theta_H_int_inc_K': em_step['theta_H_int_inc_K'],
            'emission_calculation_mode': self.emission_calculation_mode,
        }

        # merge generator outputs
        out.update({
            'max_output_g1(kWh)': _gen_value('max_output_g1(kWh)', default=0.0),
            # Aliases for EN 15316-4-1 boiler module: it returns 'Q_gen_out(kWh)',
            # 'E_gen_in(kWh)', 'E_H_gen_in(kWh)', 'E_W_gen_in(kWh)', 'W_gen_aux(kWh)'.
            'QH_gen_out(kWh)': _gen_value('QH_gen_out(kWh)', 'Q_gen_out(kWh)', 'Q_gen_out_kWh', default=0.0),
            'θX_gen_cr_flw(°C)': _gen_value('θX_gen_cr_flw(°C)', default=op['θH_dis_flw']),
            'θX_gen_cr_ret(°C)': _gen_value('θX_gen_cr_ret(°C)', default=op['θH_dis_ret']),
            'V_H_gen(m3/h)': _gen_value('V_H_gen(m3/h)', default=dist['V_H_dis']),
            'efficiency_gen(%)': _gen_value('efficiency_gen(%)', 'eta_Pn_corr(%)', default=float('nan')),
            'EHW_gen_in(kWh)': _gen_value('EHW_gen_in(kWh)', 'E_gen_in(kWh)', 'E_gen_in_kWh', default=0.0),
            'EHW_gen_aux(kWh)': _gen_value('EHW_gen_aux(kWh)', 'W_gen_aux(kWh)', default=0.0),
            'QW_gen_i_ls_rbl_H(kWh)': _gen_value('QW_gen_i_ls_rbl_H(kWh)', 'Q_gen_ls_rbl(kWh)', default=0.0),
            'QW_gen_out(kWh)': _gen_value('QW_gen_out(kWh)', default=0.0),
            'QHW_gen_out(kWh)': _gen_value('QHW_gen_out(kWh)', default=_gen_value('QH_gen_out(kWh)', 'Q_gen_out(kWh)', 'Q_gen_out_kWh', default=0.0)),
            'EH_gen_in(kWh)': _gen_value('EH_gen_in(kWh)', 'E_H_gen_in(kWh)', default=0.0),
            'EWH_gen_in(kWh)': _gen_value('EWH_gen_in(kWh)', 'E_W_gen_in(kWh)', default=0.0),
        })
        return out

    # -------------------- TIMESERIES EXECUTION --------------------
    def _temps_row_when_off(self, θint, θext):
        """
        Build a 'temperatures-only' row when Q_H <= 0:
        - compute node/supply anyway (per control),
        - set powers/flows to 0,
        - return temps coherent with no exchange (≈ θint where applicable).
        """
        common = self.calculate_common_emission_parameters(0.0, θint)

        # Emission block at zero load (still follows selected type for temps)
        if self.selected_emm_cont_circuit == 0:
            em = self.calculate_type_C2(common, θint)
        elif self.selected_emm_cont_circuit == 1:
            em = self.calculate_type_C3(common, θint)
        elif self.selected_emm_cont_circuit == 2:
            em = self.calculate_type_C4(common, θint)
        else:
            em = self.calculate_type_C5(common, θint)

        θH_nod_out = self.calculate_circuit_node_temperature(θext, em)

        # Distribution supply equals emitter supply; return ~ ambient
        θH_dis_flw = em.get('θH_em_flow', θH_nod_out)
        θH_dis_ret = θint

        # Primary: no load → ΔT→0 and flow→0
        θX_gen_cr_flw = θH_dis_flw
        θX_gen_cr_ret = θH_dis_ret

        row = {
            'ΦH_em_eff(kW)': 0.0,
            'θH_em_flow(°C)': em.get('θH_em_flow', θH_nod_out),
            'θH_em_ret(°C)': θH_dis_ret,
            'V_H_em_eff(m3/h)': 0.0,
            'θH_em_flw_min_req(°C)': em.get('θH_em_flw_min', θH_nod_out),

            'θH_nod_out(°C)': θH_nod_out,
            'θH_cr_flw(°C)': θH_nod_out,
            'θH_cr_ret(°C)': θH_dis_ret,
            'V_H_cr(m3/h)': 0.0,
            'θH_dis_flw(°C)': θH_dis_flw,
            'θH_dis_ret(°C)': θH_dis_ret,
            'βH_em(-)': 0.0,

            'Q_w_dis_i_ls(kWh)': 0.0,
            'Q_w_dis_i_aux(kWh)': 0.0,
            'Q_w_dis_i_ls_rbl_H(kWh)': 0.0,
            'QH_dis_i_req(kWh)': 0.0,
            'QH_dis_i_in(kWh)': 0.0,
            'V_H_dis(m3/h)': 0.0,

            'max_output_g1(kWh)': 0.0,
            'QH_gen_out(kWh)': 0.0,
            'θX_gen_cr_flw(°C)': θX_gen_cr_flw,
            'θX_gen_cr_ret(°C)': θX_gen_cr_ret,
            'V_H_gen(m3/h)': 0.0,
            'efficiency_gen(%)': np.nan,
            'EHW_gen_in(kWh)': 0.0,
            'EHW_gen_aux(kWh)': 0.0,
            'QW_gen_i_ls_rbl_H(kWh)': 0.0,
            'QW_gen_out(kWh)': 0.0,
            'QHW_gen_out(kWh)': 0.0,
            'EH_gen_in(kWh)': 0.0,
            'EWH_gen_in(kWh)': 0.0,
        }
        return row

    def run_timeseries(self, df=None):
        """
        Run the full computation across the loaded time series.

        Recognized columns (with aliases):
          - Q_H_kWh (aliases: Q_H, Q_h, Heating_needs)
          - T_op    (aliases: T_int, theta_int)
          - T_ext   (aliases: theta_ext)

        Returns a DataFrame indexed by timestamp with all outputs per step.
        """
        if df is None:
            if not hasattr(self, 'timeseries_data'):
                raise ValueError("Load data first via load_csv_data(...) or pass a DataFrame.")
            df = self.timeseries_data

        alias = {
            'Q_H_kWh': ['Q_H_kWh', 'Q_H', 'Q_h', 'Heating_needs'],
            'T_op': ['T_op', 'T_int', 'theta_int'],
            'T_ext': ['T_ext', 'theta_ext'],
        }

        # --- Resolve column names ONCE (instead of per-row alias lookup) ---
        def _resolve(cols, names):
            for n in names:
                if n in cols:
                    return n
            return None

        n = len(df)
        cols = df.columns
        qh_col   = _resolve(cols, alias['Q_H_kWh'])
        top_col  = _resolve(cols, alias['T_op'])
        text_col = _resolve(cols, alias['T_ext'])

        theta_int_def = float(self.input_data.get('theta_int_default', 20))
        theta_ext_def = float(self.input_data.get('theta_ext_default', 5))

        # --- Materialize numpy arrays for the hot loop ---
        if qh_col is not None:
            qh_arr = pd.to_numeric(df[qh_col], errors='coerce').to_numpy(dtype=float)
            qh_arr = np.where(np.isnan(qh_arr), 0.0, qh_arr)
        else:
            qh_arr = np.zeros(n, dtype=float)

        top_arr  = (df[top_col].to_numpy(dtype=float)  if top_col  is not None else np.full(n, theta_int_def, dtype=float))
        text_arr = (df[text_col].to_numpy(dtype=float) if text_col is not None else np.full(n, theta_ext_def, dtype=float))

        idx_arr = df.index

        # --- Precompute EN 15316-2 emission timeseries once (massive speedup) ---
        self._precompute_emission_timeseries(df)

        results = []
        for pos in range(n):
            # Tell _calculate_emission_step which row we are on, so it can look
            # up the precomputed cache instead of running EN 15316-2 again.
            self._current_emission_pos = pos

            qh    = float(qh_arr[pos])
            θint  = float(top_arr[pos])
            θext  = float(text_arr[pos])
            idx   = idx_arr[pos]

            # Filter by Q_H policy
            if qh <= 0:
                if not self.calc_when_QH_positive_only:
                    res = self.compute_step(0.0, θint, θext)
                    res_row = {'timestamp': idx}; res_row.update(res); results.append(res_row)
                    continue

                # calc_when_QH_positive_only == True
                if self.off_compute_mode == 'idle':
                    res_row = {'timestamp': idx}
                    res_row.update(self._idle_row(0.0, θint, θext))
                    results.append(res_row)
                    continue
                elif self.off_compute_mode == 'temps':
                    res_row = {'timestamp': idx}
                    res = self._temps_row_when_off(θint, θext)
                    res_row.update({'Q_h(kWh)': 0.0, 'T_op(°C)': θint, 'T_ext(°C)': θext})
                    res_row.update(res)
                    results.append(res_row)
                    continue
                elif self.off_compute_mode == 'full':
                    res = self.compute_step(0.0, θint, θext)
                    res_row = {'timestamp': idx}; res_row.update(res); results.append(res_row)
                continue

            # Normal step
            res = self.compute_step(qh, θint, θext)
            res_row = {'timestamp': idx}
            res_row.update(res)
            results.append(res_row)

        out_df = pd.DataFrame(results).set_index('timestamp')
        return out_df
