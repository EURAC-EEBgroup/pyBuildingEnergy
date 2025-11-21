import pandas as pd
import numpy as np
from pybuildingenergy.global_inputs import TB14 as TB14_backup

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

        # --- Distribution (between emitter and manifold) ---
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
        # 1) Common emission params
        common = self.calculate_common_emission_parameters(q_h_kWh, θint)
        self.current_theta_ext = float(θext)  # stored for primary control (Type A)

        # 2) Emission block per selected type
        if self.selected_emm_cont_circuit == 0:
            em = self.calculate_type_C2(common, θint)
        elif self.selected_emm_cont_circuit == 1:
            em = self.calculate_type_C3(common, θint)
        elif self.selected_emm_cont_circuit == 2:
            em = self.calculate_type_C4(common, θint)
        elif self.selected_emm_cont_circuit == 3:
            em = self.calculate_type_C5(common, θint)
        else:
            raise ValueError("selected_emm_cont_circuit must be in {0,1,2,3}")

        # 3) Node supply temp (secondary control)
        θH_nod_out = self.calculate_circuit_node_temperature(θext, em)

        # 4) Operating conditions
        op = self.calculate_operating_conditions(em, common, θint, θH_nod_out)

        # 5) Distribution block
        dist = self.calculate_distribution(op['θH_dis_flw'], op['θH_dis_ret'], θint, common['QH_em_i_in'])

        # 6) Generation (primary) block
        gen = self.calculate_generation(
            θH_dis_flw=op['θH_dis_flw'],
            θH_dis_ret=op['θH_dis_ret'],
            V_H_dis=dist['V_H_dis'],
            QH_dis_i_in=dist['QH_dis_i_in'],
        )

        # 7) Collect outputs
        out = {
            # per-step inputs
            'Q_h(kWh)': q_h_kWh,
            'T_op(°C)': θint,
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
        }

        # merge generator outputs
        out.update({
            'max_output_g1(kWh)': gen['max_output_g1(kWh)'],
            'QH_gen_out(kWh)': gen['QH_gen_out(kWh)'],
            'θX_gen_cr_flw(°C)': gen['θX_gen_cr_flw(°C)'],
            'θX_gen_cr_ret(°C)': gen['θX_gen_cr_ret(°C)'],
            'V_H_gen(m3/h)': gen['V_H_gen(m3/h)'],
            'efficiency_gen(%)': gen['efficiency_gen(%)'],
            'EHW_gen_in(kWh)': gen['EHW_gen_in(kWh)'],
            'EHW_gen_aux(kWh)': gen['EHW_gen_aux(kWh)'],
            'QW_gen_i_ls_rbl_H(kWh)': gen['QW_gen_i_ls_rbl_H(kWh)'],
            'QW_gen_out(kWh)': gen['QW_gen_out(kWh)'],
            'QHW_gen_out(kWh)': gen['QHW_gen_out(kWh)'],
            'EH_gen_in(kWh)': gen['EH_gen_in(kWh)'],
            'EWH_gen_in(kWh)': gen['EWH_gen_in(kWh)'],
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

        def pick(series, names, default=None):
            for n in names:
                if n in series:
                    return series[n]
            return default

        results = []
        for idx, row in df.iterrows():
            qh = pick(row, alias['Q_H_kWh'])
            θint = pick(row, alias['T_op'], default=self.input_data.get('theta_int_default', 20))
            θext = pick(row, alias['T_ext'], default=self.input_data.get('theta_ext_default', 5))

            if pd.isna(qh):
                qh = 0.0

            # Filter by Q_H policy
            if float(qh) <= 0:
                if not self.calc_when_QH_positive_only:
                    # Full calculation even with zero load
                    res = self.compute_step(0.0, float(θint), float(θext))
                    res_row = {'timestamp': idx}; res_row.update(res); results.append(res_row)
                    continue

                # calc_when_QH_positive_only == True
                if self.off_compute_mode == 'idle':
                    res_row = {'timestamp': idx}
                    res_row.update(self._idle_row(0.0, float(θint), float(θext)))
                    results.append(res_row)
                    continue
                elif self.off_compute_mode == 'temps':
                    res_row = {'timestamp': idx}
                    res = self._temps_row_when_off(float(θint), float(θext))
                    res_row.update({'Q_h(kWh)': 0.0, 'T_op(°C)': float(θint), 'T_ext(°C)': float(θext)})
                    res_row.update(res)
                    results.append(res_row)
                    continue
                elif self.off_compute_mode == 'full':
                    res = self.compute_step(0.0, float(θint), float(θext))
                    res_row = {'timestamp': idx}; res_row.update(res); results.append(res_row)
                continue

            # Normal step
            res = self.compute_step(float(qh), float(θint), float(θext))
            res_row = {'timestamp': idx}
            res_row.update(res)
            results.append(res_row)

        out_df = pd.DataFrame(results).set_index('timestamp')
        return out_df
