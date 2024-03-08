import sys
import os

###
sys.path.append("/".join(os.path.realpath(__file__).split("/")[0:-2]))
####

from src.utils import __ISO52016__
from data.building_archetype import Buildings_from_dictionary
from global_inputs import main_directory_
from src.functions import bar_chart_single, ePlus_shape_data, get_buildings_demos
from pyecharts.globals import ThemeType
import pandas as pd

demo_buis = get_buildings_demos()
bt_600 = [bui for bui in demo_buis if bui['building_type'] == 'BestTest600'][0]

def main(name_chart):
    # SIMULATE BUILDING
    BUI = Buildings_from_dictionary(bt_600)
    hourly_results, annual_results_df = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI) 

    # RESHAPE DATA
    # ISO 52016
    ISO52016_annual_heating = annual_results_df['Q_H_annual_per_sqm'].squeeze() / 1000
    ISO52016_annual_cooling = annual_results_df['Q_C_annual_per_sqm'].squeeze() / 1000
    ISO52016_monthly_heating_in_kWh_per_sqm = hourly_results['Q_H'].resample('ME').sum() / (1e3 * BUI.__getattribute__('a_use'))
    ISO52016_monthly_cooling_in_kWh_per_sqm = hourly_results['Q_C'].resample('ME').sum() / (1e3 * BUI.__getattribute__('a_use'))
    ISO52016_monthly_T_op = hourly_results['T_op'].resample('ME').mean()
    index = ISO52016_monthly_heating_in_kWh_per_sqm.index

    # ENERGYPLUS
    dir_energy_plus = main_directory_+"/data/energyPlus_data/eplusout.csv"
    eplus_data = ePlus_shape_data( pd.read_csv(dir_energy_plus), BUI.__getattribute__('a_use'))
    EnergyPlus_monthly_heating_in_kWh_per_sqm = eplus_data[0]
    EnergyPlus_monthly_cooling_in_kWh_per_sqm = eplus_data[1]
    ep_monthly_T_op = eplus_data[2]
    index = ISO52016_monthly_heating_in_kWh_per_sqm.index
    df_ep_monthly = pd.DataFrame(index=index, data={'Q_H EnergyPlus' : EnergyPlus_monthly_heating_in_kWh_per_sqm, 'Q_C EnergyPlus' : EnergyPlus_monthly_cooling_in_kWh_per_sqm, 'T_op EnergyPlus' : ep_monthly_T_op})
    # 

    # ISO52016 and E_plus
    df_barplot = pd.concat([ISO52016_monthly_heating_in_kWh_per_sqm, df_ep_monthly['Q_H EnergyPlus'], ISO52016_monthly_cooling_in_kWh_per_sqm, df_ep_monthly['Q_C EnergyPlus']], axis=1)
    df_barplot = df_barplot.round(2)
    
    # PLOT CHARTS with pyecharts as html file 
    graph = bar_chart_single(
        y_name = df_barplot.columns.tolist(),
        y_data_plot=df_barplot.values.T.tolist(),
        theme_type=ThemeType.SHINE
    )
    graph.render(os.getcwd()+f"/examples/{name_chart}.html")



if __name__ == "__main__":
    main(name_chart = 'BESTEST600_iso_vs_energyplus')