from src.pybuildingenergy.source.utils import ISO52016
from src.pybuildingenergy.data.building_archetype import Buildings_from_dictionary
from src.pybuildingenergy.source.functions import ePlus_shape_data
from src.pybuildingenergy.source.graphs import bar_chart_single
from pyecharts.globals import ThemeType
import pandas as pd
import os
import pickle
from .input_data.src import main_directory_, get_buildings_demos

demo_buis = get_buildings_demos(main_directory_)

bt_600 = [bui for bui in demo_buis if bui['building_type'] == 'BestTest600'][0]
weather_type ='pvgis'
latitude_bui = 44.78
longitude_bui = 9.78
path_epls_file = main_directory_ + '/energyPlus_data/Case600_V22.1.0out_Athens.csv'
path_weather_file_ = main_directory_ + '/weatherdata/2020_Athens.epw'
path_chart_name = main_directory_ + "/Result/testbed600_ISO_vs_Eplus_Athens.html"


#eplusout
def test_best600(snapshot):
    '''
    Simualte besttest600 with different weather file and comparing data with the onew coming from energy plus
    Param
    ------
    name_chart: name file to save the result of simulation
    weather_type: Specify the data source for weather data. If using the PVGIS website, indicate 'pvgis'; if loading an EPW file from the path_weather_file_, indicate 'epw'.
    latitude: mandatory if weather_type is 'pvgis'
    longitude:  mandatory if weather_type is 'epw'
    path_epls_file: specifc thte path of the energy plus simulation file 
    path_weather_file_: if weather_type ='epw', specify the folder where the epw file is uploaded.
    eplus_file_name: name of the epw file 
    '''
    # SIMULATE BUILDING
    BUI = Buildings_from_dictionary(bt_600)
    BUI.__setattr__('weather_source', weather_type)
    
    if weather_type not in ["pvgis", "epw"]:
        raise ValueError("weather_type must be 'pvgis' or 'epw'")
    else:
        if weather_type == "pvgis":
            if latitude_bui is not None:
                BUI.__setattr__('latitude', latitude_bui)
            if longitude_bui is not None:
                BUI.__setattr__('longitude', longitude_bui)

        # elif weather_type == "epw":
        #     # Check if path_weather_file_ and eplus_file_name_ are provided
        #     if not path_weather_file_ or not isinstance(path_weather_file_, str):
        #         raise ValueError("Value must be a non-empty string")
        #     if not eplus_file_name or not isinstance(eplus_file_name, str):
        #         raise ValueError("Value must be a non-empty string")


    
    hourly_results, annual_results_df = ISO52016().Temperature_and_Energy_needs_calculation(BUI, weather_source=weather_type, path_weather_file=path_weather_file_) 

    # RESHAPE DATA
    # ISO 52016
    ISO52016_annual_heating = annual_results_df['Q_H_annual_per_sqm'].squeeze() / 1000
    ISO52016_annual_cooling = annual_results_df['Q_C_annual_per_sqm'].squeeze() / 1000
    ISO52016_monthly_heating_in_kWh_per_sqm = hourly_results['Q_H'].resample('ME').sum() / (1e3 * BUI.__getattribute__('a_use'))
    ISO52016_monthly_cooling_in_kWh_per_sqm = hourly_results['Q_C'].resample('ME').sum() / (1e3 * BUI.__getattribute__('a_use'))
    ISO52016_monthly_T_op = hourly_results['T_op'].resample('ME').mean()
    index = ISO52016_monthly_heating_in_kWh_per_sqm.index

    # ENERGYPLUS
    dir_energy_plus = path_epls_file
    eplus_data = ePlus_shape_data( pd.read_csv(dir_energy_plus), BUI.__getattribute__('a_use'))
    EnergyPlus_monthly_heating_in_kWh_per_sqm = eplus_data[0]
    EnergyPlus_monthly_cooling_in_kWh_per_sqm = eplus_data[1]
    ep_monthly_T_op = eplus_data[2]
    index = ISO52016_monthly_heating_in_kWh_per_sqm.index
    # df_ep_monthly = pd.DataFrame(index=index, data={'Q_H EnergyPlus' : EnergyPlus_monthly_heating_in_kWh_per_sqm, 'Q_C EnergyPlus' : EnergyPlus_monthly_cooling_in_kWh_per_sqm, 'T_op EnergyPlus' : ep_monthly_T_op})
    df_ep_monthly = pd.DataFrame(index=index, data={'Q_H EnergyPlus' : EnergyPlus_monthly_heating_in_kWh_per_sqm, 'Q_C EnergyPlus' : EnergyPlus_monthly_cooling_in_kWh_per_sqm})
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
    graph.render(path_chart_name)

    return snapshot.assert_match(df_barplot.to_json(), "data_best600_athens.json")