from pybuildingenergy.source.utils import __ISO52016__
from pybuildingenergy.data.building_archetype import Buildings_from_dictionary
from pybuildingenergy.global_inputs import main_directory_
from pybuildingenergy.source.functions import bar_chart_single, ePlus_shape_data, get_buildings_demos
from pyecharts.globals import ThemeType
import pandas as pd

demo_buis = get_buildings_demos()
bt_600 = [bui for bui in demo_buis if bui['building_type'] == 'BestTest600'][0]

#eplusout
def main(name_chart:str,weather_type:str, 
         latitude_bui:float = None, longitude_bui:float = None, 
         path_weather_file_:str= None, eplus_file_name:str=None):
    
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


    
    hourly_results, annual_results_df = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI, weather_source=weather_type, path_weather_file=path_weather_file_) 

    # RESHAPE DATA
    # ISO 52016
    ISO52016_annual_heating = annual_results_df['Q_H_annual_per_sqm'].squeeze() / 1000
    ISO52016_annual_cooling = annual_results_df['Q_C_annual_per_sqm'].squeeze() / 1000
    ISO52016_monthly_heating_in_kWh_per_sqm = hourly_results['Q_H'].resample('ME').sum() / (1e3 * BUI.__getattribute__('a_use'))
    ISO52016_monthly_cooling_in_kWh_per_sqm = hourly_results['Q_C'].resample('ME').sum() / (1e3 * BUI.__getattribute__('a_use'))
    ISO52016_monthly_T_op = hourly_results['T_op'].resample('ME').mean()
    index = ISO52016_monthly_heating_in_kWh_per_sqm.index

    # ENERGYPLUS
    dir_energy_plus = main_directory_+f"/data/energyPlus_data/{eplus_file_name}.csv"
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
    graph.render(main_directory_+f"/examples/{name_chart}.html")



if __name__ == "__main__":
    main(
        name_chart = 'BESTEST600_iso_vs_energyplus_Athens',
        weather_type ='pvgis', 
        latitude_bui = 44.78,
        longitude_bui = 9.78,
        eplus_file_name = "Case600_V22.1.0out_Athens",
        path_weather_file_=main_directory_+"/examples/weatherdata/2020_Athens.epw"
    )