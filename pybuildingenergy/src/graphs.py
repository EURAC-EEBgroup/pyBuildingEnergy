#%%
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line, Liquid, Page, Pie
from pyecharts.commons.utils import JsCode
from pyecharts.components import Table
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType
import pandas as pd
import os
import webbrowser

# from pybuildingenergy.src.functions import capitalize_first_letter, line_and_bar, bar_chart_single, energy_gauge_chart,season_type,\
#     Scatter_with_regression, Heating_Degree_days,Simple_regeression

from src.functions import capitalize_first_letter, line_and_bar, bar_chart_single, energy_gauge_chart,season_type,\
    Scatter_with_regression, Heating_Degree_days,Simple_regeression


# ===================================================================================================================
#                                               GRAPHS
# ===================================================================================================================

class __Graphs__:
    def __init__(self, df, season:str):
        self.df = df
        self.season=season

        if season in season_type:
            self.season = season
        else:
            raise ValueError(f"Invalid choice for sesaon. Select heating or cooling")
   

    def single_variable_plot(self,folder_directory:str, name_file:str):
        '''
        Plot bar chart of monthly consumption
        Param
        ------
        name_chart: name of chart to be saved

        '''
        # filtering dataset only for heating and cooling
        df_HC = self.df.loc[:, ['Q_H', 'Q_HC', 'Q_C']]
        # Monthly resample
        dfHC_monthly = df_HC.resample('ME').sum()
        if self.season == 'heating':
            y_data = [[value/1000 for value in dfHC_monthly.loc[:,'Q_H'].tolist()]]
            y_name = "Heating consumption"
            theme_type = ThemeType.ESSOS

        elif self.season == 'cooling':
            y_data = [[value/1000 for value in dfHC_monthly.loc[:,'Q_C'].tolist()]]
            y_name = "Cooling Consumption"
            theme_type = ThemeType.WALDEN
            
        else:
            y_data = [
                [value/1000 for value in dfHC_monthly.loc[:,'Q_H'].tolist()],
                [value/1000 for value in dfHC_monthly.loc[:,'Q_C'].tolist()]
            ]
            y_name = ["Heating", "Cooling"]
            theme_type = ThemeType.ROMA
        

        Chart = bar_chart_single(y_name, y_data, theme_type)

        file_path = "{}/{}.html".format(folder_directory, name_file)
        # directory_chart = os.getcwd()+f"/charts/{chart_name}.html"
        Chart.render(file_path)
        
        return  Chart

    
    def variables_plot(self, month_selected:bool=False, _month:int=1, _frequency: str="hourly",
                       energy_var:str="heating", folder_directory:str="",name_file:str="enegry_profile"):
        '''
        Visualize data as a combination of barchart and linechart
        Example:
            barchart: for energy consuption
            linechart: for temperature
        
        Param
        ------
        month_selected = pecify whether you want to view all months or a single month. True or  False
        _month: select specific month from 1 to 12 
        _frequency: data frequency. Values: ["yearly", "monthly", "daily"]
        energy_var: energy period to be shown. A value equal to: 'heating', 'cooling', 'heating_and_cooling'
        chart_name: Name of the html file

        Return 
        -------
        line and bar chart in html
        '''
        
        # ===============================================================
        # Dataset
        df_HC = self.df.sort_index(axis=0)
        # ===============================================================
        # Filtering by month
        # Filter by month if requested
        if isinstance(month_selected, bool):
            if month_selected:
                if not 1 <= _month <= 12:
                    raise ValueError("Month must be within the range 1 to 12")
                else:
                    df_HC = df_HC[df_HC.index.month == _month]
        elif month_selected is not None:
            raise ValueError("Value must be either True or False")
        # ===============================================================
        # Filtering by frequency
        df_HC_energy = df_HC.loc[:,['Q_H', 'Q_HC', 'Q_C']]
        df_HC_temp = df_HC.loc[:,['T_op', 'T_ext']]
        frequency_mapping = {
            "yearly": 'YE',
            "monthly": 'ME',
            "daily": 'D',
            "hourly": 'h'
        }

        # Get the resampling frequency code
        freq = frequency_mapping.get(_frequency, None)

        if freq:
            # Resample energy and temperature data
            df_HC_energy_resampled = df_HC_energy.resample(freq).sum()
            df_HC_temp_resampled = df_HC_temp.resample(freq).mean()
            # Concatenate resampled data along columns
            df_HC = pd.concat([df_HC_energy_resampled, df_HC_temp_resampled], axis=1)
        else:
            raise ValueError("Invalid frequency specified")
        
        # ===============================================================
        # Filtering by energy vars
        if energy_var  == 'heating':
            df_HC_energy_resampled = [df_HC_energy_resampled['Q_H'].to_list()]
            y_name_var = ['Q_H']
        elif energy_var  == 'cooling':
            df_HC_energy_resampled = [df_HC_energy_resampled['Q_C'].to_list()]
            y_name_var = ['Q_C']
        elif energy_var == 'heating_and_cooling':
            df_HC_energy_resampled = [df_HC_energy_resampled['Q_H'].to_list(),
                                      df_HC_energy_resampled['Q_C'].to_list()]
            y_name_var = ['Q_H', 'Q_C']
        else:
            raise ValueError("Invalid energy period selected. Possible choices are 'heating','cooling','heating_and_cooling'")
        # temperature vars
        df_HC_temp_resampled = [
            df_HC_temp_resampled['T_op'].to_list(),
            df_HC_temp_resampled['T_ext'].to_list()]
        
        # GENERATE GRAPH            
        Chart = line_and_bar(
            theme_type=ThemeType.ROMA, 
            x_data=df_HC.index.strftime('%Y-%m-%d %H-%M').to_list(), 
            y_bar_plot=df_HC_energy_resampled,
            y_name=y_name_var, 
            y_title='Energy', 
            y1_data_line= df_HC_temp_resampled,
            y1_name=['T_op','T_ext'], 
            frequency = capitalize_first_letter(_frequency),
            name_chart="time_line",
        )
        # Chart.render(os.getcwd()+f"/pybuildingenergy/charts/{chart_name}.html")
        file_path = "{}/{}.html".format(folder_directory, name_file)
        # directory_chart = os.getcwd()+f"/charts/{chart_name}.html"
        Chart.render(file_path)
        return Chart

    
    def annual_charts(self, bui_area:float = 150, folder_directory:str="", name_file:str="Yearly_eNeed_gauge"):
        '''
        visualize result as gauge chart. Useful for yearly energy 
        Recommended for displaying annual energy consumption for heating or cooling according to categories defined at the national level.
        Param
        ------
        bui_area: gross building area [m2]
        name_chart: name of the html file to be saved
        '''
        # Dataset
        df_HC = self.df.sort_index(axis=0)
        df_HC_energy = df_HC.loc[:,['Q_H', 'Q_HC', 'Q_C']].resample('YE').sum()
        value = round(df_HC_energy['Q_H'].values[0]/(1000*bui_area))
        
        Chart = energy_gauge_chart('Heating', value, unit="kWh/m2", title_graph='Annual energy need for  heating')
        # Plot chart in a single html file
        # Chart.render(os.getcwd()+f"/pybuildingenergy/charts/{chart_name}.html")
        file_path = "{}/{}.html".format(folder_directory, name_file)
        # directory_chart = os.getcwd()+f"/charts/{chart_name}.html"
        Chart.render(file_path)

        return Chart

    
    def energy_signature(self, _frequency:str='daily', clean_data: bool=True, name_file:str="energy_signature", folder_directory:str=""):
        '''
        Plotting energy signature between heating consumption and HDD
        Data:
        _frequency: data frequency. Values: ["monthly", "daily"]
        clean_data: remove 0 values from dataset
        chart_name: Name of the html file

        '''
        df_HC = self.df.sort_index(axis=0)
        df_HC_energy = df_HC.loc[:,['Q_H', 'Q_HC', 'Q_C']]
        df_HC_temp = df_HC.loc[:,['T_op', 'T_ext']]
        # FREQUENCY MAPPING
        frequency_mapping = {
            "monthly": 'ME',
            "daily": 'D'
        }
        # Get the resampling frequency code
        freq = frequency_mapping.get(_frequency, None)

        if freq:
            # Resample energy and temperature data
            df_HC_energy_resampled = df_HC_energy.resample(freq).sum()
            df_HC_temp_resampled = df_HC_temp.resample(freq).mean()
            # Concatenate resampled data along columns
            df_HC = pd.concat([df_HC_energy_resampled, df_HC_temp_resampled], axis=1)
        else:
            raise ValueError("Invalid frequency specified")

        # GETTING HDD
        df_ext_temp_HDD = df_HC_temp['T_ext'].resample('D').mean().to_list()
        HDD = Heating_Degree_days(df_ext_temp_HDD, 18)
        # REMOVE 0 VALUES
        df_HDD_Q = pd.DataFrame(
            {
                'Q_H':df_HC_energy_resampled['Q_H'].to_list(),
                'HDD': HDD
            }
        )
        #  REMOVE 0 VALUES IF SELECTED
        if isinstance(clean_data, bool):
            if clean_data:
                df_HDD_Q = df_HDD_Q[df_HDD_Q['Q_H'] != 0]
            else:
                pass  # No cleaning needed
        else:
            raise ValueError("Value must be either True or False")
        # Equation of regeression 
        regress = Simple_regeression(df_HDD_Q['HDD'].to_list(), df_HDD_Q['Q_H'].to_list(), 'HDD')

        # PLOT       
        Chart = Scatter_with_regression(
            chart_title = "Heating energy need vs HDD",
            x_data = df_HDD_Q['HDD'].to_list(),
            y_data = [df_HDD_Q['Q_H'].to_list()],
            colorPoints=['red'],
            colorRegression=['green'],
            seriesName = ['Q_H'],
            
            subtitle_eq=f"{regress[1]}, r2:{regress[0]}"
        )
        
        # Chart.render(os.getcwd()+f"/pybuildingenergy/charts/{chart_name}.html")
        file_path = "{}/{}.html".format(folder_directory, name_file)
        # directory_chart = os.getcwd()+f"/charts/{chart_name}.html"
        Chart.render(file_path)

        return Chart

    #
    def bui_analysis_page(self, folder_directory:str, name_file:str):
        '''
        Create a simple report with building performance graphs:
        Param
        ------
        folder_directory : directory of the folder where to save the report
        name_file: name of the report
        
        Return 
        -------
        page: html file 
        '''

        page = Page(layout=Page.SimplePageLayout)
        page.add(
            self.variables_plot(energy_var="heating", folder_directory=folder_directory, name_file="line_chart"),
            self.variables_plot(_frequency = 'monthly', folder_directory=folder_directory, name_file="monthly_heating_need"),
            self.variables_plot(_frequency = 'monthly', folder_directory=folder_directory, energy_var = "heating_and_cooling", name_file="monthly_heating_cooling_need"),
            self.energy_signature(folder_directory=folder_directory),
            self.annual_charts(folder_directory=folder_directory)
        )

        file_path = "{}/{}.html".format(folder_directory, name_file)
        page.render(file_path)
        
        print("Report created!")
        
