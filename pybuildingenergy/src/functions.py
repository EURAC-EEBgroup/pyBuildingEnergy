__author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
__credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberagger", "Olga Somova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daniele Antonucci"

import numpy as np
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar, Scatter, Line, Gauge
from scipy.stats import linregress
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime


# ===========================================================================================
#                           FUNCTION FOR TMY
# ===========================================================================================
def get_period_from_tmy_filename(tmy_filename):
    tmp = tmy_filename.split('_')[-2:]
    start = int(tmp[0])
    end = int(tmp[1].split('.')[0])
    return [start, end]



# ===========================================================================================
def capitalize_first_letter(s):
    if not s:
        return s
    return s[0].upper() + s[1:]

def Equation_of_time(day_of_year:pd.Series) -> pd.Series:
    '''
    Equation of time according to formula 6.4.1.2 of ISO52010
    '''
    t_eq = 0.45 * (day_of_year - 359)
    mask = day_of_year < 21
    t_eq[mask] = 2.6 + 0.44 * day_of_year[mask]
    mask = (day_of_year >= 21) & (day_of_year < 136)
    t_eq[mask] = 5.2 + 9 * np.cos((day_of_year[mask] - 43) * 0.0357)
    mask = (day_of_year >= 136) & (day_of_year < 241)
    t_eq[mask] = 1.4 - 5 * np.cos((day_of_year[mask] - 135) * 0.0449)
    mask = (day_of_year >= 241) & (day_of_year < 336)
    t_eq[mask] = -6.3 - 10 * np.cos((day_of_year[mask] - 306) * 0.036)

    return t_eq


def Hour_angle_calc(solar_time:pd.Series) -> pd.Series:
    '''
    Equation of time according to formula 6.4.1.5 of ISO52010
    NOTE: South is 0, East pos; (because of shading sectors: keep between -180 and + 180)

    Param
    -------
    solar_time: defined by formula 6.4.1.4 of ISO52010
    '''
    hour_angle_deg = 180 / 12 * (12.5 - solar_time)
    hour_angle_deg[hour_angle_deg > 180] -= 360
    hour_angle_deg[hour_angle_deg < -180] += 360

    return hour_angle_deg


def Air_mass_calc(solar_altitude_angle:pd.Series)-> pd.Series:
    '''
    Calculation of air mass
    Param
    ------
    solar_altitude_angle: ....

    Return 
    ------

    '''
    solar_altitude_angle_deg = np.degrees(solar_altitude_angle)
    air_mass = 1 / (np.sin(solar_altitude_angle) + 0.15 * np.power(solar_altitude_angle_deg + 3.885, -1.253))
    mask = solar_altitude_angle_deg >= 10
    air_mass[mask] = 1 / np.sin(solar_altitude_angle[mask])
    
    return air_mass


def Get_positions(lst, value):
    positions = []
    for i, item in enumerate(lst):
        if item == value:
            positions.append(i)
    return positions

def Filter_list_by_indices(lst, indices):
    return [lst[i] for i in indices]


def Perimeter_from_area(Area, side):
    '''
    Perimeter form area assuming 10m of side
    '''
    base = Area/side
    perimeter = 2*(base + side)
    return perimeter

def Area_roof(leng_small, leng_roof):
    '''
    Area roof calculated according to: 
    formula = ((length_small*cos(28)+0.5*cos(28))*(length_roof+(0.5*2)))*2
    cos(28°) = 0.88
    '''
    Area = 2*((leng_small*0.88)+(0.5*0.88))*(leng_roof+(0.5*2))
    return Area

def Internal_gains(bui_type:str,area: float):
    '''
    Calcualtion of internal gains according to the building typology 
    Param
    --------
    bui_type: type of building. Possible choice: residential, office, commercial
    area: gross area of the building

    Return 
    --------
    int_gains: value of internal gains in W/m2

    Note
    -----
    Power value defined in the table on the top of the file 
    Only for rsedintial the data is avaialble. 
    '''
    if bui_type ==  "residential":
        # sum of power of equipments and people heat + lights
        int_gains = (120+100+2000+800+5+4*100)/area + 5*5
        
    else:
        int_gains=5
    return int_gains

def Power_heating_system(bui_volume, bui_class):
    '''
    Approssimative calculation of generator power
    p = Voume[m3] * energy needs[kW/m3]
    Param
    ------
    bui_class: could be:
        'old': No or very low insulated building
        'new': very well insulated building
        'average':medium insulated building 
    
    Return 
    ------
    heat_power: power og the generator in Watt
    '''
    if bui_class == 'old':
        p = bui_volume * 0.12
    elif bui_class == 'gold':
        p = bui_volume * 0.03
    else:
        p = bui_volume * 0.05

    return p*1000


def filter_list_by_index(lst, indices):
    return [lst[i] for i in indices]


def Check_area(elements_area:list):
    '''
    Check the areas of the opaque and transparent components for the same orientation. 
    If the opaque component is smaller than the transparent one, then the two areas are equalized.
    If the area of the floor slab on ground is higher than the area of the roof, that the two areas are equalized.
    
    The check follows the order of entry of the component as per the following reference.
    ["OP", "OP", "OP", "OP", "GR", "OP", "W", "W", "W", "W"]
    ['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR', 'NV', 'SV', 'EV', 'WV']

    where:
    OP: Opaque elements
    GR: Ground Floor
    W: Transaprent elements
    NV: Vertical North 
    SV: South Vertical 
    EV: East Vertical
    WV: West Vertical
    HOR: Horizontal for ground floor(HOR-GR), Horizontal or Slope for Roof(HOR-OP)

    Param
    ------
    elements_area: List of areas for each individual facade element.

    Returm
    check_el_area: list of corrected areas  
    '''
    check_el_area=elements_area.copy()
    for i in range(4):
        area_wall = elements_area[i]
        area_window = elements_area[i+6]
        if area_wall < area_window:
            check_el_area[i]=area_window


    area_floor = check_el_area[4]
    area_roof = check_el_area[5]
    if area_roof < area_floor: 
        check_el_area[4] = area_roof

    return check_el_area


#
# df_101_heating_hdd = df_101.loc[:,['energy_consumption_m3','heating_degree_days']]
# data_101 = [list(tuple(x)) for x in df_101_heating_hdd.itertuples(index=False, name=None)]


def Heating_Degree_days(out_temp:list, base_temperature:int) -> list:
    '''
    Getting heating degree days from daily outdoor temperature
    Param
    ------
    base_temperature: outoddor base temperature to be used as limit for calculation of Heating Degree days. 
                        e.g.  18  in Europe 
                        e.g   20  in Italy
    Return 
    -------
    HDD: Heating Degree days 
    '''

    HDD = []
    for temp in out_temp:
        if temp < base_temperature:
            HDD_value = base_temperature-temp
        else:
            HDD_value = 0

        HDD.append(HDD_value)
    return HDD


def check_dict_format(**kwargs):
    for arg_name, arg_value in kwargs.items():
        if not isinstance(arg_value, dict):
            raise TypeError(f"Argument '{arg_name}' must be a dictionary.")
    # If all inputs are in dictionary format, return True or do other operations
    return True

# ========================================================================================================
#                                   FUNCTIONS GRAPHS
# ========================================================================================================
season_type = ['heating','cooling', 'heating_cooling']
months = ['Jan','Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']

def line_and_bar(theme_type:str, x_data:list, y_bar_plot:list, y_name:list, y_title:str, 
                 y1_data_line:list, y1_name:list, frequency:str, name_chart:str):
    '''
    bar(Y-axis) and line(Y-axis-1) chart together:
    Param
    ------
    theme_type: chart theme. Possible options:
        - themeType.LIGHT
        - themeType.DARK
        - themeType.CHALK
        - themeType.ESSOS
        - themeType.INFOGRAPHIC
        - themeType.MACARONS
        - themeType.PURPLE_PASSION
        - themeType.ROMA
        - themeType.ROMANTIC
        - themeType.SHINE
        - themeType.VINTAGE
        - themeType.WALDEN
        - themeType.WESTEROS
        - themeType.WONDERLAND
    x_data: data for x-axis as str-list
    y_bar_plot: data for bar_chart. -> list of lists, e.g.[[1,2,3,4],[4,5,6,7]]
    y_name: name of the data to be visualize as bars. same length of y_bar_plot. -> list of lists, e.g [['Q_H, 'Q_HC]]
    y_title: title of the y_axes
    y1_data_line: data for line_chart. -> list of lists, e.g.[[1,2,3,4],[4,5,6,7]]
    y1_name = name of the data to be visualize as lines. same length of y_bar_plot. -> list of lists, e.g [['Q_H, 'Q_HC]]
    frequency = data frequency to be used in the graph title. e.g. daily, hourly, etc.
    name_chart: name of the html file to be saved
    
    Retrun
    -------
    html chart
    '''
    
    x_data = sorted(x_data)
    bar=(
        Bar(init_opts=opts.InitOpts(theme=theme_type))
        .add_xaxis(x_data)
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="Temperature",
                type_="value",
                min_=0,
                max_=45,
                interval=5,
                axislabel_opts=opts.LabelOpts(formatter="{value} °C"),
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{frequency} energy need"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100), opts.DataZoomOpts(type_="inside")],
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Download as Image"),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(title="View Data", lang=["Data View", "Close", "Refresh"]),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(zoom_title="Zoom In",back_title="Zoom Out"),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                )
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(
                name=y_title,
                axislabel_opts=opts.LabelOpts(formatter="{value} W")
            )
        )
    )

    for i,values in enumerate(y_bar_plot): 
        bar.add_yaxis(
            series_name = y_name[i], 
            y_axis = values,
            label_opts=opts.LabelOpts(is_show=False))

    # Y-AXIS - 1
    lineChart = (
        Line()
        .add_xaxis(x_data)  
    )
    for i, line in enumerate(y1_data_line):
        lineChart.add_yaxis(
            series_name=y1_name[i],
            yaxis_index=1,
            y_axis=line,
            label_opts=opts.LabelOpts(is_show=False),
        )


    bar.height = "600px"
    bar.width = "1400px"

    # return bar.overlap(lineChart).render(f"{os.getcwd()}/charts/{name_chart}.html")
    return bar.overlap(lineChart)
    # return line.render(f"{name_chart}.html")


def bar_chart_single(y_name:list,y_data_plot:list, theme_type:str, name_chart:str):

    c = (
        Bar(init_opts=opts.InitOpts(theme=theme_type))
        .add_xaxis(months)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Monthly consumption"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100), opts.DataZoomOpts(type_="inside")],
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Download as Image"),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(title="View Data", lang=["Data View", "Close", "Refresh"]),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(zoom_title="Zoom In",back_title="Zoom Out"),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                )
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )
    for i,values in enumerate(y_data_plot): 
        c.add_yaxis(y_name[i], values)
    c.height = "600px"
    c.width = "1200px"
    
    return c.render(f"{name_chart}.html")



def energy_gauge_chart(name_series:str, value:float, unit:str, maxLimit:float=500,
                       title_graph:str="energy need"):
    '''
    Visualize energy need as gauge chart with different levels of performance, 
    according to national energy labels
    Param
    ------
    name_series: name of the variable to be shown
    value: variable value to be visualized
    unit_ unit of measurement
    maxLimit: max limit of energy need, according to national rank. e.g 500 kWh/m2 (last category F)

    '''
    value_norm = (value*100)/maxLimit
    c = (
        Gauge()
        .add(
            series_name = name_series,
            data_pair = [(f"{value}{unit}", value_norm)],
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(
                    color=[
                         (0.1, "#024b24"), 
                         (0.2, "#00602b"), 
                         (0.3, "#1eab39"), 
                         (0.4, "#bfd804"), 
                         (0.5, "#fff500"), 
                         (0.6, "#f7ad02"), 
                         (0.7, "#e57818"), 
                         (1, "#c5271f"), 
                    ], width=30
                )
            ),
            detail_label_opts=opts.LabelOpts(formatter="{value}"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title_graph),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    return c



def Simple_regeression(x_data:list, y_data:list, x_data_name:str):
    '''
    Simple regression between two variable
    Param
    ------
    x_data = data on x-axes
    y_data = data on y-axes
    Return 
    ------
    R2 =  coefficinet of determination 
    equation = linear regression equation 
    residual = residual !!! to be included
    '''
#
    # Sample data
    x = np.array(x_data).reshape(-1, 1)  # Example independent variable
    y = np.array(y_data)  # Example dependent variable

    # Fit linear regression model
    model = LinearRegression().fit(x, y)

    # Predict y values using the fitted model
    y_pred = model.predict(x)

    # Calculate R-squared
    r2 = r2_score(y, y_pred)

    # Get the slope (coefficient) and intercept of the regression line
    slope = model.coef_[0]
    intercept = model.intercept_

    # Print R-squared and equation of the regression line
    # print("R-squared:", r2)
    # print("Equation of the regression line: y =", slope, "* x +", intercept)
    equation = f"y ={round(intercept,2)} + {x_data_name}*{round(slope,2)}"
    
    return (round(r2,2),equation)


def Scatter_with_regression(x_data:list, y_data:list, colorPoints:list, colorRegression:list, chart_title:str, 
                            seriesName:list,name_y_axes:str="", graphName:str="regressionScatter.html",
                            subtitle_eq:str = ""):
    '''
    scatter plot with regression
    Param
    ------
    x_data: list of data
    y_data: list of lists. IE [[1,2,3],[4,5,6]]
    colorPoints: list  in which length is equal to number of lists in y_data. I.E. ['red','yellow','green']
    colorRegression: list of color for the regression lines
    seriesName: list of name for each regression
    graphName: Name of the graph
    subtitle_eq: equation of regression line

    Return
    -------
    html graph
    '''
    # Create a line chart for the regression line
    line_chart = (
        Line()
        .add_xaxis(x_data)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    scatter_chart = (
        Scatter()
        .add_xaxis(xaxis_data=x_data)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=chart_title,
                subtitle=subtitle_eq,
                pos_left="left",
                pos_top="top"
            ),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Download as Image"),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(title="View Data", lang=["Data View", "Close", "Refresh"]),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(zoom_title="Zoom In",back_title="Zoom Out"),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False)
                )
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(
                type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
                name=name_y_axes
            )
        )
    )

    r_squared = []
    n=0
    for data in y_data:
        # calcuation of regression line
        # Perform linear regression using numpy
        slope, intercept, r_value, p_value, std_err = linregress(x_data, data)

        # Calculate R-squared
        r_squared.append(r_value**2)

        # Generate predicted y values based on the regression line
        
        regression_line = [slope * x_value + intercept for x_value in x_data]
        regression_line = [round(element, 2) for element in regression_line]

        line_chart.add_yaxis(
            series_name=seriesName[n], 
            y_axis=regression_line, 
            is_smooth=True,
            symbol="emptyCircle",
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=True),
            color=colorRegression[n])

        scatter_chart.add_yaxis(
            series_name=seriesName[n],
            y_axis=data,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            color=colorPoints[n]
        )
        n=n+1

    chart = scatter_chart.overlap(line_chart)
    # Set dimension of graph
    chart.height = "600px"
    chart.width = "1400px"

    return chart

