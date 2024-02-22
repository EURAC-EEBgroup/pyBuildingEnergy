import numpy as np
import pandas as pd
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Scatter, Line
from scipy.stats import linregress
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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




#
# df_101_heating_hdd = df_101.loc[:,['energy_consumption_m3','heating_degree_days']]
# data_101 = [list(tuple(x)) for x in df_101_heating_hdd.itertuples(index=False, name=None)]

def Scatter_with_regression(x_data:list, y_data:list, colorPoints:list, colorRegression:list, 
                            seriesName:list,name_y_axes="", graphName="regressionScatter.html"):
    '''
    scatter plot with regression
    Param
    ------
    x_data: list of data
    y_data: list of list. IE [[1,2,3],[4,5,6]]
    colorPoints: list  in which length is equal to number of lists in y_data. I.E. ['red','yellow','green']
    colorRegression: list of color for teh regression line
    seriesName: list of name for each regression
    graphName: Name of the graph
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
        regression_line = [slope * x + intercept for x in x_data]
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
    chart.height = "1000px"
    chart.width = "1200px"
    
    return chart.render(graphName)


def Heating_Degree_days(out_temp):
    HDD = []
    for temp in out_temp:
        if temp < 18:
            HDD_value = 18-temp
        else:
            HDD_value = 0

        HDD.append(HDD_value)
    return HDD


def Simple_regeression(x_data, y_data, x_data_name):
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
    y = np.array(y_data[0])  # Example dependent variable

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
    print("R-squared:", r2)
    print("Equation of the regression line: y =", slope, "* x +", intercept)
    equation = f"y ={intercept} + {x_data_name}*{slope}"
    return r2,equation 