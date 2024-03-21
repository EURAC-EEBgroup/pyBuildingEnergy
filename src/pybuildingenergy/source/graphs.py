#%%
from pyecharts.charts import Page
from pyecharts.globals import ThemeType
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar, Scatter, Line, Gauge
from scipy.stats import linregress
from pybuildingenergy.source.functions import (
    capitalize_first_letter,
    Heating_Degree_days,
    Simple_regeression,
)
import json

# ========================================================================================================
#                                   FUNCTIONS GRAPHS
# ========================================================================================================
season_type = ["heating", "cooling", "heating_cooling"]
months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def line_and_bar(
    theme_type: str,
    x_data: list,
    y_bar_plot: list,
    y_name: list,
    y_title: str,
    y1_data_line: list,
    y1_name: list,
    frequency: str,
):
    """
    bar(Y-axis) and line(Y-axis-1) chart together:

    :param **theme_type**: chart theme. Possible options:
            * themeType.LIGHT
            * themeType.DARK
            * themeType.CHALK
            * themeType.ESSOS
            * themeType.INFOGRAPHIC
            * themeType.MACARONS
            * themeType.PURPLE_PASSION
            * themeType.ROMA
            * themeType.ROMANTIC
            * themeType.SHINE
            * themeType.VINTAGE
            * themeType.WALDEN
            * themeType.WESTEROS
            * themeType.WONDERLAND
    :param **x_data**: data for x-axis as str-list
    :param **y_bar_plot**: data for bar_chart. -> list of lists, e.g.[[1,2,3,4],[4,5,6,7]]
    :param **y_name**: name of the data to be visualize as bars. same length of y_bar_plot. -> list of lists, e.g [['Q_H, 'Q_HC]]
    :param **y_title**: title of the y_axes
    :param **y1_data_line**: data for line_chart. -> list of lists, e.g.[[1,2,3,4],[4,5,6,7]]
    :param **y1_name**: name of the data to be visualize as lines. same length of y_bar_plot. -> list of lists, e.g [['Q_H, 'Q_HC]]
    :param **frequency**: data frequency to be used in the graph title. e.g. daily, hourly, etc.

    :return: chart in html format
    """

    x_data = sorted(x_data)
    bar = (
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
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside"),
            ],
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        title="Download as Image"
                    ),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(
                        title="View Data", lang=["Data View", "Close", "Refresh"]
                    ),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                        zoom_title="Zoom In", back_title="Zoom Out"
                    ),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                )
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            yaxis_opts=opts.AxisOpts(
                name=y_title, axislabel_opts=opts.LabelOpts(formatter="{value} W")
            ),
        )
    )

    for i, values in enumerate(y_bar_plot):
        bar.add_yaxis(
            series_name=y_name[i],
            y_axis=values,
            label_opts=opts.LabelOpts(is_show=False),
        )

    # Y-AXIS - 1
    lineChart = Line().add_xaxis(x_data)
    for i, line in enumerate(y1_data_line):
        lineChart.add_yaxis(
            series_name=y1_name[i],
            yaxis_index=1,
            y_axis=line,
            label_opts=opts.LabelOpts(is_show=False),
        )

    bar.height = "600px"
    bar.width = "1400px"

    return bar.overlap(lineChart)


def bar_chart_single(y_name: list, y_data_plot: list, theme_type: str):
    """
    Simple bar chart

    :param y_name: name of the parameters to be plot
    :param y_data_plot: list of data for each single parameter
    :param theme_type: chart theme, possible values are desribed in the parmeters of ``line_and_bar`` function

    :return: simple html bar chart
    
    """
    c = (
        Bar(init_opts=opts.InitOpts(theme=theme_type))
        .add_xaxis(months)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Monthly consumption"),
            datazoom_opts=[
                opts.DataZoomOpts(range_start=0, range_end=100),
                opts.DataZoomOpts(type_="inside"),
            ],
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        title="Download as Image"
                    ),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(
                        title="View Data", lang=["Data View", "Close", "Refresh"]
                    ),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                        zoom_title="Zoom In", back_title="Zoom Out"
                    ),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
                )
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
    )
    for i, values in enumerate(y_data_plot):
        c.add_yaxis(y_name[i], values)
    c.height = "600px"
    c.width = "1200px"

    return c


def energy_gauge_chart(
    name_series: str,
    value: float,
    unit: str,
    maxLimit: float = 500,
    title_graph: str = "energy need",
):
    """
    Visualize energy need as gauge chart with different levels of performance,
    according to national energy labels

    :param name_series: name of the variable to be shown
    :param value: variable value to be visualized
    :param unit_: unit of measurement
    :param maxLimit: max limit of energy need, according to national rank. e.g 500 kWh/m2 (last category F)
    :param title_graph: title of the graph. Default: "energy need"

    :return: gauge chart in html format
    """
    value_norm = (value * 100) / maxLimit
    c = (
        Gauge()
        .add(
            series_name=name_series,
            data_pair=[(f"{value}{unit}", value_norm)],
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
                    ],
                    width=30,
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


def Scatter_with_regression(
    x_data: list,
    y_data: list,
    colorPoints: list,
    colorRegression: list,
    chart_title: str,
    seriesName: list,
    name_y_axes: str = "",
    graphName: str = "regressionScatter.html",
    subtitle_eq: str = "",
):
    """
    scatter plot with regression line

    :param x_data: list of data
    :param y_data: list of lists. IE [[1,2,3],[4,5,6]]
    :param colorPoints: list  in which length is equal to number of lists in y_data. I.E. ['red','yellow','green']
    :param colorRegression: list of color for the regression lines
    :param seriesName: list of name for each regression
    :param graphName: Name of the graph
    :param subtitle_eq: equation of regression line

    :return: scatter plot with regression line in html format

    """
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
                title=chart_title, subtitle=subtitle_eq, pos_left="left", pos_top="top"
            ),
            toolbox_opts=opts.ToolboxOpts(
                feature=opts.ToolBoxFeatureOpts(
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                        title="Download as Image"
                    ),
                    restore=opts.ToolBoxFeatureRestoreOpts(title="Restore"),
                    data_view=opts.ToolBoxFeatureDataViewOpts(
                        title="View Data", lang=["Data View", "Close", "Refresh"]
                    ),
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(
                        zoom_title="Zoom In", back_title="Zoom Out"
                    ),
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),
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
                name=name_y_axes,
            ),
        )
    )

    r_squared = []
    n = 0
    for data in y_data:
        # calcuation of regression line
        # Perform linear regression using numpy
        slope, intercept, r_value, p_value, std_err = linregress(x_data, data)

        # Calculate R-squared
        r_squared.append(r_value ** 2)

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
            color=colorRegression[n],
        )

        scatter_chart.add_yaxis(
            series_name=seriesName[n],
            y_axis=data,
            symbol_size=10,
            label_opts=opts.LabelOpts(is_show=False),
            color=colorPoints[n],
        )
        n = n + 1

    chart = scatter_chart.overlap(line_chart)
    # Set dimension of graph
    chart.height = "600px"
    chart.width = "1400px"

    return chart


# ===================================================================================================================
#                                               REPORT
# ===================================================================================================================


class Graphs_and_report:
    def __init__(self, df, season: str):
        self.df = df
        self.season = season

        if season in season_type:
            self.season = season
        else:
            raise ValueError(f"Invalid choice for season. Select heating or cooling")

    def single_variable_plot(self, folder_directory: str, name_file: str):
        """
        Create and save in html format the bart chart of heating, cooliing or heating and cooling consumption

        :param folder_directory: directory where the save the html file
        :param name_file: name of the chart file

        :return: html chart
        """
        # filtering dataset only for heating and cooling
        df_HC = self.df.loc[:, ["Q_H", "Q_HC", "Q_C"]]
        # Monthly resample
        dfHC_monthly = df_HC.resample("ME").sum()
        if self.season == "heating":
            y_data = [[value / 1000 for value in dfHC_monthly.loc[:, "Q_H"].tolist()]]
            y_name = "Heating consumption"
            theme_type = ThemeType.ESSOS

        elif self.season == "cooling":
            y_data = [[value / 1000 for value in dfHC_monthly.loc[:, "Q_C"].tolist()]]
            y_name = "Cooling Consumption"
            theme_type = ThemeType.WALDEN

        else:
            y_data = [
                [value / 1000 for value in dfHC_monthly.loc[:, "Q_H"].tolist()],
                [value / 1000 for value in dfHC_monthly.loc[:, "Q_C"].tolist()],
            ]
            y_name = ["Heating", "Cooling"]
            theme_type = ThemeType.ROMA

        Chart = bar_chart_single(y_name, y_data, theme_type)

        file_path = "{}/{}.html".format(folder_directory, name_file)
        Chart.render(file_path)

        return Chart

    def variables_plot(
        self,
        month_selected: bool = False,
        _month: int = 1,
        _frequency: str = "hourly",
        energy_var: str = "heating",
        folder_directory: str = "",
        name_file: str = "energy_profile",
    ):
        """
        Visualize data as a combination of barchart and linechart
        Example:
            barchart: for energy consuption
            linechart: for temperature

        :param month_selected: pecify whether you want to view all months or a single month. True or  False
        :param _month: select specific month from 1 to 12
        :param _frequency: data frequency. Values: ["yearly", "monthly", "daily", "hourly"]. Deafult: 'hourly'
        :param energy_var: energy period to be shown. A value equal to: 'heating', 'cooling', 'heating_and_cooling'. Default: 'heating'
        :param folder_directory: directory where the save the html file
        :param name_file: name of the chart file. default: energy_profile

        :return: line and bar chart in html
        """

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
        df_HC_energy = df_HC.loc[:, ["Q_H", "Q_HC", "Q_C"]]
        df_HC_temp = df_HC.loc[:, ["T_op", "T_ext"]]
        frequency_mapping = {
            "yearly": "YE",
            "monthly": "ME",
            "daily": "D",
            "hourly": "h",
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
        if energy_var == "heating":
            df_HC_energy_resampled = [df_HC_energy_resampled["Q_H"].to_list()]
            y_name_var = ["Q_H"]
        elif energy_var == "cooling":
            df_HC_energy_resampled = [df_HC_energy_resampled["Q_C"].to_list()]
            y_name_var = ["Q_C"]
        elif energy_var == "heating_and_cooling":
            df_HC_energy_resampled = [
                df_HC_energy_resampled["Q_H"].to_list(),
                df_HC_energy_resampled["Q_C"].to_list(),
            ]
            y_name_var = ["Q_H", "Q_C"]
        else:
            raise ValueError(
                "Invalid energy period selected. Possible choices are 'heating','cooling','heating_and_cooling'"
            )
        # temperature vars
        df_HC_temp_resampled = [
            df_HC_temp_resampled["T_op"].to_list(),
            df_HC_temp_resampled["T_ext"].to_list(),
        ]

        # GENERATE GRAPH
        Chart = line_and_bar(
            theme_type=ThemeType.ROMA,
            x_data=df_HC.index.strftime("%Y-%m-%d %H-%M").to_list(),
            y_bar_plot=df_HC_energy_resampled,
            y_name=y_name_var,
            y_title="Energy",
            y1_data_line=df_HC_temp_resampled,
            y1_name=["T_op", "T_ext"],
            frequency=capitalize_first_letter(_frequency),
        )
        # Chart.render(os.getcwd()+f"/pybuildingenergy/charts/{chart_name}.html")
        file_path = "{}/{}.html".format(folder_directory, name_file)
        # directory_chart = os.getcwd()+f"/charts/{chart_name}.html"
        Chart.render(file_path)
        return Chart

    def annual_charts(
        self,
        bui_area: float = 150,
        folder_directory: str = "",
        name_file: str = "Yearly_eNeed_gauge",
    ):
        """
        visualize result as gauge chart. Useful for yearly energy
        Recommended for displaying annual energy consumption for heating or cooling according to categories defined at the national level.

        :param bui_area: gross building area [m2]
        :param folder_directory: directory where the save the html file
        :param name_file: name of the chart file. default: Yearly_eNeed_gauge

        :return: annual_charts in html
        """
        # Dataset
        df_HC = self.df.sort_index(axis=0)
        df_HC_energy = df_HC.loc[:, ["Q_H", "Q_HC", "Q_C"]].resample("YE").sum()
        value = round(df_HC_energy["Q_H"].values[0] / (1000 * bui_area))

        Chart = energy_gauge_chart(
            "Heating",
            value,
            unit="kWh/m2",
            title_graph="Annual energy need for  heating",
        )
        # Plot chart in a single html file
        # Chart.render(os.getcwd()+f"/pybuildingenergy/charts/{chart_name}.html")
        file_path = "{}/{}.html".format(folder_directory, name_file)
        # directory_chart = os.getcwd()+f"/charts/{chart_name}.html"
        Chart.render(file_path)

        return Chart

    def energy_signature(
        self,
        _frequency: str = "daily",
        clean_data: bool = True,
        folder_directory: str = "",
        name_file: str = "energy_signature",
    ):
        """
        Plotting energy signature between heating consumption and HDD
        Data:
        :param _frequency: data frequency. Values: ["monthly", "daily"]
        :param clean_data: remove 0 values from dataset
        :param chart_name: Name of the html file
        :param folder_directory: directory where the save the html file
        :param name_file: name of the chart file. default: energy_signature

        :return: energy signature of the building
        """
        df_HC = self.df.sort_index(axis=0)
        df_HC_energy = df_HC.loc[:, ["Q_H", "Q_HC", "Q_C"]]
        df_HC_temp = df_HC.loc[:, ["T_op", "T_ext"]]
        # FREQUENCY MAPPING
        frequency_mapping = {"monthly": "ME", "daily": "D"}
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
        df_ext_temp_HDD = df_HC_temp["T_ext"].resample("D").mean().to_list()
        HDD = Heating_Degree_days(df_ext_temp_HDD, 18)
        # REMOVE 0 VALUES
        df_HDD_Q = pd.DataFrame(
            {"Q_H": df_HC_energy_resampled["Q_H"].to_list(), "HDD": HDD}
        )
        #  REMOVE 0 VALUES IF SELECTED
        if isinstance(clean_data, bool):
            if clean_data:
                df_HDD_Q = df_HDD_Q[df_HDD_Q["Q_H"] != 0]
            else:
                pass  # No cleaning needed
        else:
            raise ValueError("Value must be either True or False")
        # Equation of regeression
        regress = Simple_regeression(
            df_HDD_Q["HDD"].to_list(), df_HDD_Q["Q_H"].to_list(), "HDD"
        )

        # PLOT
        Chart = Scatter_with_regression(
            chart_title="Heating energy need vs HDD",
            x_data=df_HDD_Q["HDD"].to_list(),
            y_data=[df_HDD_Q["Q_H"].to_list()],
            colorPoints=["red"],
            colorRegression=["green"],
            seriesName=["Q_H"],
            subtitle_eq=f"{regress[1]}, r2:{regress[0]}",
        )

        # Chart.render(os.getcwd()+f"/pybuildingenergy/charts/{chart_name}.html")
        file_path = "{}/{}.html".format(folder_directory, name_file)
        # directory_chart = os.getcwd()+f"/charts/{chart_name}.html"
        Chart.render(file_path)

        return Chart

    #
    def bui_analysis_page(self, folder_directory: str, name_file: str):
        """
        Create a simple report with building performance graphs:

        :param folder_directory: directory where the save the html file
        :param name_file: name of the chart file

        :return: html report
        """

        page = Page(layout=Page.SimplePageLayout)
        page.add(
            self.variables_plot(
                energy_var="heating",
                folder_directory=folder_directory,
                name_file="line_chart",
            ),
            self.variables_plot(
                _frequency="monthly",
                folder_directory=folder_directory,
                name_file="monthly_heating_need",
            ),
            self.variables_plot(
                _frequency="monthly",
                folder_directory=folder_directory,
                energy_var="heating_and_cooling",
                name_file="monthly_heating_cooling_need",
            ),
            self.energy_signature(folder_directory=folder_directory),
            self.annual_charts(folder_directory=folder_directory),
        )

        file_path = "{}/{}.html".format(folder_directory, name_file)
        page.render(file_path)

        print("Report created!")
        return json.dumps({"report": "created"})
