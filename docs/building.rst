Building 
--------

The building inputs should be a dictionary having the following mandatory keys:

.. list-table:: Building geometry data * general
   :widths: 20 20 40 10 10 
   :header-rows: 1

   * - Parameter
     - Key
     - Description
     - Unit
     - Mandatory
   * - **Latitude**
     - latitude
     - latitude of the building in [decimal] 
     - [-]
     - YES
   * - **Longitude**
     - longitude
     - longitude of the building location [decimal]
     - [-].
     - YES
   * - **Coldest month**
     - coldest_month
     - Define the coldest month of the building location. Value from 1 (January) to 12 (December)
     - [-].
     - YES. Default: 1
   * - **Gross building area**
     - a_use
     - gross floor area of the building
     - [m2]
     - YES
   * - **Slab on ground area** 
     - slab_on_ground_area
     - Ground floor gross area
     - [m2]
     - If not provided, the slab on ground are is calculated as useful area / number of floors
   * - **Number of floors**
     - number_of_floor
     - Number of building floors 
     - [-]
     - YES/NO if number of floors is provided
   * - **Building perimeter**
     - exposed_perimeter
     - perimeter of the building
     - [m]
     - YES/NO iIf not provided, the perimeter is calculated as if the building were rectangular with one side being 10 meters
   * - **Building height**
     - height
     - external height of the building
     - [m]
     - YES
   * - **Average thickness of wall**
     - wall_thickness
     - average thickness of building walls 
     - [m]
     - YES
   * - **Surface of envelope**
     - surface_envelope
     - gross volume of the building 
     - [m3]
     - If not provided the volume is calcuated as the slab on ground area * building height
   * - **Volume**
     - volume
     - gross volume of the building 
     - [m3]
     - If not provided the volume is calcuated as the slab on ground area * building height
   * - **Annual mean internal temperature**
     - annual_mean_internal_temperature
     - the annual mean internal temperature is the average between Heating and Cooling setpoints
     - [°C]
     - NO: if not provided, it is calculated.
   * - **Annual mean external temperature**
     - annual_mean_external_temperature
     - Annual mean axternal temperature of the building location
     - [°C]
     - NO: if not provided, it is calculated.
   * - **Heating system**
     - heating_mode
     - True if heating system is installed, False if not.
     - [True or False]
     - YES
   * - **Cooling system**
     - cooling-mode
     - True if heating system is installed, False if not.
     - [True or False]
     - YES 
   * - **Heating setpoint**
     - heating_setpoint
     - Temperature set-point of the heating system
     - [°C]
     - YES. If `heating_mode` is True
   * - **Cooling setpoint**
     - cooling_setpoint
     - Temperature set-point of the cooling system
     - [°C]
     - YES. If `cooling_mode` is True
   * - **Heating setback**
     - heating_setback
     - Temperature set-back of the heating system
     - [°C]
     - YES. If `heating_mode` is True
   * - **Cooling setback**
     - cooling_setback
     - Temperature set-back of the cooling system
     - [°C]
     - YES. If `cooling_mode` is True
   * - **Max power of heating generator**
     - power_heating_max
     - max power of heating generator
     - [W]
     - YES. If `heating_mode` is True
   * - **Max power of cooling generator**
     - power_cooling_max
     - max power of cooling generator
     - [W]
     - YES. If `cooling_mode` is True
   * - **Air change rate **
     - air_change_rate_base_value
     - value of air chnage rate
     - [m3/h*m2]
     - Yes
   * - **Air change rate extra **
     - air_change_rate_extra
     - extra value of air change rate, in specific period according to the occupancy profile
     - [m3/h*m2]
     - Yes
   * - **Internal Gains**
     - internal_gains_base_value
     - power of internal gains 
     - [W/m2] 
     - YES
   * - **Extra Internal Gains**
     - internal_gains_base_value
     - extra value of internal gains, in specific period according to the occupancy profile
     - [W/m2] 
     - YES
   * - **Thermal bridges**
     - thermal_bridge_heat
     - Overall heat transfer coefficient for thermal bridges (without groud floor)
     - [W/K] 
     - YES
   * - **Thermal resistance of floor**
     - thermal_resistance_floor
     - Average thermal resistance of internal floors
     - [m2K/W] 
     - YES
   * - **Facade elements type**
     - typology_elements
     - List of all facade elements (Walls, Roof, Ground Floor, Windows).For:
        * Wall, Roof use: "OP" (Opaque elements)
        * Ground Floor: use "GF" (Ground Floor)
        * Windows: use "W" (Windows)
     - [-] 
     - YES
   * - **Orienation of facade elements**
     - orientation_elements
     - For each elements of the facade provide the orientation, according to the following abbreviations:
        * NV: North Vertical
        * SV: South Vertical
        * EV: East Vertical
        * WV: West Vertical
        * HOR: Horizontal/Slope (for roof and ground floor)
     - [-] 
     - YES
   * - **Solar absorption coefficients**
     - solar_abs_elements
     - Solar absorption coefficient of external (Opaque) facade elements (e.g. walls)
     - [-] 
     - YES
   * - **Area of facade elements**
     - area_elements
     - Area of each facade element (e.g. Wall, Window, etc.)
     - [m2] 
     - YES
   * - **Transmittance - U**
     - transmittance_U_elements
     - Transmiattance of each facade element.
     - [W/m2K] 
     - YES
   * - **Resistance - U**
     - thermal_resistance_R_elements
     - Theraml Resistance of each facade element. 
     - [W/m2K] 
     - YES
   * - **Thermal capacity - k**
     - thermal_resistance_R_elements
     - Addition of the heat capacity of each layer (i.e. calculated by multiplying the density times its thickness times the SHC of the material)
     - [J/m2K] 
     - YES
   * - **g-value**
     - g_factor_windows
     - solar energy transmittance of windows
     - [-] 
     - YES
   * - **Heat radiative transfer coefficient - internal**
     - heat_convective_elements_internal
     - convective heat transfer coefficient internal surface for each element
     - [W/m2K] 
     - YES
   * - **Heat convective transfer coefficient - external**
     - heat_convective_elements_external
     - convective heat transfer coefficient external surface for each element
     - [W/m2K] 
     - YES
   * - **Heat radiative transfer coefficient - internal**
     - heat_radiative_elements_internal
     - radiative heat transfer coefficient internal surface for each element
     - [W/m2K] 
     - YES
   * - **Heat radiative transfer coefficient - external**
     - heat_radiative_elements_external
     - radiative heat transfer coefficient external surface for each element
     - [W/m2K] 
     - YES
   * - **View factor**
     - sky_factor_elements
     - View factor between building element and the sky
     - [-] 
     - YES
   * - **Occupancy profile workdays - internal_gains rate**
     - comf_level_we
     - Occupancy profile for workdays to evalaute the utilization of extra internal gains
     - [-] 
     - YES
   * - **Occupancy profile weekends - internal_gains rate**
     - comf_level_we
     - Occupancy profile for weekdays to evalaute the utilization of extra internal gains
     - [-] 
     - YES
   * - **Occupancy profile workdays - airflow rate**
     - comf_level_we
     - Occupancy profile for workdays to evalaute the utilization of extra air change rate
     - [-] 
     - YES
   * - **Occupancy profile weekend - airflow rate**
     - comf_level_we
     - Occupancy profile for weekend to evalaute the utilization of extra air change rate
     - [-] 
     - YES
   * - **Class of buidling construction**
     - construction_class
     - Distribution of the mass for opaque elements (vertical - walls and horizontal - floor/roof) as described in Table B.13 of ISO52016. Possible choices: class_i, class_e, class_ie, class_d
     - [-] 
     - YES
   * - **Weather source**
     - weather_source
     - In English, it would be: "Select which type of source to use for weather data. Choose 'pvgis' for connecting to the `pvgis <https://re.jrc.ec.europa.eu/pvg_tools/en/>` or 'epw' file if using an epw file, to be download from `here <https://www.ladybug.tools/epwmap/>`
     - [-] 
     - YES
   
More information about coefficients are available `here <https://github.com/EURAC-EEBgroup/pyBuildingEnergy/tree/master/src/pybuildingenergy/data>`
