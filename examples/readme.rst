=======
Examples
=======

This section showcases a selection of results and simulations achieved through the library, often juxtaposed with findings from simulation tools like EnergyPlus where feasible.

** Simulation using Best Test 600 **

The geometry of the bestest600 is: 

.. image:: https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/BESTEST600.png
   :width: 800
   :height: 400

Weather data
-------------

- **Best test 600**
 
in bestets600.py, you have to provide the following inputs:

   - name chart: name of the chart file
   - weather type. Type of weather source to get weather data. Possible choices are:
      -'pvgis': connection to pvgis website and getting data according to latitude and longitude values available in BUI object.
      -'epw': use epw files located in specific folder defined in path_weather_file
   - latitude: latitude of the building location. Mandatory if weather type = 'pvgis'
   - longitude: longitude of the building location. Mandatory if weather_type = 'pvgis'
   - path_weather_file: direcotry of .epw files. Mandatory if weather_type = 'epw'
   - eplus_file_name: name of the energy plus results to be used for comparison (mandatory)


.. code-block:: python

   if __name__ == "__main__":
      main(
         name_chart = 'BESTEST600_iso_vs_energyplus_Athens',
         weather_type ='epw', 
         latitude = "",
         longitude = "",
         path_weather_file_=main_directory_+"/examples/weatherdata/2020_Athens.epw",
         eplus_file_name = "Case600_V22.1.0out_Athens"
      )


The building has been simulated using different weather file (.epw). The results are available as html file:

   - BESTEST600_iso_vs_energyplus.html -> using epw file of Denver
   - BESTEST600_iso_vs_energyplus_Athens.html -> using epw file of Athens
   ...

The following graphs display some results for simulations using weather files (EPW) from Athens and Berlin.

**ATHENS**

.. image:: https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/iso52016_vs_EPlus_bt600_Athens.png
   :width: 800
   :height: 400

**BERLIN**

.. image:: https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/iso52016_vs_EPlus_bt600_Berlin.png
   :width: 800
   :height: 400



Run Example
-------------------

.. code-block:: python

    python -m examples.bestest600
