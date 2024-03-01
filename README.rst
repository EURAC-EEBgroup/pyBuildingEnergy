================
pyBuildingEnergy
================


.. image:: https://github.com/DanieleAntonucci20/pyBuildingEnergy/blob/master/pybuildingenergy/assets/Logo_pyBuild.png
   :width: 800
   :height: 300

|
.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - license
      - |license|
    * - downloads
      - |downloads|
    * - tests
      - | |appveyor| |codecov|
    * - package
      - | |version| |wheel|
        | |supported-ver|
        | |package-health|


.. |package-health| image:: https://snyk.io/advisor/python/pythermalcomfort/badge.svg
    :target: https://snyk.io/advisor/python/pythermalcomfort
    :alt: pythermalcomfort

.. |license| image:: https://img.shields.io/pypi/l/pythermalcomfort?color=brightgreen
    :target: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/blob/master/LICENSE
    :alt: pythermalcomfort license

.. |docs| image:: https://readthedocs.org/projects/pythermalcomfort/badge/?style=flat
    :target: https://readthedocs.org/projects/pythermalcomfort
    :alt: Documentation Status

.. |downloads| image:: https://img.shields.io/pypi/dm/pythermalcomfort?color=brightgreen
    :alt: PyPI - Downloads

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/CenterForTheBuiltEnvironment/pythermalcomfort?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/CenterForTheBuiltEnvironment/pythermalcomfort

.. |codecov| image:: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort

.. |version| image:: https://img.shields.io/pypi/v/pythermalcomfort.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pythermalcomfort

.. |wheel| image:: https://img.shields.io/pypi/wheel/pythermalcomfort.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-ver| image:: https://img.shields.io/pypi/pyversions/pythermalcomfort.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pythermalcomfort.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pythermalcomfort

.. end-badges

|

Features
--------

The new EPBD recast provides an update on building performance assessment through a methodology that must take into account various aspects such as the thermal characteristics of the building, the use of energy from renewable sources, building automation and control systems, ventilation, cooling, energy recovery, etc.

The methodology should represent the actual operating conditions, allow for the use of measured energy for accuracy and comparability purposes, and be based on hourly or sub-hourly intervals that take into account the variable conditions significantly impacting the operation and performance of the system, as well as internal conditions.
The energy performance of a building shall be expressed by a numeric indicator of primary energy use per unit of reference floor area per year, in kWh/(m2.y) for the purpose of both energy performance certification and compliance with minimum energy performance requirements. Numeric indicators of final energy use per unit of reference floor area per year, in kWh/(m2.y) and of energy needs according to ISO 52000 in kWh/(m².y) shall be used. The methodology applied for the determination of the energy performance of a building shall be transparent and open to innovation and reflect best practices, in particular from additional indicators.
Member States shall describe their national calculation methodology based on Annex A of the key European standards on energy performance of buildings, namely EN ISO 52000-1, EN ISO 52003-1, EN ISO 52010-1,EN ISO 52016-1, EN ISO 52018-1,EN 16798-1, EN 52120-1 and EN 17423 or superseding documents. This provision shall not constitute a legal codification of those standards.

**pyBuildingEnergy** aims to provide an assessment of building performance both in terms of energy and comfort. In this initial release, it is possible to assess the energy performance of the building using ISO 52106-1:2017. Additional modules will be added for a more comprehensive evaluation of performance, assessing ventilation, renewable energies, systems, etc.
The actual calculation methods for the assessment of building performance are the following:

- [x] the (sensible) energy need for heating and cooling, based on hourly or monthly calculations;

- [ ] the latent energy need for (de-)humidification, based on hourly or monthly calculations;

- [x] the internal temperature, based on hourly calculations;

- [x] the sensible heating and cooling load, based on hourly calculations;

- [ ] the moisture and latent heat load for (de-)humidification, based on hourly calculations;

- [ ] the design sensible heating or cooling load and design latent heat load using an hourly calculation interval;

- [ ] the conditions of the supply air to provide the necessary humidification and dehumidification.

The calculation methods can be used for residential or non-residential buildings, or a part of it, referred to as "the building" or the "assessed object".
ISO 52016-1:2017 also contains specifications for the assessment of thermal zones in the building or in the part of a building. The calculations are performed per thermal zone. In the calculations, the thermal zones can be assumed to be thermally coupled or not.
ISO 52016-1:2017 is applicable to buildings at the design stage, to new buildings after construction and to existing buildings in the use phase

Limitations
--------
The calculation is currently aimed at single-zone buildings with ground floors. The evaluation of multi-zone buildings is under review.

Getting Started
----------------
The following command will install the latest pyBuildinEnergy library

::

    pip install pybuildingenergy

  


Building Inputs
----------------

.. list-table:: Building geometry data - general
   :widths: 25 25 50 
   :header-rows: 1

   * - Parameter
     - Description
     - Mandatory
   * - Latitude
     - latitude of the building in [decimal].     
     - YES
   * - Longitude
     - longitude of the building location [decimal].
     - YES
   * - coldest_month
     - Define the coldest month of the building location. Value from 1 (January) to 12 (December.
     - YES. Default: 1
   * - a_use
     - gross floor area of the building [m2].
     - YES
   * - slab_on_ground_area
     - Ground floor gross area [m2].
     - If not provided, the slab on ground are is calculated as useful area / number of floors
   * - number_of_floor
     - Number of building floors [-]
     - YES/NO if number of floors is provided
   * - exposed_perimeter
     - perimeter of the building [m]
     - YES/NO iIf not provided, the perimeter is calculated as if the building were rectangular with one side being 10 meters
   * - height
     - external height of the building [m]
     - YES
   * - wall_thickness
     - average thickness of building walls [m]
     - YES
   * - volume
     - gross volume of the building [m3]
     - If not provided the volume is calcuated as the slab on ground area * building height




Documentation
--------------


Example
-------

Here some `Examples`_ on pybuildingenergy application.
For more information
.....
  

Contributing and Support
-------------------------

**Bug reports/Questions**
If you encounter a bug, kindly create a GitLab issue detailing the bug. 
Please provide steps to reproduce the issue and ideally, include a snippet of code that triggers the bug. 
If the bug results in an error, include the traceback. If it leads to unexpected behavior, specify the expected behavior.

**Code contributions**
We welcome and deeply appreciate contributions! Every contribution, no matter how small, makes a difference. Click here_ to find out more about contributing to the project.


License
--------
* Free software: MIT license
* Documentation: https://pybuildingenergy.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage