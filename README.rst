================
pyBuildingEnergy
================


.. image:: https://github.com/EURAC-EEBgroup/pyBuildingEnergy/blob/master/src/pybuildingenergy/assets/Logo_pyBuild.png
   :width: 800
   :height: 250

Citation
----------
Please cite us if you use the library

.. image:: https://zenodo.org/badge/761715706.svg
   :target: https://zenodo.org/doi/10.5281/zenodo.10887919

Features
--------

The new EPBD recast provides an update on building performance assessment through a methodology that must take into account various aspects such as the thermal characteristics of the building, the use of energy from renewable sources, building automation and control systems, ventilation, cooling, energy recovery, etc.

The methodology should represent the actual operating conditions, allow for the use of measured energy for accuracy and comparability purposes, and be based on hourly or sub-hourly intervals that take into account the variable conditions significantly impacting the operation and performance of the system, as well as internal conditions.
The energy performance of a building shall be expressed by a numeric indicator of primary energy use per unit of reference floor area per year, in kWh/(m2.y) for the purpose of both energy performance certification and compliance with minimum energy performance requirements. Numeric indicators of final energy use per unit of reference floor area per year, in kWh/(m2.y) and of energy needs according to ISO 52000 in kWh/(m².y) shall be used. The methodology applied for the determination of the energy performance of a building shall be transparent and open to innovation and reflect best practices, in particular from additional indicators.
Member States shall describe their national calculation methodology based on Annex A of the key European standards on energy performance of buildings, namely EN ISO 52000-1, EN ISO 52003-1, EN ISO 52010-1,EN ISO 52016-1, EN ISO 52018-1,EN 16798-1, EN 52120-1 and EN 17423 or superseding documents. This provision shall not constitute a legal codification of those standards.

**pyBuildingEnergy** aims to provide an assessment of building performance both in terms of energy and comfort. In this initial release, it is possible to assess the energy performance of the building using ISO 52106-1:2018. Additional modules will be added for a more comprehensive evaluation of performance, assessing ventilation, renewable energies, systems, etc.
The actual calculation methods for the assessment of building performance are the following:

- [x] the (sensible) energy need for heating and cooling, based on hourly or monthly calculations;

- [ ] the latent energy need for (de-)humidification, based on hourly or monthly calculations;

- [x] the internal temperature, based on hourly calculations;

- [x] the sensible heating and cooling load, based on hourly calculations;

- [ ] the moisture and latent heat load for (de-)humidification, based on hourly calculations;

- [ ] the design sensible heating or cooling load and design latent heat load using an hourly calculation interval;

- [ ] the conditions of the supply air to provide the necessary humidification and dehumidification.

The calculation methods can be used for residential or non-residential buildings, or a part of it, referred to as "the building" or the "assessed object".
ISO 52016-1:2018 also contains specifications for the assessment of thermal zones in the building or in the part of a building. The calculations are performed per thermal zone. In the calculations, the thermal zones can be assumed to be thermally coupled or not.
ISO 52016-1:2018 is applicable to buildings at the design stage, to new buildings after construction and to existing buildings in the use phase


Weather Data
------------
The tool can use wather data coming from 2 main sources:

- pvgis api (https://re.jrc.ec.europa.eu/pvg_tools/en/) - PHOTOVOLTAIC GEOGRAPHICAL INFORMATION SYSTEM
- .epw file from https://www.ladybug.tools/epwmap/

More details in the example folder


Domestic Hot Water - DHW
------------------------
- [x] Calculation of volume and energy need for domestic hot water according to ISO 12831-3. 
- [] Assessment of thermal load based on the type of DHW system


Primary Energy - Heating system
--------------------------------

The ISO EN 15316 series covers the calculation method for system energy requirements and system efficiencies. This family of standards is an integral part of the EPB set and covers:

### **ISO EN 15316 Modular Structure:**

- [x] ISO EN 15316-1**: General and expression of energy performance (Modules M3-1, M3-4, M3-9, M8-1, M8-4)
- [ ] ISO EN 15316-2**: Emission systems (heating and cooling)
- [ ] ISO EN 15316-3**: Distribution systems (DHW, heating, cooling)
- [ ] ISO EN 15316-4-X**: Heat generation systems:
    - 4-1: Combustion boilers
    - 4-2: Heat pumps
    - 4-3: Solar thermal and photovoltaic systems
    - 4-4: Cogeneration systems
    - 4-5: District heating
    - 4-7: Biomass
- [ ] ISO EN 15316-5**: Storage systems

For space heating, applicable standards include ISO EN 15316-1, ISO EN 15316-2-1, ISO EN 15316-2-3 and the appropriate parts of ISO EN 15316-4 depending on the system type, including losses and control aspects.

Limitations
------------
The library is developed with the intent of demonstrating specific elements of calculation procedures in the relevant standards. It is not intended to replace the regulations but to complement them, as the latter are essential for understanding the calculation. 
This library is meant to be used for demonstration and testing purposes and is therefore provided as open source, without protection against misuse or inappropriate use.

The information and views set out in this document are those of the authors and do not necessarily reflect the official opinion of the European Union. Neither the European Union institutions and bodies nor any person acting on their behalf may be held responsible for the use that may be made of the information contained herein.

The calculation is currently aimed at single-zone buildings with ground floor. The evaluation of multi-zone buildings is under evaluation.

Getting Started
----------------
The following command will install the latest pyBuildinEnergy library

::

    pip install pybuildingenergy

::

Building - System Inputs
----------------
- for building inputs refer to `Building Inputs`: <https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_52016_input/>
- for heating system input (ISO EN 15316-1) refer to `Heating System Input`: <https://eurac-eebgroup.github.io/pybuildingenergy-docs/iso_15316_input/>

Documentation
--------------
Check our doc `here <https://eurac-eebgroup.github.io/pybuildingenergy-docs/>`

Example
-------

Here some `Examples <https://github.com/EURAC-EEBgroup/pyBuildingEnergy/tree/master/examples>` on pybuildingenergy application.
For more information
.....
  

Contributing and Support
-------------------------

**Bug reports/Questions**
If you encounter a bug, kindly create a GitLab issue detailing the bug. 
Please provide steps to reproduce the issue and ideally, include a snippet of code that triggers the bug. 
If the bug results in an error, include the traceback. If it leads to unexpected behavior, specify the expected behavior.

**Code contributions**
We welcome and deeply appreciate contributions! Every contribution, no matter how small, makes a difference. Click here to find out more about contributing to the project.


License
--------
* Free software: BSD 3-Clause License
* Documentation: https://eurac-eebgroup.github.io/pybuildingenergy-docs/

Acknowledgment
---------------
This work was carried out within European projects: 
Infinite - This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 958397, 
Moderate - Horizon Europe research and innovation programme under grant agreement No 101069834, 
with the aim of contributing to the development of open products useful for defining plausible scenarios for the decarbonization of the built environment

Reagrding the DHW Calculation: 
The work was developed using the regulations and results obtained from the spreadsheet created by the EPBCenter.

Reference
----------
- EPB Center - The Energy Performance of Buildings Directive (EPBD)  
   https://epb.center/epb-standards/the-energy-performance-of-buildings-directive-epbd/
- REHVA Journal - "The new EN ISO 52000 family of standards to assess the energy performance of buildings put in practice"  
   https://www.rehva.eu/rehva-journal/chapter/the-new-en-iso-52000-family-of-standards-to-assess-the-energy-performance-of-buildings-put-in-practice
- European Commission - Energy Performance of Buildings Directive  
   https://energy.ec.europa.eu/topics/energy-efficiency/energy-performance-buildings/energy-performance-buildings-directive_en
- Directive (EU) 2024/1275 - Official text published in the Official Journal of the EU on May 8, 2024
- EN ISO 52010-1:2018 - Energy performance of buildings - External climatic conditions - Part 1: Conversion of climatic data for energy calculations
- EN ISO 52016-1:2018 - Energy performance of buildings - Energy needs for heating and cooling, internal temperatures and sensible and latent heat loads 
- EN ISO 12831-3:2018 - Energy performance of buildings - Method for calculation of the design heat load - Part 3: Domestic hot water systems heat load and characterisation of needs, Module M8-2, M8-3
- EN ISO 15316-1:2018 - Energy performance of buildings – Method for calculation of system energy requirements and system efficiencies – Part 1: General and Energy performance expression, Module M3–1, M3–4, M3–9, M8–1
- EN ISO 16798-7 - Energy performance of buildings – Ventilation for buildings – Part 7: Calculation methods for the determination of air flow rates in buildings including infiltration (Module M5–5)
- EN ISO 16798-1 - Energy performance of buildings — Ventilation of buildings — Part 1: Indoor environmental input parameters for design and assessment of energy performance of buildings addressing indoor air quality, thermal environment, lighting and acoustics (Module M1–6)



