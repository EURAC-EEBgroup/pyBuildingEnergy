Overview
============

.. image:: /images/Logo_pyBuild.png

pyBuildingEnergy aims to provide an assessment heating and cooling performance of the building using the European Standard
ISO52016. 
The actual calcaulation methods implemented are: 

- [x] the (sensible) energy need for heating and cooling, based on hourly or monthly calculations;

- [ ] the latent energy need for (de-)humidification, based on hourly or monthly calculations;

- [x] the internal temperature, based on hourly calculations;

- [x] the sensible heating and cooling load, based on hourly calculations;

- [ ] the moisture and latent heat load for (de-)humidification, based on hourly calculations;

- [ ] the design sensible heating or cooling load and design latent heat load using an hourly calculation interval;

- [ ] the conditions of the supply air to provide the necessary humidification and dehumidification.

Domestic Hot Water - DHW
------------------------
- [x] Calculation of volume and energy need for domestic hot water according to ISO 12831-3. 
- [] Assessment of thermal load based on the type of DHW system

Citation
--------------

Please cite us if you use the package: 

https://zenodo.org/doi/10.5281/zenodo.10887919

Installation 
------------

From pip: 

.. code-block:: python

    pip install pybuildingenergy

Git Code 
------------
The code is available here: 

::

    https://github.com/EURAC-EEBgroup/pyBuildingEnergy/tree/master




License
------------
Free software: MIT License


Limitations
------------
The library is developed with the intent of demonstrating specific elements of calculation procedures in the relevant standards. It is not intended to replace the regulations but to complement them, as the latter are essential for understanding the calculation. 
This library is meant to be used for demonstration and testing purposes and is therefore provided as open source, without protection against misuse or inappropriate use.

The information and views set out in this document are those of the authors and do not necessarily reflect the official opinion of the European Union. Neither the European Union institutions and bodies nor any person acting on their behalf may be held responsible for the use that may be made of the information contained herein.

The calculation is currently aimed at single-zone buildings with ground floor. The evaluation of multi-zone buildings is under evaluation.


Acknowledgment
---------------
This work was carried out within European projects: 
Infinite - This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 958397, 
Moderate - Horizon Europe research and innovation programme under grant agreement No 101069834, 
with the aim of contributing to the development of open products useful for defining plausible scenarios for the decarbonization of the built environment

