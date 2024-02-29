================
pyBuildingEnergy
================


.. image:: https://img.shields.io/pypi/v/pybuildingenergy.svg
        :target: https://pypi.python.org/pypi/pybuildingenergy

.. image:: https://img.shields.io/travis/DanieleAntonucci20/pybuildingenergy.svg
        :target: https://travis-ci.com/DanieleAntonucci20/pybuildingenergy

.. image:: https://readthedocs.org/projects/pybuildingenergy/badge/?version=latest
        :target: https://pybuildingenergy.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Energy simulation of the building using ISO52000


Building Inputs
----------------

.. list-table:: building general geometry data
   :widths: 25 25 35 35
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
   * - a_use
     - net floor area: conditioned area of the building [m2].
     - YES
   * - slab_on_ground_area
     - Ground floor gross area [m2].
     - YES/NO if number of floors is provided
   * - number_of_floor
     - Number of building floors [-]
     - YES/NO if number of floors is provided
   * - exposed_perimeter
     - perimeter of the building [m]
     - YES/NO iIf not provided, the perimeter is calculated as if the building were rectangular with one side being 10 meters
   * - exposed_perimeter
     - perimeter of the building [m]
     - YES/NO iIf not provided, the perimeter is calculated as if the building were rectangular with one side being 10 meters
..    * - Row 2, column 1
..      - Row 2, column 2
..      - Row 2, column 3

* Free software: MIT license
* Documentation: https://pybuildingenergy.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
