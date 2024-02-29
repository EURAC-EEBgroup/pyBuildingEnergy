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

Features
--------



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

  




* TODO

License
--------
* Free software: MIT license
* Documentation: https://pybuildingenergy.readthedocs.io.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage