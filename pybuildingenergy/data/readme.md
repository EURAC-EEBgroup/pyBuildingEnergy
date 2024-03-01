# Building archetype 

To define the archetypes, reference has been made to the available dataset from Tabula and Episcope (https://webtool.building-typology.eu/#bm). A more thorough study is underway to provide more detailed building models compared to those currently available, also taking into account the different construction typologies within the same country.
The physical characteristics of the building components were retrieved from the reference technical sheets provided for each building archetype according to its different use and construction period (you can download the files directly from Tabula https://webtool.building-typology.eu/#bd). The thermal capacity of the building element was calculated using the  follwoing formula:

    density[kg/m3] * Specific heat capacity[J/kgK] * tichkness[m]

- where the materials data are collecting from the following database: https://help.iesve.com/ve2021/table_6_thermal_conductivity__specific_heat_capacity_and_density.html, 
- the thickness of each single layer of the building element has been assumeed in line with what is described in the building's technical sheet.

## Description and properties of building elements
-----
The calculation of the heat capacity of each opaque element is based on the following formula:

transmittance: reference Tabula/Episcope datasheet

### ASSUMPTIONS

**FACADE ELEMENTS**
Since there is no information on the distribution of the glazed components for each orientation of the building, it was assumed to have a distribution of the same in the following way

**30% South 30% East 30% West 10% North**

Furthermore, even if the area of the building is provided, and a rectangular shape, a side width equal to:  
- Italian achetype: 10 meters for the side north/south.
- ....

-------
<br />

**AREA ROOF**
type of roof: gable roof with eaves overhang.
Area roof calcuated with slope of 28°C from the perimeter + 0.5m of eaves overhang
formula = ((length_small*cos(28)+0.5*cos(28))*length_roof)*2 with cos(28°) = 0.88

-------
<br />

**UNIT**:

- Wall thickness : meter
- Thermal transmittance: W/m2K
- Thermal_resistance: m2K/W
- Heat capacity of component: J/kgK

-------
<br />

**DEFAULT VALUES for FACADE ELEMENTS**

In the following table, the value used for each single elements of the facade (Table 25 of ISO 52016)

Parameter                                    | WALL       |ROOF     |WINDOW   | FLOOR-Slab on ground |
---------------------------------------------|------------|---------|---------|----------------------|
convective_heat_transfer_coefficient_internal|      2.50  |    5.00 |     2.50|      0.70            |
convective_heat_transfer_coefficient_external|      5.13  |    5.13 |     5.13|      5.13            |
radiative_heat_transfer_coefficient_internal |      20.0  |    5.13 |     20.0|      20.0            |
radiative_heat_transfer_coefficient_external |      4.41  |    4.14 |     4.14|      4.14            |
solar_factor*                                |      0.50  |    1.00 |     0.50|      0.00            |
solar absorption of elements                 |      0.60  |    0.60 |     1.00|      0.00            |


*Solar factor for vertical element is 0.5, horizontal 1.0, 0.0 for floor slab on ground.

--------
<br />

The Distribution of the mass for opaque elements (vertical - walls and horizontal - floor/roof) as described in Table B.13 of the standard is the following:
<br />

Class                                           |Specification of the class                                |
------------------------------------------------|-------------------------------------------               |
Class I (mass concentrated at internal side)    | Construction with external thermal insulation (main mass component near inside surface) , or equivalent
Class E (mass concentrated at external side)    | Construction with internal thermal insulation (main mass component near outside surface) , or equivalent
Class IE (mass divided over internal and external side) | Construction with thermal insulation in between two main mass components, or equivalent
Class D (mass equally distributed)              | Uninsulated construction e.g. solid or hollow bricks, heavy or lightweight concrete, or lightweight construction with negligible mass (e.g. steel sandwich panel), or equivalent"

--------
<br />

Class        |Heat Capacity J/kgK | Specification of the class|
-------------|--------------------|---------------------------|
Very Light   | 50000              |Construction containing no mass components, other than e.g. plastic board and/or wood siding, or equivalent  |
Light        | 75000              |Construction containing no mass components other than 5 to 10 cm lightweight brick or concrete, or equivalent  |
Medium       | 110000             |Construction containing no mass components other than 10 to 20 cm lightweight brick or concrete, or less than 7 cm solid brick or heavy weight concrete, or equivalent |
Hevy         | 175000             | Construction containing 7 to 12 cm solid brick or heavy weight concrete, or equivalent |
Very heavy   | 250000             | Construction containing more than 12 cm solid brick or heavy weight concrete, or equivalent


--------
<br />

**INTERNAL GAINS**

Zone                |       Equipment           |       Power       | Unit        |
--------------------|---------------------------|-------------------|------------ |
Living              |            TV             |       120         |     W       |
Living              |            Light          |        5          |     W/m2    |
Bedroom             |            Light          |        5          |     W/m2    |
Bedroom2            |            Light          |        5          |     W/m2    |
Bedroom3            |            Light          |        5          |     W/m2    |
Kitchen             |            Refrigerator   |        100        |     W       |
Kitchen             |            Washing        |        2000       |     W       |
Kitchen             |            Stove          |        800        |     W       |
Kitchen             |            Light          |        5          |     W/m2    |
People              |            people         |       100         |     W       |


Reference: 339487912_Optimization_of_passive_design_features_for_a_naturally_ventilated_residential_building_according_to_the_bioclimatic_architecture_concept_and_considering_the_northern_Morocco_climate

--------
<br />

**AIR CHANGE RATE**
Residential                     0.5 m3/h 

--------
<br />

**HEATING SYSTEM**
POWER:
    The power is calculated using the following formula with some approssimations
    
    For newly well-insulated buildings:
    - the standard energy requirement value can be 0.03 kW/m³, 
    
    for older buildings with high thermal losses:
    - the standard energy requirement value can be 0.12 kW/m³

**COOLING SYSTEM**
for the building archetype the cooling system is set off.
In this case the values of cooling setpoint and setback are the same










