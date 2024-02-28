'''
Example of energy need for heating and cooling from building archetype
'''

from pybuildingenergy.src.utils import __ISO52010__, __ISO52016__, bui_item
from pybuildingenergy.src.building_stock import Building_archetype

# GET BUILDING ARCHETYPE
inizialize = Building_archetype('single_fammily_house','before 1900',44.66345144066082, 10.323822015417987)
inputs_archetype = inizialize.get_archetype()

# SIMULATE ARCHETYPE
hourly_sim = __ISO52016__(inputs_archetype).Temperature_and_Energy_needs_calculation() 
print(hourly_sim)

