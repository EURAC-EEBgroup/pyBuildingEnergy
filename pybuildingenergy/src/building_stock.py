__author__ = "Daniele Antonucci, Ulrich Filippi Oberagger, Olga Somova"
__credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberagger", "Olga Somova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daniele Antonucci"


'''
Definition of reference tecnology and building archetypo for each nation.
Reference: 
- EIPISCOPE/TABULA
- Building stock Observatory 

# Acutal limitation
Vedere test area edificio e area del solaio controterra differenti 
# Italy:
- tasso di ventilazione fissato a 0.3  h-1
- considerato solo edifici tipo zona climatica media (E)
'''
from data import building_archetype
import numpy as np
# from src.utils import __ISO52010__, __ISO52016__, bui_item
# LIST OF BUILDING ARCHETYPE




# Filtering building archetype
class Building_archetype:

    bui_archetype = ['single_fammily_house', 'multifamily_house', 'terraced_house', 'apartment_block']
    periods = ['before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today']
    

    def __init__(self, archetype:str, period:str, latitude:float, longitude:float):
        

        if archetype in Building_archetype.bui_archetype:
                self.archetype = archetype
        else:
            raise ValueError(f"Invalid choice for archetype. Possible choices are: {', '.join(Building_archetype.bui_archetype)}")

        if period in Building_archetype.periods:
                self.built_year = period
        else:
            raise ValueError(f"Invalid choice for possible periods. Possible choices are: {', '.join(Building_archetype.periods)}")
        
        if isinstance(latitude, float):
            self.latitude = latitude
        else: 
            raise ValueError("latitude should be a float")

        if isinstance(longitude, float):
            self.longitude = longitude            
        else: 
            raise ValueError("longitude should be a float")


    def get_archetype(self, **kwargs):
        '''
        Filter list of archetype according to typology and periods
        '''
        bui_archetype = building_archetype.main(self.latitude, self.longitude)
        inputs_bui_archetype = list(filter(lambda x: x.get('type') == self.archetype and x.get('year') == self.built_year, bui_archetype))
        
        ######
        # if kwargs is provided by the user the inpu 

        return inputs_bui_archetype[0]

    
# class own_building:
#     def __init__(self, latitude, longitude, area_use, perimeter, area_floor_slab_on_ground):
#         self.lat = latitude
#         self.long = longitude
#         self.area = area_use
#         self.perimeter = perimeter
#         self.area_floor_slab_on_ground = area_floor_slab_on_ground


    
#     # QUALITY CHECK
#     def check_area_perimeter(self):
#         '''
#         Cross check between perimeter and area. 
#         If perimeter is larger than area send a warning.
#         Considering building with a surface higher than 20m2
#         '''
#         if self.area >= 20:
#             if self.perimeter > self.area:
#                 response = input("Possible error. Check perimeter and Area. Are the defined value correct?")
#                 print("")
        
#                 # Convert the user input to a boolean value
#                 try:
#                     response_bool = bool(int(response))
#                 except ValueError:
#                     response_bool = response.lower() == "true"

#                 # Check if the response is True or False
#                 if response_bool:
#                     print("You entered True.")
#                 else:
#                     input_Area = input("select the new value of area in m2")
#                     input_Perimeter = input("select the new value of area in m2")
#                     self.perimeter = input_Perimeter
#                     self.area = input_Area

#             print(self.area, self.perimeter)
                    

# class Italy:

#     bui_type = ['single_fammily_house', 'multifamily_house', 'terraced_house', 'apartment_block']
#     periods = ['before 1900', '1901-1920','1921-1945','1945-1960','1961-1875','1976-1990','1991-2005','2006-today']

#     def __init__(self, typology, year):
#         # Building typology
#         if typology in Italy.bui_type:
#             self.typology = typology
#         else:
#             raise ValueError(f"Invalid building typology. Possible choices are: {', '.join(Italy.bui_type)}")
        
#         # Building year
#         if typology in Italy.periods:
#             self.year = year
#         else:
#             raise ValueError(f"Invalid building typology. Possible choices are: {', '.join(Italy.periods)}")

#         # 




class Envelope:
    '''
    Definition of components performance of building elements according to the 
    possibile archetype of building
    '''
    def __init__(self, building_archetype, roof_area):
        self.building_archetype = building_archetype
        self.roof = roof_area,
        