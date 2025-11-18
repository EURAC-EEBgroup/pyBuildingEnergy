__author__ = "Daniele Antonucci, Ulrich Filippi Oberegger, Olga Somova"
__credits__ = ["Daniele Antonucci", "Ulrich FIlippi Oberegger", "Olga Somova"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daniele Antonucci"


'''
Static inputs
'''
import os
import pandas as pd
periods = ['before 1900', '1901-1920','1921-1945','1946-1960','1961-1875','1976-1990','1991-2005','2006-today']
bui_types = ['single_family_house']
main_directory_ = os.path.dirname(__file__)
months = ['Jan','Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov','Dec']
WATER_DENSITY = 1000 #[kg/m3]
WATER_SPECIFIC_HEAT_CAPACITY = 0.00116 # kWh/kgK 

'''
Conventional air change rate between the unconditioned spaces and the external environemnt
ISO 13798:2018
'''
df_n_ue = pd.DataFrame(
    {
        'Air thigtness type': [
            "No doors no windows, no ventilation openings provided",
            "All joints between components well-sealed, no ventilation opening provided",
            "All joints well-sealed, small opneing provided for ventilation",
            "Not airtight due to some localized open joints or permanent ventialtion openings",
            "Not Airtight due to numerous opne joints, or large or numerous permanent ventilation openings" 
        ],
        'code': [1,2,3,4,5],
        'n_ue':[0.1, 0.5, 1, 3, 10]
    }
)

TB14 = pd.DataFrame({
    "Emitters_nominale_deltaTeta_air_C": [50, 15, 25],
    "Emitters_exponent_n": [1.3, 1.1, 1.0],
    "Emitters_nominal_deltaTeta_Water_C": [20, 5, 10],
}, index=["Radiator", "Floor heating", "Fan coil"])

