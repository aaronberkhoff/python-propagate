import sqlite3

class Planet:

    def __init__(self, name : str, radius: float, J2: float, J3: float, spice_id:int, mu: float):
        self._radius = radius
        self._mu = mu
        self._J2 = J2
        self._J3 = J3
        self._name = name
        self._spice_id = spice_id

    @property
    def radius(self):
        return self._radius
    
    @property
    def mu(self):
        return self._mu
    
    @property
    def J2(self):
        return self._J2
    
    @property
    def J3(self):
        return self._J3
    
    @property
    def spice_id(self):
        return self._spice_id
    
    @property
    def name(self):
        return self._name
    
    
def initialize_planet(name:str):

    #load data
    connection = sqlite3.connect('data/planets.db')

    cursor = connection.cursor()

    # Query to fetch data for Earth
    cursor.execute("SELECT * FROM planets WHERE name = ?", (name,))

    planet_data = cursor.fetchone()

    # Close the connection
    connection.close()

    planet = Planet(*planet_data)

    return planet
    

        
        
    
    

        



    



