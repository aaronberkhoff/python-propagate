import sqlite3
from collections import namedtuple

Planet = namedtuple("Planet", ["name", 
                               "radius", 
                               "J2",
                               "J3", 
                               "spice_id",
                               "mu"])

earth = Planet(name     = 'Earth',
               radius   = 6378.1363,
               J2       = 0.0010826267,
               J3       = -0.0000025327,
               spice_id = 399,
               mu       = 398600.4415)


# Connect to SQLite database (it will create the database file if it doesn't exist)
connection = sqlite3.connect('data/planets.db')

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Create a table for planets
cursor.execute('''
CREATE TABLE IF NOT EXISTS planets (
    name TEXT NOT NULL,           -- Name of planet
    radius REAL NOT NULL,         -- Radius of the planet (in km)
    J2 REAL NOT NULL,             -- Second zonal
    J3 REAL NOT NULL,             -- Second zonal
    spice_id INTEGER,             -- spice identifier
    mu REAL                       -- Gravitational constant
);
''')

# Insert data for various planets
planets_data = [
    (earth.name, 
    earth.radius, 
    earth.J2,
    earth.J3,
    earth.spice_id,
    earth.mu)  # Gravitational constants in m/s^2
]

# Insert the planet data into the table
cursor.executemany('''
INSERT INTO planets (name, radius, J2, J3, spice_id, mu)
VALUES (?, ?, ?, ?, ?, ?);
''', planets_data)

# Commit the transaction (to save the changes)
connection.commit()

# Query to fetch all planet data
cursor.execute("SELECT * FROM planets")

# Fetch and display all the planets
planets = cursor.fetchall()
for planet in planets:
    print(f"ID: {planet[0]}, Name: {planet[1]} J3")

# Close the connection to the database
connection.close()
