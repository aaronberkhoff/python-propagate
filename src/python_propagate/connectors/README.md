# AstriaConnector

`AstriaConnector` is a Python class that facilitates secure access to the [Astria Graph](http://astria.tacc.utexas.edu/AstriaGraph/) Neo4j database. It provides simple methods for retrieving orbital data for space objects, including Cartesian states, orbital elements, and full dynamic state representations, all linked to their `NoradId`.

---

## Features

- 🔐 **Secure credential handling** (via YAML or interactive prompt)
- 🌐 **Connects to Astria Neo4j DB** using the Bolt+S protocol
- 📦 **Retrieves**:
  - Cartesian state vectors
  - Orbital elements and epoch
  - Full state objects as `State` instances
- 🧠 Integrates with `python_propagate` for downstream analysis
- 📝 Queries and credentials are abstracted for clean reuse

---
## Usage

### **1. Create API instance**

```python 
astria = AstriaConnector()
```
This will prompt the user for credentials. Provide your username and credentials.

### **2. Input Parameters**

Once a connector is create one can input parameters to return results.

```python
state = astria.get_full_state(norad_id=norad_id,time=time)
```
Where "norad_id" is the unique identifier of the space object and "time" is a string with the time of interest. Note that Astria does not query the exact time of interest, but extracts data from a time closest to the input time. 

### Example of valid time strings
- "2019-07-18"
- "2018-09-21"
- "2021-05-21"

### Example NORAD IDs for GEO Satellites

| Satellite Name             | NORAD ID | Launch Year | Description                     |
|---------------------------|----------|-------------|---------------------------------|
| GOES-16 (GOES-R)          | 41866    | 2016        | NOAA weather satellite          |
| GOES-17 (GOES-S)          | 43226    | 2018        | NOAA weather satellite          |
| Intelsat 35e              | 42818    | 2017        | Commercial communications       |
| SES-20                    | 54039    | 2022        | Communications satellite        |
| Amazonas Nexus            | 55251    | 2023        | Spanish broadband satellite     |
| EchoStar 24 (Jupiter-3)   | 57350    | 2023        | Large GEO internet satellite    |
| Galaxy 37                 | 57355    | 2023        | Communications satellite        |
| GSAT-24                   | 52816    | 2022        | Indian communications satellite |
| Eutelsat Hotbird 13F      | 54035    | 2022        | TV broadcasting satellite       |
| Thuraya 3                 | 32500    | 2008        | Mobile communications satellite |

### Example NORAD IDs for Starlink Satellites

| Satellite Name    | NORAD ID | Launch Year | Description                        |
|------------------|----------|-------------|------------------------------------|
| Starlink-1001     | 44713    | 2020        | Early operational satellite        |
| Starlink-1130     | 45178    | 2020        | Part of 6th launch batch           |
| Starlink-1435     | 45726    | 2020        | 10th mission                       |
| Starlink-1934     | 47410    | 2021        | L28 batch                          |
| Starlink-30067    | 56326    | 2023        | Group 6-4 (v2 Mini)                |
| Starlink-30100    | 56367    | 2023        | Group 6-5 (v2 Mini)                |
| Starlink-30677    | 58716    | 2024        | Latest v2 Mini deployment          |
| Starlink-30678    | 58717    | 2024        | Group 6-53 (v2 Mini)               |
| Starlink-1601     | 45929    | 2020        | Later addition to L13 batch        |
| Starlink-2531     | 51899    | 2022        | Group 4-11                         |




### **3. Query the database**

There are several options for querying the database:

***3.1***

```python
state = astria.get_cart_state(norad_id,time)
```
This will return the a list containing the cartesian state of the spacecraft
***3.2***

```python
oe = astria.get_oe_and_epoch(norad_id,time)
```
This will return a list containg the orbital elements of a spacecraft
***3.3***
RECOMMENDED 

```python
state = astria.get_full_state(norad_id=norad_id,time=time)
```
This will return the full state information of the spacecraft including ECI and ECEF position and velocity along with latitude and longitude. The altitude of the spacecraft can be extracted by taking the norm of either the ECI or ECEF position. NOTE that the state values are returned in kilometers and radians. 

```python
state = astria.get_full_state(norad_id=norad_id,time=time)

position_eci = state.position_eci
velocity_eci = state.velocity_eci

position_ecef = state.position_ecef
velocity_ecef = state.velocity_ecef

latitude, longitude = state.latlong

altitude = np.linalg.norm(position_eci)

```

***3.4***

Putting it all together: 

```python
from python_propagate.connectors.astria import AstriaConnector
astria = AstriaConnector()

norad_id = 25544 #ISS
time = "2019-10-22"

state = astria.get_cart_state(norad_id,time)
oe = astria.get_oe_and_epoch(norad_id,time)
state = astria.get_full_state(norad_id=norad_id,time=time)


state = astria.get_full_state(norad_id=norad_id,time=time)

position_eci = state.position_eci
velocity_eci = state.velocity_eci

position_ecef = state.position_ecef
velocity_ecef = state.velocity_ecef

latitude, longitude = state.latlong

altitude = np.linalg.norm(position_eci)

```

### **4. Query a time range and multiple IDs**

To query a range of times and with multiple objects use the following process:

```python
astria = AstriaConnector()

norad_ids = [41866, 43226, 42818]

start_time = "2019-10-22"
end_time = "2019-11-22"

csv_path = "tests/results/astria/test_csv.csv" #optional, default is None
pandas_dataframe = astria.get_data(norad_ids=norad_ids,start_time=start_time,end_time=end_time,csv_path=csv_path)
```

This will produce a pandas dataframe containing the available data for each object.






