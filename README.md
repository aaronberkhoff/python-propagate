# AstroForge Simulation Configuration

## Overview
This file defines a simulation scenario for the **AstroForge** framework, modeling the motion of a spacecraft in Earth's orbit with applied thrust maneuvers. The configuration includes spacecraft properties, thrust maneuvers, ground stations, and data generation parameters.

---

## File Structure
The configuration consists of the following sections:
- **Central Body:** Defines the central celestial body (Earth in this case).
- **Maneuvers:** Specifies thrust maneuvers for the spacecraft.
- **Stations:** Lists ground stations for observation.
- **Scenario:** Defines the simulation duration, time step, and stations.
- **Agents:** Specifies the spacecraft and its parameters.
- **Genes:** Defines data generation parameters for optimization.
- **Forge:** Configures AstroForge for simulation output and data handling.

---

## Configuration Details

### **1. Central Body**
Defines the gravitational body:
```yaml
central_body: &earth !Earth
  flattening_bool: False
```
- `flattening_bool`: If `True`, accounts for Earth's oblateness.

### **2. Maneuvers**
Defines thrust maneuvers applied to the spacecraft:
```yaml
manuevers:
  - &manuever1 !ThrustManuever
    magnitude: -0.001 # km/s
    direction_ric: [0.0, 1.0, 0.0]
    execution_time:
      days: 2
    execution_duration:
      minutes: 1
```
- `magnitude`: Change in velocity (delta-V) in km/s.
- `direction_ric`: Direction in the **Radial-Intrack-Crosstrack** (RIC) frame.
- `execution_time`: When the maneuver occurs.
- `execution_duration`: How long the thrust is applied.

### **3. Stations**
Defines ground stations for tracking:
```yaml
stations:
    - &arecibo !Station
      name: 'Arecibo'
      lat_long_alt: [18.344, -66.752, 0.0]
      minimum_elevation_angle: 15.0
      identity: 0
```
- `lat_long_alt`: Latitude, longitude, and altitude of the station.
- `minimum_elevation_angle`: Minimum tracking angle in degrees.

### **4. Scenario**
Configures the simulation environment:
```yaml
scenario: &scenario !Scenario
  central_body: *earth
  start_time: "2025-01-15T12:30:00"
  duration:
    days: 10
  dt:
    seconds: 30
  name: 'Base'
  stations: [*arecibo]
```
- `start_time`: Start time of the simulation.
- `duration`: Total simulation time.
- `dt`: Time step in seconds.

### **5. Agents (Spacecraft)**
Defines spacecraft properties:
```yaml
agents:
  - &sat1 !Spacecraft
    name: "GEO_Sat1_nominal"
    start_time: "2025-01-15T12:30:00"
    state: !OrbitalElements
              sma: 42164
              ecc: 0.0
              inc: 0.0001
              arg: 0.0
              raan: 0.0
              nu: 180.0
    dt:
      seconds: 30
    duration:
      days: 3
    coefficient_of_drag: 2.0
    mass: 150.0 # kg
    area: 2.0 # m^2
    dynamics: ['kepler']
    scenario: *scenario
    manuevers: [*manuever1]
```
- `state`: Initial orbital elements (semi-major axis, eccentricity, etc.).
- `dynamics`: Defines motion model (`kepler` for Keplerian orbits).
- `manuevers`: List of applied maneuvers.

### **6. Genes (Data Generation Parameters)**
Defines the range of parameters per spacecraft:
```yaml
gene_magnitude: &gene_magnitude !Gene
  attribute: 'magnitude'
  magnitude_min: -0.001
  magnitude_max: 0.001
  samples: 5
  agent_base: *sat1
```
- `attribute`: The parameter being changed (e.g., `magnitude`).
- `samples`: Number of samples for genetic variation.
- `agent_base`: The spacecraft associated with this gene.
- `<attribute>_min`: The lower bound of the parameter to be changed in the base agent. The name before the underscore must be consistent with the attribute field. 
- `<attribute>_max`: The upper bound of the parameter to be changed in the base agent. The name before the underscore must be consistent with the attribute field. 


### **7. Forge (Simulation Output Configuration)**
Defines output data types and storage:
```yaml
forge:
  !AstroForge
  name: "AstroForge"
  scenario: *scenario
  genes: *genes
  data_types:
     - 'RA_DEG'
     - 'DEC_DEG'
     - 'RANGE_KM'
     - 'RANGE_RATE_KMS'
     - 'AZ_DEG'
     - 'EL_DEG'
  plots: ['orbital_elements','orbit','ground_track']
  output_directory: 'examples/results/AstroForgeExample'
  output_types: ['csv', 'h5']
```
- `data_types`: Defines what data is recorded (e.g., position, velocity, range).
- `plots`: Types of plots generated.
- `output_directory`: Where results are stored.
- `output_types`: Output formats (`csv`, `h5`).

---

## Running the Simulation
To run this configuration, execute AstroForge with the provided config file:
```sh
forge --infile examples/forge_example.yaml
```
Optional run in parallel

```sh
forge --infile examples/forge_example.yaml --parallel <number_of_cores>
```
Ensure that:
- python-propagate is installed in editable mode. (ie. pip install -e .)
- The configuration file is correctly formatted.

---

## Output Files
Results will be stored in `examples/results/AstroForgeExample/` and may include:
- **CSV Files**: Numerical output for analysis. NOT RECOMMENDED FOR LARGE RUNS
- **HDF5 Files (`.h5`)**: Structured data for further processing. RECOMMENDED FOR LARGE RUNS
- **Plots**: Visualizations of orbits and elements.

---

# Todos
1. Photometric Forge
2. Add TLE option to agent state
3. Make process_agent in forge unique to each forge
4. Gravity field model

