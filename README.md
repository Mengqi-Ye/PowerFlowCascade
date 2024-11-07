# Map2Grid

## Build power network
1. **Extract Power Infrastructure Data**: Retrieve power infrastructure data from OpenStreetMap, including power plants, substations, and transmission lines (including minor lines and cables). Supplement the power plant data by combining it with the Global Power Plant Database.

2. **Construct Power Flow Network**: Develop a network for power flow model by using substations as buses and transmission lines as lines. Establish the connectivity between buses, determining which two buses each line connects.

## 1_osm_preparation
clone `osm_flex` to local machine, and revise:
1. osmconf.ini: add specific attributes needed by personal projects. For example, cables, frequency, etc.
2. Add attributes you added to osmconf.ini file to corresponding `osm_keys` in config.py file. Note: `plant:output:electricity` in osmconf.ini file and `plant_output_electricity` in `osm_keys`.
3. Change the data directory in both config.py and __init__.py file.