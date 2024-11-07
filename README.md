# Map2Grid

Map2Grid extracts power grid data from [OpenStreetMap](https://www.openstreetmap.org/#map=7/52.154/5.295) and add topology among power lines and substations. This tool is programmed in Python based on [GridTool](https://github.com/IEE-TUGraz/GridTool) [[1]](https://www.sciencedirect.com/science/article/pii/S2352711023000109?via%3Dihub), and can be easily integrated into power system modelling. 

## Manual

1. **Extract Power Grid Data**: Retrieve power grid data from OpenStreetMap, including power plants, substations, and transmission lines (including lines, minor lines, and cables). Supplement the power plant data by combining it with the Global Power Plant Database.

2. **Construct Power System Model**: Develop a power system model by establishing topolygy of extracted power infrastructures.

### Extract power grid data from OpenStreetMap

We use Python package [`osm-flex`](https://github.com/osm-flex/osm-flex) [[2]](https://zenodo.org/records/10204123) to download, extract, and simplify OSM data with specific features.

***Notes***: To set up the `osm_flex` package on your local machine and make the necessary adjustments:
1. Clone the repository to your local machine.

2. Edit *osmconf.ini* file: Add specific power-related attributes needed for your project, such as `cables`, `frequency`, etc.

3. Add attributes you added to *osmconf.ini* file to corresponding `osm_keys` in *config.py* file.

    *Note*: For example, `plant:output:electricity` attribute in *osmconf.ini* should correspond to `plant_output_electricity` in `osm_keys` in *config.py*.

4. Modify the data directory paths: Update the data directory paths in both *config.py* and *__init__.py*.






## References
[1] Gaugl R., Wogrin S., Bachhiesl U., Frauenlob L. (2023). GridTool: An open-source tool to convert electricity grid data. SoftwareX, Volume 21, ISSN 2352-7110. https://doi.org/10.1016/j.softx.2023.101314

[2] Koks. E.E., Mühlhofer E., Kropf, C.M., Riedel, L. (2023). OSM-flex: A Python package for flexible data extraction from OpenStreetMap. Zenodo. (https://doi.org/10.5281/zenodo.8082963)