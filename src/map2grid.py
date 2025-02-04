import os
os.environ['USE_PYGEOS'] = '0'
import sys
import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, mapping
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

current_dir = os.getcwd()
osm_flex_path = os.path.abspath(os.path.join(current_dir, '../../osm-flex/src'))
sys.path.insert(0, osm_flex_path)
import osm_flex.download as dl
import osm_flex.extract as ex
import osm_flex.config
import osm_flex.simplify as sy


###############################################################
###############################################################
#################### MODULE 1： IMPORT DATA ###################
###############################################################
###############################################################

def load_osm_data(iso3):
    """
    Load and process OpenStreetMap (OSM) data for a given country ISO3 code.
    Steps include removing duplicates and filtering by geometry type.

    Args:
        iso3 (str): The ISO3 code for the country.
        dump_filename (str): Name of the OSM dump file.

    Returns:
        gpd.GeoDataFrame: Processed GeoDataFrame.
    """
    dump_filename = dl.get_country_geofabrik(iso3)
    print(dump_filename)
    path_dump = osm_flex.config.OSM_DATA_DIR.joinpath(dump_filename.name)

    gdf = ex.extract_cis(path_dump, 'power')
    print(f'Initial number of results: {len(gdf)}')

    gdf = sy.remove_contained_points(gdf)
    print(f'After removing contained points: {len(gdf)}')

    gdf = sy.remove_contained_polys(gdf)
    print(f'After removing contained polygons: {len(gdf)}')

    gdf = sy.remove_exact_duplicates(gdf)
    print(f'After removing exact duplicates: {len(gdf)}')

    gdf.set_crs(epsg=4326)
    gdf = gdf.to_crs(epsg=32648)

    return gdf


def import_osm_infras(gdf, osm_key, osm_values, output_path, driver="GPKG"):
    """
    Filter geometries based on column values and save to file.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to filter.
        osm_key (str): Column name to filter on.
        osm_values (list): List of values to filter.
        output_path (str): Filepath to save the filtered data.
        driver (str): File driver (default: "GPKG").

    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame.
    """
    filtered_gdf = gdf[gdf[osm_key].isin(osm_values)]
    print(f"Number of {osm_values}: {len(filtered_gdf)}")

    if filtered_gdf['osm_id'].isnull().any():
        print("Column 'osm_id' has None values")
        if osm_values[0] == 'plant':
            reference_gdf = gpd.read_file(f'../data/osm/power_plant_power_generator_vietnam.gpkg')
        elif osm_values[0] == 'substation':
            reference_gdf = gpd.read_file(f'../data/osm/power_substation_vietnam.gpkg')
        add_missing_osm_ids(filtered_gdf, reference_gdf)
    else:
        print("Column 'osm_id' doesn't have None values")

    filtered_gdf.to_file(f"{os.path.join(output_path, osm_values[0])}.gpkg", driver=driver)

    return filtered_gdf


def add_missing_osm_ids(target_gdf, reference_gdf, key_column='osm_id'):
    """
    Add missing OSM IDs to a target GeoDataFrame using a reference GeoDataFrame via spatial join.

    Args:
        target_gdf (gpd.GeoDataFrame): Target GeoDataFrame with missing OSM IDs.
        reference_gdf (gpd.GeoDataFrame): Reference GeoDataFrame with OSM IDs.
        key_column (str): Column name for OSM IDs (default: 'osm_id').

    Returns:
        gpd.GeoDataFrame: Updated target GeoDataFrame with missing IDs filled.
    """
    if target_gdf.crs != reference_gdf.crs:
        reference_gdf = reference_gdf.to_crs(target_gdf.crs)

    matched = gpd.sjoin(target_gdf, reference_gdf[['geometry', key_column]], op='intersects')
    target_gdf.loc[matched.index, key_column] = matched[f'{key_column}_right']

    print(f"Number of unmatched geometries: {len(target_gdf[target_gdf[key_column].isna()])}")
    return target_gdf


###############################################################
###############################################################
################### MODULE 2： DATA ANALYSIS ##################
###############################################################
###############################################################

def extract_unique_endpoints(gdf):
    """
    Extract unique start and end coordinates from each LineString in a GeoDataFrame (optimized version).
    
    Parameters:
    gdf (GeoDataFrame): A GeoDataFrame containing LINESTRING geometries.
    
    Returns:
    GeoDataFrame: A GeoDataFrame containing unique start and end points.
    """
    unique_points = {}  # Dictionary to store unique points
    point_id = 0
    
    # Iterate through each LineString to extract start and end points
    for line in gdf['geometry']:
        if line.geom_type == 'LineString':
            # Get the coordinates of the start and end points
            start_point = line.coords[0]
            end_point = line.coords[-1]
            
            # Use a dictionary to check for duplicates and avoid adding the same point multiple times
            for point in [start_point, end_point]:
                if point not in unique_points:
                    unique_points[point] = {
                        "nodeID": point_id,
                        "lon": point[0],
                        "lat": point[1],
                        "geometry": Point(point[0], point[1])
                    }
                    point_id += 1
    
    # Create and return a GeoDataFrame containing unique start and end points
    return gpd.GeoDataFrame(list(unique_points.values()), 
                            columns=["nodeID", "lon", "lat", "geometry"], 
                            crs=gdf.crs)


# def extract_all_nodes(gdf):
#     """
#     Extract all nodes (coordinates) from each LineString in a GeoDataFrame.
    
#     Parameters:
#     gdf (GeoDataFrame): A GeoDataFrame containing LINESTRING geometries.
    
#     Returns:
#     GeoDataFrame: A GeoDataFrame containing all unique nodes.
#     """
#     unique_points = {}  # Dictionary to store unique points
#     point_id = 0  # Counter for generating unique node IDs
    
#     # Iterate through each LineString geometry in the GeoDataFrame
#     for line in gdf['geometry']:
#         if line.geom_type == 'LineString':
#             # Iterate through all coordinates in the LineString
#             for point in line.coords:
#                 # Add the point to the dictionary if it is not already present
#                 if point not in unique_points:
#                     unique_points[point] = {
#                         "nodeID": point_id,
#                         "lon": point[0],
#                         "lat": point[1],
#                         "geometry": Point(point[0], point[1])
#                     }
#                     point_id += 1  # Increment the node ID
    
#     # Create and return a GeoDataFrame containing all unique nodes
#     return gpd.GeoDataFrame(list(unique_points.values()), 
#                             columns=["nodeID", "lon", "lat", "geometry"], 
#                             crs=gdf.crs)


def add_endnodes_to_lines(gdf, nodes_gdf):
    """
    Adds the starting and ending node IDs (node1 and node2) and their corresponding geometric coordinates (geom_node1, geom_node2) 
    to each line in the original GeoDataFrame.

    Parameters:
    - gdf (geopandas.GeoDataFrame): The original GeoDataFrame containing LINESTRING geometries.
    - nodes_gdf (geopandas.GeoDataFrame): A GeoDataFrame containing unique coordinate points and their corresponding node IDs.

    Returns:
    - gdf (geopandas.GeoDataFrame): The updated GeoDataFrame with added node1, node2, geom_node1, and geom_node2 columns.
    """

    # Create a mapping from coordinate to nodeID
    point_to_nodeid = { (row['lon'], row['lat']): row['nodeID'] for _, row in nodes_gdf.iterrows() }

    gdf = gdf.copy()

    # Add node1 and node2 columns for each LineString
    gdf['node1'] = gdf['geometry'].astype(object).apply(
        lambda line: point_to_nodeid.get((line.coords[0][0], line.coords[0][1]), None))
    gdf['node2'] = gdf['geometry'].astype(object).apply(
        lambda line: point_to_nodeid.get((line.coords[-1][0], line.coords[-1][1]), None))

    # Add geometric coordinates for node1 and node2
    gdf['geom_node1'] = gdf['geometry'].astype(object).apply(
        lambda line: (line.coords[0][0], line.coords[0][1]))
    gdf['geom_node2'] = gdf['geometry'].astype(object).apply(
        lambda line: (line.coords[-1][0], line.coords[-1][1]))

    # Calculate the length of each line in kilometers
    gdf['Length'] = gdf['geometry'].length * 1.2 / 1000
    
    return gdf


def transform_osm_subs(gdf, buffer_distance=500):
    """
    Transforms the geometries of a GeoDataFrame by buffering them with a specified distance.

    Parameters:
    - gdf (geopandas.GeoDataFrame): The input GeoDataFrame containing geometries to be transformed.
    - buffer_distance (int, optional): The buffer distance to apply around each geometry. Default is 500.

    Returns:
    - gdf (geopandas.GeoDataFrame): The updated GeoDataFrame with transformed geometries.
    """

    gdf = gdf.copy()
    gdf['geometry_update'] = gdf['geometry'].astype(object).apply(lambda geom: transform_geometry(geom, buffer_distance))

    gdf['geometry'] = gdf['geometry_update']
    gdf = gdf.drop(columns=['geometry_update'])

    gdf['geom_centroid'] = gdf.geometry.centroid

    return gdf


def transform_geometry(geom, buffer_distance=500):
    """
    Applies a buffer to the given geometry based on its type.

    Parameters:
    - geom (shapely.geometry): The geometry to transform.
    - buffer_distance (int, optional): The buffer distance to apply. Default is 500.

    Returns:
    - buffered_geom (shapely.geometry): The buffered geometry.
    """

    # Convert LineString to Polygon
    geom = linestring_to_polygon(geom)
    
    # Apply buffer based on geometry type
    if geom.geom_type == 'Point':
        buffered_geom = geom.buffer(buffer_distance, cap_style=3)  # Square buffer for Point
    elif geom.geom_type in ['Polygon', 'MultiPolygon']:
        buffered_geom = geom.buffer(buffer_distance)
    else:
        buffered_geom = geom   # Keep other geometry types unchanged
    return buffered_geom


# https://stackoverflow.com/questions/2964751/how-to-convert-a-geos-multilinestring-to-polygon
def linestring_to_polygon(geom):
    # gdf['geometry'] = [Polygon(mapping(x)['coordinates']) for x in gdf.geometry]
    if geom.geom_type == 'LineString':
        return Polygon(mapping(geom)['coordinates'])
    return geom


def add_osm_ids_to_data(data, osm_subs):
    """
    Updates a DataFrame with OSM (OpenStreetMap) node IDs and geometry information for nodes that fall within 
    a specified geographical area, represented by `osm_subs`.

    The function iterates through each row in the provided DataFrame (`data`), where each row contains 
    information about two nodes (node1 and node2) along with their geometric coordinates. The function 
    attempts to match these nodes with their corresponding OSM IDs and geometry within the `osm_subs` 
    boundary using the helper function `get_osm_info_for_point`. If a match is found, it updates the 
    node's ID and geometry in the DataFrame.

    The function also keeps track of which nodes were updated (i.e., those whose IDs and/or geometry 
    were modified), returning a list of these nodes at the end.

    Parameters:
    - data (pandas.DataFrame): A DataFrame containing node information (node1, node2, their geometries).
      The DataFrame should have columns `node1`, `node2`, `geom_node1`, and `geom_node2`, where the 
      geometry columns contain the coordinates of each node.
    - osm_subs (object): A geographical area or boundary used to check if nodes are within this area. 
      The specific type and structure of this object depend on the implementation of the `get_osm_info_for_point` function.

    Returns:
    - data (pandas.DataFrame): The updated DataFrame with OSM IDs and geometry information for nodes 
      that are located within `osm_subs`.
    - nodes_in_osm_subs (list): A list of nodes (node1 and node2) that were updated with new OSM IDs 
      and geometry within the specified area.
    """

    nodes_in_osm_subs = set()

    for idx, row in data.iterrows():
        # Convert node1 and node2 geometry to Point objects
        node1_point = Point(row['geom_node1'][0], row['geom_node1'][1])
        node2_point = Point(row['geom_node2'][0], row['geom_node2'][1])
        
        # Check if node1 is within the osm_subs area, and replace ID and geometry if so
        osmID_node1, geom_node1 = get_osm_info_for_point(node1_point, row['node1'], osm_subs)
        osmID_node2, geom_node2 = get_osm_info_for_point(node2_point, row['node2'], osm_subs)
        # print(type(osmID_node1))

        # If node1 is within osm_subs, update ID and geometry
        if osmID_node1 != row['node1']:  # If replacement occurred
            data.at[idx, 'node1'] = osmID_node1
            data.at[idx, 'geom_node1'] = geom_node1
            nodes_in_osm_subs.add(row['node1'])
        
        # If node2 is within osm_subs, update ID and geometry
        if osmID_node2 != row['node2']:  # If replacement occurred
            data.at[idx, 'node2'] = osmID_node2
            data.at[idx, 'geom_node2'] = geom_node2
            nodes_in_osm_subs.add(row['node2'])

    return data, list(nodes_in_osm_subs)


def get_osm_info_for_point(point, original_id, osm_subs):
    """
    Checks if the given point is within the osm_subs range.
    If it is, returns the osm_id and the coordinates of the centroid; otherwise, returns the original ID and point coordinates.
    If a point falls within multiple ranges, the closest osm_sub is chosen.

    Parameters:
    point : Point
        The point for which OSM information is being checked.
    original_id : any type
        The original identifier associated with the point.
    osm_subs : GeoDataFrame
        A GeoDataFrame containing OSM subsets with geometry information.

    Returns:
    osm_id : int or original_id
        The OSM ID of the closest osm_sub if the point is within its range, otherwise the original ID.
    centroid_coords : tuple
        The coordinates (x, y) of the centroid of the matching osm_sub, or the coordinates of the original point if not matched.
    """

    # if point is None:
    #     return original_id, None

    # Find OSM subsets that contain the point
    matches = osm_subs[osm_subs['geometry'].contains(point)]

    if not matches.empty:
        # If there are multiple matching osm_subs, select the nearest one
        distances = matches['geom_centroid'].distance(point)

        nearest_idx = distances.idxmin()  # Index of the nearest osm_sub
        osm_id = matches.loc[nearest_idx, 'osm_id']
        centroid = matches.loc[nearest_idx, 'geom_centroid']
        return int(osm_id), (centroid.x, centroid.y)
    else:
        return original_id, (point.x, point.y)


def count_voltage_levels(data, voltage_levels_selected=[110000,220000,500000]):
    """
    Counts and processes voltage levels in the dataset, expanding rows with multiple voltage levels.

    Parameters:
    data : DataFrame
        The input dataset containing a 'voltage' column.
    voltage_levels_selected : list of int, optional
        A list of voltage levels to filter and count, in volts (e.g., [110000, 220000, 500000]).

    Returns:
    expanded_data : DataFrame
        The updated dataset with rows expanded for multiple voltage levels and additional columns 
        for 'vlevels' (count of voltage levels) and filtered 'voltage'.
    """

    print('Start counting voltage levels...')

    data['vlevels'] = 0
    
    expanded_data = []
    voltage_levels_count = {}
    none_voltage_count = 0

    for _, row in data.iterrows():
        voltage = row['voltage']
        vlevels = 0 # Initialize voltage levels count for this row


        if isinstance(voltage, str) and voltage:
            # Handle rows with multiple voltage levels
            if ';' in voltage:
                voltage_levels = voltage.split(';')
                valid_voltages = [int(v.strip()) for v in voltage_levels if v.strip().isdigit()]
                filtered_voltages = [v for v in valid_voltages if v in voltage_levels_selected]

                vlevels = len(filtered_voltages) # Set the count of selected voltage levels

                # Clone and add rows for each valid voltage level
                for v in filtered_voltages:
                    new_row = row.copy()
                    new_row['voltage'] = v  # Set the single voltage value
                    new_row['vlevels'] = vlevels  # Set the count of voltage levels
                    expanded_data.append(new_row)

                    # Count the occurrences of this voltage level
                    voltage_levels_count[v] = voltage_levels_count.get(v, 0) + 1
            
            else:
                # Handle rows with a single voltage level
                row['voltage'] = int(voltage.strip()) if voltage.strip().isdigit() else None
                if row['voltage'] in voltage_levels_selected:  # Only include selected voltage levels
                    row['vlevels'] = 1
                    expanded_data.append(row)

                    # Count the occurrences of this voltage level
                    voltage_level = row['voltage']
                    if voltage_level is not None:
                        voltage_levels_count[voltage_level] = voltage_levels_count.get(voltage_level, 0) + 1
        
        else:
            # Handle rows with no voltage level
            # print(f'WARNING: Way with ID {row.get("id", "Unknown ID")} does not have a voltage level.')
            none_voltage_count += 1
            row['voltage'] = None
            row['vlevels'] = 0
            expanded_data.append(row)

    # Convert the expanded data into a DataFrame
    expanded_data = pd.DataFrame(expanded_data, columns=data.columns)
    expanded_data['voltage'] = expanded_data['voltage'].astype('Int64')
    expanded_data['vlevels'] = expanded_data['vlevels'].astype('Int64')

    # unique_voltage_levels = sorted(expanded_data['voltage'].dropna().unique(), reverse=True)

    print('Voltage levels count:')
    for v_level, count in voltage_levels_count.items():
        print(f'Voltage level {int(v_level/1000)} kV: {count}')
    
    print(f'Count of rows with None voltage: {none_voltage_count}')

    return expanded_data #, unique_voltage_levels


def delete_busbars(data, bool_options, busbar_max_length=0.2): # busbar_max_length=0.5??
    """
    Delete busbars and bays from the dataset based on their length.
    
    Parameters:
        data (GeoDataFrame): Input dataset of selected ways.
        bool_options (bool): If True, plot a histogram of busbar lengths.
        busbar_max_length (float): The maximum length a busbar can have, 0.5km.
    
    Returns:
        GeoDataFrame: Updated dataset without busbars.
        GeoDataFrame: All busbars extracted from the original dataset.
    """
    print('Start deleting ways with type "busbar" or "bay"...')
    start_time = time.time()
    
    # Initialize counters for busbars
    i_busbars_bays = 0  # number of busbars or bays
    d_busbars_bays = 0  # number of deleted busbars or bays
    lengths_of_busbars = []

    # Add a new column to flag busbars
    data['busbar'] = False
    
    # Iterate through all way-elements
    for index, row in data.iterrows():
        # Check if "line" field is "busbar" or "bay" and if length is within limit
        is_busbar = pd.notna(row['line']) and row['line'].lower() in ['busbar', 'bay']
        length_ok = row['Length'] < busbar_max_length

        if is_busbar:
            i_busbars_bays += 1
            if length_ok:
                data.at[index, 'busbar'] = True
                d_busbars_bays += 1
                lengths_of_busbars.append(row['Length'])
            else:
                print(f'   ATTENTION! Way Element ID {row["osm_id"]} has type "busbar" or "bay", '
                      f'but is too long.\n               Length: {row["Length"]:.2f} km of max. '
                      f'{busbar_max_length:.1f} km\n               This way won\'t be added to the '
                      '"busbar" exception list.')

    # Extract and remove all busbars/bays from the dataset
    data_busbars = data[data['busbar']].copy()
    data = data[~data['busbar']].copy()

    data = data.reset_index(drop=True)
    
    # Optional: Histogram of busbar/bays lengths
    if bool_options.get('histogram_length_busbars', True):
        plt.figure(figsize=(10, 6))  # You can customize the figure size if needed
        plt.hist(lengths_of_busbars, bins=200, color='blue', alpha=0.7)
        plt.title('Lengths of busbars/bays below busbar-max-length-threshold')
        plt.xlabel('Length [km]')
        plt.ylabel('Number of busbars with that length')
        plt.grid(True)
        plt.show()
    
    print(f'   ... there are {i_busbars_bays} busbars/bays in total')
    print(f'   ... {d_busbars_bays} busbars have been deleted')
    print(f'   ... finished! ({time.time() - start_time:.3f} seconds) \n')
 
    return data, data_busbars


def count_possible_dc(data):
    """
    Identify potential DC lines in the dataset.
    
    Parameters:
        data (GeoDataFrame): Input dataset of selected ways.
        
    Returns:
        GeoDataFrame: Updated dataset, including a flag if a way may be a DC line.
        list of dict: List of potential DC candidates with id, reason, and voltage level.
    """
    print('Start detecting lines which could be DC lines...')
    start_time = time.time()

    # Initialize list to store DC candidates
    dc_candidates = []
    
    # Add a new column 'dc_candidate' to flag possible DC lines
    data['dc_candidate'] = False

    # Go through each row in the dataset
    for index, row in data.iterrows():
        # Initialize the reason for this row
        reason = ''

        # Check for frequency condition
        if 'frequency' in row and pd.notna(row['frequency']) and int(row['frequency']) == 0:
            data.at[index, 'dc_candidate'] = True
            reason = 'frequency is 0'
        
        # Check for name condition (case insensitive for 'dc')
        if 'name' in row and isinstance(row['name'], str) and 'dc' in row['name'].lower():
            data.at[index, 'dc_candidate'] = True
            reason = 'name contains "DC"'

        # Check for cables condition
        if 'cables' in row and pd.notna(row['cables']) and int(row['cables']) == 1:
            data.at[index, 'dc_candidate'] = True
            reason = 'cables is 1'

        # print(data.at[index, 'dc_candidate'])
        # If the row meets any DC condition, add it to dc_candidates
        if data.at[index, 'dc_candidate']:
            dc_candidates.append({
                'id': row['osm_id'],
                'reason': reason,
                'voltage_level': row.get('voltage', 'unknown')
            })
    
    # Output result based on whether candidates were found
    if len(dc_candidates) == 0:
        print('   ... no potentially DC lines found.')
    else:
        print(f'   ... {len(dc_candidates)} ways could potentially be a DC line.')
        print('   ... Please refer to dc_candidates for further checks.')
    
    print(f'   ... finished! ({time.time() - start_time:.3f} seconds) \n')

    return data, dc_candidates


def count_cables(data):
    print('Start counting cables per way...')

    # Initialize the cables list
    cables_per_way = []

    # Go through every way
    for index, row in data.iterrows():
        # Check if "cables" field exists and is not NaN
        if 'cables' in row and pd.notna(row['cables']):
            # Handle NaN and convert to integer
            try:
                num_of_cables = int(row['cables'])  # Convert cables to int

                cables_per_way.append({'ID': row['osm_id'], 'num_of_cables': num_of_cables})

                # Set the systems flag accordingly
                if num_of_cables == 6:
                    data.at[index, 'systems'] = 2
                elif num_of_cables == 9:
                    data.at[index, 'systems'] = 3
                elif num_of_cables == 12:
                    data.at[index, 'systems'] = 4
                else:
                    data.at[index, 'systems'] = None  # None equivalent in Python
                
                data.at[index, 'cables'] = num_of_cables

            except ValueError:
                print(f'   ATTENTION! Unknown cable number ("{row["cables"]}") in ID {row["osm_id"]}. This way wont be cloned automatically.')
                continue
        
        else:
            data.at[index, 'systems'] = None  # None equivalent in Python

    # Print cable occurrence information
    if cables_per_way:
        cables_df = pd.DataFrame(cables_per_way)
        cables_count = cables_df['num_of_cables'].value_counts().reset_index()
        cables_count.columns = ['cables_per_way', 'number_of_ways']

        print('\n', cables_count)
        print(f'   ... {data.shape[0] - cables_count["number_of_ways"].sum()} ways with unknown number of cables.')
        
        print('   ... ways with 6 cables will be doubled, ways with 9 cables tripled and ways with 12 cables quadrupled.')
    else:
        print('   ... no cables per way info was found.')

    print('   ... finished!')

    return data


###############################################################
###############################################################
#################### MODULE 3: GROUP NODES ####################
###############################################################
###############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import geopandas as gpd

def calc_distances_between_endpoints(data, bool_options=False):
    data = gpd.GeoDataFrame(data).set_crs(epsg=3426, inplace=True)
    data = data.to_crs(epsg=32648)

    # Initialize distance matrix
    n = len(data)
    M = np.full((2 * n, 2 * n), np.nan)  # Initialize the distance matrix with NaN

    # Extract start and end points
    start_points = np.array([way.coords[0] for way in data.geometry])
    end_points = np.array([way.coords[-1] for way in data.geometry])

    # Combine start and end points
    all_points = np.empty((2 * n, 2))  # Each point has two coordinates (x, y)
    all_points[0::2] = start_points     # Odd rows are start points
    all_points[1::2] = end_points       # Even rows are end points

    # Calculate distances using cdist
    distances = cdist(all_points, all_points) / 1000  # Convert from meters to kilometers

    # Fill in the distance matrix
    for i in range(n):
        M[2 * i, 2 * i + 1] = -1       # Start to end distance is -1
        M[2 * i + 1, 2 * i] = -1       # End to start distance is -1
        M[2 * i, 2 * i] = -1           # Start to itself distance is -1
        M[2 * i + 1, 2 * i + 1] = -1   # End to itself distance is -1

        # Fill in distances between different paths
        for j in range(i + 1, n):
            M[2 * i, 2 * j] = distances[2 * i, 2 * j]         # Start to start distance
            M[2 * i, 2 * j + 1] = distances[2 * i, 2 * j + 1] # Start to end distance
            M[2 * i + 1, 2 * j] = distances[2 * i + 1, 2 * j] # End to start distance
            M[2 * i + 1, 2 * j + 1] = distances[2 * i + 1, 2 * j + 1] # End to end distance

    # Optional: Plot a histogram of all the distances
    if bool_options.get('histogram_distances_between_endpoints', True):
        print('   ... start visualizing all distances in a histogram ...')

        # Flatten the matrix and exclude NaN values
        distance_values = M[~np.isnan(M)]

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(distance_values, bins=200, color='skyblue', edgecolor='black')
        plt.title('Distances Between All Endpoints')
        plt.xlabel('Distance (km)')
        plt.ylabel('Number of Pairs')
        plt.grid(True)
        plt.show()

    return M


def calc_stacked_endnodes(data, distances, bool_options):
    """
    DESCRIPTION
    This function searches every distance combination between all
    endpoints which have the value "0", indicating that two endpoints
    have the same coordinates and are stacked on top of each other.

    INPUT
    data ... input dataset (DataFrame)
    distances ... distance Matrix M (numpy array)
    bool_options ... dictionary with boolean flags for visualizations

    OUTPUT
    data ... updated dataset with new flags: node1_stacked and node2_stacked
    nodes_stacked_pairs ... a raw list of all pairs of stacked endnodes
    """
    
    start_time = time.time()
    print('Start finding all stacked endnodes...')
    
    # Create boolean logical index of all distance combinations that equal 0
    b_dist_is_zero = distances == 0

    # If no distance element has value 0, cancel that function since no two endpoints are stacked
    if not np.any(b_dist_is_zero):
        data['node1_stacked'] = False
        data['node2_stacked'] = False
        nodes_stacked_pairs = []
        print('... no endnode is stacked!')
        print(f'... finished! ({time.time() - start_time:.3f} seconds)\n')
        return data, nodes_stacked_pairs
    
    # Get the indices of this boolean matrix
    dist_row, dist_column = np.where(b_dist_is_zero)

    # Combine the row and column indices and sort them
    dist_combined = np.sort(np.concatenate((dist_row, dist_column)))

    # Remove duplicates and calculate occurrences
    unique, unique_counts = np.unique(dist_combined, return_counts=True)
    
    print(f'... {len(unique)} endnodes are stacked!')

    # Create a DataFrame for stacked nodes
    nodes_stacked = pd.DataFrame({
        'index': unique,
        'way_ID': unique // 2,  # np.ceil(unique / 2).astype(int),  # Convert indices to Wayelement ID
        'endnode1': unique % 2 == 0  # Convert indices to boolean indicator if it's endnode1
    })

    # Return all pairs for grouping
    nodes_stacked_pairs = np.column_stack((dist_row, dist_column))

    # Initialize boolean columns with correct type
    data['node1_stacked'] = False
    data['node2_stacked'] = False
    data = data.astype({'node1_stacked': 'bool', 'node2_stacked': 'bool'})

    # Add stacked information to dataset
    i_stacked_nodes = 0
    numel_way_IDs = len(nodes_stacked)

    for i_ways in range(len(data)):
        if i_stacked_nodes >= numel_way_IDs:
            break
        
        if i_ways == nodes_stacked['way_ID'].iloc[i_stacked_nodes]:
            # Yes, at least one endnode is stacked
            if (i_stacked_nodes < numel_way_IDs - 1) and \
               (nodes_stacked['way_ID'].iloc[i_stacked_nodes] == nodes_stacked['way_ID'].iloc[i_stacked_nodes + 1]):
                # Both endnodes are stacked
                data.at[i_ways, 'node1_stacked'] = True
                data.at[i_ways, 'node2_stacked'] = True
                i_stacked_nodes += 1  # Skip one index
            elif nodes_stacked['endnode1'].iloc[i_stacked_nodes]:
                # Only endnode 1 is stacked
                data.at[i_ways, 'node1_stacked'] = True
                data.at[i_ways, 'node2_stacked'] = False
            else:
                # Only endnode 2 is stacked
                data.at[i_ways, 'node1_stacked'] = False
                data.at[i_ways, 'node2_stacked'] = True
            
            # Select next index to compare against way_ID
            i_stacked_nodes += 1
        else:
            # No, none of both endnodes are stacked
            data.at[i_ways, 'node1_stacked'] = False
            data.at[i_ways, 'node2_stacked'] = False
    
    print(f'... finished! ({time.time() - start_time:.3f} seconds)\n')

    data = gpd.GeoDataFrame(data, geometry='geometry', crs='EPSG:32648')

    # Visualize this stacked data
    if bool_options.get('plot_stacked_endnodes', True):
        plot_stacked_lines(data, bool_options=bool_options)

    # Plotting histogram of stacked endnodes
    if bool_options.get('histogram_stacked_endnodes', False):
        plt.figure(figsize=(8, 6))
        plt.hist(unique_counts + 1, bins=np.arange(1, unique_counts.max() + 2) - 0.5, edgecolor='black')
        plt.title('Stacked endnodes: If stacked, how many are stacked?')
        plt.xlabel('Nodes stacked on top of each other')
        plt.ylabel('Number of different positions this occurs in')
        plt.show()

    return data, nodes_stacked_pairs


def group_nodes(pairs_input):
    """
    Groups nodes based on pairs of connections. Nodes that are directly or indirectly 
    connected will be grouped together.

    Parameters:
    pairs_input : list of pairs
        Each element is a tuple (or list) containing a pair of nodes that are connected.
        
    Returns:
    list_groups : list of lists
        A list where each element represents a group of connected nodes.
    """

    start_time = time.time()
    print(f'Start grouping all pairs from "pairs_input" (may take a few seconds)...')

    # Initialize an empty list to store groups
    list_groups = []
    
    # Sort each pair for consistency
    pairs_sorted = [sorted(pair) for pair in pairs_input]
    
    # Sort pairs by the first element to facilitate grouping
    pairs_sorted.sort()

    # Process each pair to form groups of connected nodes
    for partner1, partner2 in pairs_sorted:
        row_partner1 = row_partner2 = None

        # Check if either partner1 or partner2 already belongs to an existing group
        for i, group in enumerate(list_groups):
            if partner1 in group:
                row_partner1 = i
            if partner2 in group:
                row_partner2 = i

        if row_partner1 is not None:
            if row_partner2 is not None:
                if row_partner1 == row_partner2:
                    # Both partners are in the same group; no further action required
                    continue
                else:
                    # Merge two groups and remove the redundant group
                    list_groups[row_partner1] = sorted(set(list_groups[row_partner1] + list_groups[row_partner2]))
                    del list_groups[row_partner2]
            else:
                # Add partner2 to the group containing partner1
                list_groups[row_partner1].append(partner2)
                list_groups[row_partner1] = sorted(set(list_groups[row_partner1]))
        elif row_partner2 is not None:
            # Add partner1 to the group containing partner2
            list_groups[row_partner2].append(partner1)
            list_groups[row_partner2] = sorted(set(list_groups[row_partner2]))
        else:
            # Create a new group for both partners
            list_groups.append(sorted([partner1, partner2]))

    # Summary output
    num_groups = len(list_groups)
    total_nodes = sum(len(group) for group in list_groups)
    avg_nodes_per_group = total_nodes / num_groups if num_groups > 0 else 0
    print(f"   ... {total_nodes} nodes will be grouped together in {num_groups} groups,")
    print(f"       with an average of {avg_nodes_per_group:.2f} nodes per group.")
    print(f'   ... finished! ({time.time() - start_time:.3f} seconds)\n')

    return list_groups


def group_stacked_endnodes(data, points_gdf, nodes_stacked_grouped):
    """
    Updates the coordinates and IDs of stacked end nodes in the dataset. 
    For each group of stacked nodes, their IDs and coordinates are replaced 
    with those of the first node in the group.

    Parameters:
    data : DataFrame
        The dataset containing information about ways, including start and end nodes.
    points_gdf : GeoDataFrame
        GeoDataFrame containing point data, typically used for spatial operations.
    nodes_stacked_grouped : list of lists
        Each sublist contains the IDs of nodes that belong to the same stacked group.

    Returns:
    data : DataFrame
        The updated dataset with new columns for grouped IDs and coordinates.
    """

    # Add new columns for grouped IDs and coordinates, initialized to NaN
    data['ID_node1_grouped'] = np.nan
    data['ID_node2_grouped'] = np.nan
    data['lon1_grouped'] = np.nan
    data['lat1_grouped'] = np.nan
    data['lon2_grouped'] = np.nan
    data['lat2_grouped'] = np.nan

    # Iterate over each group of stacked nodes
    for group in nodes_stacked_grouped:
        # Get the first node ID in the current group
        first_node_id = group[0]
        way_id = first_node_id // 2  # Calculate the way ID (node IDs start from 0 in Python, DIFFERENT from Matlab)
        is_endnode1 = (first_node_id % 2) == 0  # Even IDs correspond to start nodes

        # Retrieve the coordinates for the start or end node
        if is_endnode1:
            # Start node
            grouped_node_id = data.at[way_id, 'node1']
            grouped_lon, grouped_lat = data.at[way_id, 'geometry'].coords[0]
        else:
            # End node
            grouped_node_id = data.at[way_id, 'node2']
            grouped_lon, grouped_lat = data.at[way_id, 'geometry'].coords[-1]

        # Update all members of the group
        for member_node_id in group:
            way_id = member_node_id // 2  # Calculate the corresponding way ID
            is_endnode1 = (member_node_id % 2) == 0  # Determine if it's a start nod

            if is_endnode1:
                data.at[way_id, 'ID_node1_grouped'] = grouped_node_id
                data.at[way_id, 'lon1_grouped'] = grouped_lon
                data.at[way_id, 'lat1_grouped'] = grouped_lat
            else:
                data.at[way_id, 'ID_node2_grouped'] = grouped_node_id
                data.at[way_id, 'lon2_grouped'] = grouped_lon
                data.at[way_id, 'lat2_grouped'] = grouped_lat

    print("Completed updating coordinates for stacked groups.")
    return data


def add_final_coordinates(data):
    """
    This function selects the final coordinates: If one or both endnodes
    got grouped (because they were stacked and/or in a neighbourhood), 
    those new grouped coordinates will be the final coordinates. If not, 
    then the original coordinates will be taken as the final coordinates. 
    The final coordinate will consist of the ID, the lon/lat and the x/y
    coordinates.

    Parameters:
    data : DataFrame
        The original dataset containing ways and their coordinates.

    Returns:
    DataFrame
        The updated dataset with new final coordinates fields.
    """
    print('Start adding final coordinates...')
    
    # Iterate through each way in the dataset
    for i_ways in range(len(data)):
        start_node_id = data.at[i_ways, 'node1']
        end_node_id = data.at[i_ways, 'node2']
        geometry = data.at[i_ways, 'geometry']

        # Check if a new grouped node1 exists; if not, use the original data
        if pd.isna(data.at[i_ways, 'ID_node1_grouped']):
            data.at[i_ways, 'ID_node1_final'] = start_node_id
            data.at[i_ways, 'lon1_final'] = geometry.coords[0][0]  # lon
            data.at[i_ways, 'lat1_final'] = geometry.coords[0][1]  # lat
        else:
            data.at[i_ways, 'ID_node1_final'] = data.at[i_ways, 'ID_node1_grouped']
            data.at[i_ways, 'lon1_final'] = data.at[i_ways, 'lon1_grouped']
            data.at[i_ways, 'lat1_final'] = data.at[i_ways, 'lat1_grouped']

        # Check if a new grouped node2 exists; if not, use the original data
        if pd.isna(data.at[i_ways, 'ID_node2_grouped']):
            data.at[i_ways, 'ID_node2_final'] = end_node_id
            data.at[i_ways, 'lon2_final'] = geometry.coords[-1][0]  # lon
            data.at[i_ways, 'lat2_final'] = geometry.coords[-1][1]  # lat
        else:
            data.at[i_ways, 'ID_node2_final'] = data.at[i_ways, 'ID_node2_grouped']
            data.at[i_ways, 'lon2_final'] = data.at[i_ways, 'lon2_grouped']
            data.at[i_ways, 'lat2_final'] = data.at[i_ways, 'lat2_grouped']

    print('... finished!')
    return data


###############################################################
###############################################################
####################### MODULE 4: Export ######################
###############################################################
###############################################################


def delete_singular_ways(data, node1_col, node2_col):
    """
    Deletes all lines (ways) that have the same start and end points after grouping, 
    i.e., lines that have been reduced to a single point.

    Parameters:
    - data (DataFrame): The original dataset containing all lines.
    - node1_col (str): The column name representing the start point of the line.
    - node2_col (str): The column name representing the end point of the line.

    Returns:
    - data (GeoDataFrame): A new dataset after removing singular lines.
    - data_singular_ways (DataFrame): A subset of the original dataset containing the removed singular lines.
    """
    
    start_time = time.time()
    print("Start deleting ways which have the same endpoints after grouping...")

    # Identify rows where the start and end points are the same, i.e., "singular lines"
    singular_ways_indices = data[data[node1_col] == data[node2_col]].index

    # Extract all singular lines from the original data
    data_singular_ways = data.loc[singular_ways_indices].copy()

    # Remove the singular lines from the original data
    data = data.drop(singular_ways_indices).reset_index(drop=True)

    print(f"   ... {len(singular_ways_indices)} ways were deleted!")
    print(f"   ... finished! ({time.time() - start_time:.3f} seconds)")

    return gpd.GeoDataFrame(data, geometry='geometry'), data_singular_ways


def add_lineID_clone_ways(data, country_code='VN'):
    """
    Creates a unique 'LineID' for each way element in the dataset.
    If a way needs to be cloned (has more than one system), it will be duplicated, tripled, or quadrupled.
    
    Parameters:
    - data (DataFrame): Input dataset containing way elements.
    - country_code (str): Two-letter country code.
    
    Returns:
    - DataFrame: New dataset with cloned ways and 'LineID' column.
    """
    start_time = time.time()
    print('Start adding "LineID" and cloning ways...')
    
    # Fill NaN values with 1 (indicating no clone) and convert to int
    data['systems'] = data['systems'].fillna(1).astype(int)

    # Create unique LineID prefix
    lineID_prefix = f'LINE{country_code}'
    
    # Initialize list for new data
    data_new = []
    
    # Process each row in data
    for i, row in data.iterrows():
        num_clones = row['systems'] # Determine number of clones based on 'systems' value
        base_lineID = f"{lineID_prefix}{str(i+1).zfill(4)}"  # Base LineID with four digits
        
        if num_clones == 1:
            # For rows where systems = 1, add only the base LineID
            row['LineID'] = base_lineID
            data_new.append(row)  # Add the original row to data_new
       
        else:
            # For rows where systems > 1, create clones as per 'systems' 
            # and add suffixes 'a', 'b', 'c', 'd' as needed
            clones = [row.copy() for _ in range(num_clones)]
            for j, clone in enumerate(clones):
                clone['LineID'] = f"{base_lineID}{chr(97 + j)}"  # Append 'a', 'b', 'c', 'd'
                data_new.append(clone)
    
    # Convert list of expanded data back to a DataFrame
    data_new = pd.DataFrame(data_new).reset_index(drop=True)

    # Print cloning summary
    print(f"   ... {sum(row['systems'] == 2 for _, row in data.iterrows())} ways doubled, "
          f"{sum(row['systems'] == 3 for _, row in data.iterrows())} tripled, "
          f"{sum(row['systems'] == 4 for _, row in data.iterrows())} quadrupled.")
    print(f'   ... finished! ({time.time() - start_time:.3f} seconds) \n')
    
    return gpd.GeoDataFrame(data_new, geometry='geometry')


def fill_line_voltage(gdf):
    """
    Fill missing Voltage values in the dataframe by finding matching rows
    with the same fromNode and toNode (in either order) that have a non-null voltage.

    Parameters:
        gdf (pd.DataFrame): Input dataframe with columns 'ID_node1_final', 'ID_node2_final', and 'voltage'.

    Returns:
        pd.DataFrame: The dataframe with missing voltage values filled.
    """
    for index, row in gdf.iterrows():
        if pd.isna(row['voltage']):
            # Find rows where ID_node1_final and ID_node2_final match (in either direction)
            matching_rows = gdf[((gdf['ID_node1_final'] == row['ID_node1_final']) & (gdf['ID_node2_final'] == row['ID_node2_final'])) |
                                 ((gdf['ID_node1_final'] == row['ID_node2_final']) & (gdf['ID_node2_final'] == row['ID_node1_final']))]
            # Extract voltage values from matching rows
            for _, match in matching_rows.iterrows():
                if not pd.isna(match['voltage']):
                    gdf.at[index, 'voltage'] = match['voltage']
                    break
    return gdf


def create_unique_index(df, id_column):
    """
    Creates unique indices for a given column by appending alphabetical suffixes to duplicates.

    Parameters:
    - df: DataFrame containing the column to process.
    - id_column: Column name for which unique indices are created.

    Returns:
    - List of unique indices as strings.
    """    
    counts = df[id_column].value_counts()
    suffix = list(string.ascii_lowercase)  # Suffix for duplicates
    indices = []

    for node_id in df[id_column]:
        count = counts[node_id]
        if count == 1:
            indices.append(f"{int(node_id):04d}")
        else:
            position = sum([1 for i in indices if i.startswith(f"{int(node_id):04d}")])
            indices.append(f"{int(node_id):04d}{suffix[position]}")

    return indices


def export_data(data, output_dir, buffer_distance=500, export_excel_country_code='VN'):
    """ 
    Exports the data to Excel and GeoPackage formats, adding unique indices for NodeID.

    Parameters:  
    - data: DataFrame containing the dataset to export. 
    - output_dir: Directory for saving exported files.
    - buffer_distance: Distance used for naming files.
    - export_excel_country_code: Country code to be used for naming. 

    Returns:
    - None
    """
    print('Start exporting data to Excel files... (may take a few seconds)')
    data = data.copy()

    # Ensure numeric columns are of proper types
    for col in data.select_dtypes(include=["Float32", "Int64"]).columns:
        data[col] = data[col].astype(np.float64 if "float" in str(data[col].dtype) else np.int64)

    data['fromNode'] = data['ID_node1_final'].astype(int)
    data['toNode'] = data['ID_node2_final'].astype(int)

    # Prepare the main data export
    data['Annotation'] = ''

    # Create strings for the Annotation "Bemerkung" column
    for index, row in data.iterrows():
        annotations = []

        if pd.isna(row['vlevels']) or row['vlevels'] != 1:
            annotations.append("multiple vlevels")

        if row['systems'] == 2:
            annotations.append("6 cables - 2 systems")
        elif row['systems'] == 3:
            annotations.append("9 cables - 3 systems")
        elif row['systems'] == 4:
            annotations.append("12 cables - 4 systems")

        if row['dc_candidate']:
            annotations.append("potentially DC")

        data.at[index, 'Annotation'] = ', '.join(annotations) if annotations else ' '

    data['Voltage'] = data['voltage'] / 1000  # Convert voltage to kV
    data['Country'] = export_excel_country_code
    data['R'] = ''
    data['XL'] = ''
    data['XC'] = ''
    data['Itherm'] = ''
    data['Capacity'] = ''

    table_lines = data.drop(columns=['voltage'])

    desired_order = [
        'Country', 'osm_id', 'LineID', 'fromNode', 'toNode', 
        'Voltage', 'Length', 'R', 'XL', 'XC', 'Itherm',
        'Capacity', 'frequency', 'Annotation', 'geometry'
    ]

    other_columns = [col for col in table_lines.columns if col not in desired_order]
    new_order = desired_order + other_columns
    table_lines = table_lines[new_order]

    # Generate filename for lines
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename_lines = os.path.join(output_dir, f"tbl_Lines_{export_excel_country_code}_{buffer_distance}m.xlsx")
    table_lines.to_excel(filename_lines, index=False)
    print(f'INFO: Exported lines to {filename_lines}')

    # Extract and deduplicate nodes
    node1_data = data[['ID_node1_final', 'Voltage', 'lon1_final', 'lat1_final']].rename(columns={
        'ID_node1_final': 'NodeID', 'lon1_final': 'lon', 'lat1_final': 'lat'})
    node2_data = data[['ID_node2_final', 'Voltage', 'lon2_final', 'lat2_final']].rename(columns={
        'ID_node2_final': 'NodeID', 'lon2_final': 'lon', 'lat2_final': 'lat'})

    endnodes_data = pd.concat([node1_data, node2_data])
    endnodes_data = endnodes_data.drop_duplicates(subset=['NodeID', 'Voltage']).reset_index(drop=True)

    # Add unique indices
    endnodes_data['Index'] = create_unique_index(endnodes_data, 'NodeID')

    # Convert voltage units and create geometry
    endnodes_data['geometry'] = endnodes_data.apply(
        lambda row: Point(row['lon'], row['lat']) if pd.notnull(row['lon']) and pd.notnull(row['lat']) else None, axis=1
    )

    # Generate filename for nodes
    filename_nodes = os.path.join(output_dir, f"tbl_Nodes_{export_excel_country_code}_{buffer_distance}m.xlsx")
    endnodes_data.to_excel(filename_nodes, index=False)
    print(f'INFO: Exported Nodes to {filename_nodes}')

    # Save nodes to GeoPackage
    gdf_nodes = gpd.GeoDataFrame(endnodes_data, geometry='geometry', crs='EPSG:32648')
    for col in gdf_nodes.select_dtypes(include=["Float32", "Int64"]).columns:
        gdf_nodes[col] = gdf_nodes[col].astype(np.float64 if "float" in str(gdf_nodes[col].dtype) else np.int64)

    gdf_nodes.to_file(os.path.join(output_dir, f"table_nodes_{buffer_distance}m.gpkg"), layer='nodes', driver='GPKG')

    # Save lines to GeoPackage
    gdf_lines = gpd.GeoDataFrame(table_lines, geometry='geometry', crs='EPSG:32648')
    for col in gdf_lines.columns:
        if isinstance(gdf_lines[col].iloc[0], tuple):
            gdf_lines[col] = gdf_lines[col].apply(lambda x: str(x) if isinstance(x, tuple) else x)

    gdf_lines.to_file(os.path.join(output_dir, f"table_lines_{buffer_distance}m.gpkg"), layer='lines', driver='GPKG')

    print('... finished!')


###############################################################
###############################################################
##################### MODULE 5: Plotting ######################
###############################################################
###############################################################

def plot_stacked_lines(data, bool_options):
    """
    This function plots lines from the geometry column and highlights stacked nodes.
    
    INPUT:
    - data: GeoDataFrame containing LineString geometries and node stack information.
    - bool_options: Dictionary with flags for plotting options (not used here but kept for flexibility).
    """
    # Convert node geometries (if stored as tuples) to Point objects
    def convert_to_point(geom):
        return Point(geom) if isinstance(geom, tuple) else geom

    # Ensure the CRS is EPSG:32648 (UTM Zone 48N), reproject if necessary
    if data.crs != 'EPSG:4326':
        print(f"Reprojecting CRS from {data.crs} to EPSG:32648.")
        data = data.to_crs(epsg=4326)

    data['geom_node1'] = data['geom_node1'].apply(convert_to_point)
    data['geom_node2'] = data['geom_node2'].apply(convert_to_point)

    # Set CRS for the node geometries and reproject to EPSG:4326 (lat/lon)
    data['geom_node1'] = gpd.GeoSeries(data['geom_node1'], crs="EPSG:32648").to_crs(epsg=4326)
    data['geom_node2'] = gpd.GeoSeries(data['geom_node2'], crs="EPSG:32648").to_crs(epsg=4326)

    # Extract latitudes and longitudes from the node geometries
    lat_node1, lon_node1 = zip(*[(geom.y, geom.x) for geom in data['geom_node1']])
    lat_node2, lon_node2 = zip(*[(geom.y, geom.x) for geom in data['geom_node2']])

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('All Lines with Stacked Endnodes', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True)

    # Plot the lines (using the 'geometry' column)
    data.plot(ax=ax, color='green', linewidth=1, linestyle='-', label="Lines")

    # Extract stacked node coordinates (based on the flags)
    lat_node1_stacked = [lat for lat, stacked in zip(lat_node1, data['node1_stacked']) if stacked]
    lon_node1_stacked = [lon for lon, stacked in zip(lon_node1, data['node1_stacked']) if stacked]
    
    lat_node2_stacked = [lat for lat, stacked in zip(lat_node2, data['node2_stacked']) if stacked]
    lon_node2_stacked = [lon for lon, stacked in zip(lon_node2, data['node2_stacked']) if stacked]

    # Plot stacked nodes with different markers
    ax.plot(lon_node1_stacked, lat_node1_stacked, 'xr', label="Node 1 Stacked", markersize=5)
    ax.plot(lon_node2_stacked, lat_node2_stacked, '+b', label="Node 2 Stacked", markersize=5)

    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()


def plot_ways_original(data, data_busbars, bool_options, data_singular_ways, voltage_levels_selected=[110000, 115000, 220000, 230000, 500000]):
    """
    This function plots the original dataset as it was with selected voltage levels.
    Two plots will be generated if the flag in bool_options is set:
    A plot with a lon/lat coordinate system and a plot with a more intuitive x/y plot in km.
    """
    data.set_crs(epsg=32648, inplace=True)
    data = data.to_crs(epsg=4326)
    data_busbars = gpd.GeoDataFrame(data_busbars, geometry='geometry', crs='EPSG:32648').to_crs(epsg=4326)
    data_singular_ways = gpd.GeoDataFrame(data_singular_ways, geometry='geometry', crs='EPSG:32648').to_crs(epsg=4326)

    print(data.crs, data_busbars.crs, data_singular_ways.crs)

    if bool_options.get('plot_ways_original', False):
        print('Start plotting original ways... (takes a few seconds)')
        
        # Create custom 12-color qualitative colormap for better distinctness
        colormap = np.array([
            [51, 160, 44], [31, 120, 180], [177, 89, 40], [106, 61, 154],
            [255, 127, 0], [178, 223, 138], [227, 26, 28], [255, 255, 153],
            [166, 206, 227], [202, 178, 214], [251, 154, 153], [253, 191, 111]
        ]) / 255
        
        # Create figure for deg Plot
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.set_title('Original Ways, Only Selected Voltages')
        ax.set_xlabel('Longitude [°]')
        ax.set_ylabel('Latitude [°]')
        
        # Add data_busbars with legend
        label_set_busbars = False
        for geom in data_busbars['geometry']:
            coords = np.array(geom.coords)
            lon = coords[:, 0]
            lat = coords[:, 1]
            if not label_set_busbars:
                ax.plot(lon, lat, 'cx-', color="#dec5e3", linewidth=1, label="Busbars")
                label_set_busbars = True
            else:
                ax.plot(lon, lat, 'cx-', color="#dec5e3", linewidth=1)

        # Add data_singular_ways with legend
        label_set_singular_ways = False
        for geom in data_singular_ways['geometry']:
            coords = np.array(geom.coords)
            lon = coords[:, 0]
            lat = coords[:, 1]
            if not label_set_singular_ways:
                ax.plot(lon, lat, 'kx-', color="#73d2de", linewidth=1, label="Deleted Ways")
                label_set_singular_ways = True
            else:
                ax.plot(lon, lat, 'kx-', color="#73d2de", linewidth=1)

        # Plot voltage levels with legend
        voltage_labels_added = set()  # Keep track of added labels
        for i_vlevel in range(len(voltage_levels_selected) - 1, -1, -1):
            i_colormap = i_vlevel - (i_vlevel // 12) * 12
            current_color = colormap[i_colormap]
            current_voltage = int(voltage_levels_selected[i_vlevel])
            label = f"{current_voltage // 1000} kV"  # Voltage level label
            
            # Get all ways with the current voltage level
            current_ways = data[data['voltage'] == current_voltage]
            
            # Plot the actual LineString data
            for geom in current_ways['geometry']:
                coords = np.array(geom.coords)
                lon = coords[:, 0]
                lat = coords[:, 1]
                if label not in voltage_labels_added:
                    ax.plot(lon, lat, '-o', color=current_color, markersize=1, label=label)
                    voltage_labels_added.add(label)
                else:
                    ax.plot(lon, lat, '-o', color=current_color, markersize=1)

        # Set plot limits
        xlim = (107.631, 107.640)
        ylim = (11.964, 11.9722)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        # Add legend
        ax.legend(loc='upper left', frameon=False)
        
        # Show the plot
        plt.show()
