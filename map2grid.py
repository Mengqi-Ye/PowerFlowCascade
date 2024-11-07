import os
os.environ['USE_PYGEOS'] = '0'

import json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point, LineString
from datetime import datetime


# Function to import JSON data
def import_json(file_path):
    """
    Imports data from a JSON file and extracts relevant elements.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    data_raw (list): All data from the imported JSON file.
    """
    print('Start importing Data (*.json file)...')
    
    # Read the JSON file
    with open(file_path, 'r') as file:
        data_raw_json = json.load(file)
    
    # Extract relevant elements
    data_raw = data_raw_json['elements']
    
    print('   ... finished!')

    return data_raw

# Function to separate nodes and ways and add UID
def separate_raw_data(data_raw):
    """
    Separates 'node' and 'way' elements from raw data and adds UID.

    Parameters:
    data_raw (list): Imported JSON data.

    Returns:
    data_nodes_all (list): All node elements.
    data_ways_all (list): All way elements.
    """
    start_time = time.time()
    print('Start separating raw data into way- and node-elements... (takes a few seconds)')
    
    data_nodes = []
    data_ways = []

    # Preallocation of counter variables
    num_node_elements = 0
    num_way_elements = 0

    # Separate nodes and ways
    for element in data_raw:
        if element['type'] == 'node':
            num_node_elements += 1
            data_nodes.append(element)
        elif element['type'] == 'way': # or 'relation'
            num_way_elements += 1
            data_ways.append(element)

    # Convert lists to DataFrames for easier manipulation
    data_nodes_df = pd.DataFrame(data_nodes)
    data_ways_df = pd.DataFrame(data_ways)

    # # Remove rows with missing fields
    # data_ways_df.dropna(inplace=True)
    # data_nodes_df.dropna(inplace=True)

    # Add UID
    # data_ways_df['UID'] = np.arange(1, len(data_ways_df) + 1) # 暂时删除！！！！！！！！！

    print(f'   ... finished! ({time.time() - start_time:.3f} seconds) \n')
    
    return data_nodes_df, data_ways_df

# Function to add coordinates and calculate lengths
def add_coordinates_length(data_ways_all, data_nodes_all, way_length_multiplier=1.2):
    """
    This function adds the coordinates of all nodes in the way elements
    and calculates the distance between start and end nodes.
    
    Parameters:
    data_ways_all: DataFrame with way elements
    data_nodes_all: DataFrame with all node elements
    
    Returns:
    data_ways_all: GeoDataFrame with updated way elements
    """

    print('Start adding coordinates to each way...')

    # Create a dictionary for quick access to node coordinates
    node_dict = {row['id']: (row['lon'], row['lat']) for _, row in data_nodes_all.iterrows()}

    # Initialize lists for coordinates and geometries
    all_geometries = []
    distances = []

    # Process each way element
    for _, way in data_ways_all.iterrows():
        # Extract node IDs for the current way
        node_ids = way['nodes']

        # Get coordinates for all nodes
        coords = [node_dict[node_id] for node_id in node_ids]

        # Create a LineString for the way
        line = LineString(coords)
        all_geometries.append(line)

    # Add geometries and distances to the DataFrame
    data_ways_all['geometry'] = all_geometries

    # Convert to GeoDataFrame and set CRS
    data_ways_all_gdf = gpd.GeoDataFrame(data_ways_all, geometry='geometry', crs="EPSG:4326")
    data_ways_all_gdf = data_ways_all_gdf.to_crs(epsg=32648)

    # # Calculate the distance between the first and last node
    # start_point = coords[0]
    # end_point = coords[-1]
    # distance = LineString([start_point, end_point]).length
    # distances.append(distance)

    # Calculate the distance of the entire route
    data_ways_all_gdf['Length'] = data_ways_all_gdf['geometry'].length * way_length_multiplier / 1000

    print('Finished adding coordinates!')
    return data_ways_all_gdf


def expand_tags_to_column(data):
    """
    Expands a column containing dictionaries into separate columns, 
    where each key-value pair in the dictionary becomes its own column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'tags' column, 
                       where each entry in 'tags' is a dictionary.
    tags_column (str): The name of the column containing dictionaries, default is 'tags'.
    
    Returns:
    pd.DataFrame: A DataFrame with the 'tags' column expanded, 
                  where each unique key from the dictionaries becomes a new column.
    """

    # Identify all unique keys across all dictionaries in the 'tags' column
    all_keys = set()
    for tags in data['tags']:
        if isinstance(tags, dict):
            all_keys.update(tags.keys())
    print("Unique keys found in tags:", all_keys)
    
    # Create a new column for each key, and fill it with the corresponding values from the dictionaries
    for key in all_keys:
        data[key] = data['tags'].apply(lambda x: x.get(key, '') if isinstance(x, dict) else '')
    
    # Drop the original 'tags' column (optional)
    data.drop(columns='tags', inplace=True)
    
    return data


def replace_empty_with_nan(df):
    """
    将DataFrame中的所有空字符串、仅含空格的字符串和None值替换为NaN。
    
    参数:
        df (pd.DataFrame): 待处理的DataFrame
        
    返回:
        pd.DataFrame: 处理后的DataFrame
    """
    # 替换空字符串或仅包含空格的字符串为NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # 替换None值为NaN
    df = df.where(pd.notnull(df), np.nan)
    
    return df


# def extract_unique_points(gdf):
#     """
#     从GeoDataFrame中提取唯一的坐标点。
    
#     参数:
#     gdf (GeoDataFrame): 包含LINESTRING geometry的GeoDataFrame
    
#     返回:
#     GeoDataFrame: 包含唯一坐标点的GeoDataFrame, 列包括 nodeID, lon, lat, geometry
#     """
#     # 用于存储唯一点的集合
#     unique_points = {}
#     point_id  = 0  # 用于生成唯一的 nodeID
    
#     # 遍历每一行的 geometry 并提取坐标点
#     for line in gdf['geometry']:
#         # 如果 geometry 是 LINESTRING，则提取坐标
#         if line.geom_type == 'LineString':
#             for x, y in line.coords:
#                 point = (x, y)
#                 # 如果该点还未存储到 unique_points 中，则添加
#                 if point not in unique_points:
#                     unique_points[point] = {
#                         "nodeID": point_id ,
#                         "lon": x,
#                         "lat": y,
#                         "geometry": Point(x, y)
#                     }
#                     point_id  += 1
    
#     # 创建包含唯一点的 GeoDataFrame
#     points_gdf = gpd.GeoDataFrame(list(unique_points.values()), 
#                                   columns=["nodeID", "lon", "lat", "geometry"], 
#                                   crs=gdf.crs)

#     return points_gdf


def extract_unique_endpoints(gdf):
    """
    从GeoDataFrame中提取每条LineString的唯一起点和终点坐标，优化版。
    
    参数:
    gdf (GeoDataFrame): 包含LINESTRING geometry的GeoDataFrame
    
    返回:
    GeoDataFrame: 包含唯一起点和终点的GeoDataFrame, 列包括 nodeID, lon, lat, geometry
    """
    unique_points = {}  # 用字典存储唯一点
    point_id = 0  # 用于生成唯一的 nodeID
    
    # 遍历每条 LineString，提取起点和终点
    for line in gdf['geometry']:
        if line.geom_type == 'LineString':
            # 获取起点和终点坐标
            start_point = line.coords[0]
            end_point = line.coords[-1]
            
            # 使用字典直接判断是否已经添加，避免重复添加
            for point in [start_point, end_point]:
                if point not in unique_points:
                    unique_points[point] = {
                        "nodeID": point_id,
                        "lon": point[0],
                        "lat": point[1],
                        "geometry": Point(point[0], point[1])
                    }
                    point_id += 1  # 递增 nodeID
    
    # 创建并返回包含唯一起点和终点的 GeoDataFrame
    return gpd.GeoDataFrame(list(unique_points.values()), 
                            columns=["nodeID", "lon", "lat", "geometry"], 
                            crs=gdf.crs)


def add_node_ids_to_lines(gdf, nodes_gdf):
    """
    为原始的GeoDataFrame中的每条LineString添加node1和node2列，表示每条线的起点和终点的nodeID。
    
    参数:
    gdf (GeoDataFrame): 包含LINESTRING geometry的原始GeoDataFrame
    unique_points_gdf (GeoDataFrame): 包含唯一坐标点的GeoDataFrame（包含nodeID等信息）
    
    返回:
    GeoDataFrame: 更新后的原始GeoDataFrame，添加了node1和node2列
    """
    # 创建一个坐标到nodeID的映射字典
    point_to_nodeid = { (row['lon'], row['lat']): row['nodeID'] for _, row in nodes_gdf.iterrows() }

    # 为每条LineString添加node1和node2列
    gdf['node1'] = gdf['geometry'].apply(lambda line: point_to_nodeid.get((line.coords[0][0], line.coords[0][1]), None))
    gdf['node2'] = gdf['geometry'].apply(lambda line: point_to_nodeid.get((line.coords[-1][0], line.coords[-1][1]), None))

    return gdf


# def count_voltage_levels(data):
#     """
#     Count voltage levels in ways and update the dataset accordingly.

#     Parameters:
#     - data: DataFrame containing way elements with a 'voltage' column.

#     Returns:
#     - data: updated dataset of all way elements with cloned rows for multiple voltage levels.
#     - voltage_levels_unique: unique voltage levels found in the dataset.
#     """
#     print('Start counting voltage levels...')
    
#     # 预定义的电压等级
#     voltage_levels = [110000, 220000, 500000]

#     # 为每个“way”元素提取和处理电压等级
#     for index, row in data.iterrows():
#         # 检查'voltage'字段
#         if 'voltage' not in row or pd.isna(row['voltage']):
#             print(f'ATTENTION! Way element UID {row["UID"]} does not contain a field "voltage". This way wont be selected.')
#             continue
        
#         # 解析电压等级
#         row_voltage_levels = str(row['voltage']).split(';')
#         row_voltage_levels = [int(v) for v in row_voltage_levels if v.isdigit()]

#         # 如果无法解析出有效电压等级，跳过该行
#         if len(row_voltage_levels) == 0:
#             print(f'ATTENTION! UNKNOWN voltage level in UID {row["UID"]}. This way wont be selected.')
#             continue
        
#         # 处理单一电压等级的行
#         if len(row_voltage_levels) == 1:
#             data.at[index, 'voltage'] = row_voltage_levels[0]
#             data.at[index, 'vlevels'] = 1
#         else:
#             # 多电压等级的行设置 'vlevels' 为电压等级数量
#             data.at[index, 'vlevels'] = len(row_voltage_levels)
#             print(f'ATTENTION! Multiple voltage levels in UID {row["UID"]}. This way will be duplicated.')

#             # 克隆多电压等级行，每个克隆行包含一个单一的电压等级
#             for v in row_voltage_levels:
#                 if v in voltage_levels:
#                     new_row = row.copy()
#                     new_row['voltage'] = v
#                     data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

#     # 统计每个电压等级的出现次数
#     voltage_levels_count = data['voltage'].value_counts()
#     print(voltage_levels_count)

#     # 计算缺少电压信息的行数
#     unknown_voltage_count = len(data) - voltage_levels_count.sum()
#     print(f'There are {unknown_voltage_count} way(s) with unknown voltage level.')
#     print('Finished!')
    
#     # 返回更新后的 data 和唯一电压等级列表
#     voltage_levels_unique = sorted(data['voltage'].dropna().unique(), reverse=True)
#     return data, voltage_levels_unique


def count_voltage_levels(data, voltage_levels_selected=[110000, 220000, 500000]):
    print('Start counting voltage levels...')

    data['vlevels'] = 0
    
    expanded_data = []
    voltage_levels_count = {}
    none_voltage_count = 0

    for _, row in data.iterrows():
        voltage = row['voltage']

        # 在处理每一行之前，初始化 vlevels
        vlevels = 0

        if isinstance(voltage, str) and voltage:
            if ';' in voltage:  # 处理包含多个电压等级的情况
                voltage_levels = voltage.split(';')
                valid_voltages = [int(v.strip()) for v in voltage_levels if v.strip().isdigit()]
                filtered_voltages = [v for v in valid_voltages if v in voltage_levels_selected]

                # 设置 vlevels
                vlevels = len(filtered_voltages)

                # 克隆并添加符合条件的电压等级行
                for v in filtered_voltages:
                    new_row = row.copy()
                    new_row['voltage'] = v  # 设置为单一电压值
                    new_row['vlevels'] = vlevels  # 添加 vlevels 列
                    expanded_data.append(new_row)

                    # 统计电压等级出现次数
                    voltage_levels_count[v] = voltage_levels_count.get(v, 0) + 1

            else:  # 只有一个电压等级的情况
                row['voltage'] = int(voltage.strip()) if voltage.strip().isdigit() else None
                if row['voltage'] in voltage_levels_selected:  # 只选择选定的电压等级
                    row['vlevels'] = 1  # 明确设置 vlevels
                    expanded_data.append(row)

                    # 统计电压等级出现次数
                    voltage_level = row['voltage']
                    if voltage_level is not None:
                        voltage_levels_count[voltage_level] = voltage_levels_count.get(voltage_level, 0) + 1
        else:
            # 处理没有电压等级的情况
            # print(f'WARNING: Way with ID {row.get("id", "Unknown ID")} does not have a voltage level.')
            none_voltage_count += 1  # 计数无电压等级的行
            row['voltage'] = None
            row['vlevels'] = 0  # 对于没有电压等级的行，设置 vlevels 为 0
            expanded_data.append(row)  # 在没有电压等级的情况下仍添加行

    # 将新数据列表转为 DataFrame
    expanded_data = pd.DataFrame(expanded_data, columns=data.columns)
    expanded_data['voltage'] = expanded_data['voltage'].astype('Int64')  # 处理缺失值
    expanded_data['vlevels'] = expanded_data['vlevels'].astype('Int64')  # 确保 vlevels 的类型

    # unique_voltage_levels = sorted(expanded_data['voltage'].dropna().unique(), reverse=True)

    # 打印电压等级统计信息
    print('Voltage levels count:')
    for v_level, count in voltage_levels_count.items():
        print(f'Voltage level {int(v_level/1000)} kV: {count}')
    
    print(f'Count of rows with None voltage: {none_voltage_count}')

    return expanded_data #, unique_voltage_levels


def delete_busbars(data, plot_histogram=False, busbar_max_length=0.5):
    """
    Delete busbars and bays from the dataset based on their length.
    
    Parameters:
        data (GeoDataFrame): Input dataset of selected ways.
        plot_histogram (bool): If True, plot a histogram of busbar lengths.
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
    if plot_histogram:
        plt.figure()
        plt.hist(lengths_of_busbars, bins=200)
        plt.title('Lengths of busbars/bays below busbar-max-length-threshold')
        plt.xlabel('Length [km]')
        plt.ylabel('Number of busbars with that length')
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
        # Check if "cables" field exists
        if 'cables' in row:
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


# def my_calc_distances_between_endpoints(data: pd.DataFrame, crs='EPSG:32648', plot_histogram: bool = False):
#     """
#     Calculate a distance matrix between endpoints of geometries in the given DataFrame.

#     Parameters:
#         data (DataFrame): Dataset of all selected ways with geometries.
#         crs (str): Coordinate reference system to use for the GeoDataFrame.
#         plot_histogram (bool): If True, plot histogram of distances.

#     Returns:
#         np.ndarray: Matrix with distances between all endpoints.
#     """

#     print('Start calculating distances between all endpoints... (takes a few seconds)')
    
#     # Convert DataFrame to GeoDataFrame
#     if 'geometry' not in data.columns:
#         raise ValueError("The input DataFrame must contain a 'geometry' column.")
    
#     gdf = gpd.GeoDataFrame(data, geometry='geometry', crs=crs)

#     # Preallocate the distance matrix with NaN elements
#     num_elements = len(gdf) * 2
#     M = np.full((num_elements, num_elements), np.nan)

#     # Fetch coordinates from the GeoDataFrame
#     all_points = gdf['geometry'].apply(lambda geom: list(geom.coords))  # Extract coordinates for each LineString
#     all_lon1, all_lat1 = zip(*[coords[0] for coords in all_points])  # First endpoint of each LineString
#     all_lon2, all_lat2 = zip(*[coords[-1] for coords in all_points])  # Last endpoint of each LineString

#     # Go through each row of the distance matrix
#     for i_row in range(len(gdf)):
#         # Create the current row data
#         lon1, lat1 = all_lon1[i_row], all_lat1[i_row]
#         lon2, lat2 = all_lon2[i_row], all_lat2[i_row]

#         # Prepare coordinates for calculation
#         data_row_lon = np.concatenate([all_lon1[i_row + 1:], all_lon2[i_row + 1:]])
#         data_row_lat = np.concatenate([all_lat1[i_row + 1:], all_lat2[i_row + 1:]])

#         # Calculate distances directly using geospatial distances
#         distances_to_lon1 = [gpd.points_from_xy([lon1], [lat1])[0].distance(gpd.points_from_xy([lon], [lat])[0]) for lon, lat in zip(data_row_lon, data_row_lat)]
#         distances_to_lon2 = [gpd.points_from_xy([lon2], [lat2])[0].distance(gpd.points_from_xy([lon], [lat])[0]) for lon, lat in zip(data_row_lon, data_row_lat)]

#         # Apply the newly calculated distance row to the distance matrix
#         M[i_row * 2, i_row * 2 + 1:] = [-1] + distances_to_lon1  # Set diagonal element to -1
#         M[i_row * 2 + 1, i_row * 2 + 1:] = [-1] + distances_to_lon2  # Set diagonal element to -1

#     # Plot a histogram of all the distances if required
#     if plot_histogram:
#         print('... start visualizing all distances in a histogram ...')
#         plt.figure(figsize=(10, 12))
        
#         plt.subplot(5, 1, 1)
#         plt.hist(M[~np.isnan(M)], bins=200)
#         plt.title('Distances between all endnodes')
#         plt.ylabel('number of pairs')
#         plt.xlabel('distance [m]')  # 由于我们在 CRS 中计算，单位将是米
        
#         plt.subplot(5, 1, 2)
#         plt.hist(M[~np.isnan(M)], bins=200, range=(0, 10000))
#         plt.ylabel('number of pairs')
#         plt.xlabel('distance [m]')

#         plt.subplot(5, 1, 3)
#         plt.hist(M[~np.isnan(M)], bins=400, range=(-1.5, 2000))
#         plt.ylabel('number of pairs')
#         plt.xlabel('distance [m]')

#         plt.subplot(5, 1, 4)
#         plt.hist(M[~np.isnan(M)], bins=300, range=(0, 300))
#         plt.ylabel('number of pairs')
#         plt.xlabel('distance [m]')

#         plt.subplot(5, 1, 5)
#         plt.hist(M[~np.isnan(M)], bins=300, range=(0 + np.finfo(float).eps, 300))
#         plt.ylabel('number of pairs')
#         plt.xlabel('distance [m]')

#         plt.tight_layout()
#         plt.show()

#     print('... finished!')

#     return M


def my_calc_distances_between_endpoints(data, plot_histogram=False):
   
    data = gpd.GeoDataFrame(data).set_crs(epsg=3426, inplace=True)
    data = data.to_crs(epsg=32648)

    # Initialize distance matrix
    n = len(data)
    M = np.full((2 * n, 2 * n), np.nan)  # Initialize the distance matrix with NaN

    # Extract start and end points
    start_points = np.array([way.coords[0] for way in data.geometry])
    end_points = np.array([way.coords[-1] for way in data.geometry])

    # Combine start and end points
    # all_points = np.vstack((start_points, end_points))
    all_points = np.empty((2 * n, 2))  # 每个点有两个坐标 (x, y)
    all_points[0::2] = start_points     # 奇数行为起点
    all_points[1::2] = end_points       # 偶数行为终点

    # Calculate distances using cdist
    distances = cdist(all_points, all_points) / 1000

    # # 填充对角块的上三角部分和对角线
    # # 对角线为-1，计算线的start_point到end_point的距离
    # for i in range(n):
    #     M[2 * i, 2 * i] = -1       # 起点到自身
    #     M[2 * i + 1, 2 * i + 1] = -1  # 终点到自身
        
    #     # 填充起点和终点之间的距离
    #     M[2 * i, 2 * i + 1] = distances[2 * i, 2 * i + 1]  # 同条路径的起点到终点
    #     M[2 * i + 1, 2 * i] = distances[2 * i + 1, 2 * i]  # 同条路径的终点到起点

    #     # 填充不同路径的距离
    #     for j in range(i + 1, n):
    #         # 起点和终点之间的距离
    #         M[2 * i, 2 * j] = distances[2 * i, 2 * j]      # 起点到起点
    #         M[2 * i, 2 * j + 1] = distances[2 * i, 2 * j + 1]  # 起点到终点
    #         M[2 * i + 1, 2 * j] = distances[2 * i + 1, 2 * j]  # 终点到起点
    #         M[2 * i + 1, 2 * j + 1] = distances[2 * i + 1, 2 * j + 1]  # 终点到终点

    # 填充对角块的上三角部分和对角线
    # 对角块为-1，不计算线的start_point到end_point的距离
    for i in range(n):
        # 设置同一路径的对角块
        M[2 * i, 2 * i + 1] = -1       # 起点到终点距离为 -1
        M[2 * i + 1, 2 * i] = -1       # 终点到起点距离为 -1
        M[2 * i, 2 * i] = -1           # 起点到自身距离为 -1
        M[2 * i + 1, 2 * i + 1] = -1   # 终点到自身距离为 -1

        # 填充不同路径之间的起点和终点距离
        for j in range(i + 1, n):
            M[2 * i, 2 * j] = distances[2 * i, 2 * j]         # 起点到起点距离
            M[2 * i, 2 * j + 1] = distances[2 * i, 2 * j + 1] # 起点到终点距离
            M[2 * i + 1, 2 * j] = distances[2 * i + 1, 2 * j] # 终点到起点距离
            M[2 * i + 1, 2 * j + 1] = distances[2 * i + 1, 2 * j + 1] # 终点到终点距离

    return M


def my_calc_stacked_endnodes(data, distances, bool_options=False):
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

    # # Visualize this stacked data
    # if bool_options.get('plot_stacked_endnodes', False):
    #     plt.figure()
    #     plt.title('All ways with endnodes STACKED on XY-Map')
    #     plt.grid(True)
    #     plt.xlabel('x - distance from midpoint [km]')
    #     plt.ylabel('y - distance from midpoint [km]')
        
    #     x = np.concatenate([data['x1'], data['x2']])
    #     y = np.concatenate([data['y1'], data['y2']])
        
    #     # Extract node1 if it is stacked, else ignore it    
    #     x_node1_stacked = x[data['node1_stacked']]
    #     y_node1_stacked = y[data['node1_stacked']]
        
    #     # Extract node2 if it is stacked, else ignore it
    #     x_node2_stacked = x[data['node2_stacked']]
    #     y_node2_stacked = y[data['node2_stacked']]
        
    #     plt.plot(x, y, 'ok-')
    #     plt.plot(x_node1_stacked, y_node1_stacked, 'xr')
    #     plt.plot(x_node2_stacked, y_node2_stacked, '+b')
    #     plt.show()

    # # Plot histogram of how many endnodes are stacked
    # if bool_options.get('histogram_stacked_endnodes', False):
    #     plt.figure()
    #     plt.hist(unique_counts + 1, bins=np.arange(1, unique_counts.max() + 2) - 0.5, edgecolor='black')
    #     plt.title('Stacked endnodes: If stacked, how many are stacked?')
    #     plt.xlabel('Nodes stacked on top of each other')
    #     plt.ylabel('Number of different positions this occurs in')
    #     plt.show()

    return data, nodes_stacked_pairs


def my_calc_neighbouring_endnodes(data, distances, neighbourhood_threshold=0.5, bool_options=False):
    """
    This function searches every distance combination between all endpoints 
    which have a distance value bigger than "0" and lower than the threshold 
    in "neighbourhood_threshold", indicating two endpoints are in the vicinity.

    Parameters:
    - data: input dataset (DataFrame)
    - distances: distance matrix (DataFrame)
    - neighbourhood_threshold (km): threshold to determine if an endnode is in neighbourhood
    - bool_options: dictionary containing visualization options

    Returns:
    - updated data: DataFrame with new flags for endnode1/2 being neighbours
    - nodes_neighbouring_pairs: list of all pairs of neighbouring endnodes
    """
    print('Start finding all neighbouring endnodes...')
    
    # Create a boolean mask for distances in the neighbourhood range
    b_dist_neighbourhood = (distances < neighbourhood_threshold) & (distances > 0)
    
    # Check if any elements are in the neighbourhood
    if not np.any(b_dist_neighbourhood):
        data['node1_neighbour'] = False
        data['node2_neighbour'] = False
        print('... no endnode is in a neighbourhood!')
        return data, []

    # Get indices of the boolean matrix
    nodes_neighbour_indices = np.argwhere(b_dist_neighbourhood)
    
    # Extract unique indices and their occurrences
    dist_combined = np.concatenate((nodes_neighbour_indices[:, 0], nodes_neighbour_indices[:, 1]))
    unique, unique_counts = np.unique(dist_combined, return_counts=True)
    
    print(f'... {len(unique)} endnodes are in the same neighbourhood!')

    # Create a DataFrame for neighbouring nodes
    nodes_neighbouring = pd.DataFrame({
        'index': unique,
        'way_ID': unique // 2,  # np.ceil(unique / 2).astype(int),
        'endnode1': unique % 2 == 0
    })

    # Initialize pairs of neighbouring nodes
    nodes_neighbouring_pairs = nodes_neighbour_indices.tolist()
    
    # Initialize flags in the data DataFrame
    data['node1_neighbour'] = False
    data['node2_neighbour'] = False

    # Iterate over the data to check for neighbourhood
    i_neighbouring_nodes = 0
    numel_way_IDs = len(nodes_neighbouring)

    for i_ways in range(len(data)):
        if i_neighbouring_nodes >= numel_way_IDs:
            break
        
        if i_ways == nodes_neighbouring['way_ID'].iloc[i_neighbouring_nodes]:
            # Check if both endnodes are neighbours
            if (i_neighbouring_nodes < numel_way_IDs - 1 and 
                nodes_neighbouring['way_ID'].iloc[i_neighbouring_nodes] == 
                nodes_neighbouring['way_ID'].iloc[i_neighbouring_nodes + 1]):
                
                data.at[i_ways, 'node1_neighbour'] = True
                data.at[i_ways, 'node2_neighbour'] = True
                i_neighbouring_nodes += 1  # Skip one index
            
            elif nodes_neighbouring['endnode1'].iloc[i_neighbouring_nodes]:
                data.at[i_ways, 'node1_neighbour'] = True
                data.at[i_ways, 'node2_neighbour'] = False
            else:
                data.at[i_ways, 'node1_neighbour'] = False
                data.at[i_ways, 'node2_neighbour'] = True
            
            i_neighbouring_nodes += 1
        else:
            data.at[i_ways, 'node1_neighbour'] = False
            data.at[i_ways, 'node2_neighbour'] = False
    
    print('... finished!')

    # Visualize neighbouring data if specified
    if bool_options.get('plot_neighbouring_endnodes', False):
        print('Start visualizing all neighbouring endnodes (takes a few seconds)...')
        
        # Extract nodes' coordinates
        x = np.concatenate([data['x1'], data['x2']])
        y = np.concatenate([data['y1'], data['y2']])
        
        # Extract node1 and node2 if in neighbourhood
        x_node1_neighbour = x[data['node1_neighbour']]
        y_node1_neighbour = y[data['node1_neighbour']]
        
        x_node2_neighbour = x[data['node2_neighbour']]
        y_node2_neighbour = y[data['node2_neighbour']]

        # Plot all nodes
        plt.figure()
        plt.title('All ways with endnodes NEIGHBOURING on XY-Map')
        plt.grid(True)
        plt.xlabel('x - distance from midpoint [km]')
        plt.ylabel('y - distance from midpoint [km]')
        
        plt.plot(x, y, 'ok-')
        plt.plot(x_node1_neighbour, y_node1_neighbour, '*g', label='Node 1 Neighbour')
        plt.plot(x_node2_neighbour, y_node2_neighbour, '*g', label='Node 2 Neighbour')
        plt.legend()
        plt.show()

    # Plot histogram if specified
    if bool_options.get('histogram_neighbouring_endnodes', False):
        plt.figure()
        plt.hist(unique_counts + 1, bins=np.arange(1, max(unique_counts) + 2) - 0.5, rwidth=0.8)
        plt.title('Neighbouring endnodes: How many will be in one group?')
        plt.xlabel('Number of nodes which will be grouped together')
        plt.ylabel('Number of different positions this occurs in')
        plt.show()

    return data, nodes_neighbouring_pairs


def my_group_nodes(pairs_input):
    """
    Group nodes based on pairs of connections.
    
    Parameters:
    pairs_input : list of pairs
        Each element is a tuple (or list) containing a pair of nodes that are connected.
        
    Returns:
    list_groups : list of lists
        A list where each element is a list representing a group of connected nodes.
    """
    start_time = time.time()
    print(f'Start grouping all pairs from "pairs_input" (may take a few seconds)...')

    # Initialize empty list
    list_groups = []
    
    # Sort each pair for consistency
    pairs_sorted = [sorted(pair) for pair in pairs_input]
    
    # Sort pairs by the first element for easier grouping
    pairs_sorted.sort()

    # Go through each pair to group connected nodes
    for partner1, partner2 in pairs_sorted:
        row_partner1 = row_partner2 = None

        # Check if partner1 and partner2 already belong to any groups
        for i, group in enumerate(list_groups):
            if partner1 in group:
                row_partner1 = i
            if partner2 in group:
                row_partner2 = i

        if row_partner1 is not None:
            if row_partner2 is not None:
                if row_partner1 == row_partner2:
                    continue  # Both are in the same group, no action needed
                else:
                    # Merge groups and remove the redundant group
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
    # 新增列，初始化为 NaN
    data['ID_node1_grouped'] = np.nan
    data['ID_node2_grouped'] = np.nan
    data['lon1_grouped'] = np.nan
    data['lat1_grouped'] = np.nan
    data['lon2_grouped'] = np.nan
    data['lat2_grouped'] = np.nan

    # 遍历 stacked group
    for group in nodes_stacked_grouped:
        # 获取当前 group 的首个成员 node ID
        first_node_id = group[0]
        way_id = first_node_id // 2  # 在 Python 中，节点 ID 从 0 开始
        is_endnode1 = (first_node_id % 2) == 0  # 偶数 ID 判定为起点

        # 获取起点或终点的坐标信息
        if is_endnode1:
            # 起点
            grouped_node_id = data.at[way_id, 'node1']
            grouped_lon, grouped_lat = data.at[way_id, 'geometry'].coords[0]
            
            # # 从 points_gdf 中读取该节点的坐标
            # point_row = points_gdf[points_gdf['nodeID'] == grouped_node_id]
            # if not point_row.empty:
            #     points_lon = point_row.iloc[0]['lon']
            #     points_lat = point_row.iloc[0]['lat']
            #     # 比较两种方式获取的经纬度信息
            #     if np.isclose(grouped_lon, points_lon) and np.isclose(grouped_lat, points_lat):
            #         continue
            #     else:
            #         print(f"Node {grouped_node_id} 的经纬度信息不一致: "
            #             f"geometry中为({grouped_lon}, {grouped_lat}), "
            #             f"points_gdf中为({points_lon}, {points_lat})")
        else:
            # 终点
            grouped_node_id = data.at[way_id, 'node2']
            grouped_lon, grouped_lat = data.at[way_id, 'geometry'].coords[-1]

            # point_row = points_gdf[points_gdf['nodeID'] == grouped_node_id]
            # if not point_row.empty:
            #     points_lon = point_row.iloc[0]['lon']
            #     points_lat = point_row.iloc[0]['lat']
                # # 比较两种方式获取的经纬度信息
                # if np.isclose(grouped_lon, points_lon) and np.isclose(grouped_lat, points_lat):
                #     continue
                # else:
                #     print(f"Node {grouped_node_id} 的经纬度信息不一致: "
                #         f"geometry中为({grouped_lon}, {grouped_lat}), "
                #         f"points_gdf中为({points_lon}, {points_lat})")
                        
        # 更新 group 中的所有成员
        for member_node_id in group:
            way_id = member_node_id // 2  # 计算对应的 way ID
            is_endnode1 = (member_node_id % 2) == 0  # 偶数 ID 判定为起点

            if is_endnode1:
                # 更新起点
                data.at[way_id, 'ID_node1_grouped'] = grouped_node_id
                data.at[way_id, 'lon1_grouped'] = grouped_lon
                data.at[way_id, 'lat1_grouped'] = grouped_lat
            else:
                # 更新终点
                data.at[way_id, 'ID_node2_grouped'] = grouped_node_id
                data.at[way_id, 'lon2_grouped'] = grouped_lon
                data.at[way_id, 'lat2_grouped'] = grouped_lat

    print("Completed updating coordinates for stacked groups.")
    return data

###### IGNORE my_group_neighbouring_endnodes() function in original matlab program ######

def my_add_final_coordinates(data):
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
    
    # Iterate through each way
    for i_ways in range(len(data)):
        start_node_id = data.at[i_ways, 'node1']  # 起点 ID
        end_node_id = data.at[i_ways, 'node2']    # 终点 ID
        geometry = data.at[i_ways, 'geometry']

        # 检查是否有新的节点 1，若没有则从原数据中提取
        if pd.isna(data.at[i_ways, 'ID_node1_grouped']):
            data.at[i_ways, 'ID_node1_final'] = start_node_id
            data.at[i_ways, 'lon1_final'] = geometry.coords[0][0]  # 从 geometry 中提取经度
            data.at[i_ways, 'lat1_final'] = geometry.coords[0][1]  # 从 geometry 中提取纬度
        else:
            data.at[i_ways, 'ID_node1_final'] = data.at[i_ways, 'ID_node1_grouped']
            data.at[i_ways, 'lon1_final'] = data.at[i_ways, 'lon1_grouped']
            data.at[i_ways, 'lat1_final'] = data.at[i_ways, 'lat1_grouped']

        # 检查是否有新的节点 2，若没有则从原数据中提取
        if pd.isna(data.at[i_ways, 'ID_node2_grouped']):
            data.at[i_ways, 'ID_node2_final'] = end_node_id
            data.at[i_ways, 'lon2_final'] = geometry.coords[-1][0]  # 从 geometry 中提取经度
            data.at[i_ways, 'lat2_final'] = geometry.coords[-1][1]  # 从 geometry 中提取纬度
        else:
            data.at[i_ways, 'ID_node2_final'] = data.at[i_ways, 'ID_node2_grouped']
            data.at[i_ways, 'lon2_final'] = data.at[i_ways, 'lon2_grouped']
            data.at[i_ways, 'lat2_final'] = data.at[i_ways, 'lat2_grouped']

    print('... finished!')
    return data

# def my_delete_singular_ways(data):
#     """
#     删除所有在分组后具有相同起点和终点的线路（ways），即已收缩成一个点的线路。
    
#     Parameters:
#     - data (DataFrame): 包含所有线路的原始数据。
    
#     Returns:
#     - data (DataFrame): 删除单一性线路后的新数据集。
#     - data_singular_ways (DataFrame): 包含被删除的单一性线路的集合。
#     """
#     start_time = time.time()
#     print("Start deleting ways which have the same endpoints after grouping...")

#     # 找到起点和终点相同的行的索引，即“单一性线路”
#     singular_ways_indices = data[data['ID_node1_final'] == data['ID_node2_final']].index

#     # 从原始数据中提取所有单一性线路
#     data_singular_ways = data.loc[singular_ways_indices].copy()

#     # 从原始数据中删除这些单一性线路
#     data = data.drop(singular_ways_indices).reset_index(drop=True)

#     # 输出删除信息
#     print(f"   ... {len(singular_ways_indices)} ways were deleted!")
#     print(f"   ... finished! ({time.time() - start_time:.3f} seconds)")

#     return data, data_singular_ways


def my_delete_singular_ways(data, node1_col, node2_col):
    """
    删除所有在分组后具有相同起点和终点的线路（ways），即已收缩成一个点的线路。
    
    Parameters:
    - data (DataFrame): 包含所有线路的原始数据。
    
    Returns:
    - data (DataFrame): 删除单一性线路后的新数据集。
    - data_singular_ways (DataFrame): 包含被删除的单一性线路的集合。
    """
    start_time = time.time()
    print("Start deleting ways which have the same endpoints after grouping...")

    # 找到起点和终点相同的行的索引，即“单一性线路”
    singular_ways_indices = data[data[node1_col] == data[node2_col]].index

    # 从原始数据中提取所有单一性线路
    data_singular_ways = data.loc[singular_ways_indices].copy()

    # 从原始数据中删除这些单一性线路
    data = data.drop(singular_ways_indices).reset_index(drop=True)

    # 输出删除信息
    print(f"   ... {len(singular_ways_indices)} ways were deleted!")
    print(f"   ... finished! ({time.time() - start_time:.3f} seconds)")

    return data, data_singular_ways


# def add_lineID_clone_ways(data, country_code):
#     """
#     Creates a unique 'LineID' for each way element in the dataset.
#     If a way needs to be cloned (has more than one system), it will be duplicated, tripled, or quadrupled.
    
#     Parameters:
#     - data (DataFrame): Input dataset containing way elements.
#     - country_code (str): Two-letter country code.
    
#     Returns:
#     - DataFrame: New dataset with cloned ways and 'LineID' column.
#     """
#     from time import time
#     start_time = time()
#     print('Start adding "LineID" and cloning ways...')

#     # Initialize variables
#     num_of_doubled_ways = 0
#     num_of_tripled_ways = 0
#     num_of_quadrupled_ways = 0
    
#     # Create unique LineID
#     LineID_prefix = f'LINE{country_code}'
#     data['LineID'] = [f"{LineID_prefix}{str(i+1).zfill(4)}" for i in range(len(data))]

#     # Create a new list to hold the expanded data
#     data_new = []
    
#     for index, row in data.iterrows():
#         # Base LineID without suffix
#         base_lineID = row['LineID']
        
#         # Clone the way based on the 'systems' value
#         if row['systems'] == 2:
#             row_a = row.copy()
#             row_b = row.copy()
#             row_a['LineID'] = f"{base_lineID}a"
#             row_b['LineID'] = f"{base_lineID}b"
#             data_new.extend([row_a, row_b])
#             num_of_doubled_ways += 1
            
#         elif row['systems'] == 3:
#             row_a = row.copy()
#             row_b = row.copy()
#             row_c = row.copy()
#             row_a['LineID'] = f"{base_lineID}a"
#             row_b['LineID'] = f"{base_lineID}b"
#             row_c['LineID'] = f"{base_lineID}c"
#             data_new.extend([row_a, row_b, row_c])
#             num_of_tripled_ways += 1
            
#         elif row['systems'] == 4:
#             row_a = row.copy()
#             row_b = row.copy()
#             row_c = row.copy()
#             row_d = row.copy()
#             row_a['LineID'] = f"{base_lineID}a"
#             row_b['LineID'] = f"{base_lineID}b"
#             row_c['LineID'] = f"{base_lineID}c"
#             row_d['LineID'] = f"{base_lineID}d"
#             data_new.extend([row_a, row_b, row_c, row_d])
#             num_of_quadrupled_ways += 1
            
#         else:
#             # If 'systems' is 1, no cloning needed
#             data_new.append(row)

#     # Convert the list of expanded data back into a DataFrame
#     data_new = pd.DataFrame(data_new).reset_index(drop=True)

#     # Print cloning summary
#     print(f'   ... {num_of_doubled_ways} ways have been doubled, {num_of_tripled_ways} tripled, {num_of_quadrupled_ways} quadrupled.')
#     print(f'   ... finished! ({time() - start_time:.3f} seconds) \n')
    
#     return data_new

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
    
    return data_new


# def export_to_excel(data, export_excel_country_code='VN'): #, neighbourhood_threshold=0.5
#     """
#     Exports the data to two Excel files. This function processes the input DataFrame,
#     retains the original node IDs, and generates relevant columns for export.

#     Parameters:
#     - data: DataFrame containing the dataset to export.
#     - export_excel_country_code: Country code to be used for naming.
#     - neighbourhood_threshold: Threshold for naming files.
#     - way_length_multiplier: Multiplier for adjusting line lengths.

#     Returns:
#     - None
#     """

#     print('Start exporting data to Excel files... (may take a few seconds)')

#     data['fromNode'] = data['osmID_node1']
#     data['toNode'] = data['osmID_node2']

#     # Prepare the main data export
#     data['Annotation'] = ''
    
#     # Create strings for the Annotation "Bemerkung" column
#     for index, row in data.iterrows():
#         annotations = []
        
#         # Check for multiple voltage levels
#         if row['vlevels'] != 1:
#             annotations.append("multiple vlevels")
        
#         # Check for systems
#         if row['systems'] == 2:
#             annotations.append("6 cables - 2 systems")
#         elif row['systems'] == 3:
#             annotations.append("9 cables - 3 systems")
#         elif row['systems'] == 4:
#             annotations.append("12 cables - 4 systems")
        
#         # Check for DC candidate
#         if row['dc_candidate']:
#             annotations.append("potentially DC")

#         # Join annotations
#         data.at[index, 'Annotation'] = ', '.join(annotations) if annotations else ' '

#     # Prepare export columns
#     data['Voltage'] = data['voltage'] / 1000  # Convert voltage to kV
#     # data['Length'] = data['distance_m']

#     # print(type(data['distance_m']))
#     # Compensate for slack
#     # data['Length'] = round(data['distance_m'] * float(way_length_multiplier) / 1000, 2)

#     # Prepare final columns
#     data['Country'] = export_excel_country_code
#     # data['LineID'] = data['LtgsID']  # Rename LtgsID to LineID

#     data['R'] = ''
#     data['XL'] = ''
#     data['XC'] = ''
#     data['Itherm'] = ''
#     data['Capacity'] = ''
    
#     table_lines = data[list(data.columns)].drop(columns=['voltage'])

#     desired_order = [
#         'Country', 'osm_id', 'LineID', 'fromNode', 'toNode', 
#         'Voltage', 'Length', 'R', 'XL', 'XC', 'Itherm',
#         'Capacity', 'frequency', 'Annotation', 'geometry'
#     ]

#     # 选择所需的列，并重新排列
#     # '其他列' 将是 DataFrame 中除了上述列以外的所有列
#     other_columns = [col for col in table_lines.columns if col not in desired_order]
#     new_order = desired_order + other_columns  # 将其他列添加到新顺序的末尾

#     # 重新排列列
#     table_lines = table_lines[new_order]

#     # Generate filename
#     timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
#     # filename_lines = f"../outputs/tbl_Lines_{export_excel_country_code}_thres-{neighbourhood_threshold}km_{timestamp}.xlsx"
#     filename_lines = f"../outputs/tbl_Lines_{export_excel_country_code}.xlsx"

#     # Export to Excel
#     table_lines.to_excel(filename_lines, index=False)
#     print(f'INFO: Exported lines to {filename_lines}')

#     # # Assign relevant columns
#     # node1_data = data[['ID_node1_final', 'voltage', 'lon1_final', 'lat1_final']].copy()
#     # node2_data = data[['ID_node2_final', 'voltage', 'lon2_final', 'lat2_final']].copy()

#     # # Combine node data and find unique nodes
#     # nodes_unique = pd.concat([node1_data, node2_data.rename(columns={
#     #     'ID_node1_final': 'ID_node2_final',
#     #     'lon1_final': 'lon2_final',
#     #     'lat1_final': 'lat2_final'
#     # })]).drop_duplicates().reset_index(drop=True)

#     # # Prepare data for Nodes.xlsx
#     # node1_ids = data['ID_node1_final']
#     # node2_ids = data['ID_node2_final']

#     # # 创建table_nodes，初始长度与node1_ids和node2_ids相同
#     # table_nodes = pd.DataFrame({
#     #     'NodeID': pd.concat([node1_ids, node2_ids]),
#     #     'Country': export_excel_country_code
#     # })

#     # # 添加Voltage、lon和lat列并根据NodeID填充
#     # # 假设你有以下信息用于填充
#     # voltage_data = data[['ID_node1_final', 'voltage']].drop_duplicates()
#     # voltage_data.columns = ['NodeID', 'Voltage']

#     # lon_data = data[['ID_node1_final', 'lon1_final']].drop_duplicates()
#     # lon_data.columns = ['NodeID', 'lon']

#     # lat_data = data[['ID_node1_final', 'lat1_final']].drop_duplicates()
#     # lat_data.columns = ['NodeID', 'lat']

#     # # 合并数据
#     # table_nodes = table_nodes.merge(voltage_data, on='NodeID', how='left')
#     # table_nodes = table_nodes.merge(lon_data, on='NodeID', how='left')
#     # table_nodes = table_nodes.merge(lat_data, on='NodeID', how='left')

#     # # 删除重复的NodeID，只保留一行
#     # table_nodes = table_nodes.drop_duplicates(subset='NodeID').reset_index(drop=True)

#     # # 你可以继续添加其他需要的列，比如将Voltage转换为kV等
#     # table_nodes['geometry'] = table_nodes.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

#     # 提取起点（node1）和终点（node2）的数据
#     node1_data = data[['osmID_node1', 'Voltage', 'lon1_final_1', 'lat1_final_1']].rename(columns={
#         'osmID_node1': 'NodeID', 'lon1_final_1': 'lon', 'lat1_final_1': 'lat'})
#     node2_data = data[['osmID_node2', 'Voltage', 'lon2_final_1', 'lat2_final_1']].rename(columns={
#         'osmID_node2': 'NodeID', 'lon2_final_1': 'lon', 'lat2_final_1': 'lat'})
#     # node1_data = data[['ID_node1_final', 'Voltage', 'lon1_final', 'lat1_final']].rename(columns={
#     #     'ID_node1_final': 'NodeID', 'lon1_final': 'lon', 'lat1_final': 'lat'})
#     # node2_data = data[['ID_node2_final', 'Voltage', 'lon2_final', 'lat2_final']].rename(columns={
#     #     'ID_node2_final': 'NodeID', 'lon2_final': 'lon', 'lat2_final': 'lat'})

#     # 合并起点和终点的数据
#     table_nodes = pd.concat([node1_data, node2_data])

#     # 删除重复的 NodeID 并重置索引
#     table_nodes = table_nodes.drop_duplicates(subset='NodeID').reset_index(drop=True)

#     # 转换电压单位并生成几何数据
#     table_nodes['geometry'] = table_nodes.apply(lambda row: Point(row['lon'], row['lat']) if pd.notnull(row['lon']) and pd.notnull(row['lat']) else None, axis=1)

#     # Generate filename for Nodes
#     # filename_nodes = f"../outputs/tbl_Nodes_{export_excel_country_code}_thres-{neighbourhood_threshold}km_{timestamp}.xlsx"
#     filename_nodes = f"../outputs/tbl_Nodes_{export_excel_country_code}.xlsx"

#     # Export Nodes to Excel
#     table_nodes.to_excel(filename_nodes, index=False)
#     print(f'INFO: Exported Nodes to {filename_nodes}')

#     # for df in [table_lines, table_nodes]:
#     #     for col in df.columns:
#     #         if df[col].apply(lambda x: isinstance(x, list)).any():
#     #             df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)

#     gdf_nodes = gpd.GeoDataFrame(table_nodes, geometry='geometry', crs='EPSG:32648')
#     gdf_nodes.to_file("../outputs/table_nodes.gpkg", layer='nodes', driver='GPKG')
    

#     gdf_lines = gpd.GeoDataFrame(table_lines, geometry='geometry', crs='EPSG:32648')
#     # 创建节点的 DataFrame
#     endnodes_data = {
#         'NodeID': pd.concat([table_lines['osmID_node1'], table_lines['osmID_node2']]).unique(),
#         'lon': pd.concat([table_lines['lon1_final_1'], table_lines['lon2_final_1']]).unique(),
#         'lat': pd.concat([table_lines['lat1_final_1'], table_lines['lat2_final_1']]).unique()
#     }
#     print(len(endnodes_data['NodeID']), len(endnodes_data['lon']), len(endnodes_data['lat']))

#     # 通过经纬度创建几何点
#     endnodes_df = pd.DataFrame(endnodes_data)
#     endnodes_df['geometry'] = endnodes_df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

#     # 创建 GeoDataFrame
#     gdf_endnodes = gpd.GeoDataFrame(endnodes_df, geometry='geometry', crs='EPSG:32648')
#     gdf_endnodes = gdf_endnodes.drop_duplicates(subset=['NodeID', 'geometry'])

#     # gdf_lines = gdf_lines.to_crs(epsg=32648)
#     # gdf_nodes = gdf_nodes.to_crs(epsg=32648)

#     # 保存为 GeoPackage 文件
#     gdf_lines.to_file("../outputs/table_lines.gpkg", layer='lines', driver='GPKG')
#     gdf_endnodes.to_file("../outputs/table_lines.gpkg", layer='nodes', driver='GPKG')


#     print('... finished!')


def export_to_excel(data, export_excel_country_code='VN'): #, neighbourhood_threshold=0.5 
    """ 
    Exports the data to two Excel files. This function processes the input DataFrame, 
    retains the original node IDs, and generates relevant columns for export. 
 
    Parameters:  
    - data: DataFrame containing the dataset to export. 
    - export_excel_country_code: Country code to be used for naming. 
    - neighbourhood_threshold: Threshold for naming files.
    - way_length_multiplier: Multiplier for adjusting line lengths.

    Returns:
    - None
    """

    print('Start exporting data to Excel files... (may take a few seconds)')

    data['fromNode'] = data['osmID_node1']
    data['toNode'] = data['osmID_node2']

    # Prepare the main data export
    data['Annotation'] = ''
    
    # Create strings for the Annotation "Bemerkung" column
    for index, row in data.iterrows():
        annotations = []
        
        # Check for multiple voltage levels
        if row['vlevels'] != 1:
            annotations.append("multiple vlevels")
        
        # Check for systems
        if row['systems'] == 2:
            annotations.append("6 cables - 2 systems")
        elif row['systems'] == 3:
            annotations.append("9 cables - 3 systems")
        elif row['systems'] == 4:
            annotations.append("12 cables - 4 systems")
        
        # Check for DC candidate
        if row['dc_candidate']:
            annotations.append("potentially DC")

        # Join annotations
        data.at[index, 'Annotation'] = ', '.join(annotations) if annotations else ' '

    # Prepare export columns
    data['Voltage'] = data['voltage'] / 1000  # Convert voltage to kV

    # Prepare final columns
    data['Country'] = export_excel_country_code
    data['R'] = ''
    data['XL'] = ''
    data['XC'] = ''
    data['Itherm'] = ''
    data['Capacity'] = ''
    
    table_lines = data[list(data.columns)].drop(columns=['voltage'])

    desired_order = [
        'Country', 'osm_id', 'LineID', 'fromNode', 'toNode', 
        'Voltage', 'Length', 'R', 'XL', 'XC', 'Itherm',
        'Capacity', 'frequency', 'Annotation', 'geometry'
    ]

    # 选择所需的列，并重新排列
    other_columns = [col for col in table_lines.columns if col not in desired_order]
    new_order = desired_order + other_columns  # 将其他列添加到新顺序的末尾
    table_lines = table_lines[new_order]

    # Generate filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename_lines = f"../outputs/tbl_Lines_{export_excel_country_code}.xlsx"

    # Export to Excel
    table_lines.to_excel(filename_lines, index=False)
    print(f'INFO: Exported lines to {filename_lines}')

    # 提取起点（node1）和终点（node2）的数据
    node1_data = data[['osmID_node1', 'Voltage', 'lon1_final_1', 'lat1_final_1']].rename(columns={
        'osmID_node1': 'NodeID', 'lon1_final_1': 'lon', 'lat1_final_1': 'lat'})
    node2_data = data[['osmID_node2', 'Voltage', 'lon2_final_1', 'lat2_final_1']].rename(columns={
        'osmID_node2': 'NodeID', 'lon2_final_1': 'lon', 'lat2_final_1': 'lat'})

    # 将所有的ID, lon, lat组合保存到endnodes_data
    endnodes_data = pd.concat([node1_data, node2_data])

    # 删除重复的 NodeID 和经纬度组合
    endnodes_data = endnodes_data.drop_duplicates(subset=['NodeID', 'lon', 'lat']).reset_index(drop=True)

    # 转换电压单位并生成几何数据
    endnodes_data['geometry'] = endnodes_data.apply(lambda row: Point(row['lon'], row['lat']) if pd.notnull(row['lon']) and pd.notnull(row['lat']) else None, axis=1)

    # Generate filename for Nodes
    filename_nodes = f"../outputs/tbl_Nodes_{export_excel_country_code}.xlsx"

    # Export Nodes to Excel
    endnodes_data.to_excel(filename_nodes, index=False)
    print(f'INFO: Exported Nodes to {filename_nodes}')

    # Save nodes to GeoPackage (GPKG)
    gdf_nodes = gpd.GeoDataFrame(endnodes_data, geometry='geometry', crs='EPSG:32648')
    gdf_nodes.to_file("../outputs/table_nodes.gpkg", layer='nodes', driver='GPKG')

    # 保存为GeoPackage
    gdf_lines = gpd.GeoDataFrame(table_lines, geometry='geometry', crs='EPSG:32648')

    # 保存为GeoPackage
    gdf_lines.to_file("../outputs/table_lines.gpkg", layer='lines', driver='GPKG')

    print('... finished!')
