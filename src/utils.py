import pandas as pd
import re

# Function to convert different units of electricity output to MW
def convert_to_mw(value):
    if pd.isnull(value):
        return None
    
    # Use regular expression to extract the numeric part (including decimal points)
    match = re.search(r"([\d.]+)", value)
    if not match:
        raise ValueError(f"Unexpected value: {value}")
    
    number = float(match.group(1))  # Extract the numeric part
    
    # Convert based on units
    if "MW" in value or "MWp" in value:
        return number
    elif "kW" in value:
        return number / 1000  # kW to MW
    elif "GW" in value:
        return number * 1000  # GW to MW
    else:
        raise ValueError(f"Unexpected value: {value}")

# Function to convert different units of electricity output to GW
def convert_to_gw(value):
    if pd.isnull(value):
        return None
    
    # Use regular expression to extract the numeric part (including decimal points)
    match = re.search(r"([\d.]+)", value)
    if not match:
        raise ValueError(f"Unexpected value: {value}")
    
    number = float(match.group(1))  # Extract the numeric part
    
    # Convert based on units
    if "MW" in value or "MWp" in value:
        return number / 1000 # MW to GW
    elif "kW" in value:
        return number / 1000000  # kW to GW
    # elif "GW" in value:
    #     return number * 1000  # GW to MW
    else:
        raise ValueError(f"Unexpected value: {value}")


def convert_to_point(geom):
    """
    If the geometry is a MultiPolygon, Polygon, or Linestring, return its centroid
    """
    if geom.geom_type == 'MultiPolygon' or geom.geom_type == 'Polygon':
        return geom.centroid
    elif geom.geom_type == 'LineString':
        return geom.centroid
    elif geom.geom_type == 'Point':
        return geom
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")