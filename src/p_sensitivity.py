"""
Sensitivity analysis for parameter p in industrial–substation assignment
p ∈ [0, 1] with step = 0.05

Outputs:
- CSV summary table (Table Sx)
- Assignment stability plot
- Load injection sensitivity plot
"""
import os,sys
os.environ['USE_PYGEOS'] = '0'

import numpy as np
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

parent_dir = Path().resolve().parent
print(parent_dir)
sys.path.insert(0, str(os.path.join(parent_dir,'src')))
from utils import *

# Import landuse sites
landuse_sites = gpd.read_file('../../cascading_failure/mapping_industries_economic_sectors/outputs/landuse_sites_osm_boundary.gpkg')
landuse_sites_gdf = gpd.GeoDataFrame(landuse_sites, geometry='geometry', crs='EPSG:4326')
landuse_sites_gdf = landuse_sites_gdf.to_crs(epsg=32648)
print(len(landuse_sites_gdf))

# Import industries
industries_df = gpd.read_file('../../cascading_failure/mapping_industries_economic_sectors/outputs/intersected_polygons.gpkg')
industries_gdf = gpd.GeoDataFrame(industries_df, geometry='geometry', crs='EPSG:4326')
industries_gdf = industries_gdf.rename(columns={'Unnamed: 0': 'industry_id', 'name_1': 'name'})
industries_gdf = industries_gdf.to_crs(epsg=32648)
# industries_gdf['area'] = industries_gdf['geometry'].apply(lambda geom: geom.area)
industries_gdf['geometry'] = industries_gdf['geometry'].apply(convert_to_point)
industries_gdf = industries_gdf[industries_gdf['business_status'] == 'OPERATIONAL']

# Import nodes_gdftations from table_nodes_500m.gpkg
nodes_gdf =  gpd.read_file("../outputs/table_nodes_200m_update_remove_disconnected.gpkg")

# Import electricity demand statistics
sector_demand_df = pd.read_excel('../data/sectors_searchstrings_full.xlsx')

# Step 1: Count the number of each Sector_id in industries_gdf
sector_counts = industries_gdf.groupby('Sector_id').size()
print(sector_counts)

# Step 2: Distribute the total electricity_demand in sector_demand_df to each industry_id under each Sector_id
# Create a mapping from sector_id to total_demand
sector_demand_dict = sector_demand_df.set_index('sector_id')['electricity_demand'].to_dict()

# Step 3: Add electricity_demand and required_capacity to industries_gdf
# The total electricity demand for each Sector_id / the number of industries under that Sector_id, to get the electricity demand for each industry_id
industries_gdf['electricity_demand'] = industries_gdf['Sector_id'].map(sector_demand_dict) / industries_gdf['Sector_id'].map(sector_counts)
# industries_gdf['required_capacity'] = nodes_gdf['capacity'].sum() * 0.54 / len(industries_gdf)

# Step 4: Sum the electricity_demand and required_capacity, respectively,
# for all industries_gdf records with the same osmid, and assign it to the corresponding osmid in landuse_sites_gdf
# Aggregate the electricity_demand of all industries in industries_gdf by osmid
aggregated_demand = industries_gdf.groupby('osmid')['electricity_demand'].sum().rename('total_electricity_demand')
# aggregated_capacity = industries_gdf.groupby('osmid')['required_capacity'].sum().rename('total_required_capacity')

# Merge the aggregated results into landuse_sites_gdf
landuse_sites_gdf = landuse_sites_gdf.merge(aggregated_demand, how='left', left_on='osmid', right_index=True)
# landuse_sites_gdf = landuse_sites_gdf.merge(aggregated_capacity, how='left', left_on='osmid', right_index=True)
print(len(landuse_sites_gdf))
landuse_sites_gdf_cleaned = landuse_sites_gdf.dropna(subset=['total_electricity_demand'])
print(len(landuse_sites_gdf_cleaned))


def assign_buses_to_loads_with_k(
    landuse_sites_gdf,
    nodes_gdf,
    p=0.8,
    seed=42
):
    nodes_gdf = nodes_gdf.copy().set_index('NodeID')
    landuse_sites_gdf = landuse_sites_gdf.copy()

    rng = np.random.default_rng(seed)
    results_dict = {}

    for idx, landuse_site in landuse_sites_gdf.iterrows():
        landuse_site_id = landuse_site['osmid']
        distances = nodes_gdf.geometry.distance(landuse_site.geometry)

        results_dict[landuse_site_id] = {}
        weighted_probs = {}
        P_ks = {}

        # sort substations by distance
        sorted_idx = np.argsort(distances.values)

        sum_weighted_prob = 0.0

        for k, sub_idx in enumerate(sorted_idx, start=1):
            sub_id = nodes_gdf.index[sub_idx]
            capacity = nodes_gdf.at[sub_id, 'sn_mva']

            P_k = p * (1 - p) ** (k - 1)
            weighted_prob = capacity * P_k

            weighted_probs[sub_id] = weighted_prob
            P_ks[sub_id] = P_k
            sum_weighted_prob += weighted_prob

        # normalize
        norm_probs = {
            sub_id: wp / sum_weighted_prob
            for sub_id, wp in weighted_probs.items()
        }

        # build CDF (ordered by distance rank)
        cumulative = 0.0
        cdf = []
        for sub_idx in sorted_idx:
            sub_id = nodes_gdf.index[sub_idx]
            cumulative += norm_probs[sub_id]
            cdf.append((sub_id, cumulative))

        r = rng.random()
        selected_substation = None
        for sub_id, cdf_value in cdf:
            if r <= cdf_value:
                selected_substation = sub_id
                break

        landuse_sites_gdf.loc[
            landuse_sites_gdf['osmid'] == landuse_site_id, 'BusID'
        ] = selected_substation

        for sub_id in norm_probs:
            results_dict[landuse_site_id][sub_id] = {
                'norm_prob': norm_probs[sub_id],
                'P_k': P_ks[sub_id],
                'distance': distances.loc[sub_id]
            }

    return results_dict, landuse_sites_gdf


def run_p_sensitivity(
    landuse_sites_gdf,
    nodes_gdf,
    load_col="total_electricity_demand",
    p_min=0.0,
    p_max=1.0,
    p_step=0.05,
    base_p=0.8,
    seed=42
):
    p_values = np.round(
        np.arange(p_min, p_max + p_step, p_step), 2
    )

    bus_maps = {}
    injections = {}

    # run assignment for each p
    for p in p_values:
        _, assigned = assign_buses_to_loads_with_k(
            landuse_sites_gdf,
            nodes_gdf,
            p=p,
            seed=seed
        )

        bus_maps[p] = assigned.set_index("osmid")["BusID"]

        inj = (
            assigned
            .groupby("BusID")[load_col]
            .sum()
        )
        injections[p] = inj

    base_map = bus_maps[base_p]
    base_inj = injections[base_p]

    records = []

    for p in p_values:
        cur_map = bus_maps[p]

        # assignment stability
        same_as_base = (cur_map == base_map).mean()

        # always same (across all p)
        always_same = (
            pd.concat(bus_maps.values(), axis=1)
            .nunique(axis=1)
            .eq(1)
            .mean()
        )

        # injection sensitivity (normalized L1)
        inj = injections[p].reindex(base_inj.index, fill_value=0.0)
        delta_L1 = (inj - base_inj).abs().sum() / base_inj.sum()

        records.append({
            "p": p,
            "same_as_base_fraction": same_as_base,
            "always_same_fraction": always_same,
            "delta_L1_injection": delta_L1
        })

    df = pd.DataFrame(records)
    return df

def plot_sensitivity(df):
    # Assignment stability
    plt.figure(figsize=(6,4))
    plt.plot(df["p"], df["same_as_base_fraction"], marker="o")
    plt.xlabel("p")
    plt.ylabel("Fraction of loads unchanged (relative to p=0.8)")
    plt.title("Assignment stability under different p values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures/Fig_S_assignment_stability.png", dpi=600)
    plt.close()

    # Injection sensitivity
    plt.figure(figsize=(6,4))
    plt.plot(df["p"], df["delta_L1_injection"], marker="o")
    plt.xlabel("p")
    plt.ylabel("Normalized difference in load injection")
    plt.title("Sensitivity of substation load injection to p")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../figures/Fig_S_injection_sensitivity.png", dpi=600)
    plt.close()


if __name__ == "__main__":

    # TODO: load your data here
    # landuse_sites_gdf = ...
    # nodes_gdf = ...

    df_sensitivity = run_p_sensitivity(
        landuse_sites_gdf,
        nodes_gdf,
        load_col="total_electricity_demand",
        p_min=0.0,
        p_max=1.0,
        p_step=0.05,
        base_p=0.8,
        seed=42
    )

    df_sensitivity.to_csv(
        "../figures/Table_S_p_sensitivity.csv", index=False
    )

    plot_sensitivity(df_sensitivity)

    print("Sensitivity analysis completed.")
