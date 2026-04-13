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
from matplotlib.ticker import MaxNLocator, FuncFormatter
from pathlib import Path

gpd.options.io_engine = "pyogrio"

# 1) landuse
landuse_sites_gdf = gpd.read_file('../../cascading_failure/mapping_industries_economic_sectors/outputs/landuse_sites_osm_boundary.gpkg')
landuse_sites_gdf = landuse_sites_gdf.to_crs(epsg=32648)

# 2) industries
industries_gdf = gpd.read_file('../../cascading_failure/mapping_industries_economic_sectors/outputs/intersected_polygons.gpkg')
industries_gdf = industries_gdf.rename(columns={'Unnamed: 0': 'industry_id', 'name_1': 'name'})

# 先筛选（减少后续几何运算规模）
industries_gdf = industries_gdf[industries_gdf['business_status'] == 'OPERATIONAL'].copy()

# 先把 polygon 变成点（用向量化方法，替代 apply）
# 若你要点在面内，用 representative_point；若无所谓在内，用 centroid
industries_gdf['geometry'] = industries_gdf.geometry.centroid

# 最后再投影（点的投影很快）
industries_gdf = industries_gdf.to_crs(epsg=32648)

# 3) nodes
nodes_gdf = gpd.read_file("../outputs/table_nodes_200m_update_remove_disconnected.gpkg")

# 4) demand stats
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


def plot_sensitivity(df, out_dir="../figures", color="#34D1BF", base_p=0.8):
    """
    Expected df columns:
      - p
      - same_as_base_fraction
      - delta_L1_injection
    """
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # Publication-style settings
    # -------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "savefig.dpi": 800,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    def _style_ax(ax):
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    def _add_base_line(ax):
        # Mark the baseline p (if within plotting range)
        ax.axvline(base_p, linestyle="--", linewidth=1.0, color="0.35", alpha=0.8, zorder=1)
        ax.text(
            base_p, 0.98, f"baseline p={base_p:g}",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top",
            fontsize=12, color="0.35"
        )

    # Ensure sorted by p for clean lines
    df_plot = df.sort_values("p")

    # -------------------------
    # 1) Assignment stability
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(
        df_plot["p"], df_plot["same_as_base_fraction"],
        marker="o", markersize=4.5,
        lw=1.6, color=color,
        label="Stability",
        zorder=3
    )

    # If it's a fraction in [0,1], optionally show as %
    # Comment this out if you prefer raw fraction.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax.set_xlabel("p (distance preference parameter)")
    ax.set_ylabel("Loads unchanged vs baseline")
    # ax.set_title("Assignment stability across p values")

    # Baseline marker
    if (df_plot["p"] == base_p).any():
        _add_base_line(ax)

    _style_ax(ax)
    ax.legend(frameon=False, loc="best")

    fig.savefig(os.path.join(out_dir, "Fig_S4_p_sensitivity_load_assignment.png"), dpi=800, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "Fig_S4_p_sensitivity_load_assignment.svg"), bbox_inches="tight")
    plt.close(fig)

    # -------------------------
    # 2) Injection sensitivity
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(
        df_plot["p"], df_plot["delta_L1_injection"],
        marker="o", markersize=4.5,
        lw=1.6, color=color,
        label="Sensitivity",
        zorder=3
    )

    ax.set_xlabel("p (distance preference parameter)")
    ax.set_ylabel("Normalized L1 difference in load injection")
    # ax.set_title("Sensitivity of substation load injection to p")

    if (df_plot["p"] == base_p).any():
        _add_base_line(ax)

    _style_ax(ax)
    ax.legend(frameon=False, loc="best")

    fig.savefig(os.path.join(out_dir, "Fig_S4_p_sensitivity_load_injection.png"), dpi=800, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "Fig_S4_p_sensitivity_load_injection.svg"), bbox_inches="tight")
    plt.close(fig)


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
        "../figures/Fig_S4_p_sensitivity.csv", index=False
    )

    plot_sensitivity(df_sensitivity)

    print("Sensitivity analysis completed.")
