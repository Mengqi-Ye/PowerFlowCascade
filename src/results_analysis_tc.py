# results_analysis_tc.py

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import bootstrap
from matplotlib.patches import Patch
from shapely import wkt


def assign_adm1_codes_with_ref_and_pcode(nodes_gdf, lines_gdf, loads_gdf, basemap):
    """
    Assign ADM1_EN, ADM1_REF, and ADM1_PCODE to nodes, lines, and loads by spatial overlay proportion.
    """
    nodes_gdf = nodes_gdf.to_crs(epsg=32648)
    lines_gdf = lines_gdf.to_crs(epsg=32648)
    loads_gdf = loads_gdf.to_crs(epsg=32648)
    basemap = basemap.to_crs(epsg=32648)

    nodes_gdf = gpd.sjoin(nodes_gdf, basemap[["ADM1_EN", "ADM1_REF", "ADM1_PCODE", "geometry"]], how="left", predicate="within")
    nodes_gdf.drop(columns=["index_right"], inplace=True)

    line_overlay = gpd.overlay(lines_gdf, basemap, how="intersection")
    line_overlay["length"] = line_overlay.geometry.length
    total_lengths = line_overlay.groupby("LineID")["length"].sum().rename("total")
    line_overlay = line_overlay.join(total_lengths, on="LineID")
    line_overlay["ratio"] = line_overlay["length"] / line_overlay["total"]
    major_parts = line_overlay[line_overlay["ratio"] > 0.5].copy()
    line_adm = major_parts[["LineID", "ADM1_EN", "ADM1_REF", "ADM1_PCODE"]].drop_duplicates()
    lines_gdf = lines_gdf.merge(line_adm, on="LineID", how="left")

    load_overlay = gpd.overlay(loads_gdf, basemap, how="intersection")
    load_overlay["area"] = load_overlay.geometry.area
    total_areas = load_overlay.groupby("osmid")["area"].sum().rename("total")
    load_overlay = load_overlay.join(total_areas, on="osmid")
    load_overlay["ratio"] = load_overlay["area"] / load_overlay["total"]
    major_parts = load_overlay[load_overlay["ratio"] > 0.5].copy()
    load_adm = major_parts[["osmid", "ADM1_EN", "ADM1_REF", "ADM1_PCODE"]].drop_duplicates()
    loads_gdf = loads_gdf.merge(load_adm, on="osmid", how="left")

    return nodes_gdf, lines_gdf, loads_gdf


def compute_failure_probabilities(gdf):
    """
    计算每条线路或节点的 Direct 和 Indirect failure 概率。
    输入 gdf 包含 impact_1, impact_2, ..., impact_n 列，每列值为字符串:
    "Direct Failed", "Indirect Failed", 或 "Not Failed"
    """
    impact_cols = [col for col in gdf.columns if col.startswith("impact_")]

    gdf["p_direct"] = gdf[impact_cols].apply(
        lambda row: sum(val == "Direct Failed" for val in row) / len(impact_cols),
        axis=1
    )
    gdf["p_indirect"] = gdf[impact_cols].apply(
        lambda row: sum(val == "Indirect Failed" for val in row) / len(impact_cols),
        axis=1
    )

    return gdf


def plot_combined_failure_maps_extended(lines_gdf, nodes_gdf, basemap, fig_path=None):
    # Ensure proper CRS
    lines_gdf = lines_gdf.to_crs("EPSG:4326")
    nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    # Filter region
    pcode_list = ['VN106', 'VN111', 'VN101', 'VN107', 'VN103', 'VN109', 'VN113', 'VN117', 'VN115', 'VN104']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    # Compute majority failure status and probability
    def compute_majority_with_prob(gdf):
        impact_cols = [col for col in gdf.columns if col.startswith("impact_")]
        mode_vals = gdf[impact_cols].mode(axis=1)[0]
        probs = []
        for idx, row in gdf.iterrows():
            status_counts = row[impact_cols].value_counts(normalize=True)
            mode = mode_vals.loc[idx]
            probs.append(status_counts.get(mode, 0.0))
        gdf['majority_status'] = mode_vals
        gdf['majority_prob'] = probs
        return gdf

    lines_gdf = compute_majority_with_prob(lines_gdf)
    nodes_gdf = compute_majority_with_prob(nodes_gdf)

    # Color definitions
    status_colors = {
        'Not Failed': '#98DF8A',
        'Direct Failed': '#D62728',
        'Indirect Failed': '#1F77B4'
    }

    cmap = cm.Reds
    norm = colors.Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18), gridspec_kw={'hspace': 0.05, 'wspace': 0.01})
    row_labels = ["Majority failure status", "Direct failure probability", "Indirect failure probability"]
    col_labels = ["Line", "Bus"]

    col_pairs = [
        (lines_gdf, 'majority_status', 'majority_prob'),
        (nodes_gdf, 'majority_status', 'majority_prob'),
        (lines_gdf, 'p_direct', None),
        (nodes_gdf, 'p_direct', None),
        (lines_gdf, 'p_indirect', None),
        (nodes_gdf, 'p_indirect', None),
    ]

    # for ax, title, (gdf, status_col, prob_col) in zip(axes.flat, titles, col_pairs):
    for i, (ax, (gdf, status_col, prob_col)) in enumerate(zip(axes.flat, col_pairs)):
        row, col = divmod(i, 2)
        target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.5, zorder=0)
        other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.3, zorder=0)

        if prob_col is not None:
            # Plot each status
            for status, color in status_colors.items():
                subset = gdf[gdf['majority_status'] == status]
                if subset.empty:
                    continue
                if subset.geometry.iloc[0].geom_type == 'Point':
                    subset.plot(ax=ax, color=color, markersize=15, label=status, zorder=3)
                else:
                    subset.plot(ax=ax, color=color, linewidth=1.5, label=status, zorder=3)
                
        else:
            if gdf.geometry.iloc[0].geom_type == 'Point':
                gdf[gdf[status_col] == 0].plot(ax=ax, color='#98DF8A', markersize=10, alpha=0.5, zorder=2)
                gdf[gdf[status_col] > 0].plot(ax=ax, column=status_col, cmap=cmap, norm=norm,
                                              markersize=30, alpha=0.9, zorder=3)
            else:
                gdf[gdf[status_col] == 0].plot(ax=ax, color='#98DF8A', linewidth=0.5, alpha=0.5, zorder=2)
                gdf[gdf[status_col] > 0].plot(ax=ax, column=status_col, cmap=cmap, norm=norm,
                                              linewidth=1.2, zorder=3)

        ax.set_xlim(105, 107.5)
        ax.set_ylim(19.85, 21.7)

        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        if row == 2:
            ax.set_xlabel("Longitude", fontsize=15)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("Latitude", fontsize=15)
        else:
            ax.tick_params(labelleft=False)

        if prob_col is not None:
            ax.legend(loc='lower right', fontsize=15)

    # Add super labels
    for ax, col_label in zip(axes[0], col_labels):
        ax.annotate(col_label, xy=(0.5, 1.03), xycoords='axes fraction',
                    ha='center', va='center', fontsize=15, fontweight='bold') #, fontweight='bold'

    for ax, row_label in zip(axes[:, 0], row_labels):
        ax.annotate(row_label, xy=(-0.17, 0.5), xycoords='axes fraction',
                    ha='center', va='center', fontsize=15, fontweight='bold', rotation=90) #, fontweight='bold'

    # Add single colorbar for probability maps
    # 自定义 colorbar，使其高度对齐中下两行图（行1和行2）
    # 获取中间两行右边子图的位置（axes[1, 1] 和 axes[2, 1]）
    pos1 = axes[1, 1].get_position()
    pos2 = axes[2, 1].get_position()
    cbar_ax = fig.add_axes([pos2.x1 + 0.01, pos2.y0, 0.015, pos1.y1 - pos2.y0])

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Failure probability", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    # Save or show
    if fig_path:
        os.makedirs(fig_path, exist_ok=True)
        output_file = os.path.join(fig_path, "combined_failure_maps_extended.png")
        plt.savefig(output_file, dpi=600, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")

def plot_overload_by_adm_ref_fixed_bins(lines_with_ratio, output_path=None, basemap=None):
    os.makedirs(output_path, exist_ok=True)

    lines_with_ratio = lines_with_ratio.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    pcode_list = ['VN106','VN111','VN101','VN107','VN103','VN109','VN113','VN117','VN115','VN104',
                  'VN717','VN711','VN707','VN713','VN701','VN709']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    vmin = lines_with_ratio["overload_prob"].min()
    vmax = lines_with_ratio["overload_prob"].max()
    
    bounds = [0, 0.2, 0.4, 0.6, 0.8, vmax + 0.01]
    cmap = ListedColormap(plt.cm.YlOrRd(np.linspace(0.3, 1, len(bounds)-1)))
    norm = BoundaryNorm(bounds, cmap.N)

    overloaded = lines_with_ratio[lines_with_ratio["overload_prob"] > 0]
    non_overloaded = lines_with_ratio[lines_with_ratio["overload_prob"] == 0]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))

    # ax.set_title("Red River Delta", fontsize=15)
    ax.set_xlabel("Longitude", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)
    ax.set_xlim(105, 107.5)
    ax.set_ylim(19.85, 21.7)
    ax.set_aspect('auto')
    ax.tick_params(axis='both', labelsize=12)
    target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.5, zorder=1)
    other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.3, zorder=2)
    non_overloaded.plot(ax=ax, color='#98DF8A', linewidth=0.5, alpha=0.5, zorder=3, label='Not overloaded')
    overloaded.plot(ax=ax, column="overload_prob", cmap=cmap, norm=norm, linewidth=1.5, zorder=4)

    ax.set_aspect('auto')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(loc='lower right', fontsize=13)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    tick_locs = [(bounds[i] + bounds[i+1]) / 2 for i in range(len(bounds)-1)]
    tick_labels = [f"{bounds[i]:.1f}-{bounds[i+1]:.1f}" for i in range(len(bounds)-1)]
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", location='right', fraction=0.04, ticks=tick_locs, pad=0.03)
    cbar.ax.set_yticklabels(tick_labels, fontsize=13)
    cbar.set_label("Overload ratio", fontsize=13)

    plt.savefig(os.path.join(output_path, "overload_map_region_split_fixed_bins.png"), dpi=600, bbox_inches="tight")


def plot_overload_by_adm_ref_continous(lines_with_ratio, output_path=None, basemap=None):
    os.makedirs(output_path, exist_ok=True)

    lines_with_ratio = lines_with_ratio.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    pcode_list = ['VN106','VN111','VN101','VN107','VN103','VN109','VN113','VN117','VN115','VN104',
                  'VN717','VN711','VN707','VN713','VN701','VN709']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    vmin = lines_with_ratio["overload_prob"].min()
    vmax = lines_with_ratio["overload_prob"].max()
    cmap = plt.colormaps.get_cmap("YlOrRd")
    norm = colors.Normalize(vmin=0.1, vmax=vmax)

    overloaded = lines_with_ratio[lines_with_ratio["overload_prob"] > 0]
    non_overloaded = lines_with_ratio[lines_with_ratio["overload_prob"] == 0]

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))

    ax.set_xlabel("Longitude", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)
    ax.set_xlim(105, 107.5)
    ax.set_ylim(19.85, 21.7)
    ax.set_aspect('auto')
    ax.tick_params(axis='both', labelsize=12)
    target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.5, zorder=1)
    other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.3, zorder=2)
    non_overloaded.plot(ax=ax, color='#98DF8A', linewidth=0.5, alpha=0.5, zorder=3, label='Not overloaded')
    overloaded.plot(ax=ax, column="overload_prob", cmap=cmap, norm=norm, linewidth=1.5, zorder=4)

    ax.set_aspect('auto')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.legend(loc='lower right', fontsize=13)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", location='right', fraction=0.04, pad=0.01)
    cbar.set_label("Overload Ratio", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(os.path.join(output_path, "overload_map_region_split_continous.png"), dpi=600, bbox_inches="tight")


def plot_load_failure_probability(load_status_df, loads_gdf, results_df, basemap, output_path=None):
    # === Ensure ID formats match ===
    # loads_gdf["osmid"] = pd.to_numeric(loads_gdf["osmid"], errors="coerce").astype("Int64")
    # load_status_df["load_id"] = pd.to_numeric(load_status_df["load_id"], errors="coerce").astype("Int64")

    # === Filter out iterations with failed powerflow ===
    successful_iters = results_df.loc[results_df["powerflow_success"], "fail_iter"]
    load_status_df = load_status_df[load_status_df["fail_iter"].isin(successful_iters)]

    # === Calculate failure probability ===
    failure_stats = (
        load_status_df.groupby("load_id")["served"]
        .mean()
        .reset_index(name="served_ratio")
    )
    failure_stats["failure_probability"] = 1 - failure_stats["served_ratio"]

    # === Merge & fill missing with 0 ===
    loads_gdf = loads_gdf.merge(failure_stats[["load_id", "failure_probability", "served_ratio"]],
                                left_on="osmid", right_on="load_id", how="left")
    loads_gdf["failure_probability"] = loads_gdf["failure_probability"].fillna(0)

    # === Basemap preprocessing ===
    loads_gdf = loads_gdf.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    pcode_list = ['VN106','VN111','VN101','VN107','VN103','VN109','VN113','VN117','VN115','VN104']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    # === Split data ===
    always_served_loads = loads_gdf[loads_gdf["failure_probability"] == 0]
    failed_loads = loads_gdf[loads_gdf["failure_probability"] > 0]

    # === Colormap ===
    cmap = cm.YlOrRd
    norm = colors.Normalize(vmin=0, vmax=1)

    # === Plot ===
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6))

    # ax.set_title("Load Failure Probability", fontsize=15)
    ax.set_xlim(105, 107.5)
    ax.set_ylim(19.85, 21.7)
    ax.set_xlabel("Longitude", fontsize=13)
    ax.set_ylabel("Latitude", fontsize=13)
    ax.tick_params(axis='both', labelsize=12)

    # Basemap
    target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, zorder=1)
    other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.3, zorder=0)

    # Plot zero-failure loads (fixed color)
    always_served_loads.plot(ax=ax, color='green', markersize=10, label="Failure p = 0", zorder=2)

    # # Plot partial failures (gradient color)
    failed_loads.plot(ax=ax, column="failure_probability", cmap=cmap, norm=norm, zorder=3)
    # loads_gdf.plot(ax=ax, column="failure_probability", cmap=cmap, norm=norm, zorder=3)

    # Colorbar for only failed loads
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
    cbar.set_label("Failure probability", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # ax.legend(loc='lower right', fontsize=13)
    # Manually create legend for green loads
    legend_elements = [
        Patch(facecolor='green', label='Failure p = 0')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=13)

    if output_path:
        plt.savefig(os.path.join(output_path, "load_failure_probability.png"), dpi=600, bbox_inches="tight")
        print(f"✓ Saved to {output_path}")

    return loads_gdf


def analyze_load_served_ratio_by_adm(loads_gdf, load_status_df, results_df, net_dir, output_path):
    # === Step 1: 读取成功迭代 ===
    passed_iters = (
        results_df[results_df["powerflow_success"] == True]["fail_iter"]
        .str.extract(r"fail_iter_(\d+)")
        .astype(int)
        .squeeze()
        .tolist()
    )
    passed_iters_str = [f"fail_iter_{i}" for i in passed_iters]
    print(f"✓ Found {len(passed_iters_str)} successful iterations")

    all_records = []

    for i in passed_iters:
        filepath = os.path.join(net_dir, f"net_results_fail_iter{i}.xlsx")
        try:
            xls = pd.read_excel(filepath, sheet_name=None)
            df_load = xls["load"]
            df_res = xls["res_load"]
        except Exception as e:
            print(f"\u2717 Skipping {filepath}: {e}")
            continue

        df = df_load[["name", "p_mw"]].merge(
            df_res[["p_mw"]], left_index=True, right_index=True, suffixes=("", "_res")
        ).rename(columns={"name": "load_id", "p_mw": "p_mw", "p_mw_res": "res_p_mw"})

        id_adm_df = loads_gdf[["osmid", "ADM1_EN"]].rename(columns={"osmid": "load_id"})
        df = pd.merge(df, id_adm_df, on="load_id", how="left").dropna(subset=["ADM1_EN"])

        grouped = df.groupby("ADM1_EN")[["res_p_mw", "p_mw"]].sum()
        grouped["served_ratio"] = grouped["res_p_mw"] / grouped["p_mw"]
        grouped["iteration"] = i
        all_records.append(grouped.reset_index()[["ADM1_EN", "iteration", "served_ratio"]])

    df_all = pd.concat(all_records, ignore_index=True)

    # === Step 3: Prepare average failed ratio per region ===
    id_adm_df = loads_gdf[['osmid', 'ADM1_EN']].rename(columns={'osmid': 'load_id'})
    status_with_adm = pd.merge(load_status_df, id_adm_df, on='load_id', how='left')
    status_with_adm['served'] = status_with_adm['served'].astype(bool)
    status_with_adm = status_with_adm[status_with_adm['fail_iter'].isin(passed_iters_str)]

    failure_ratio = (
        status_with_adm
        .groupby(['fail_iter', 'ADM1_EN'])['served']
        .apply(lambda x: (~x).sum() / len(x))
        .reset_index(name='failed_ratio')
    )

    # avg_failed_ratio = (
    #     failure_ratio
    #     .groupby('ADM1_EN')['failed_ratio']
    #     .mean()
    #     .fillna(0)
    # )

    # Filter provinces with failure
    adm_with_failures = failure_ratio[failure_ratio["failed_ratio"] > 0]["ADM1_EN"].unique()
    df_all = df_all[df_all["ADM1_EN"].isin(adm_with_failures)]
    # avg_failed_ratio = avg_failed_ratio.loc[adm_with_failures]
    
    df_avg = df_all.groupby("ADM1_EN")["served_ratio"].mean().reset_index(name="served_ratio_mean")
    df_median = df_all.groupby("ADM1_EN")["served_ratio"].median().reset_index(name="served_ratio_median")
    df_std = df_all.groupby("ADM1_EN")["served_ratio"].std().reset_index(name="served_ratio_std")
    df_min = df_all.groupby("ADM1_EN")["served_ratio"].min().reset_index(name="served_ratio_min")

    # Unstable (high std)
    df_unstable = df_std.merge(df_min, on="ADM1_EN")
    df_unstable = df_unstable.sort_values(by="served_ratio_std", ascending=False).head(5)
    print("\nTop 5 unstable provinces (by std of served_ratio):")
    print(df_unstable)

    # Stable (low std)
    df_stable = df_std.merge(df_min, on="ADM1_EN")
    df_stable = df_stable.sort_values(by="served_ratio_std", ascending=True).head(5)
    print("\nTop 5 stable provinces (by std of served_ratio):")
    print(df_stable)

    # # Sort order by instability (std)
    # sort_order = df_std[df_std["ADM1_EN"].isin(adm_with_failures)].sort_values("served_ratio_std", ascending=True)["ADM1_EN"]

    # Sort order by median served_ratio
    sort_order = df_median.sort_values("served_ratio_median")["ADM1_EN"]

    # === Step 4: 绘图 ===
    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=df_all,
        x="ADM1_EN",
        y="served_ratio",
        order=sort_order,
        ax=ax1,
        color="skyblue"
    )

    # means = df_all.groupby("ADM1_EN")["served_ratio"].mean()

    # for xtick, adm1 in enumerate(sort_order):
    #     group = df_all[df_all["ADM1_EN"] == adm1]["served_ratio"].values
    #     if len(group) > 1:
    #         ci = bootstrap((group,), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    #         ci_low, ci_high = ci.confidence_interval
    #         ax1.plot(xtick, means[adm1], 'ro')
    #         ax1.vlines(xtick, ci_low, ci_high, color='black', linewidth=1.2)
    #     else:
    #         ax1.plot(xtick, means[adm1], 'ro')

    # 设置 x tick label 颜色（红色为最不稳定，绿色为最稳定）
    xtick_labels = ax1.get_xticklabels()
    for label in xtick_labels:
        adm = label.get_text()
        if adm in df_unstable["ADM1_EN"].values:
            label.set_fontweight("bold")
        elif adm in df_stable["ADM1_EN"].values:
            label.set_color("#2CA02C")

    ax1.set_ylabel("Served ratio", fontsize=14)
    ax1.set_xlabel("Province (sorted by median served ratio)", fontsize=14)
    ax1.set_xticklabels(sort_order, rotation=45, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.tick_params(axis='both', labelsize=13)

    # # Right Y-axis: avg failed ratio
    # ax2 = ax1.twinx()
    # ax2.bar(sort_order, avg_failed_ratio.loc[sort_order].values, alpha=0.25, color="gray", label="Average failed ratio")
    # ax2.set_ylabel("Average failed load ratio", fontsize=14)
    # ax2.legend(loc="lower right", fontsize=14)
    # ax2.tick_params(axis='both', labelsize=13)

    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "load_served_ratio_boxplot_by_adm1.png"), dpi=600, bbox_inches='tight')

    return df_all, df_median, df_std, df_avg, failure_ratio, avg_failed_ratio


def export_summary_table(df_median, df_std, df_avg, avg_failed_ratio, loads_gdf, output_path):
    # Convert avg_failed_ratio Series to DataFrame
    df_failed = avg_failed_ratio.reset_index().rename(columns={'failed_ratio': 'avg_failed_ratio'})

    summary_df = (
        df_median
        .merge(df_std, on='ADM1_EN', how='outer')
        .merge(df_avg, on='ADM1_EN', how='outer')
        .merge(df_failed, on='ADM1_EN', how='outer')
    )

    # 添加每省负荷数量
    load_counts = (
        loads_gdf[loads_gdf["ADM1_EN"].isin(summary_df["ADM1_EN"])]
        .groupby("ADM1_EN")["osmid"]
        .nunique()
        .reset_index()
        .rename(columns={"osmid": "total_loads"})
    )

    summary_df = summary_df.merge(load_counts, on="ADM1_EN", how="left")
    summary_df["served_ratio_mean"] = (summary_df["served_ratio_mean"] * 100).round(1)
    summary_df["avg_failed_ratio"] = (summary_df["avg_failed_ratio"] * 100).round(1)
    summary_df = summary_df.sort_values(by="served_ratio_mean", ascending=True)

    summary_df.to_excel(os.path.join(output_path, "adm1_served_summary.xlsx"), index=False)

    return summary_df

def main():
    # INPUT PARAMETERS
    base_dir = f"../outputs/20250725_hazard_iter_1000"

    lines_gdf = gpd.read_file('../outputs/table_lines_200m_update_remove_disconnected.gpkg')
    nodes_gdf = gpd.read_file('../outputs/table_nodes_200m_update_remove_disconnected.gpkg')
    loads_gdf = gpd.read_file('../outputs/landuse_sites_gdf_add_bus.gpkg')
    gens_gdf = gpd.read_file('../outputs/plant_update.gpkg')
    basemap = gpd.read_file("../data/base_map/vnm_admbnda_adm1_gov_20201027.shp")

    overload_lines_df = pd.read_csv(f"{base_dir}/overloaded_lines_records.csv")
    failed_lines_df = pd.read_csv(f"{base_dir}/failed_lines_records.csv")
    failed_nodes_df = pd.read_csv(f"{base_dir}/failed_buses_records.csv")
    # load_ratio_df = pd.read_csv(f"{base_dir}/load_served_ratio.csv")
    load_status_df = pd.read_csv(f"{base_dir}/load_status_records.csv")
    results_df = pd.read_csv(f"{base_dir}/percolation_overall_results.csv")

    nodes_gdf, lines_gdf, loads_gdf = assign_adm1_codes_with_ref_and_pcode(nodes_gdf, lines_gdf, loads_gdf, basemap)

    # 读取 Excel 为 DataFrame
    lines_df = pd.read_excel(f"{base_dir}/lines_gdf_sim_update.xlsx")
    nodes_df = pd.read_excel(f"{base_dir}/nodes_gdf_sim_update.xlsx")

    # 将 WKT 字符串转换为 Shapely geometry
    lines_df["geometry"] = lines_df["geometry"].apply(wkt.loads)
    nodes_df["geometry"] = nodes_df["geometry"].apply(wkt.loads)

    # 转为 GeoDataFrame
    lines_gdf_sim_update = gpd.GeoDataFrame(lines_df, geometry="geometry", crs="EPSG:4326")
    nodes_gdf_sim_update = gpd.GeoDataFrame(nodes_df, geometry="geometry", crs="EPSG:4326")
    
    lines_gdf_update = compute_failure_probabilities(lines_gdf_sim_update)
    nodes_gdf_update = compute_failure_probabilities(nodes_gdf_sim_update)

    plot_combined_failure_maps_extended(lines_gdf_update, nodes_gdf_update, basemap, fig_path=base_dir)

    # Merge overload ratio stats
    # === Step 1: 统计每条线路超载次数 ===
    overload_count = overload_lines_df.groupby("line_name").size().rename("overload_count").reset_index()
    # 真实参与仿真的总次数（成功且线路有结果）
    sim_combinations = overload_lines_df["fail_iter"].drop_duplicates()
    num_simulations = sim_combinations.shape[0]

    overload_count["overload_prob"] = overload_count["overload_count"] / num_simulations
    overload_count = overload_count.sort_values("overload_prob", ascending=False)
    overload_count.to_csv(os.path.join(base_dir, "overloaded_line_stats.csv"))

    lines_gdf = lines_gdf.merge(overload_count.rename(columns={"line_name": "LineID"}), on="LineID", how="left")
    lines_gdf["overload_prob"] = lines_gdf["overload_prob"].fillna(0)

    plot_overload_by_adm_ref_fixed_bins(lines_gdf, output_path=base_dir, basemap=basemap)
    plot_overload_by_adm_ref_continous(lines_gdf, output_path=base_dir, basemap=basemap)

    plot_load_failure_probability(load_status_df, loads_gdf, results_df, basemap=basemap, output_path=base_dir)

    # # --------------------------------------
    # # Step 1: Load successful fail_iters
    # passed_iters = (
    #     results_df[results_df["powerflow_success"] == True]["fail_iter"]
    #     .str.extract(r"fail_iter_(\d+)")
    #     .astype(int)
    #     .squeeze()
    #     .tolist()
    # )

    # print(f"✓ Found {len(passed_iters)} successful iterations")

    # # Step 2: Calculate failure ratio for each
    # fail_stats = []

    # for i in passed_iters:
    #     filepath = os.path.join(base_dir, f"net_results_fail_iter{i}.xlsx")
    #     try:
    #         xls = pd.read_excel(filepath, sheet_name=None)
    #         df_res = xls["res_load"]
    #         total = len(df_res)
    #         zero_count = (df_res["p_mw"] == 0).sum()
    #         ratio = zero_count / total if total > 0 else None
    #         fail_stats.append({
    #             "fail_iter": i,
    #             "total_loads": total,
    #             "zero_p_mw": zero_count,
    #             "zero_ratio": ratio
    #         })
    #     except Exception as e:
    #         fail_stats.append({
    #             "fail_iter": i,
    #             "total_loads": None,
    #             "zero_p_mw": None,
    #             "zero_ratio": None,
    #             "error": str(e)
    #         })

    # df_fail_ratios = pd.DataFrame(fail_stats)
    # df_fail_ratios.to_csv(os.path.join(base_dir, "df_fail_ratios.csv"))
    # print("{df_fail_ratios} zero_ratio mean: ", df_fail_ratios['zero_ratio'].mean())

    df_all, df_median, df_std, df_avg, failure_ratio, avg_failed_ratio = analyze_load_served_ratio_by_adm(
        loads_gdf=loads_gdf,
        load_status_df=load_status_df,
        results_df=results_df,
        net_dir=base_dir,
        output_path=base_dir
    )

    summary_df = export_summary_table(
        df_median=df_median, 
        df_std=df_std,
        df_avg=df_avg,
        avg_failed_ratio=avg_failed_ratio,
        loads_gdf=loads_gdf,
        output_path=base_dir
    )

    print("summary_df: ", summary_df)

if __name__ == "__main__":
    main()