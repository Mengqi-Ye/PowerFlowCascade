# failure_analysis_module.py

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib import colors, cm
import seaborn as sns
from scipy.stats import bootstrap
from matplotlib.cm import ScalarMappable
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString, MultiLineString

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
    major_parts = line_overlay[line_overlay["ratio"] > 0.6].copy()
    line_adm = major_parts[["LineID", "ADM1_EN", "ADM1_REF", "ADM1_PCODE"]].drop_duplicates()
    lines_gdf = lines_gdf.merge(line_adm, on="LineID", how="left")

    load_overlay = gpd.overlay(loads_gdf, basemap, how="intersection")
    load_overlay["area"] = load_overlay.geometry.area
    total_areas = load_overlay.groupby("osmid")["area"].sum().rename("total")
    load_overlay = load_overlay.join(total_areas, on="osmid")
    load_overlay["ratio"] = load_overlay["area"] / load_overlay["total"]
    major_parts = load_overlay[load_overlay["ratio"] > 0.6].copy()
    load_adm = major_parts[["osmid", "ADM1_EN", "ADM1_REF", "ADM1_PCODE"]].drop_duplicates()
    loads_gdf = loads_gdf.merge(load_adm, on="osmid", how="left")

    return nodes_gdf, lines_gdf, loads_gdf


def plot_failure_frequency_histograms(failed_lines_df, failed_nodes_df, output_dir):
    # Histogram for lines and nodes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    removal_fractions = [0.01, 0.05, 0.1, 0.15, 0.2]
    colors = ['lightgreen', 'skyblue', 'orange', 'gray', 'salmon']

    for rf, color in zip(removal_fractions, colors):
        df = failed_lines_df[np.isclose(failed_lines_df["removal_fraction"], rf, atol=1e-5)]
        freq = df["name"].dropna().value_counts() / df["iteration"].nunique()
        ax1.hist(freq, bins=20, alpha=0.5, label=f"removal={rf}", color=color)

    ax1.set_title("Line failure probability distribution", fontsize=13)
    ax1.set_ylabel("Line count", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='both', labelsize=12)

    for rf, color in zip(removal_fractions, colors):
        df = failed_nodes_df[np.isclose(failed_nodes_df["removal_fraction"], rf, atol=1e-5)]
        freq = df["name"].dropna().value_counts() / df["iteration"].nunique()
        ax2.hist(freq, bins=20, alpha=0.5, label=f"removal={rf}", color=color)

    ax2.set_title("Bus failure probability distribution", fontsize=13)
    ax2.set_xlabel("Failure probability", fontsize=13)
    ax2.set_ylabel("Bus count", fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.legend(fontsize=13)

    # Add subplot labels
    ax1.text(0.95, 0.02, "(a)", transform=ax1.transAxes,
            fontsize=13, va="bottom", ha="right") #, fontweight="bold"

    ax2.text(0.95, 0.02, "(b)", transform=ax2.transAxes,
            fontsize=13, va="bottom", ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "failure_hist_compare_removal.png"), dpi=600)


def combine_failure_and_overload_maps(lines_gdf, nodes_gdf, basemap, output_path):
    # Failure probability
    nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    lines_gdf = lines_gdf.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")
    # nodes_gdf = nodes_gdf[nodes_gdf.geometry.notnull() & nodes_gdf.geometry.is_valid]
    # lines_gdf = lines_gdf[lines_gdf.geometry.notnull() & lines_gdf.geometry.is_valid]

    pcode_list = ['VN106','VN111','VN101','VN107','VN103','VN109','VN113','VN117','VN115','VN104',
                  'VN717','VN711','VN707','VN713','VN701','VN709']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    fail_vmin = max(nodes_gdf["fail_prob_0_05"].min(), lines_gdf["fail_prob_0_05"].min())
    fail_vmax = max(nodes_gdf["fail_prob_0_05"].max(), lines_gdf["fail_prob_0_05"].max())
    fail_cmap = cm.Reds
    fail_norm = colors.Normalize(vmin=fail_vmin, vmax=fail_vmax)    
    # fail_bounds = np.linspace(0, max(0.3, round(fail_vmax + 0.01, 2)), 6)
    # fail_cmap = ListedColormap(plt.cm.Reds(np.linspace(0.3, 1, len(fail_bounds)-1)))
    # fail_norm = BoundaryNorm(fail_bounds, fail_cmap.N)

    nodes_zero = nodes_gdf[nodes_gdf["fail_prob_0_05"] == 0]
    nodes_nonzero = nodes_gdf[nodes_gdf["fail_prob_0_05"] > 0]

    lines_zero = lines_gdf[lines_gdf["fail_prob_0_05"] == 0]
    lines_nonzero = lines_gdf[lines_gdf["fail_prob_0_05"] > 0]

    node_threshold = nodes_gdf["fail_prob_0_05"].quantile(0.95)
    line_threshold = lines_gdf["fail_prob_0_05"].quantile(0.95)
    nodes_highlight = nodes_gdf[nodes_gdf["fail_prob_0_05"] >= node_threshold]
    lines_highlight = lines_gdf[lines_gdf["fail_prob_0_05"] >= line_threshold]
    
    print("nodes_zero:", len(nodes_zero), nodes_zero.geometry.isna().sum())
    print("nodes_nonzero:", len(nodes_nonzero), nodes_nonzero.geometry.isna().sum())
    print("lines_zero:", len(lines_zero), lines_zero.geometry.isna().sum())
    print("lines_nonzero:", len(lines_nonzero), lines_nonzero.geometry.isna().sum())

    print("nodes_highlight:", len(nodes_highlight))
    print("lines_highlight:", len(lines_highlight))

    # Overload probability
    overload_vmin = lines_gdf["overload_prob_0_05"].min()
    overload_vmax = lines_gdf["overload_prob_0_05"].max()
    # overload_bounds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, overload_vmax + 0.01]
    # overload_cmap = ListedColormap(plt.cm.YlOrRd(np.linspace(0.4, 1, len(overload_bounds)-1)))
    # overload_norm = BoundaryNorm(overload_bounds, overload_cmap.N)
    overload_cmap = cm.YlOrRd
    overload_norm = colors.Normalize(vmin=overload_vmin, vmax=overload_vmax)  

    overloaded = lines_gdf[lines_gdf["overload_prob_0_05"] > 0]
    non_overloaded = lines_gdf[lines_gdf["overload_prob_0_05"] == 0]

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={'hspace': 0.08, 'wspace': 0.02})

    legend_elements = [
        Line2D([0], [0], color='#98DF8A', lw=2, label='Failure/Overload p=0 (line)'),
        Line2D([0], [0], marker='o', color='#98DF8A', linestyle='None',
            markersize=8, label='Failure/Overload p=0 (bus)'),
        Line2D([0], [0], color='#9467BD', lw=2, label='Top 5% failure (line)'),
        Line2D([0], [0], marker='x', color='#9467BD', linestyle='None',
            markersize=10, label='Top 5% failure (bus)'),
    ]

    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if i == 0:
                ax.set_xticklabels([])
            if i == 1:
                ax.set_xlabel("Longitude", fontsize=13)
            if j == 0:
                ax.set_ylabel("Latitude", fontsize=13)

    for ax, title, xlim, ylim in [
        (axes[0, 0], "Red River Delta", (105, 107.2), (19.75, 21.7)),
        (axes[0, 1], "Southeast Region", (105.5, 108), (10.25, 12.5)),
        (axes[1, 0], "", (105, 107.2), (19.75, 21.7)),
        (axes[1, 1], "", (105.5, 108), (10.25, 12.5))]:
        if title:
            ax.set_title(title, fontsize=13)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # === 上排：Failure Map ===
    for ax in axes[0]:
        target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.5, zorder=1)
        other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.3, zorder=2)
        if not lines_zero.empty:
            lines_zero.plot(ax=ax, color='#98DF8A', linewidth=0.4, alpha=0.4, zorder=3)
        if not lines_nonzero.empty:
            lines_nonzero.plot(ax=ax, column="fail_prob", cmap=fail_cmap, norm=fail_norm, linewidth=1.0, zorder=4)
        if not lines_highlight.empty:
            # lines_highlight.plot(ax=ax, color='#9467BD', linewidth=2.0, zorder=9)
            for geom in lines_highlight.geometry:
                if isinstance(geom, LineString):
                    x, y = geom.xy
                    ax.plot(x, y, color='#9467BD', linewidth=1.2, zorder=9)
                elif isinstance(geom, MultiLineString):
                    for line in geom.geoms:
                        x, y = line.xy
                        ax.plot(x, y, color='#9467BD', linewidth=1.2, zorder=9)

        if not nodes_zero.empty:
            nodes_zero.plot(ax=ax, color='#98DF8A', markersize=8, alpha=0.5, zorder=6)
        if not nodes_nonzero.empty:
            nodes_nonzero.plot(ax=ax, column="fail_prob_0_05", cmap=fail_cmap, norm=fail_norm, markersize=12, zorder=7)
        # if nodes_highlight.empty:
        # #     nodes_highlight.plot(ax=ax, color='#9467BD', markersize=200, zorder=10, marker='x')
        #     plot_highlight_nodes(ax, nodes_highlight)

        # 替换 plot 为 scatter 显示 Top 5% failure nodes
        highlight_coords = nodes_highlight.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
        x_vals, y_vals = zip(*highlight_coords)
        ax.scatter(x_vals, y_vals, color='#9467BD', s=20, marker='x', zorder=10, linewidths=1.2)

    # === 下排：Overload Map ===
    for ax in axes[1]:
        target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.5, zorder=1)
        other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.3, zorder=2)
        non_overloaded.plot(ax=ax, color='#98DF8A', linewidth=0.5, alpha=0.5, zorder=3)
        overloaded.plot(ax=ax, column="overload_prob_0_05", cmap=overload_cmap, norm=overload_norm, linewidth=1.5, zorder=4)

    axes[1, 1].legend(handles=legend_elements, loc='upper left', fontsize=12)

    # === Colorbars ===
    sm1 = cm.ScalarMappable(cmap=fail_cmap, norm=fail_norm)
    sm1.set_array([])
    # tick_locs1 = [(fail_bounds[i] + fail_bounds[i+1]) / 2 for i in range(len(fail_bounds)-1)]
    # tick_labels1 = [f"{fail_bounds[i]:.2f}-{fail_bounds[i+1]:.2f}" for i in range(len(fail_bounds)-1)]
    # cbar1 = fig.colorbar(sm1, ax=axes[0], orientation="vertical", location='right', shrink=0.6, pad=0.01, ticks=tick_locs1)
    cbar1 = fig.colorbar(sm1, ax=axes[0], orientation="vertical", location='right', pad=0.01)
    # cbar1.ax.set_yticklabels(tick_labels1, fontsize=12)
    cbar1.set_label("Failure probability", fontsize=13)

    sm2 = cm.ScalarMappable(cmap=overload_cmap, norm=overload_norm)
    sm2.set_array([])
    # tick_locs2 = [(overload_bounds[i] + overload_bounds[i+1]) / 2 for i in range(len(overload_bounds)-1)]
    # tick_labels2 = [f"{overload_bounds[i]:.1f}-{overload_bounds[i+1]:.1f}" for i in range(len(overload_bounds)-1)]
    # cbar2 = fig.colorbar(sm2, ax=axes[1], orientation="vertical", location='right', shrink=0.6, pad=0.01, ticks=tick_locs2)
    cbar2 = fig.colorbar(sm2, ax=axes[1], orientation="vertical", location='right', pad=0.01)
    # cbar2.ax.set_yticklabels(tick_labels2, fontsize=12)
    cbar2.set_label("Overload probability", fontsize=13)

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "combined_failure_overload_map.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")



def plot_overload_histogram(lines_with_ratio, output_dir, basemap):
    """
    Plot overload ratios across all ADM1_REF regions.
    """
    os.makedirs(output_dir, exist_ok=True)
    lines_with_ratio = lines_with_ratio.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    # Histogram of overload ratio bins (vs. total lines)
    bins = np.arange(0.01, 1.01, 0.1)
    lines_with_ratio["overload_prob"] = lines_with_ratio["overload_prob"].fillna(0)

    total_lines_all = len(lines_with_ratio)
    lines_positive = lines_with_ratio[lines_with_ratio["overload_prob"] > 0]

    counts, bin_edges = np.histogram(lines_positive["overload_prob"], bins=bins)
    bin_labels = [f"{round(bin_edges[i],1)}–{round(bin_edges[i+1],1)}" for i in range(len(counts))]
    proportions = [f"{(c/total_lines_all*100):.2f}%" for c in counts]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(bin_labels, counts, color="skyblue", edgecolor="lightgray")
    plt.xlabel("Overload probability interval (excl. 0)", fontsize=13)
    plt.ylabel("Number of lines", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Overload probability distribution (% of all lines)", fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar, pct in zip(bars, proportions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, pct,
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overload_prob_hist_vs_all_lines.png"), dpi=600)


def analyze_failure_distribution_by_adm(nodes_gdf, lines_gdf, output_dir):
    """
    Generate failure boxplots for ADM1_EN regions (provinces).
    """

    node_groups = nodes_gdf.groupby("ADM1_EN")["fail_prob"]
    line_groups = lines_gdf.groupby("ADM1_EN")["fail_prob"]

    node_df = pd.DataFrame({
        "ADM1_EN": [k for k, v in node_groups],
        "fail_prob_list": [list(v) for k, v in node_groups]
    }).explode("fail_prob_list")

    line_df = pd.DataFrame({
        "ADM1_EN": [k for k, v in line_groups],
        "fail_prob_list": [list(v) for k, v in line_groups]
    }).explode("fail_prob_list")

    # Sort by median in ascending order
    node_order = node_df.groupby("ADM1_EN")["fail_prob_list"].median().sort_values(ascending=True).index.tolist()
    line_order = line_df.groupby("ADM1_EN")["fail_prob_list"].median().sort_values(ascending=True).index.tolist()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    sns.boxplot(data=node_df, x="ADM1_EN", y="fail_prob_list", ax=ax1, color='lightblue', order=node_order)
    ax1.set_title("Bus failure probability by province", fontsize=13)
    ax1.set_ylabel("Failure probability", fontsize=13)
    ax1.set_xlabel("")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.tick_params(axis='both', labelsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    sns.boxplot(data=line_df, x="ADM1_EN", y="fail_prob_list", ax=ax2, color='lightcoral', order=line_order)
    ax2.set_title("Line failure probability by province", fontsize=13)
    ax2.set_ylabel("Failure probability", fontsize=13)
    ax2.set_xlabel("Province", fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.tick_params(axis='both', labelsize=12)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # Add subplot labels
    ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes,
            fontsize=13, va="top", ha="left")

    ax2.text(0.01, 0.98, "(b)", transform=ax2.transAxes,
            fontsize=13, va="top", ha="left")

    # plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adm1_failure_boxplot.png"), dpi=600)
    

def summarize_top_failure_adm_regions(nodes_gdf, lines_gdf, output_dir, top=0.95):
    """
    Summarize top (1-top)*100% failure frequency lines/nodes per ADM1_REF region.
    """
    node_threshold = nodes_gdf["fail_prob_0_05"].quantile(top)
    line_threshold = lines_gdf["fail_prob_0_05"].quantile(top)

    nodes_top = nodes_gdf[nodes_gdf["fail_prob_0_05"] >= node_threshold]
    lines_top = lines_gdf[lines_gdf["fail_prob_0_05"] >= line_threshold]

    node_counts = nodes_top["ADM1_REF"].value_counts().rename("HighFailProbBuses")
    line_counts = lines_top["ADM1_REF"].value_counts().rename("HighFailProbLines")

    stats_df = pd.concat([node_counts, line_counts], axis=1).fillna(0).astype(int)
    stats_df = stats_df.sort_values(by=["HighFailProbBuses", "HighFailProbLines"], ascending=False)
    stats_df.to_csv(os.path.join(output_dir, f"adm1_top{int((1 - top) * 100)}%_failure_counts.csv"))

    ax = stats_df.plot(kind="bar", figsize=(12, 6), color=["steelblue", "salmon"]) #, edgecolor="#8C564B"

    ax.set_title(f"Number of top {int((1 - top) * 100)}% failure probability lines and buses", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=13)
    ax.tick_params(axis='y', labelsize=13)

    ax.legend(fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, f"adm1_top{(1-top)*100}_failure_bar.png"), dpi=600)


def analyze_load_served_ratio_by_adm(load_ratio_df, loads_gdf, output_dir):
    df = load_ratio_df[np.isclose(load_ratio_df["removal_fraction"], 0.05, atol=1e-5)]
    avg_served = df.groupby("osmid")["served_ratio_mean"].mean().rename("served_ratio")
    loads_gdf = loads_gdf.copy()
    loads_gdf["served_ratio"] = loads_gdf["osmid"].map(avg_served)
    filtered = loads_gdf.dropna(subset=["ADM1_EN", "served_ratio"])

    median_order = (
        filtered.groupby("ADM1_EN")["served_ratio"]
        .median()
        .sort_values(ascending=True)
        .reset_index()
        .rename(columns={"served_ratio": "served_ratio_median"})
    )
    median_order_list = median_order["ADM1_EN"].astype(str).tolist()
    counts = filtered["ADM1_EN"].value_counts().reindex(median_order_list).fillna(0)

    # Calculate standard deviation per ADM1_EN
    std_dev = filtered.groupby("ADM1_EN")["served_ratio"].std().reset_index(name="served_ratio_std")
    merged_stats = pd.merge(median_order, std_dev, on="ADM1_EN")

    # Identify top 5 unstable and stable provinces
    df_unstable = merged_stats.sort_values(by="served_ratio_std", ascending=False).head(5)
    df_stable = merged_stats.sort_values(by="served_ratio_std", ascending=True).head(5)
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Boxplot
    sns.boxplot(
        data=filtered,
        x="ADM1_EN",
        y="served_ratio",
        order=median_order_list,
        ax=ax1,
        color="skyblue"
    )

    # 均值 + CI
    # means = filtered.groupby("ADM1_EN")["served_ratio"].mean()
    # for xtick, adm1 in enumerate(median_order_list):
    #     group = filtered[filtered["ADM1_EN"] == adm1]["served_ratio"].values
    #     if len(group) > 1:
    #         ci = bootstrap((group,), np.mean, confidence_level=0.95, n_resamples=1000, method="basic")
    #         ci_low, ci_high = ci.confidence_interval
    #         ax1.plot(xtick, means[adm1], 'ro')  # mean
    #         ax1.vlines(xtick, ci_low, ci_high, color='#8C564B', linewidth=1.2)
    #     else:
    #         ax1.plot(xtick, means[adm1], 'ro')

    # 平均线
    # ax1.axhline(filtered["served_ratio"].mean(), color="green", linestyle="--", linewidth=1.2, label="Global Mean")
    ax1.set_ylabel("Served ratio", fontsize=13)
    ax1.set_xlabel("Provinces (sorted by median served ratio)", fontsize=13)
    ax1.set_xticklabels(median_order_list, rotation=45, ha="right")
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Color xtick labels for stable/unstable
    xtick_labels = ax1.get_xticklabels()
    for label in xtick_labels:
        adm = label.get_text()
        if adm in df_unstable["ADM1_EN"].values:
            label.set_fontweight("bold")  # red
        elif adm in df_stable["ADM1_EN"].values:
            label.set_color("#2CA02C")  # green

    # === 右侧 y 轴 ===
    ax2 = ax1.twinx()
    ax2.bar(median_order_list, counts.values, alpha=0.25, color="gray", label="Load count")
    ax2.set_ylabel("Load count", fontsize=13)
    # ax2.legend(loc="upper right", fontsize=13)
    ax2.legend(loc='center right', bbox_to_anchor=(1, 0.25), fontsize=13)

    plt.title("Loads served ratio distribution and load count by province (removal_fraction = 0.05)", fontsize=13)
    plt.tight_layout()
    # os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "load_served_ratio_boxplot_by_adm1_count.png"), dpi=400)

    return filtered, median_order, std_dev, df_unstable, df_stable

def analyze_powerflow_success(results_df, output_dir):
    df = results_df.copy()
    df["removal_fraction"] = df["removal_fraction"].round(4)

    powerflow_summary = (
        df.groupby("removal_fraction")["powerflow_success"]
        .mean().reset_index(name="success_rate")
    )

    plt.figure(figsize=(7, 4))
    plt.plot(powerflow_summary["removal_fraction"], powerflow_summary["success_rate"], marker="o", color="green")
    plt.xlabel("Bus removal fraction")
    plt.ylabel("Powerflow success rate")
    plt.title("Powerflow feasibility vs Removal fraction")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    powerflow_summary.to_csv(os.path.join(output_dir, f"powerflow_success_summary.csv"))
    plt.savefig(os.path.join(output_dir, "powerflow_success_rate.png"), dpi=600)


def analyze_connectivity_vs_supply(results_df, output_dir):
    cmap = cm.get_cmap("GnBu")  # GnBu
    norm = Normalize(vmin=results_df["removal_fraction"].min(),
                     vmax=results_df["removal_fraction"].max())

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=results_df,
        x="giant_component_fraction",
        y="load_served_ratio",
        hue="removal_fraction",
        palette="GnBu",
        alpha=0.5,
        edgecolor="white",
        linewidth=0.2,
        legend=False,
        ax=ax
    )

    # # 添加拟合线
    # sns.regplot(
    #     data=results_df,
    #     x="giant_component_fraction",
    #     y="load_served_ratio",
    #     scatter=False,
    #     lowess=True,
    #     color="#888888",
    #     line_kws={"linewidth": 1.5},
    #     ax=ax
    # )

    # 添加 colorbar 替代 legend
    smap = ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])
    cbar = fig.colorbar(smap, ax=ax)
    cbar.set_label("Removal fraction", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # lowess_line = mlines.Line2D([], [], color="#888888", linewidth=1.5, label="Fitted curve")
    # plt.legend(handles=[lowess_line], loc="lower right", fontsize=13)
    
    ax.set_xlabel("Giant component fraction", fontsize=13)
    ax.set_ylabel("Load served ratio", fontsize=13)
    ax.set_title("Structural connectivity vs Load supply", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "connectivity_vs_supply.png"), dpi=600)

def export_summary_table(df_median, df_std, df_avg, avg_failed_ratio, loads_gdf, output_dir):
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
        .rename(columns={"osmid": "number of loads"})
    )

    summary_df = summary_df.merge(load_counts, on="ADM1_EN", how="left")
    summary_df["served_ratio_mean"] = (summary_df["served_ratio_mean"] * 100).round(1)
    summary_df["avg_failed_ratio"] = (summary_df["avg_failed_ratio"] * 100).round(1)
    summary_df = summary_df.sort_values(by="served_ratio_mean", ascending=True)

    summary_df.to_excel(os.path.join(output_dir, "adm1_served_summary.xlsx"), index=False)

    return summary_df


def main():
    # Load input files
    base_dir = "../outputs/20250729_0_0.3_31_1000"
    # base_dir = "../outputs/20250729_0_0.3_7_10"
    overload_lines_df = pd.read_csv(f"{base_dir}/overloaded_lines_records.csv")
    failed_lines_df = pd.read_csv(f"{base_dir}/failed_lines_records.csv")
    failed_nodes_df = pd.read_csv(f"{base_dir}/failed_buses_records.csv")
    load_ratio_df = pd.read_csv(f"{base_dir}/load_served_ratio.csv")
    load_status_df = pd.read_csv(f"{base_dir}/load_status_records.csv")
    results_df = pd.read_csv(f"{base_dir}/percolation_overall_results.csv")
    # percolation_result_csv=(f"{base_dir}/percolation_overall_results.csv")

    lines_gdf = gpd.read_file('../outputs/table_lines_200m_update_remove_disconnected.gpkg')
    nodes_gdf = gpd.read_file('../outputs/table_nodes_200m_update_remove_disconnected.gpkg')
    loads_gdf = gpd.read_file('../outputs/landuse_sites_gdf_add_bus.gpkg')
    basemap = gpd.read_file("../data/base_map/vnm_admbnda_adm1_gov_20201027.shp")

    output_dir = "../figures/20250729_0_0.3_31_1000"
    # output_dir = "../figures/overload_analysis_250729"
    os.makedirs(output_dir, exist_ok=True)

    # Assign ADM1_REF and ADM1_PCODE
    nodes_gdf, lines_gdf, loads_gdf = assign_adm1_codes_with_ref_and_pcode(nodes_gdf, lines_gdf, loads_gdf, basemap)

    # Calculate failure frequency at 0.05
    iter_lines = failed_lines_df[np.isclose(failed_lines_df["removal_fraction"], 0.05)]
    iter_nodes = failed_nodes_df[np.isclose(failed_nodes_df["removal_fraction"], 0.05)]
    num_iter = iter_lines["iteration"].nunique()

    lines_gdf["fail_prob"] = lines_gdf["LineID"].map(iter_lines["name"].value_counts() / num_iter).fillna(0)
    nodes_gdf["fail_prob"] = nodes_gdf["NodeID"].map(iter_nodes["name"].value_counts() / num_iter).fillna(0)

    # Merge overload ratio stats
    # 仅选取 removal_fraction 为 0.05 的模拟结果
    overload_lines_df_0_05 = overload_lines_df[np.isclose(overload_lines_df["removal_fraction"], 0.05)]
    
    # 统计每条线路超载次数
    overload_count = overload_lines_df_0_05.groupby("line_name").size().rename("overload_count").reset_index()
    # 真实参与仿真的总次数（成功且线路有结果）
    sim_combinations = overload_lines_df_0_05[["removal_fraction", "iteration"]].drop_duplicates()
    num_simulations = sim_combinations.shape[0]
    
    overload_count["overload_prob"] = overload_count["overload_count"] / num_simulations
    overload_count = overload_count.sort_values("overload_prob", ascending=False)
    overload_count.to_csv(os.path.join(output_dir, "overloaded_line_stats.csv"))
    
    lines_gdf = lines_gdf.merge(overload_count.rename(columns={"line_name": "LineID"}), on="LineID", how="left")
    lines_gdf["overload_prob"] = lines_gdf["overload_prob"].fillna(0)

    # Run all analysis modules
    # plot_failure_maps_and_histograms(failed_lines_df, failed_nodes_df, lines_gdf, nodes_gdf, basemap, output_dir)
    # plot_overload_by_adm_ref(lines_gdf, output_dir, basemap)
    plot_failure_frequency_histograms(failed_lines_df, failed_nodes_df, output_dir)
    combine_failure_and_overload_maps(lines_gdf, nodes_gdf, basemap, output_dir)
    plot_overload_histogram(lines_gdf, output_dir, basemap)

    summarize_top_failure_adm_regions(nodes_gdf, lines_gdf, output_dir)
    analyze_failure_distribution_by_adm(nodes_gdf, lines_gdf, output_dir)
    # analyze_load_served_ratio_by_adm(load_ratio_df, loads_gdf, output_dir)
    # analyze_enip_curve(results_df, output_dir)
    analyze_powerflow_success(results_df, output_dir)
    analyze_connectivity_vs_supply(results_df, output_dir)

    df_all, df_median, df_std, df_avg, failure_ratio, avg_failed_ratio = analyze_load_served_ratio_by_adm(
        loads_gdf=loads_gdf,
        load_status_df=load_status_df,
        results_df=results_df,
        output_dir=output_dir
    )

    summary_df = export_summary_table(
        df_median=df_median, 
        df_std=df_std,
        df_avg=df_avg,
        avg_failed_ratio=avg_failed_ratio,
        loads_gdf=loads_gdf,
        output_dir=base_dir
    )

if __name__ == "__main__":
    main()
