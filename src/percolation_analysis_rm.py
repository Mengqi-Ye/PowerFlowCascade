# percolation_analysis.py

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import geopandas as gpd
import pandapower as pp
import pandapower.topology as top
import matplotlib.cm as cm
import matplotlib.colors as colors
from pandapower.plotting.plotly import pf_res_plotly
import plotly.io as pio
import argparse
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from pandapower.plotting import pf_res_plotly

from power_flow import build_pandapower_model

def simulate_percolation_with_powerflow_and_loadstatus(nodes_gdf, lines_gdf, gens_gdf, loads_gdf, trafo_gdf, 
                                                       num_iterations=10, #target_iteration=1,
                                                       use_column_in_service = False,
                                                       remove_fractions=None, fig_path=None): #, plot_target=None
    if remove_fractions is None:
        # remove_fractions = np.linspace(0, 0.15, 16)
        remove_fractions = np.linspace(0, 0.02, 3) #for test

    base_net = build_pandapower_model(nodes_gdf, lines_gdf, gens_gdf, loads_gdf, trafo_gdf, use_column_in_service)
    all_buses = base_net.bus.index.tolist()
    total_bus_count = len(base_net.bus)
    total_line_count = len(base_net.line)

    results = []
    load_status_records = []
    failed_buses_records = []
    failed_lines_records = []  # To store failed lines
    overloaded_lines_records = []

    # os.makedirs("../figures/red_river", exist_ok=True)
    # os.makedirs("../figures/southeast", exist_ok=True)
    # os.makedirs("../figures/raw", exist_ok=True)

    for frac in tqdm(remove_fractions, desc="Processing removal fractions"):
        n_remove = int(frac * total_bus_count)
        # if n_remove == 0:
        #     continue  # Skip iteration if no buses are to be removed

        # for i in range(num_iterations):
        for i in tqdm(range(num_iterations), desc=f"Simulating frac={frac:.4f}", leave=False):
            net = copy.deepcopy(base_net)
            # net.bus["in_service"] = True
            # net.line["in_service"] = True
            
            failed_buses = np.random.choice(all_buses, size=n_remove, replace=False)
            net.bus.loc[failed_buses, "in_service"] = False

            for fb in failed_buses:
                failed_buses_records.append({
                    "iteration": i,
                    "removal_fraction": frac,
                    "bus_index": fb,
                    "name": net.bus.loc[fb, "name"]
                })

            # # Randomly fail lines based on bus failures
            # failed_lines = []
            # for idx, line in net.line.iterrows():
            #     if not net.bus.loc[line["from_bus"], "in_service"] or not net.bus.loc[line["to_bus"], "in_service"]:
            #         net.line.at[idx, "in_service"] = False
            #         failed_lines.append((line['name'], line["from_bus"], line["to_bus"]))

            # # Track the failed lines
            # for fl in failed_lines:
            #     failed_lines_records.append({
            #         "iteration": i,
            #         "removal_fraction": frac,
            #         "name": fl[0],
            #         "from_bus": fl[1],
            #         "to_bus": fl[2]
            #     })

            failed_lines_mask = ~net.bus.loc[net.line["from_bus"], "in_service"].values | \
                                ~net.bus.loc[net.line["to_bus"], "in_service"].values
            net.line.loc[failed_lines_mask, "in_service"] = False

            for idx in net.line[failed_lines_mask].index:
                line = net.line.loc[idx]
                failed_lines_records.append({
                    "iteration": i,
                    "removal_fraction": frac,
                    "name": line['name'],
                    "from_bus": line["from_bus"],
                    "to_bus": line["to_bus"]
                })

            try:
                G = top.create_nxgraph(net, include_switches=True)
                comps = list(top.connected_components(G))
                giant_size = max((len(c) for c in comps), default=0)
                num_components = len(comps)
            except Exception:
                giant_size = 0
                num_components = np.nan

            try:
                # pp.runpp(net, init="auto", calculate_voltage_angles=True)
                pp.runpp(net)
                success = True
                error_msg = ""
            except Exception as e:
                success = False
                error_msg = str(e)

            if success:
                # pp.to_excel(net, os.path.join(fig_path, f"net_results_{int(frac*100)}%_iter{i}.xlsx"))
                voltages = net.res_bus.vm_pu[net.bus.in_service]
                under = sum(voltages < 0.95)
                over = sum(voltages > 1.05)
                voltage_violations = under + over

                total_load = net.load.p_mw.sum()
                total_supplied = net.res_load.p_mw.sum()
                load_served_ratio = total_supplied / total_load if total_load > 0 else np.nan

                overload_lines = net.res_line[(net.res_line.loading_percent > 100) & net.line["in_service"]]
                for idx, row in overload_lines.iterrows():
                    overloaded_lines_records.append({
                        "iteration": i,
                        "removal_fraction": frac,
                        "line_index": idx,
                        "line_name": net.line.at[idx, "name"],
                        "loading_percent": row["loading_percent"]
                    })
                    
                overload_lines_count = len(overload_lines)
                overload_lines_ratio = overload_lines_count / net.line.in_service.sum() if net.line.in_service.sum() > 0 else 0

                for idx, row in net.load.iterrows():
                    load_bus = row["bus"]
                    is_served = (
                        net.bus.at[load_bus, "in_service"]
                        and not np.isnan(net.res_load.at[idx, "p_mw"])
                        and net.res_load.at[idx, "p_mw"] > 0
                    )
                    load_status_records.append({
                        "iteration": i,
                        "removal_fraction": frac,
                        "load_id": net.load.at[idx, "name"],
                        "bus": load_bus,
                        "served": is_served
                    })

                # # Generate the plot for power flow results
                # fig_pf = pf_res_plotly(net, cmap='Jet', use_line_geo=True, projection='epsg:4326',
                #                     line_width=0.5, bus_size=5) # , climits_volt=(0.95, 1.05)
                # pio.write_html(fig_pf, 
                #                file=os.path.join(fig_path, f"Network_pf_VN_{int(frac*100)}%_iter{target_iteration}.html"),
                #                auto_open=False, include_plotlyjs='cdn')
                
                # fig_pf_1 = go.Figure(data=fig_pf.data)
                # fig_pf_1.update_layout(
                #     width=800,
                #     height=600,
                #     # title=f"Power flow results (Failure {int(frac*100)}%, Iteration {i})",
                #     title=f"Power flow results - Red River Delta (Failure fraction {int(frac*100)}%, Iteration {target_iteration})",
                #     margin=dict(l=10, r=10, t=50, b=10),
                #     xaxis=dict(range=[521492, 735117]),  # Set x-axis range (min, max values)
                #     yaxis=dict(range=[2189746, 2392830]),   # Set y-axis range (min, max values)
                #     showlegend=False  # Remove the legend
                # )
                # fig_pf_1.update_yaxes(scaleanchor="x", scaleratio=1)
                
                # fig_pf_2 = go.Figure(data=fig_pf.data)
                # fig_pf_2.update_layout(
                #     width=800,
                #     height=600,
                #     # title=f"Power flow results (Failure {int(frac*100)}%, Iteration {i})",
                #     title=f"Power flow results - Southeast Region (Failure fraction {int(frac*100)}%, Iteration {target_iteration})",
                #     margin=dict(l=10, r=10, t=50, b=10),
                #     xaxis=dict(range=[577612, 794647]),  # Set x-axis range (min, max values)
                #     yaxis=dict(range=[1127822, 1371522]),   # Set y-axis range (min, max values)
                #     showlegend=False  # Remove the legend
                # )
                # fig_pf_2.update_yaxes(scaleanchor="x", scaleratio=1)
    
                # # Save the figure PNG
                # fig_pf_1.write_image(os.path.join(fig_path, f"Network_pf_subset_RedRiver_{int(frac*100)}%_iter{target_iteration}.png"),scale=6)  # Save as PNG with higher resolution
                # fig_pf_2.write_image(os.path.join(fig_path, f"Network_pf_subset_Southeast_{int(frac*100)}%_iter{target_iteration}.png"), scale=6)  # Save as PNG with higher resolution

            else:
                under = over = voltage_violations = np.nan
                overload_lines_count = overload_lines_ratio = 0
                total_load = net.load.p_mw.sum()
                total_supplied = np.nan
                load_served_ratio = np.nan
                max_voltage = min_voltage = np.nan
                num_components = np.nan

                load_status_records.extend([
                    {
                        "iteration": i,
                        "removal_fraction": frac,
                        "load_id": row["name"],
                        "bus": row["bus"],
                        "served": False
                    }
                    for _, row in net.load.iterrows()
                ])

            results.append({
                "removal_fraction": frac,
                "iteration": i,
                "giant_component_fraction": giant_size / total_bus_count,
                "num_components": num_components,
                "powerflow_success": success,
                "num_failed_buses": n_remove,
                "num_failed_lines": failed_lines_mask.sum(),
                "failed_line_ratio": failed_lines_mask.sum()/total_line_count,
                "voltage_under_0.95": under,
                "voltage_over_1.05": over,
                "voltage_violations": voltage_violations,
                "mean_voltage": voltages.mean() if success else np.nan,
                "std_voltage": voltages.std() if success else np.nan,
                "max_voltage": voltages.max() if success else np.nan,
                "min_voltage": voltages.min() if success else np.nan,
                "total_bus_count": total_bus_count,
                "total_line_count": total_line_count,
                "total_load": total_load,
                "total_supplied": total_supplied,
                "load_served_ratio": load_served_ratio,
                "supply_loss_ratio": 1 - load_served_ratio if total_load > 0 else np.nan,
                "line_overload_violations": overload_lines_count,
                "overload_line_ratio": overload_lines_ratio,
                "avg_line_loading": net.res_line.loading_percent[net.line["in_service"]].mean() if success else np.nan,
                "max_line_loading": net.res_line.loading_percent[net.line["in_service"]].max() if success else np.nan,
                "min_line_loading": net.res_line.loading_percent[net.line["in_service"]].min() if success else np.nan,
                "error_msg": error_msg
            })

            if success and fig_path:
                pp.to_excel(net, os.path.join(fig_path, f"net_results_frac_{int(round(frac*100))}_fail_iter{i+1}.xlsx"))

    return (
        pd.DataFrame(results),
        pd.DataFrame(load_status_records),
        pd.DataFrame(failed_buses_records),
        pd.DataFrame(failed_lines_records),
        pd.DataFrame(overloaded_lines_records)
    )

# # OLD version
# def prepare_loads_map(load_status_df, loads_gdf, target_iteration=None, target_fraction=None):
#     # Ensure ID types are consistent
#     loads_gdf["osmid"] = loads_gdf["osmid"].astype(int)

#     load_status_df = load_status_df.rename(columns={"load_id": "osmid"})
#     load_status_df["osmid"] = load_status_df["osmid"].astype(int)

#     # Filter based on selected iteration and removal fraction
#     # target_fraction = round(target_fraction, 4)
#     load_status_df["removal_fraction"] = load_status_df["removal_fraction"].round(4)
    
#     # filtered = load_status_df[
#     #     (load_status_df["iteration"] == target_iteration) &
#     #     (load_status_df["removal_fraction"] == target_fraction)
#     # ]
#     # print(f"Selected {len(filtered)} rows for iteration={target_iteration}, fraction={target_fraction}")
#     # print("Served value counts: ", filtered["served"].value_counts())

#     filtered = load_status_df[load_status_df["iteration"] == target_iteration]
#     # print(f"Selected {len(filtered)} rows for iteration={target_iteration}")

#     # Map binary served status to loads
#     served_map = dict(zip(filtered.osmid, filtered.served))
#     loads_gdf["served"] = loads_gdf["osmid"].map(served_map).fillna(False).astype(bool)

#     # Calculate average served ratio across simulations, grouped by osmid and target_fraction
#     # avg_status = load_status_df.groupby("osmid")["served"].mean().rename("served_ratio").reset_index()
#     avg_status = load_status_df.groupby(["osmid", "removal_fraction"])["served"].mean().rename("served_ratio").reset_index()
#     # print(avg_status)

#     loads_gdf = loads_gdf.merge(avg_status, on="osmid", how="left")
#     # loads_gdf["removal_fraction"] = loads_gdf["removal_fraction"].fillna(0)
#     # loads_gdf["served_ratio"] = loads_gdf["served_ratio"].fillna(0)

#     return loads_gdf


# Successful
# def prepare_loads_map(load_status_df, loads_gdf, target_iteration=None):
#     loads_gdf = loads_gdf.copy()
#     load_status_df = load_status_df.copy()

#     # Ensure ID types are consistent
#     loads_gdf["osmid"] = loads_gdf["osmid"].astype(int)

#     load_status_df = load_status_df.rename(columns={"load_id": "osmid"})
#     load_status_df["osmid"] = load_status_df["osmid"].astype(int)

#     # Filter based on selected iteration and removal fraction
#     load_status_df["removal_fraction"] = load_status_df["removal_fraction"].round(4)
#     filtered = load_status_df[load_status_df["iteration"] == target_iteration]

#     # Map binary served status to loads
#     served_map = dict(zip(filtered.osmid, filtered.served))
#     loads_gdf["served"] = loads_gdf["osmid"].map(served_map).fillna(False).astype(bool)

#     # Compute average served ratio per osmid per frac
#     # avg_status = load_status_df.groupby(["osmid", "removal_fraction"])["served"].mean().rename("served_ratio").reset_index()
#     avg_status = (
#         load_status_df
#         .groupby(["osmid", "removal_fraction"], as_index=False)["served"]
#         .mean()
#         .rename(columns={"served": "served_ratio"})
#     )
#     loads_gdf = loads_gdf.merge(avg_status, on="osmid", how="left")

#     return loads_gdf


def prepare_loads_map(load_status_df, loads_gdf):
    loads_gdf = loads_gdf.copy()
    load_status_df = load_status_df.copy()

    loads_gdf["osmid"] = pd.to_numeric(loads_gdf["osmid"], errors="coerce").dropna().astype(int)
    load_status_df = load_status_df.rename(columns={"load_id": "osmid"})
    load_status_df["osmid"] = pd.to_numeric(load_status_df["osmid"], errors="coerce").dropna().astype(int)

    load_status_df["removal_fraction"] = load_status_df["removal_fraction"].round(4)

    # Compute average, min, and max served ratio per osmid per frac
    avg_status = (
        load_status_df.groupby(["osmid", "removal_fraction"])["served"]
        .agg(served_ratio_mean="mean", served_ratio_min="min", served_ratio_max="max")
        .reset_index()
    )

    loads_gdf = loads_gdf.merge(avg_status, on="osmid", how="left")

    return loads_gdf


def plot_served_ratio_maps(loads_gdf, fig_path=None, basemap=None):
    # 将所有数据转换为 EPSG:4326 坐标系
    loads_gdf = loads_gdf.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    pcode_list = ['VN106','VN111','VN101','VN107','VN103','VN109','VN113','VN117','VN115','VN104','VN717','VN711','VN707','VN713','VN701','VN709']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    # 获取所有唯一的 removal_fraction 值
    # unique_fractions = loads_gdf["removal_fraction"].unique()
    # 聚合：按 geometry 平均 served_ratio_mean（也可替换为 load_id）
    avg_served = loads_gdf.groupby(loads_gdf.geometry).agg({
        'served_ratio_mean': 'mean'
    }).reset_index()
    avg_served_gdf = gpd.GeoDataFrame(avg_served, geometry='geometry', crs=loads_gdf.crs)

    vmin = avg_served_gdf["served_ratio_mean"].min()
    vmax = avg_served_gdf["served_ratio_mean"].max()
    # vmin = loads_gdf["served_ratio_mean"].min()
    # vmax = loads_gdf["served_ratio_mean"].max()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps.get_cmap("RdYlGn")
    
    os.makedirs(fig_path, exist_ok=True)

    # for fraction in unique_fractions:
    #     # 筛选出当前 removal_fraction 的数据
    #     fraction_data = loads_gdf[loads_gdf["removal_fraction"] == fraction]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)  # 创建1行2列的子图

    # Red River Delta (ADM1_PCODE: VN1xx)
    ax1.set_title("Red River Delta", fontsize=15)
    ax1.set_xlabel("Longitude", fontsize=15)
    ax1.set_ylabel("Latitude", fontsize=15)
    ax1.set_xlim(105, 107.25)
    ax1.set_ylim(19.75, 21.75)
    ax1.set_aspect('auto')
    ax1.tick_params(axis='both', which='major', labelsize=14)

    target_regions.plot(ax=ax1, color='lightgray', edgecolor='white', linewidth=0.8, zorder=1)
    other_regions.plot(ax=ax1, color='white', edgecolor='lightgray', linewidth=0.5, zorder=2)
    # fraction_data.plot(ax=ax1, column="served_ratio_mean", cmap=cmap, norm=norm, markersize=100, legend=False, zorder=3)
    avg_served_gdf.plot(ax=ax1, column="served_ratio_mean", cmap=cmap, norm=norm, markersize=100, legend=False, zorder=3)

    # Southeast Vietnam (ADM1_PCODE: VN7xx)
    ax2.set_title("Southeast Region", fontsize=15)
    ax2.set_xlabel("Longitude", fontsize=15)
    ax2.set_xlim(105.5, 108)
    ax2.set_ylim(10.25, 12.5)
    ax2.set_aspect('auto')
    ax2.tick_params(axis='both', which='major', labelsize=14)

    target_regions.plot(ax=ax2, color='lightgray', edgecolor='white', linewidth=0.8, zorder=1)
    other_regions.plot(ax=ax2, color='white', edgecolor='lightgray', linewidth=0.5, zorder=2)
    # fraction_data.plot(ax=ax2, column="served_ratio_mean", cmap=cmap, norm=norm, markersize=100, legend=False, zorder=3)
    avg_served_gdf.plot(ax=ax2, column="served_ratio_mean", cmap=cmap, norm=norm, markersize=100, legend=False, zorder=3)

    # Add shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # 伪数据，仅为触发色标绘制
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation="vertical",location='right', fraction=0.02)
    cbar.set_label("Served load ratio", fontsize=15)

    if fig_path:
        # output_path = f"{fig_path}/served_ratio_map_frac{int(fraction*100)}%.png"
        output_path = os.path.join(fig_path, 'served_ratio_map_average.png')
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        # print(f"Saved map for removal_fraction {fraction} at {output_path}")
    

# def debug_load_status_mapping(load_status_df, loads_gdf, loads_result):
#     print("===== DEBUG: served 列分布 =====")
#     print(load_status_df["served"].value_counts(dropna=False))
#     print("\n===== DEBUG: load_id 和 osmid 类型 =====")
#     print("load_id:", load_status_df["load_id"].dtype)
#     print("osmid:", loads_gdf["osmid"].dtype)
#     print("\n===== DEBUG: removal_fraction 示例值 =====")
#     print(load_status_df["removal_fraction"].unique()[:10])
#     print("\n===== DEBUG: 映射后的 served 分布 =====")
#     print(loads_result["served"].value_counts(dropna=False))
#     print("\n===== DEBUG: 映射后 served_ratio 范围 =====")

#     print("样例 load_id in load_status_df:")
#     print(load_status_df["load_id"].astype(str).unique()[:5])

#     print("样例 osmid in loads_gdf:")
#     print(loads_gdf["osmid"].astype(str).unique()[:5])

#     # 交集长度
#     intersection = set(load_status_df["load_id"].astype(str)) & set(loads_gdf["osmid"].astype(str))
#     print(f"可匹配的 load_id 数量：{len(intersection)} / {len(load_status_df['load_id'].unique())}")

#     if "served_ratio" in loads_result.columns:
#         print("min:", loads_result["served_ratio"].min())
#         print("max:", loads_result["served_ratio"].max())
#     else:
#         print("served_ratio 未计算")
#     print("\n===== DEBUG: 映射后样例行 =====")
#     print(loads_result[["osmid", "served"]].head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--target_iteration", type=int, default=0, help="Target iteration to visualize")
    parser.add_argument("--num_iterations", type=int, default=2, help="Number of Monte Carlo simulation iterations")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--use_column_in_service", action='store_true')

    # Add the --remove_fractions argument, expect a string that can be converted into a list of floats
    parser.add_argument("--remove_fractions", type=str, 
                        default="np.linspace(0, 0.3, 31)", 
                        help="List of fractions for node removal, e.g. 'np.linspace(0, 0.6, 7)'")
    
    args = parser.parse_args()

    # Parse the remove_fractions argument and evaluate the expression if it's a valid numpy command
    remove_fractions = eval(args.remove_fractions)

    # === Read power network data ===
    lines_gdf = gpd.read_file('../outputs/table_lines_200m_update_remove_disconnected.gpkg')
    nodes_gdf = gpd.read_file('../outputs/table_nodes_200m_update_remove_disconnected.gpkg')
    gens_gdf = gpd.read_file('../outputs/plant_update.gpkg')
    loads_gdf = gpd.read_file('../outputs/landuse_sites_gdf_add_bus.gpkg') # updated version data
    trafo_gdf = gpd.read_file('../outputs/table_transformers_remove_disconnected.gpkg')

    basemap = gpd.read_file("../data/base_map/vnm_admbnda_adm1_gov_20201027.shp")
    
    fig_path = f"../figures/{args.timestamp}"
    output_path = f"../outputs/{args.timestamp}"

    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Run simulation
    results, load_status_df, failed_buses_df, failed_lines_df, overloaded_lines_df = simulate_percolation_with_powerflow_and_loadstatus(
        nodes_gdf, lines_gdf, gens_gdf, loads_gdf, trafo_gdf,
        num_iterations=args.num_iterations,
        # target_iteration=args.target_iteration,
        use_column_in_service=args.use_column_in_service,
        remove_fractions=remove_fractions,
        fig_path = fig_path
    )

    # Save the results
    results.to_csv(os.path.join(output_path, "percolation_overall_results.csv"), index=False)
    load_status_df.to_csv(os.path.join(output_path, "load_status_records.csv"), index=False)
    failed_buses_df.to_csv(os.path.join(output_path, "failed_buses_records.csv"), index=False)
    failed_lines_df.to_csv(os.path.join(output_path, "failed_lines_records.csv"), index=False)
    overloaded_lines_df.to_csv(os.path.join(output_path, "overloaded_lines_records.csv"), index=False)

    # Visulization
    df_mean = results.groupby("removal_fraction")["giant_component_fraction"].mean()
    df_std = results.groupby("removal_fraction")["giant_component_fraction"].std()

    plt.figure(figsize=(8,5))
    plt.plot(df_mean.index, df_mean, label="Mean giant component")
    plt.fill_between(df_mean.index, df_mean - df_std, df_mean + df_std, alpha=0.3, label="±1 std")
    plt.xlabel("Fraction of buses removed", fontsize=15)
    plt.ylabel("Giant component fraction", fontsize=15)
    plt.title("Percolation analysis", fontsize=15)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "percolation_curve_random_node_removal.png"))

    loads_gdf_update = prepare_loads_map(load_status_df, loads_gdf)
    loads_gdf_update.to_csv(os.path.join(output_path, "load_served_ratio.csv"), index=False)

    plot_served_ratio_maps(loads_gdf_update, fig_path=fig_path, basemap=basemap)

    # debug_load_status_mapping(load_status_df, loads_gdf, loads_result_gdf)
