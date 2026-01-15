# percolation_analysis_tc.py

import geopandas as gpd
import os
import rasterio
import pandas as pd
import numpy as np
import pandapower as pp
import sys
import copy
import argparse
import plotly.io as pio
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from rasterstats import zonal_stats
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
from pandapower.plotting.plotly import pf_res_plotly

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'src')))
sys.path.append(os.path.abspath(os.getcwd()))
from power_flow import build_pandapower_model

# === Read fragility curves ===
def load_curves(curve_path, sheet_name=None):
    # 先读取前5行用于获取正确的列名（ID number 行）
    raw = pd.read_excel(curve_path, sheet_name=sheet_name, nrows=5)
    
    # 用第1列为 index（即 'Wind speed (m/s)'），第2~n列为 curve ID
    id_row = raw.iloc[0, 1:]  # “ID number” 行，作为列名
    id_row.index = range(1, len(id_row)+1)

    # 读取真正的数据，从第6行开始
    df = pd.read_excel(curve_path, sheet_name=sheet_name, skiprows=5)

    # 将列名替换为 ['Wind speed', ID1, ID2, ...]
    df.columns = ['Wind speed'] + list(id_row)

    # 保证所有数值列都是 float
    df = df.apply(pd.to_numeric, errors='coerce')

    # # design wind speed (dws)
    # original_dws = 44
    # target_dws = 60
    # # shift design wind speed of all curves to target_dws (60 m/s)
    # scaling_factor = original_dws / target_dws
    # df['Wind speed'] = df['Wind speed'] * scaling_factor

    # # interpolate the curves to fill missing values
    # df = df.interpolate()

    return df


def extract_point_wind_speed(gdf, src):
    """
        gdf: infrastructure vector data
        src: hazard map raster data
    """
    gdf = gdf.to_crs(src.crs)
    coords = [(geom.x, geom.y) for geom in gdf.geometry]
    values = list(src.sample(coords))
    nodata = src.nodata

    gdf["wind_speed"] = [v[0] if v[0] != nodata else 0 for v in values]
    return gdf


def extract_zonal_wind_speed(gdf, src, stats_method="max"):
    gdf = gdf.to_crs(src.crs)

    # 读取风速数据和transform信息
    wind_speed = src.read(1)
    transform = src.transform
    nodata = src.nodata

    # 处理 nodata
    if nodata is not None:
        wind_speed = np.where(wind_speed == nodata, 0, wind_speed)

    # 进行 zonal stats
    stats = zonal_stats(gdf, wind_speed, affine=transform, stats=stats_method)
    gdf["wind_speed"] = [s[stats_method] if s[stats_method] is not None else 0 for s in stats]
    
    return gdf


def wind_speed_extraction(gdf, tif_path, stats_method="max"):
    with rasterio.open(tif_path) as src:
        results = []

        # 将不同类型的geometry分开处理
        print(gdf.geometry.geom_type.unique())
        for geom_type in gdf.geometry.geom_type.unique():
            sub_gdf = gdf[gdf.geometry.geom_type == geom_type].copy()

            if sub_gdf.empty:
                continue
            
            print(f"Processing {len(sub_gdf)} {geom_type}s...")
            
            sub_gdf = sub_gdf.to_crs(src.crs)

            if geom_type == "Point":
                sub_gdf = extract_point_wind_speed(sub_gdf, src)
            elif geom_type in ["LineString", "MultiLineString", "Polygon", "MultiPolygon"]:
                sub_gdf = extract_zonal_wind_speed(sub_gdf, src, stats_method=stats_method)
            else:
                print(f"Unsupported geometry type: {geom_type}, skipping...")
                continue

            results.append(sub_gdf)

        # 合并所有结果
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            raise ValueError("No supported geometries found.")


def monte_carlo_simulation(gdf, curves, dam_class, num_iterations=2):
    if dam_class == 'Bus' and 'substation' not in gdf.columns:
        gdf['substation'] = gdf['OriginalID'].astype(int) > 99999

    simulation_result = gdf.copy()
    fail_results = []

    for iteration in range(num_iterations):
        def simulate_failure(row):
            wind_speed = row['wind_speed']
            if dam_class == 'Line':
                curve_id = 'W6.1' if row['voltage'] == 110.0 else (
                           'W6.2' if row['voltage'] in [220.0, 500.0] else None)
            elif dam_class == 'Bus':
                if row['substation']:
                    curve_id = 'W1'
                else:
                    curve_id = 'W3.15' if row['voltage'] in [110.0, 220.0] else (
                               'W3.45' if row['voltage'] == 500.0 else None)
            else:
                curve_id = None

            if not curve_id:
                raise ValueError(f"Unsupported voltage or dam_class for row: {row}")

            frag_curve = curves[["Wind speed", curve_id]]
            interp_func = interp1d(frag_curve["Wind speed"], frag_curve[curve_id], kind='linear', fill_value='extrapolate')
            fail_prob = interp_func(wind_speed)
            return 1 if np.random.rand() <= fail_prob else 0

        fail_column = f'fail_iter_{iteration+1}'
        fail_results.append(simulation_result.apply(simulate_failure, axis=1).rename(fail_column))

    # 合并所有 fail_iter 列，避免碎片化
    fail_df = pd.concat(fail_results, axis=1)
    simulation_result = pd.concat([simulation_result.reset_index(drop=True), fail_df.reset_index(drop=True)], axis=1)

    return simulation_result


def plot_combined_failure_probability_map(lines_gdf, nodes_gdf, basemap, fig_path=None):
    lines_gdf = lines_gdf.to_crs("EPSG:4326")
    nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    # 区域筛选
    pcode_list = ['VN106','VN111','VN101','VN107','VN103','VN109','VN113','VN117','VN115','VN104']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    cmap = cm.Reds
    norm = colors.Normalize(vmin=0, vmax=1)

    # 自动计算 failure_probability（若未计算）
    for df in [lines_gdf, nodes_gdf]:
        fail_cols = [col for col in df.columns if col.startswith('fail_iter_')]
        if 'failure_probability' not in df.columns:
            df['failure_probability'] = df[fail_cols].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(19, 7), constrained_layout=True)
    titles = ["Line failure probability", "Bus failure probability"]
    gdfs = [lines_gdf, nodes_gdf]

    for ax, gdf, title in zip(axes, gdfs, titles):
        failed = gdf[gdf['failure_probability'] > 0]
        non_failed = gdf[gdf['failure_probability'] == 0]

        ax.set_title(title, fontsize=16)
        ax.set_xlim(105, 107.5)
        ax.set_ylim(19.85, 21.7)
        ax.set_xlabel("Longitude", fontsize=16)
        if ax == axes[0]:
            ax.set_ylabel("Latitude", fontsize=16)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        # ax.set_aspect('auto')
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.8, zorder=1)
        other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.5, zorder=0)

        if gdf.geometry.iloc[0].geom_type == 'Point':
            if not non_failed.empty:
                ax.scatter(non_failed.geometry.x, non_failed.geometry.y,
                           s=8, color='lightgreen', alpha=0.5, label='Failure p = 0', zorder=2)
            if not failed.empty:
                ax.scatter(failed.geometry.x, failed.geometry.y,
                           s=failed['failure_probability'] * 10,
                           c=failed['failure_probability'],
                           cmap=cmap, norm=norm, alpha=0.8, zorder=3)
        elif gdf.geometry.iloc[0].geom_type == 'LineString':
            if not non_failed.empty:
                non_failed.plot(ax=ax, color='lightgreen', linewidth=0.8, zorder=2, alpha=0.5, label='Failure p = 0')
            if not failed.empty:
                failed.plot(ax=ax, column='failure_probability',
                            cmap=cmap, norm=norm, linewidth=1.5, zorder=3)

        ax.legend(loc='lower right', fontsize=16)
    
    # Set colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", location='right', fraction=0.02, pad=0.0, aspect=30)
    cbar.set_label("Failure probability", fontsize=16)
    cbar.ax.tick_params(labelsize=15)

    if fig_path:
        plt.savefig(os.path.join(fig_path, 'failure_probability_combined.png'), dpi=600, bbox_inches="tight")
        print(f"✓ Saved to: {fig_path}")

    plt.show()


def simulate_power_flows_over_failures(nodes_gdf_sim, lines_gdf_sim, gens_gdf, loads_gdf, trafo_gdf,
                                       num_iterations=1, use_column_in_service=True, outputh_path=None):
    base_net = build_pandapower_model(nodes_gdf_sim, lines_gdf_sim, gens_gdf, loads_gdf, trafo_gdf, use_column_in_service)

    results, load_status_records = [], []
    failed_buses_records, failed_lines_records, overloaded_lines_records = [], [], []

    for i in tqdm(range(num_iterations), desc="Simulating failure iterations"):
        fail_iter = f"fail_iter_{i+1}"
        print(f"Running simulation for: {fail_iter}")
        net = copy.deepcopy(base_net)

        net.line['in_service'] = (lines_gdf_sim[fail_iter] == 0)
        net.bus['in_service'] = (nodes_gdf_sim[fail_iter] == 0)

        failed_buses = net.bus.index[~net.bus['in_service']].tolist()
        failed_buses_records.extend([
            {"fail_iter": fail_iter, "bus_index": fb, "bus_name": net.bus.at[fb, 'name']} for fb in failed_buses
        ])

        failed_lines_mask = ~net.bus.loc[net.line.from_bus, "in_service"].values | \
                            ~net.bus.loc[net.line.to_bus, "in_service"].values
        net.line.loc[failed_lines_mask, "in_service"] = False

        failed_lines_records.extend([
            {
                "fail_iter": fail_iter,
                "line_name": net.line.at[idx, 'name'],
                "from_bus": net.line.at[idx, "from_bus"],
                "to_bus": net.line.at[idx, "to_bus"]
            } for idx in net.line[failed_lines_mask].index
        ])

        try:
            pp.runpp(net)
            success = True
            print(f"✓ Power flow successful for {i+1}")
        except Exception as e:
            success = False
            print(f"✗ Power flow failed for {i+1}: {e}")

        voltages = net.res_bus.vm_pu[net.bus.in_service] if success else pd.Series(dtype=float)
        under, over, voltage_violations = (voltages < 0.95).sum(), (voltages > 1.05).sum(), (voltages < 0.95).sum() + (voltages > 1.05).sum()

        total_load = net.load.p_mw.sum()
        total_supplied = net.res_load.p_mw.sum() if success else np.nan
        load_served_ratio = total_supplied / total_load if total_load > 0 else np.nan

        overload_lines = net.res_line[(net.res_line.loading_percent > 100) & net.line.in_service] if success else pd.DataFrame()
        overloaded_lines_records.extend([
            {
                "fail_iter": fail_iter,
                "line_index": idx,
                "line_name": net.line.at[idx, "name"],
                "loading_percent": row.loading_percent
            } for idx, row in overload_lines.iterrows()
        ])

        overload_lines_count = len(overload_lines)
        overload_lines_ratio = overload_lines_count / net.line.in_service.sum() if net.line.in_service.sum() > 0 else 0

        load_status_records.extend([
            {
                "fail_iter": fail_iter,
                "load_id": row["name"],
                "bus": row["bus"],
                "served": bool(net.bus.at[row["bus"], "in_service"] and \
                                success and not np.isnan(net.res_load.at[idx, "p_mw"]) and \
                                net.res_load.at[idx, "p_mw"] > 0)
            } for idx, row in net.load.iterrows()
        ])

        results.append({
            "fail_iter": fail_iter,
            "powerflow_success": success,
            "voltage_under_0.95": under if success else np.nan,
            "voltage_over_1.05": over if success else np.nan,
            "voltage_violations": voltage_violations if success else np.nan,
            "mean_voltage": voltages.mean() if success else np.nan,
            "std_voltage": voltages.std() if success else np.nan,
            "max_voltage": voltages.max() if success else np.nan,
            "min_voltage": voltages.min() if success else np.nan,
            "total_load": total_load,
            "total_supplied": total_supplied,
            "load_served_ratio": load_served_ratio,
            "supply_loss_ratio": 1 - load_served_ratio if total_load > 0 and success else np.nan,
            "line_overload_violations": overload_lines_count,
            "overload_line_ratio": overload_lines_ratio,
            "avg_line_loading": net.res_line.loading_percent[net.line.in_service].mean() if success else np.nan,
            "max_line_loading": net.res_line.loading_percent[net.line.in_service].max() if success else np.nan,
            "min_line_loading": net.res_line.loading_percent[net.line.in_service].min() if success else np.nan
        })

        if success and outputh_path:
            pp.to_excel(net, os.path.join(outputh_path, f"net_results_fail_iter{i+1}.xlsx"))

    return (
        pd.DataFrame(results),
        pd.DataFrame(load_status_records),
        pd.DataFrame(failed_buses_records),
        pd.DataFrame(failed_lines_records),
        pd.DataFrame(overloaded_lines_records)
    )


def prepare_loads_map(load_status_df, loads_gdf):
    loads_gdf = loads_gdf.copy()
    load_status_df = load_status_df.copy()

    loads_gdf["osmid"] = pd.to_numeric(loads_gdf["osmid"], errors="coerce")
    load_status_df = load_status_df.rename(columns={"load_id": "osmid"})
    load_status_df["osmid"] = pd.to_numeric(load_status_df["osmid"], errors="coerce")

    # Compute average, min, and max served ratio per osmid per frac
    avg_status = (
        load_status_df.groupby(["osmid", "fail_iter"])["served"]
        .agg(served_ratio_mean="mean", served_ratio_min="min", served_ratio_max="max")
        .reset_index()
    )

    loads_gdf = loads_gdf.merge(avg_status, on="osmid", how="left")

    return loads_gdf


def plot_served_ratio_maps(loads_gdf, fig_path=None, basemap=None):
    # Average served ratio of all fail_iters
    loads_gdf = loads_gdf.to_crs("EPSG:4326")
    basemap = basemap.to_crs("EPSG:4326")

    pcode_list = ['VN106','VN111','VN101','VN107','VN103','VN109','VN113','VN117','VN115','VN104']
    target_regions = basemap[basemap["ADM1_PCODE"].isin(pcode_list)]
    other_regions = basemap[~basemap["ADM1_PCODE"].isin(pcode_list)]

    # unique_iters = loads_gdf["fail_iter"].dropna().unique()
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

    # for iter_label in unique_iters:
    #     iter_data = loads_gdf[loads_gdf["fail_iter"] == iter_label]

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.set_xlim(105, 107.5)
    ax.set_ylim(19.85, 21.7)
    ax.set_aspect('auto')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    target_regions.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.8, zorder=1)
    other_regions.plot(ax=ax, color='white', edgecolor='lightgray', linewidth=0.5, zorder=0)
    # iter_data.plot(ax=ax, column="served_ratio_mean", cmap=cmap, norm=norm, markersize=100, legend=False, zorder=2)
    avg_served_gdf.plot(ax=ax, column="served_ratio_mean", cmap=cmap, norm=norm,
                        markersize=100, legend=False, zorder=2)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.035)
    cbar.set_label("Served load ratio", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    if fig_path:
        os.makedirs(fig_path, exist_ok=True)
        # output_path = f"{fig_path}/served_ratio_map_{iter_label}.png"
        output_path = os.path.join(fig_path, 'served_ratio_map_average.png')
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
    


# def plot_lines_based_on_failures(gdf_sim, failed_records, gdf_type='line', num_iterations=1):
#     # 遍历所有 fail_iter_ 列
#     for i in range(num_iterations):  # 从 fail_iter_1 到 fail_iter_num_iterations
#         fail_iter_column = f'fail_iter_{i+1}'
#         impact_column = f'impact_{i+1}'

#         gdf_sim[impact_column] = 'Not Failed'

#         # 标记为 Failed（红色）：fail_iter_X == 1
#         gdf_sim.loc[gdf_sim[fail_iter_column] == 1, impact_column] = 'Direct Failed'

#         if gdf_type == 'line':
#             index = 'LineID'
#             name = 'line_name'
#         elif gdf_type == 'node':
#             index = 'NodeID'
#             name = 'bus_name'

#         # 标记 'Indirect Failed'（蓝色）：在 failed_lines_records 中存在的线路，且 fail_iter_X == 0
#         for _, record in failed_records.iterrows():
#             gdf_sim.loc[(gdf_sim[index] == record[name]) & 
#                               (gdf_sim[fail_iter_column] == 0), impact_column] = 'Indirect Failed'

#         # 标记为 Not Failed（绿色）：fail_iter_X == 0，且 LineID 不在 failed_lines_records 中
#         gdf_sim.loc[(gdf_sim[fail_iter_column] == 0) & 
#                           (~gdf_sim[index].isin(failed_records[name])), impact_column] = 'Not Failed'
    
#     return gdf_sim


def plot_lines_based_on_failures(gdf_sim, failed_records, gdf_type='line', num_iterations=1):
    # 确定使用的唯一标识字段
    if gdf_type == 'line':
        index = 'LineID'
        name = 'line_name'
    elif gdf_type == 'node':
        index = 'NodeID'
        name = 'bus_name'
    else:
        raise ValueError("gdf_type must be either 'line' or 'node'.")

    # 创建一个 impact 列的字典
    impact_dict = {}

    # 提前提取失败记录的名称集合，加快判断速度
    failed_names = set(failed_records[name].values)

    for i in range(num_iterations):
        fail_iter_column = f'fail_iter_{i+1}'
        impact_column = f'impact_{i+1}'

        # 初始为 Not Failed
        status_col = pd.Series(['Not Failed'] * len(gdf_sim), index=gdf_sim.index)

        # 条件布尔筛选
        direct_failed_mask = gdf_sim[fail_iter_column] == 1
        indirect_failed_mask = (gdf_sim[fail_iter_column] == 0) & gdf_sim[index].isin(failed_names)

        # 更新状态
        status_col[direct_failed_mask] = 'Direct Failed'
        status_col[indirect_failed_mask] = 'Indirect Failed'

        # 存入字典
        impact_dict[impact_column] = status_col

    # 合并 impact 所有列（避免频繁 insert 导致的 fragmentation）
    impact_df = pd.DataFrame(impact_dict)

    # 合并结果
    gdf_sim = pd.concat([gdf_sim.reset_index(drop=True), impact_df], axis=1)

    return gdf_sim


def analyze_failure_impact(lines_gdf, nodes_gdf, impact_cols):
    summary_records = []

    lines_gdf.to_crs(epsg=4326)

    total_line_length = lines_gdf["Length"].sum()  # ["Length"]
    total_node_count = nodes_gdf.shape[0]
    print("Total line length: ", total_line_length)
    print("Total node count: ", total_node_count)

    for col in impact_cols:
        if col not in lines_gdf.columns or col not in nodes_gdf.columns:
            continue

        # Line failure lengths
        direct_lines = lines_gdf[lines_gdf[col] == "Direct Failed"]
        indirect_lines = lines_gdf[lines_gdf[col] == "Indirect Failed"]

        direct_len = direct_lines["Length"].sum()
        indirect_len = indirect_lines["Length"].sum()

        # Node failure counts
        direct_nodes = nodes_gdf[nodes_gdf[col] == "Direct Failed"]
        indirect_nodes = nodes_gdf[nodes_gdf[col] == "Indirect Failed"]

        direct_node_count = len(direct_nodes)
        indirect_node_count = len(indirect_nodes)

        summary_records.append({
            "impact_column": col,
            "direct_line_length": direct_len,
            "indirect_line_length": indirect_len,
            "direct_line_ratio": direct_len / total_line_length * 100,
            "indirect_line_ratio": indirect_len / total_line_length * 100,
            "direct_node_count": direct_node_count,
            "indirect_node_count": indirect_node_count,
            "direct_node_ratio": direct_node_count / total_node_count * 100,
            "indirect_node_ratio": indirect_node_count / total_node_count * 100
        })

    df_summary = pd.DataFrame(summary_records)

    return df_summary


if __name__ == "__main__":
    # sys.argv = [""]  # 清空 sys.argv 避免解析 Jupyter 参数
    # sys.argv = ["percolation_analysis_tc.py", "--num_iterations", "1"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=2, help="Number of Monte Carlo simulation iterations")
    args = parser.parse_args()
    print(args.num_iterations)

    # === Read power network data ===
    lines_gdf = gpd.read_file('../outputs/table_lines_200m_update_remove_disconnected.gpkg')
    nodes_gdf = gpd.read_file('../outputs/table_nodes_200m_update_remove_disconnected.gpkg')
    gens_gdf = gpd.read_file('../outputs/plant_update.gpkg')
    loads_gdf = gpd.read_file('../outputs/landuse_sites_gdf_add_bus.gpkg')
    trafo_gdf = gpd.read_file('../outputs/table_transformers_remove_disconnected.gpkg')

    basemap = gpd.read_file("../data/base_map/vnm_admbnda_adm1_gov_20201027.shp")

    tc_path = '../data/tc/tc_yagi.tif'

    curve_path = '../data/fragility_curves.xlsx'
    curves = load_curves(curve_path, sheet_name='test')

    output_path = f'../outputs/20250725_hazard_iter_{args.num_iterations}'
    os.makedirs(output_path, exist_ok=True)

    fig_path = f'../figures/20250725_hazard_iter_{args.num_iterations}'
    os.makedirs(fig_path, exist_ok=True)

    lines_gdf_wind = wind_speed_extraction(lines_gdf, tc_path)
    nodes_gdf_wind = wind_speed_extraction(nodes_gdf, tc_path)

    lines_gdf_sim = monte_carlo_simulation(lines_gdf_wind, curves, 'Line', num_iterations=args.num_iterations)
    nodes_gdf_sim = monte_carlo_simulation(nodes_gdf_wind, curves, 'Bus', num_iterations=args.num_iterations)


    results, load_status_records, failed_buses_records, failed_lines_records, overloaded_lines_records = simulate_power_flows_over_failures(
        nodes_gdf_sim, lines_gdf_sim, gens_gdf, loads_gdf, trafo_gdf,
        num_iterations=args.num_iterations, use_column_in_service=True, outputh_path=output_path
    )

    # Save the results
    results.to_csv(os.path.join(output_path, "percolation_overall_results.csv"), index=False)
    load_status_records.to_csv(os.path.join(output_path, "load_status_records.csv"), index=False)
    failed_buses_records.to_csv(os.path.join(output_path, "failed_buses_records.csv"), index=False)
    failed_lines_records.to_csv(os.path.join(output_path, "failed_lines_records.csv"), index=False)
    overloaded_lines_records.to_csv(os.path.join(output_path, "overloaded_lines_records.csv"), index=False)

    lines_gdf_sim_update = plot_lines_based_on_failures(lines_gdf_sim, failed_lines_records, gdf_type='line', num_iterations=args.num_iterations)
    nodes_gdf_sim_update = plot_lines_based_on_failures(nodes_gdf_sim, failed_buses_records, gdf_type='node', num_iterations=args.num_iterations)
    lines_gdf_sim_update.to_excel(os.path.join(output_path, "lines_gdf_sim_update.xlsx"), index=False)
    nodes_gdf_sim_update.to_excel(os.path.join(output_path, "nodes_gdf_sim_update.xlsx"), index=False)

    # plot_combined_failure_probability_map(lines_gdf_sim, nodes_gdf_sim, basemap, fig_path)
    # plot_failure_impact_maps(lines_gdf_sim_update, nodes_gdf_sim_update, basemap, fig_path)
    # plot_direct_indirect_failure_probability(lines_gdf_sim_update, nodes_gdf_sim_update, basemap, fig_path)
    # plot_majority_failure_status_map(lines_gdf_sim_update, nodes_gdf_sim_update, basemap, fig_path)
    # plot_majority_status_with_probability(lines_gdf_sim_update, nodes_gdf_sim_update, basemap, fig_path)

    loads_gdf_update = prepare_loads_map(load_status_records, loads_gdf)
    plot_served_ratio_maps(loads_gdf_update, fig_path=fig_path, basemap=basemap)

    impact_cols = [col for col in lines_gdf_sim_update.columns if col.startswith("impact_")]
    df_summary = analyze_failure_impact(lines_gdf_sim_update, nodes_gdf_sim_update, impact_cols)
    df_summary.to_excel(os.path.join(output_path, "failure_status_ratio.xlsx"), index=False)
