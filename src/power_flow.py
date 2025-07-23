# === Build power flow model ===
import pandas as pd
import pandapower as pp
import math

def build_pandapower_model(nodes_gdf, lines_gdf, gens_gdf, loads_gdf, trafo_gdf,
                           use_column_in_service = False):
    """
        use_column_in_service = False: randomly remove nodes from network
        use_column_in_service = True: remove nodes and lines exposed to hazard maps
    """
    net = pp.create_empty_network()

    # create buses
    buses = nodes_gdf.copy()
    for i in range(buses.shape[0]):
        bus = buses.iloc[i]
        coordinates = (bus.geometry.x, bus.geometry.y)
        in_service = bool(bus['in_service']) if use_column_in_service and 'in_service' in bus else True

        # Create bus with the formatted values
        pp.create_bus(
            net,
            vn_kv=float(buses.iloc[i]['voltage']),
            name=str(buses.iloc[i]['NodeID']),
            min_vm_pu=0.9,
            max_vm_pu=1.05,
            in_service=in_service,
            geodata=coordinates)

    # create lines
    lines = lines_gdf.copy()
    voltage_to_std_type = {
        110: "490-AL1/64-ST1A 110.0",
        220: "490-AL1/64-ST1A 220.0",
        500: "490-AL1/64-ST1A 380.0"
    }
    lines['std_type'] = lines['voltage'].map(voltage_to_std_type)

    for i in range(lines.shape[0]):
        line = lines_gdf.iloc[i]
        coordinates = list(line.geometry.coords)
        coordinates = [(x, y) for x, y in coordinates]

        from_bus_name = line['fromNode']
        to_bus_name = line['toNode']

        # Find the index in net.bus corresponding to fromNode and toNode
        try:
            from_bus = net.bus.loc[net.bus['name'] == from_bus_name].index[0]
            to_bus = net.bus.loc[net.bus['name'] == to_bus_name].index[0]
        except IndexError:
            print(f"Skipping line {lines.loc[i,'LineID']} due to missing bus in net.bus")
            continue

        in_service = bool(line['in_service']) if use_column_in_service and 'in_service' in line else True

        pp.create_line(
            net,
            name = line['LineID'],
            from_bus = from_bus,
            to_bus = to_bus,
            # don't neet to add parallel
            length_km = line['Length'],
            std_type = line['std_type'],
            in_service = in_service,   #lines.loc[i,'in_service'],
            geodata = coordinates
        )

    # create loads
    loads = loads_gdf.copy()

    for i in range(loads.shape[0]):
        bus_name = str(loads.loc[i, 'BusID'])

        # Find the index in net.bus corresponding to fromNode and toNode
        try:
            bus = net.bus.loc[net.bus['name'] == bus_name].index[0]
        except IndexError:
            # print(f"Skipping load {loads.loc[i,'osmid']} due to missing bus in net.bus")
            continue
        
        # Calculate p_mw
        peak_factor = 1.3  # 1.3 for industrial
        avg_p = loads.loc[i,'total_electricity_demand'] * 1000 / 8760
        p_mw = avg_p * peak_factor # peak_factor is optional

        # Calculate q_mvar
        pf = 0.9  # industrial power factor
        q_mvar = p_mw * math.tan(math.acos(pf))

        load_idx = pp.create_load(
            net, 
            name = loads.loc[i, 'osmid'],
            bus = bus,
            p_mw = p_mw, # loads.loc[i, 'peak_demand_MW'],
            # q_mvar = loads.loc[i, 'peak_demand_MW'] * (1 - 0.8 * 0.8),
            in_service = True,   #loads.loc[i, 'in_service']
        )

        if "load_geodata" not in net:
            net["load_geodata"] = pd.DataFrame(columns=["name", "geometry"])

        net["load_geodata"].loc[load_idx] = {
            "name": loads.loc[i, "osmid"],
            "geometry": loads.loc[i, "geometry"]
        }

    # create generators
    gens = gens_gdf.copy()
    gen_geodata_list = []

    for i in range(gens.shape[0]):
        bus_name = str(gens.loc[i, 'BusID']).replace('.0', '')

        # Find the index in net.bus corresponding to fromNode and toNode
        try:
            bus = net.bus.loc[net.bus['name'] == bus_name].index[0]
        except IndexError:
            # print(f"Skipping generator {gens.loc[i,'osm_id']} due to missing bus in net.bus")
            continue

        gen_idx = pp.create_gen(net, name=gens.loc[i,'osm_id'],
                                bus = bus,
                                # p_mw is the active power of the generator (positive for generation!)
                                p_mw = gens.loc[i,'installed_capacity'], # * 0.9, # Typically, peak demand is 80% to 100% of the rated capacity
                                # q_mvar = gens.loc[i,'installed_capacity'],
                                type = gens.loc[i,'energy_source'],
                                in_service = True,   #gens.loc[i,'in_service'],
                                slack = True
                                #    slack = False # CHECK: network disconnected or the lines are not overloaded
                    )
        
        if "gen_geodata" not in net:
            net["gen_geodata"] = pd.DataFrame(columns=["name", "geometry"])

        net["gen_geodata"].loc[gen_idx] = {
            "name": gens.loc[i, "osm_id"],
            "geometry": gens.loc[i, "geometry"]
        }

    # create external grid
    # pp.create_ext_grid(net, bus=buses.loc[buses['NodeID']=='NODEVN3423'].index[0], vm_pu=1, va_degree=0) # Create an external grid connection
    pp.create_ext_grid(net, bus=net.bus.loc[net.bus['name']=='NODEVN3748a'].index[0], vm_pu=1.0, va_degree=0) # Create an external grid connection

    # create transformers
    def get_bus_index_by_name(bus_name):
        matches = net.bus[net.bus['name'] == bus_name]
        if not matches.empty:
            return matches.index[0]
        else:
            return None

    trafo_gdf['hv_bus_id'] = trafo_gdf['hv_bus'].apply(get_bus_index_by_name)
    trafo_gdf['lv_bus_id'] = trafo_gdf['lv_bus'].apply(get_bus_index_by_name)

    for _, row in trafo_gdf.iterrows():
        try:
            if pd.isna(row['hv_bus_id']) or pd.isna(row['lv_bus_id']):
                print(f"Skipping transformer {row['name']} due to missing bus index.")
                continue

            pp.create_transformer_from_parameters(
                net,
                hv_bus=int(row['hv_bus_id']),
                lv_bus=int(row['lv_bus_id']),
                vn_hv_kv=row['vn_hv_kv'],
                vn_lv_kv=row['vn_lv_kv'],
                sn_mva=row['sn_mva'],
                vk_percent=row['vk_percent'],
                vkr_percent=row['vkr_percent'],
                pfe_kw=row['pfe_kw'],
                i0_percent=row['i0_percent'],
                # std_type='row['std_type']',
                name=row['name']
            )
        except Exception as e:
            print(f"Error creating transformer for row {row.to_dict()}: {e}")

    return net
    

# def build_pandapower_model_hazard(nodes_gdf, lines_gdf, gens_gdf, loads_gdf, trafo_gdf):
#     net = pp.create_empty_network()

#     # create buses
#     buses = nodes_gdf.copy()
#     for i in range(buses.shape[0]):
#         coordinates = buses.iloc[i]['geometry']
#         coordinates = (coordinates.x, coordinates.y)

#         # Create bus with the formatted values
#         pp.create_bus(net, vn_kv=float(buses.iloc[i]['voltage']),
#                         name=str(buses.iloc[i]['NodeID']),
#                         min_vm_pu=0.9, max_vm_pu=1.05,
#                         in_service = buses.loc[i,'in_service'], # buses.iloc[i]['in_service'],
#                         geodata=coordinates)

#     # create lines
#     lines = lines_gdf.copy()
#     voltage_to_std_type = {
#         110: "490-AL1/64-ST1A 110.0",
#         220: "490-AL1/64-ST1A 220.0",
#         500: "490-AL1/64-ST1A 380.0"
#         }
#     lines['std_type'] = lines['voltage'].map(voltage_to_std_type)

#     for i in range(lines.shape[0]):
#         coordinates = list(lines.loc[i,'geometry'].coords)
#         coordinates = [(x, y) for x, y in coordinates]

#         from_bus_name = lines.loc[i, 'fromNode']
#         to_bus_name = lines.loc[i, 'toNode']

#         # Find the index in net.bus corresponding to fromNode and toNode
#         try:
#             from_bus = net.bus.loc[net.bus['name'] == from_bus_name].index[0]
#             to_bus = net.bus.loc[net.bus['name'] == to_bus_name].index[0]
#         except IndexError:
#             print(f"Skipping line {lines.loc[i,'LineID']} due to missing bus in net.bus")
#             continue

#         pp.create_line(net, name = lines.loc[i,'LineID'],
#                     from_bus = from_bus,
#                     to_bus = to_bus,
#                     # don't neet to add parallel
#                     length_km = lines.loc[i,'Length'],
#                     std_type = lines.loc[i,'std_type'],
#                     in_service = lines.loc[i,'in_service'],
#                     geodata = coordinates
#         )

#     # create loads
#     loads = loads_gdf.copy()

#     for i in range(loads.shape[0]):
#         # centroid = loads.iloc[i]['geometry'].centroid
#         # coordinates = (centroid.x, centroid.y)  # Convert to a tuple
#         # coordinates = list(loads.loc[i,'geometry'].coords)
#         # coordinates = [(x, y) for x, y in coordinates]

#         bus_name = str(loads.loc[i, 'BusID'])

#         # Find the index in net.bus corresponding to fromNode and toNode
#         try:
#             bus = net.bus.loc[net.bus['name'] == bus_name].index[0]
#         except IndexError:
#             print(f"Skipping load {loads.loc[i,'osmid']} due to missing bus in net.bus")
#             continue
        
#         # Calculate p_mw
#         peak_factor = 1.3  # 1.3 for industrial
#         avg_p = loads.loc[i,'total_electricity_demand'] * 1000 / 8760
#         p_mw = avg_p * peak_factor # peak_factor is optional

#         # Calculate q_mvar
#         pf = 0.9  # industrial power factor
#         q_mvar = p_mw * math.tan(math.acos(pf))

#         pp.create_load(net, name=loads.loc[i,'osmid'],
#                     bus=bus,
#                     # The active power of the load (positive value -> load; negative value -> generation)
#                     p_mw = p_mw,
#                     #    q_mvar = q_mvar,
#                     in_service = True,
#         )

#     # create generators
#     gens = gens_gdf.copy()

#     for i in range(gens.shape[0]):
#         bus_name = str(gens.loc[i, 'BusID']).replace('.0', '')

#         # Find the index in net.bus corresponding to fromNode and toNode
#         try:
#             bus = net.bus.loc[net.bus['name'] == bus_name].index[0]
#         except IndexError:
#             print(f"Skipping generator {gens.loc[i,'osm_id']} due to missing bus in net.bus")
#             continue

#         gen_idx = pp.create_gen(net, name=gens.loc[i,'osm_id'],
#                                 bus = bus,
#                                 # p_mw is the active power of the generator (positive for generation!)
#                                 p_mw = gens.loc[i,'installed_capacity'] * 0.9, # Typically, peak demand is 80% to 100% of the rated capacity
#                                 q_mvar = gens.loc[i,'installed_capacity'],
#                                 type = gens.loc[i,'energy_source'],
#                                 in_service = True, # gens.loc[i,'in_service'],
#                                 slack = True
#                                 #    slack = False # CHECK: network disconnected or the lines are not overloaded
#                     )
        
#         if "gen_geodata" not in net:
#             net["gen_geodata"] = pd.DataFrame(columns=["name", "geometry"])

#         net["gen_geodata"].loc[gen_idx] = {
#             "name": gens.loc[i, "osm_id"],
#             "geometry": gens.loc[i, "geometry"]
#         }

#     # create external grid
#     pp.create_ext_grid(net, bus=buses.loc[buses['NodeID']=='NODEVN3748a'].index[0], vm_pu=1, va_degree=0) # Create an external grid connection

#     # create transformers
#     def get_bus_index_by_name(bus_name):
#         matches = net.bus[net.bus['name'] == bus_name]
#         if not matches.empty:
#             return matches.index[0]
#         else:
#             return None

#     trafo_gdf['hv_bus_id'] = trafo_gdf['hv_bus'].apply(get_bus_index_by_name)
#     trafo_gdf['lv_bus_id'] = trafo_gdf['lv_bus'].apply(get_bus_index_by_name)

#     for _, row in trafo_gdf.iterrows():
#         try:
#             if pd.isna(row['hv_bus_id']) or pd.isna(row['lv_bus_id']):
#                 print(f"Skipping transformer {row['name']} due to missing bus index.")
#                 continue

#             pp.create_transformer_from_parameters(
#                 net,
#                 hv_bus=int(row['hv_bus_id']),
#                 lv_bus=int(row['lv_bus_id']),
#                 vn_hv_kv=row['vn_hv_kv'],
#                 vn_lv_kv=row['vn_lv_kv'],
#                 sn_mva=row['sn_mva'],
#                 vk_percent=row['vk_percent'],
#                 vkr_percent=row['vkr_percent'],
#                 pfe_kw=row['pfe_kw'],
#                 i0_percent=row['i0_percent'],
#                 # std_type='row['std_type']',
#                 name=row['name']
#             )
#         except Exception as e:
#             print(f"Error creating transformer for row {row.to_dict()}: {e}")

#     return net