import argparse # Handles command-line input arguments
import time # Measures routing execution time
import math # Performs logarithmic risk calculations
import os # Checks file existence
import pandas as pd # Reads and processes CSV datasets
import geopandas as gpd # Spatial data processing 
from shapely.geometry import Point # Creates geographic point objects
import osmnx as ox # Downloads OpenStreetMap road networks
import networkx as nx # Implements Dijkstra shortest path algorithm
import folium # Produces interactive HTML route maps

# Settings
ox.settings.use_cache = False
ox.settings.log_console = False

# Command Line Interface
parser = argparse.ArgumentParser(
    description="Dijkstra routing using incident severity comparison."
)
parser.add_argument("city", type=str, help="e.g., 'Dublin, Ireland'") # Accept the city name as input
parser.add_argument("start_location", type=str, help="Start address or place name") # Accept the start location
parser.add_argument("end_location", type=str, help="End address or place name") # Accept the destination
args = parser.parse_args() # Read the user inputs

city_name = args.city # Store the city name

print("\n=== INPUT RECEIVED ===")
print(f"City: {city_name}")
print(f"Start location: {args.start_location}")
print(f"End location: {args.end_location}\n")

# Geocode
start_query = f"{args.start_location}, {city_name}"
end_query = f"{args.end_location}, {city_name}"

start_latitude, start_longitude = ox.geocode(start_query)
end_latitude, end_longitude = ox.geocode(end_query)

print(f"Geocoded Start: {(start_latitude, start_longitude)}")
print(f"Geocoded Finish: {(end_latitude, end_longitude)}\n")

# Download Road Network
print(f"Downloading road network for {city_name}...")
Graph = ox.graph_from_place(
    city_name,
    network_type="drive",
    retain_all=True,
    simplify=False
)
Graph = ox.distance.add_edge_lengths(Graph)

city_gdf = ox.geocode_to_gdf(city_name)
city_poly = city_gdf.geometry.union_all()

# Snap Nodes
start_node = ox.distance.nearest_nodes(Graph, start_longitude, start_latitude)
finish_node = ox.distance.nearest_nodes(Graph, end_longitude, end_latitude)

print(f"Nearest start node: {start_node}")
print(f"Nearest finish node: {finish_node}")

if not nx.has_path(Graph, start_node, finish_node):
    raise ValueError("No path exists between the selected points.")

# Helper Functions
def severity_to_probability(sev, p0=0.001, lam=0.02):
    return max(0.0, min(0.99, p0 + lam * sev))

def qualitative_safety(sev):
    if sev == 0: return "No recorded incidents"
    elif sev == 1: return "Very low risk"
    elif sev == 2: return "Low to moderate risk"
    elif sev == 3: return "Moderate to high risk"
    else: return "High risk"

def path_metrics(G, path):
    edges = [G[u][v] for u, v in zip(path[:-1], path[1:])]
    total_distance = sum(e["length"] for e in edges)
    max_severity = max(e["severity"] for e in edges) if edges else 0
    log_safe = sum(math.log(1 - severity_to_probability(e["severity"])) for e in edges)
    return total_distance, max_severity, log_safe

# Dijkstra Routing
def run_routing_pipeline(df_source, case_label):
    print("\n==============================================")
    print(f"RUNNING: {case_label}")
    print("==============================================")

    G_local = Graph.copy()

    # Process Incident Data
    if df_source.empty:
        print("[INFO] Incident dataset is empty for this run.")
        gdf_inc = gpd.GeoDataFrame(columns=["Latitude", "Longitude", "severity"], geometry=[])
    else:
        gdf_inc = gpd.GeoDataFrame( # Convert the incident data into a GeoDataFrame
            df_source,
            geometry=[Point(xy) for xy in zip(df_source["Longitude"], df_source["Latitude"])],
            crs="EPSG:4326"
        )
        gdf_inc = gdf_inc[gdf_inc.within(city_poly)].copy() # Keep only incidents located inside the selected city

    print(f"Incidents mapped inside {city_name}: {len(gdf_inc)}")

    # Initialise edge severities
    for _, _, _, data in G_local.edges(keys=True, data=True): # Set the default severity of every road edge to zero
        data.setdefault("severity", 0)

    if not gdf_inc.empty:
        gdf_inc["nearest_node"] = ox.distance.nearest_nodes( # Find the nearest road node for each incident
            G_local, gdf_inc["Longitude"], gdf_inc["Latitude"]
        )
        severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict() # Create a lookup table from node ID to severity value
    else:
        severity_map = {}

    for u, v, key, data in G_local.edges(keys=True, data=True):
        data["severity"] = max(severity_map.get(u, 0), severity_map.get(v, 0))

    # Probabilistic risk weights
    for _, _, _, data in G_local.edges(keys=True, data=True):
        p_inc = severity_to_probability(data["severity"])
        data["risk_weight"] = -math.log(1.0 - p_inc)

    # Collapse MultiDiGraph to DiGraph
    H = nx.DiGraph()
    H.add_nodes_from(G_local.nodes(data=True))

    for u, v, data in G_local.edges(data=True):
        length = data.get("length", 1.0)
        risk = data.get("risk_weight", 0.0)
        sev = data.get("severity", 0)

        if H.has_edge(u, v):
            H[u][v]["length"] = min(H[u][v]["length"], length)
            H[u][v]["risk_weight"] = min(H[u][v]["risk_weight"], risk)
            H[u][v]["severity"] = max(H[u][v]["severity"], sev)
        else:
            H.add_edge(u, v, length=length, risk_weight=risk, severity=sev)

    # Compute paths
    t0 = time.perf_counter()
    shortest_path = nx.dijkstra_path(H, start_node, finish_node, weight="length")
    shortest_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    safest_path = nx.dijkstra_path(H, start_node, finish_node, weight="risk_weight")
    safest_time = time.perf_counter() - t1

    # Metrics
    sd, ss, slog = path_metrics(H, shortest_path)
    fd, fs, flog = path_metrics(H, safest_path)

    threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0
    shortest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    safest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(safest_path[:-1], safest_path[1:]))
    risk_reduction = 1 - (safest_high / shortest_high) if shortest_high else 0

    # Output
    print(f"\n=== RESULTS FOR {city_name.upper()} ({case_label}) ===")
    print(f"Shortest Path: Distance = {sd:.2f} m | Max Severity = {ss} | Safety = {qualitative_safety(ss)} | Log P(no incident) = {slog:.4f} | Time = {shortest_time:.4f} s")
    print(f"Safest Path: Distance = {fd:.2f} m | Max Severity = {fs} | Safety = {qualitative_safety(fs)} | Log P(no incident) = {flog:.4f} | Time = {safest_time:.4f} s")

    print("\n=== ROUTE RISK SUMMARY (THRESHOLD-BASED) ===")
    print(f"High-risk edges on shortest path: {shortest_high}")
    print(f"High-risk edges on safest path: {safest_high}")
    print(f"Relative risk reduction: {risk_reduction:.2%}")

    # Map
    m = folium.Map(location=(H.nodes[start_node]["y"], H.nodes[start_node]["x"]), zoom_start=14)

    folium.Marker((H.nodes[start_node]["y"], H.nodes[start_node]["x"]), popup="Start",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((H.nodes[finish_node]["y"], H.nodes[finish_node]["x"]), popup="Finish",
                  icon=folium.Icon(color="red")).add_to(m)

    def plot_path(path, label, color):
        for u, v in zip(path[:-1], path[1:]):
            coords = [(H.nodes[u]["y"], H.nodes[u]["x"]), (H.nodes[v]["y"], H.nodes[v]["x"])]
            folium.PolyLine(coords, color=color, weight=6, opacity=0.8,
                            popup=f"{label}: severity {H[u][v]['severity']}").add_to(m)

    plot_path(shortest_path, "Shortest Path", "blue")
    plot_path(safest_path, "Safest Path", "cyan")

    clean_label = case_label.lower().replace(" ", "_")
    output_file = f"{city_name.replace(',', '').replace(' ', '_')}_{clean_label}_routes.html"
    m.save(output_file)
    print(f"\nHTML map saved as: {output_file}")

# Data Loading And City Exclusion 
file_path = "C:/Users/bencr/Downloads/Practicum/Incidents_With_OSM_IDs.csv"

if not os.path.exists(file_path):
    print(f"[ERROR] Base incident CSV file not found at {file_path}.")
    raise SystemExit(1)

df_all = pd.read_csv(file_path)
df_all = df_all.rename(columns={"Lat": "Latitude", "Long": "Longitude"})

gdf_all = gpd.GeoDataFrame(
    df_all,
    geometry=[Point(xy) for xy in zip(df_all["Longitude"], df_all["Latitude"])],
    crs="EPSG:4326"
)

print(f"\nExcluding incidents inside routing city: {city_name}")
exclude_gdf = ox.geocode_to_gdf(city_name)
exclude_poly = exclude_gdf.geometry.union_all()

gdf_excluded = gdf_all[~gdf_all.within(exclude_poly)].copy()
df_excluded = gdf_excluded.drop(columns="geometry")

# Run Both Cases
run_routing_pipeline(df_excluded, f"Case 1 - Excluding {city_name}")
run_routing_pipeline(df_all, "Case 2 - Full Dataset")

print("\nAll routing comparisons completed successfully.")