import argparse # Handles command-line arguments
import time # Used to measure execution time
import math # Provides mathematical functions
import pandas as pd # Data analysis and CSV handling
import geopandas as gpd # Spatial/geographic data handling
from shapely.geometry import Point # Creates geometric point objects
import osmnx as ox # Downloads and works with OpenStreetMap road networks
import networkx as nx # Graph algorithms including A*
import folium # Creates interactive HTML maps

# Disable OSMnx caching/logging for clean CLI output
ox.settings.use_cache = False
ox.settings.log_console = False

# Command Line Interface Arguments
parser = argparse.ArgumentParser(description="Probabilistic A* routing using incident severity.")
parser.add_argument("city", type=str, help="City name, e.g. 'Dublin, Ireland'") # City name
parser.add_argument("start_location", type=str, help="Start address or place name") # Start location
parser.add_argument("end_location", type=str, help="End address or place name") # End location
args = parser.parse_args() # Parse user input

city_name = args.city # Store city name

# Print user inputs for confirmation
print("\n=== INPUT RECEIVED ===")
print(f"City: {city_name}")
print(f"Start location: {args.start_location}")
print(f"End location: {args.end_location}\n")

# Geocode Start and End Locations
start_query = f"{args.start_location}, {city_name}" # Add city for accurate geocoding
end_query = f"{args.end_location}, {city_name}"

# Convert place names into latitude/longitude coordinates
start_latitude, start_longitude = ox.geocode(start_query)
end_latitude, end_longitude = ox.geocode(end_query)

# Store coordinates as tuples
start = (start_latitude, start_longitude)
finish = (end_latitude, end_longitude)

# Print coordinates
print(f"Geocoded Start: {start}")
print(f"Geocoded Finish: {finish}\n")

# Load Incident Data
file_path = "C:/Users/bencr/Downloads/Practicum/Incidents_With_OSM_IDs.csv"
dataframe = pd.read_csv(file_path) # Read CSV file
df = dataframe.rename(columns={"Lat": "Latitude", "Long": "Longitude"}) # Standardise column names

# Convert DataFrame into GeoDataFrame with point geometry
gdf_inc = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])],
    crs="EPSG:4326"
)

# Download City Network and Boundary
print(f"Downloading road network for {city_name}...")
Graph = ox.graph_from_place(
    city_name,
    network_type="drive", # Drivable roads only
    retain_all=True, # Keep disconnected components
    simplify=False # Preserve raw edges for severity mapping
)

# Add edge lengths to graph (required when simplify=False)
Graph = ox.distance.add_edge_lengths(Graph)

# Retrieve city boundary polygon
city_gdf = ox.geocode_to_gdf(city_name)
city_poly = city_gdf.geometry.union_all()

# Filter incidents to those inside the city boundary
gdf_inc = gdf_inc[gdf_inc.within(city_poly)].copy()

if gdf_inc.empty:
    print(f"[INFO] No incidents found inside {city_name}. Routing will be distance-only.\n")

# Find the nearest graph nodes to the start and end
start_node = ox.distance.nearest_nodes(Graph, start[1], start[0]) # longitude, latitude
finish_node = ox.distance.nearest_nodes(Graph, finish[1], finish[0])

print(f"Nearest start node: {start_node}")
print(f"Nearest finish node: {finish_node}")

# Ensure a valid path exists
if not nx.has_path(Graph, start_node, finish_node):
    raise ValueError("No path exists between the selected points.")

# Initialise Edge Severity
for _, _, _, data in Graph.edges(keys=True, data=True):
    data.setdefault("severity", 0) # Default severity = 0

# Map Incident Severity to graph
if not gdf_inc.empty:
    # Find nearest graph node for each incident
    gdf_inc["nearest_node"] = ox.distance.nearest_nodes(
        Graph,
        gdf_inc["Longitude"],
        gdf_inc["Latitude"]
    )
    severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict() # node → severity
else:
    severity_map = {}

# Assign severity to edges based on connected nodes
for u, v, key, data in Graph.edges(keys=True, data=True):
    if u in severity_map:
        data["severity"] = severity_map[u]
    elif v in severity_map:
        data["severity"] = severity_map[v]
    else:
        data.setdefault("severity", 0)

# Probabilistic Model
def severity_to_probability(sev, p0=0.001, lam=0.02):
    p = p0 + lam * sev # Base probability + severity scaling
    return max(0.0, min(0.99, p)) # Clamp to avoid log(0)

def edge_probability_incident(data):
    return severity_to_probability(data.get("severity", 0))

# Precompute risk weight for every edge
for _, _, _, data in Graph.edges(keys=True, data=True):
    p_inc = edge_probability_incident(data)
    data["risk_weight"] = -math.log(1.0 - p_inc) # Negative log survival probability

# Collapse MultiGraph to DiGraph
H = nx.DiGraph()
H.add_nodes_from(Graph.nodes(data=True)) # Copy nodes

for u, v, data in Graph.edges(data=True):
    length = data.get("length", 1.0)
    risk = data.get("risk_weight", 0.0)
    sev = data.get("severity", 0)

    if H.has_edge(u, v):
        # Keep shortest distance
        if length < H[u][v]["length"]:
            H[u][v]["length"] = length
        # Keep lowest risk
        if risk < H[u][v]["risk_weight"]:
            H[u][v]["risk_weight"] = risk
        # Keep highest severity for reporting
        H[u][v]["severity"] = max(H[u][v]["severity"], sev)
    else:
        H.add_edge(u, v, length=length, risk_weight=risk, severity=sev)

# Path Metrics and Safety
def path_log_safe_probability(G, path):
    log_p = 0.0
    for u, v in zip(path[:-1], path[1:]):
        sev = G[u][v].get("severity", 0)
        p_inc = severity_to_probability(sev)
        log_p += math.log(1.0 - p_inc)
    return log_p

def path_metrics(G, path):
    edges = [G[i][j] for i, j in zip(path[:-1], path[1:])]
    total_distance = sum(e.get("length", 0.0) for e in edges)
    max_severity = max(e.get("severity", 0) for e in edges) if edges else 0
    return total_distance, max_severity

def qualitative_safety(sev):
    if sev == 0:
        return "No recorded incidents along this route"
    elif sev == 1:
        return "Very low-risk route"
    elif sev == 2:
        return "Low to moderate risk"
    elif sev == 3:
        return "Moderate to high risk"
    elif sev == 4:
        return "High-risk route with severe incidents"
    else:
        return "Unknown risk level"

# A* Heuristic (Great Circle Distance)
def astar_heuristic(u, v):
    y1, x1 = H.nodes[u]['y'], H.nodes[u]['x']
    y2, x2 = H.nodes[v]['y'], H.nodes[v]['x']
    return ox.distance.great_circle(y1, x1, y2, x2)

# Shortest Path
t0 = time.perf_counter()
shortest_path = nx.astar_path(
    H,
    start_node,
    finish_node,
    heuristic=astar_heuristic,
    weight="length"
)
t1 = time.perf_counter()

# Safest Path
t2 = time.perf_counter()
safest_path = nx.astar_path(
    H,
    start_node,
    finish_node,
    heuristic=astar_heuristic,
    weight="risk_weight"
)
t3 = time.perf_counter()

# Store execution times
shortest_time = t1 - t0
safest_time = t3 - t2

# Metrics
shortest_distance, shortest_severity = path_metrics(H, shortest_path)
safest_distance, safest_severity = path_metrics(H, safest_path)

shortest_rating = qualitative_safety(shortest_severity)
safest_rating = qualitative_safety(safest_severity)

shortest_log_safe = path_log_safe_probability(H, shortest_path)
safest_log_safe = path_log_safe_probability(H, safest_path)

# Risk Summary (Threshold-Based)
def label_edges(G, path, threshold):
    labels = []
    for i, j in zip(path[:-1], path[1:]):
        sev = G[i][j].get("severity", 0)
        labels.append(1 if sev > threshold else 0)
    return labels

threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0 # Median severity threshold

shortest_high = sum(label_edges(H, shortest_path, threshold))
safest_high = sum(label_edges(H, safest_path, threshold))

# Compute relative risk reduction
if shortest_high == 0 and safest_high == 0:
    risk_reduction = 0.0
elif shortest_high == 0 and safest_high > 0:
    risk_reduction = -1.0
else:
    risk_reduction = 1 - (safest_high / shortest_high)

# Html Map Output
start_coordinates = (H.nodes[start_node]['y'], H.nodes[start_node]['x'])
finish_coordinates = (H.nodes[finish_node]['y'], H.nodes[finish_node]['x'])

m = folium.Map(location=start_coordinates, zoom_start=14)

# Add start and end markers
folium.Marker(start_coordinates, popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(finish_coordinates, popup="Finish", icon=folium.Icon(color="red")).add_to(m)

def plot_path(path, color, label):
    coordinates = [(H.nodes[n]['y'], H.nodes[n]['x']) for n in path]
    folium.PolyLine(coordinates, color=color, weight=5, opacity=0.8, popup=label).add_to(m)

plot_path(shortest_path, "blue", "Shortest Path")
plot_path(safest_path, "cyan", "Safest Path")

output_file = f"{city_name.replace(',', '').replace(' ', '_')}_prob_routes.html"
m.save(output_file)

# Output Summary
print(f"=== RESULTS FOR {city_name.upper()} ===")
print(f"Shortest Path: Distance = {shortest_distance:.2f} m | "
      f"Max Severity = {shortest_severity} | Safety = {shortest_rating} | "
      f"Log P(no incident) = {shortest_log_safe:.4f} | Time = {shortest_time:.4f} s")

print(f"Safest Path: Distance = {safest_distance:.2f} m | "
      f"Max Severity = {safest_severity} | Safety = {safest_rating} | "
      f"Log P(no incident) = {safest_log_safe:.4f} | Time = {safest_time:.4f} s")

print("\n=== ROUTE RISK SUMMARY (THRESHOLD-BASED) ===")
print(f"High-risk edges on shortest path: {shortest_high}")
print(f"High-risk edges on safest path: {safest_high}")
print(f"Relative risk reduction: {risk_reduction:.2%}")

print(f"\nHTML map saved as: {output_file}\n")