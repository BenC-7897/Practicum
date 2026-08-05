import argparse # Handles command-line arguments
import time # Used to measure execution time
import math # Provides mathematical functions
import pandas as pd # Data analysis and CSV handling
import geopandas as gpd # Spatial/geographic data handling
from shapely.geometry import Point # Creates geometric point objects
import osmnx as ox # Downloads and works with OpenStreetMap road networks
import networkx as nx # Graph algorithms such as Dijkstra
import folium # Creates interactive HTML maps

# Disable OSMnx caching/logging
ox.settings.use_cache = False
ox.settings.log_console = False

# Create parser for user inputs
parser = argparse.ArgumentParser(description="Probabilistic Dijkstra routing using incident severity.")
parser.add_argument("city", type=str, help="City name, e.g. 'Dublin, Ireland'") # City Name
parser.add_argument("start_location", type=str, help="Start address or place name") # Starting Location 
parser.add_argument("end_location", type=str, help="End address or place name") # Finishing Location 
args = parser.parse_args() # User Entry

city_name = args.city # Store city name separately

# Print user inputs
print("\n=== INPUT RECEIVED ===")
print(f"City: {city_name}")
print(f"Start location: {args.start_location}")
print(f"End location: {args.end_location}\n")

# Geocode Start and End: Combine locations with city name for accurate geocoding
start_query = f"{args.start_location}, {city_name}"
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
df = pd.read_csv(file_path) # Read CSV file
df = df.rename(columns={"Lat": "Latitude", "Long": "Longitude"}) # Rename latitude/longitude columns for consistency

gdf_inc = gpd.GeoDataFrame( # Convert DataFrame into GeoDataFrame
    df,
    geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])],
    crs="EPSG:4326" # Coordinate reference system
)

# Download City Network and Boundary
print(f"Downloading road network for {city_name}...")
Graph = ox.graph_from_place( # Download drivable road network from OpenStreetMap
    city_name,
    network_type="drive",
    retain_all=True,
    simplify=False
)

# Add edge lengths to all roads
Graph = ox.distance.add_edge_lengths(Graph)

# Retrieve city boundary polygon
city_gdf = ox.geocode_to_gdf(city_name)
city_poly = city_gdf.geometry.union_all()

# Filter Incidents to City
gdf_inc = gdf_inc[gdf_inc.within(city_poly)].copy() # Keep only incidents located inside city boundary

if gdf_inc.empty: # Warn user if no incidents are found
    print(f"[INFO] No incidents found inside {city_name}. Routing will be distance-only.\n")

# Find nearest road-network node to start and finish coordinates
start_node = ox.distance.nearest_nodes(Graph, start[1], start[0])
finish_node = ox.distance.nearest_nodes(Graph, finish[1], finish[0])

# Print node IDs
print(f"Nearest start node: {start_node}")
print(f"Nearest finish node: {finish_node}")

# Check whether a valid path exists
if not nx.has_path(Graph, start_node, finish_node):
    raise ValueError("No path exists between the selected points.")

# Initialise Edge Attributes
for _, _, _, data in Graph.edges(keys=True, data=True):
    data.setdefault("severity", 0)

# Map Graph Incident Severity
if not gdf_inc.empty:
    gdf_inc["nearest_node"] = ox.distance.nearest_nodes( # Find nearest graph node for every incident
        Graph,
        gdf_inc["Longitude"],
        gdf_inc["Latitude"]
    )
    severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict() # node_id -> severity value
else:
    severity_map = {}

# Assign severity values to graph edges
for u, v, key, data in Graph.edges(keys=True, data=True): 
    if u in severity_map: # If start node of edge has severity
        data["severity"] = severity_map[u]
    elif v in severity_map: # Else if end node has severity
        data["severity"] = severity_map[v]
    else: # Otherwise keep severity at zero
        data.setdefault("severity", 0)

# Probabilistic Model
def severity_to_probability(sev, p0=0.001, lam=0.02): # Convert severity score into incident probability
    p = p0 + lam * sev # Base probability and severity scaling
    return max(0.0, min(0.99, p)) # Clamp probability between 0 and 0.99

def edge_probability_incident(data): # Calculate edge incident probability
    return severity_to_probability(data.get("severity", 0))

# Precompute risk weight for every edge
for _, _, _, data in Graph.edges(keys=True, data=True):
    p_inc = edge_probability_incident(data) # Incident Probability
    data["risk_weight"] = -math.log(1.0 - p_inc) # Convert to additive Dijkstra-compatible weight

# Collapse MultiDiGraph to DiGraph: Create simplified directed graph
H = nx.DiGraph()
H.add_nodes_from(Graph.nodes(data=True)) # Copy nodes into new graph

for u, v, data in Graph.edges(data=True): # Iterate through all edges
    # Extract attributes
    length = data.get("length", 1.0)
    risk = data.get("risk_weight", 0.0)
    sev = data.get("severity", 0)

    # If edge already exists
    if H.has_edge(u, v):
        # Keep smallest length and lowest risk; keep max severity
        if length < H[u][v]["length"]: # Keep shortest distance
            H[u][v]["length"] = length
        if risk < H[u][v]["risk_weight"]: # Keep lowest risk
            H[u][v]["risk_weight"] = risk
        H[u][v]["severity"] = max(H[u][v]["severity"], sev) # Keep highest severity
    else:
        H.add_edge(u, v, length=length, risk_weight=risk, severity=sev) # Add edge to graph

# Path Metrics And Safety: Compute log probability of avoiding incidents
def path_log_safe_probability(G, path):
    log_p = 0.0
    for u, v in zip(path[:-1], path[1:]): # Iterate through route edges
        sev = G[u][v].get("severity", 0)
        p_inc = severity_to_probability(sev) # Convert severity into incident probability
        log_p += math.log(1.0 - p_inc) # Add log-safe probability
    return log_p

def path_metrics(G, path): # Calculate route distance and max severity
    edges = [G[i][j] for i, j in zip(path[:-1], path[1:])] # Extract all edges in path
    total_distance = sum(e.get("length", 0.0) for e in edges) # Total route distance
    max_severity = max(e.get("severity", 0) for e in edges) if edges else 0 # Maximum severity encountered
    return total_distance, max_severity

def qualitative_safety(sev): # Convert severity to human-readable safety description
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

# Shortest Path (Distance)
t0 = time.perf_counter() # Start timing
shortest_path = nx.dijkstra_path(H, start_node, finish_node, weight="length") # Compute shortest route
t1 = time.perf_counter() # End timing

# Safest Path (Risk Weight)
def negative_log_safe_weight(u, v, data): # Custom weight function using risk values
    return data.get("risk_weight", 0.0)

t2 = time.perf_counter() # Start timing
safest_path = nx.dijkstra_path(H, start_node, finish_node, weight="risk_weight") # Compute safest route
t3 = time.perf_counter() # End timing

# Store execution times
shortest_time = t1 - t0
safest_time = t3 - t2

# Metrics
shortest_distance, shortest_severity = path_metrics(H, shortest_path) # Shortest route metrics
safest_distance, safest_severity = path_metrics(H, safest_path) # Safest route metrics

# Human-readable safety labels
shortest_rating = qualitative_safety(shortest_severity)
safest_rating = qualitative_safety(safest_severity)

# Log probabilities
shortest_log_safe = path_log_safe_probability(H, shortest_path)
safest_log_safe = path_log_safe_probability(H, safest_path)

# Risk Summary
def label_edges(G, path, threshold): # Label high-risk edges
    labels = []
    for i, j in zip(path[:-1], path[1:]):
        sev = G[i][j].get("severity", 0)
        labels.append(1 if sev > threshold else 0) # Mark edge as high risk if above threshold
    return labels

threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0 # Use median severity as threshold

# Count high-risk edges
shortest_high = sum(label_edges(H, shortest_path, threshold))
safest_high = sum(label_edges(H, safest_path, threshold))

# Compute relative risk reduction
if shortest_high == 0 and safest_high == 0:
    risk_reduction = 0.0
elif shortest_high == 0 and safest_high > 0:
    risk_reduction = -1.0
else:
    risk_reduction = 1 - (safest_high / shortest_high)

# HTML Map Output: Extract coordinates from graph nodes
start_coordinates = (H.nodes[start_node]['y'], H.nodes[start_node]['x'])
finish_coordinates = (H.nodes[finish_node]['y'], H.nodes[finish_node]['x'])

m = folium.Map(location=start_coordinates, zoom_start=14) # Create folium map

# Add start and destination marker
folium.Marker(start_coordinates, popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(finish_coordinates, popup="Finish", icon=folium.Icon(color="red")).add_to(m)

def plot_path(path, color, label): # Function to draw route on map
    coordinates = [(H.nodes[n]['y'], H.nodes[n]['x']) for n in path] # Convert node IDs to coordinates
    folium.PolyLine(coordinates, color=color, weight=5, opacity=0.8, popup=label).add_to(m) # Draw polyline

plot_path(shortest_path, "blue", "Shortest Path") # Plot shortest route
plot_path(safest_path, "cyan", "Safest Path") # Plot safest route

output_file = f"{city_name.replace(',', '').replace(' ', '_')}_prob_routes.html"
m.save(output_file) # Save map as HTML file

# Output Summary
print(f"=== RESULTS FOR {city_name.upper()} ===")
# Print shortest-path metrics
print(f"Shortest Path: Distance = {shortest_distance:.2f} m | "
      f"Max Severity = {shortest_severity} | Safety = {shortest_rating} | "
      f"Log P(no incident) = {shortest_log_safe:.4f} | Time = {shortest_time:.4f} s")

# Print safest-path metrics
print(f"Safest Path: Distance = {safest_distance:.2f} m | "
      f"Max Severity = {safest_severity} | Safety = {safest_rating} | "
      f"Log P(no incident) = {safest_log_safe:.4f} | Time = {safest_time:.4f} s")

# Print risk comparison summary
print("\n=== ROUTE RISK SUMMARY (THRESHOLD-BASED) ===")
print(f"High-risk edges on shortest path: {shortest_high}")
print(f"High-risk edges on safest path: {safest_high}")
print(f"Relative risk reduction: {risk_reduction:.2%}")

print(f"\nHTML map saved as: {output_file}\n") # Print saved map file name