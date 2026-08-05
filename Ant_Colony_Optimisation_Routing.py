import argparse # Handles command-line arguments
import time # Used to measure execution time
import math # Mathematical functions
import random # Random sampling for probabilistic transitions
import os # File path checks
import pandas as pd # CSV loading and data handling
import geopandas as gpd # Spatial/geographic data handling
from shapely.geometry import Point # Creates geometric point objects
import osmnx as ox # Downloads and processes OpenStreetMap road networks
import networkx as nx # Graph algorithms and data structures
import folium # Creates interactive HTML maps

# Disable caching/logging for reproducibility and cleaner output
ox.settings.use_cache = False
ox.settings.log_console = False
random.seed(42) # Ensure deterministic ACO behaviour

# Parse command-line inputs: city, start location, end location
parser = argparse.ArgumentParser(
    description="Probabilistic Ant Colony Optimisation (ACO) routing using incident severity."
)
parser.add_argument("city", type=str)
parser.add_argument("start_location", type=str)
parser.add_argument("end_location", type=str)
args = parser.parse_args()

city_name = args.city

# Print user inputs for clarity
print("\n=== INPUT RECEIVED ===")
print(f"City: {city_name}")
print(f"Start location: {args.start_location}")
print(f"End location: {args.end_location}\n")

# Convert user-provided place names into latitude/longitude coordinates
start_query = f"{args.start_location}, {city_name}"
end_query = f"{args.end_location}, {city_name}"

start_lat, start_lon = ox.geocode(start_query)
end_lat, end_lon = ox.geocode(end_query)

print(f"Geocoded Start: {(start_lat, start_lon)}")
print(f"Geocoded Finish: {(end_lat, end_lon)}\n")

# Load CSV containing incident locations and severity scores
file_path = "C:/Users/bencr/Downloads/Practicum/Incidents_With_OSM_IDs.csv"
if not os.path.exists(file_path): # If file missing, continue with empty dataset
    print(f"[WARNING] CSV not found at {file_path}. Proceeding with dummy data/empty set.")
    df = pd.DataFrame(columns=["Latitude", "Longitude", "severity"])
else:
    df = pd.read_csv(file_path)
    df = df.rename(columns={"Lat": "Latitude", "Long": "Longitude"})

gdf_inc = gpd.GeoDataFrame( # Convert incident table into a GeoDataFrame
    df,
    geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])],
    crs="EPSG:4326"
)

# Download Network
print(f"Downloading road network for {city_name}...")
Graph = ox.graph_from_place( # Download drivable road network from OpenStreetMap
    city_name,
    network_type="drive",
    retain_all=True,
    simplify=False
)

Graph = ox.distance.add_edge_lengths(Graph) # Add edge lengths (metres) to all roads
city_gdf = ox.geocode_to_gdf(city_name) # Retrieve city boundary polygon
city_poly = city_gdf.geometry.union_all()
gdf_inc = gdf_inc[gdf_inc.within(city_poly)].copy() # Keep only incidents located inside the city boundary

if gdf_inc.empty:
    print(f"[INFO] No incidents found inside {city_name}. Routing will be distance-only.\n")

# Find nearest graph nodes to start and end coordinates
start_node = ox.distance.nearest_nodes(Graph, start_lon, start_lat)
finish_node = ox.distance.nearest_nodes(Graph, end_lon, end_lat)

print(f"Nearest start node: {start_node}")
print(f"Nearest finish node: {finish_node}")

if not nx.has_path(Graph, start_node, finish_node): # Ensure a valid path exists
    raise ValueError("No path exists between the selected points.")

# Ensure every edge has a severity attribute
for _, _, _, data in Graph.edges(keys=True, data=True):
    data.setdefault("severity", 0)

if not gdf_inc.empty: # Map incidents to nearest graph nodes
    gdf_inc["nearest_node"] = ox.distance.nearest_nodes(
        Graph,
        gdf_inc["Longitude"],
        gdf_inc["Latitude"]
    )
    severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict()
else:
    severity_map = {}

for u, v, key, data in Graph.edges(keys=True, data=True): # Assign severity to edges based on endpoint nodes
    data["severity"] = max(
        severity_map.get(u, 0),
        severity_map.get(v, 0)
    )

# Convert severity score to probability of incident
def severity_to_probability(sev, p0=0.001, lam=0.02):
    return max(0.0, min(0.99, p0 + lam * sev))

# Precompute risk weight for every edge (negative log-safe probability)
for _, _, _, data in Graph.edges(keys=True, data=True):
    p_inc = severity_to_probability(data["severity"])
    data["risk_weight"] = -math.log(1.0 - p_inc)

# Convert MultiDiGraph to DiGraph by keeping: shortest length, lowest risk and highest severity
H = nx.DiGraph()
H.add_nodes_from(Graph.nodes(data=True))
for u, v, data in Graph.edges(data=True):
    length = data.get("length", 1.0)
    risk = data.get("risk_weight", 0.0)
    sev = data.get("severity", 0)
    if H.has_edge(u, v):
        H[u][v]["length"] = min(H[u][v]["length"], length)
        H[u][v]["risk_weight"] = min(H[u][v]["risk_weight"], risk)
        H[u][v]["severity"] = max(H[u][v]["severity"], sev)
    else:
        H.add_edge(u, v, length=length, risk_weight=risk, severity=sev)

# Ant-Colony Optimisation
def ant_colony_path(G, start, goal, lambda_weight=0.5):
    "ACO routing: multi-objective (distance and risk), pheromone evaporation/reinforcement and probabilistic transitions."

    # Parameters
    n_ants = 35 # Ant numbers per iteration
    max_iterations = 80 # Maximum ACO iterations
    alpha = 1.0 # Pheromone influence
    beta = 2.5 # Heuristic influence
    rho = 0.15 # Evaporation rate
    Q = 100.0 # Pheromone deposit constant
    max_steps = 300 # Max steps per ant
    k_neighbours = 6 # Random neighbour sampling for speed

    # Weighted combination of distance and risk
    def edge_cost(u, v):
        d = G[u][v].get("length", 1.0)
        r = G[u][v].get("risk_weight", 0.0)
        return lambda_weight * d + (1 - lambda_weight) * r

    # Straight-line distance to goal
    def goal_heuristic(n):
        return math.hypot(
            G.nodes[n]['x'] - G.nodes[goal]['x'],
            G.nodes[n]['y'] - G.nodes[goal]['y']
        )

    def eta(u, v): # Inverse cost heuristic 
        return 1.0 / (edge_cost(u, v) + 1e-6)

    # Pheromones
    pheromone = {edge: 1.0 for edge in G.edges()}

    best_path = None
    best_cost = float("inf")

    stagnation = 0 # Early stopping counter

    # Main Loop
    for iteration in range(max_iterations):
        # Initialise ants at start node
        ants = [{
            "node": start,
            "path": [start],
            "cost": 0.0
        } for _ in range(n_ants)]

        for ant in ants: # Move each ant through the graph

            for _ in range(max_steps):

                if ant["node"] == goal:
                    break

                neighbors = list(G.successors(ant["node"])) # Get outgoing neighbours
                neighbors = [n for n in neighbors if n not in ant["path"]]

                if not neighbors:
                    break

                # Speed optimisation: sample neighbours
                if len(neighbors) > k_neighbours:
                    neighbors = random.sample(neighbors, k_neighbours)

                probs = [] # Compute transition probabilities
                total = 0.0

                for n in neighbors:
                    tau = pheromone[(ant["node"], n)] # pheromone level
                    visibility = eta(ant["node"], n) # heuristic
                    goal_bias = 1.0 / (goal_heuristic(n) + 1e-6)

                    val = (tau ** alpha) * ((visibility + 0.5 * goal_bias) ** beta)
                    probs.append(val)
                    total += val

                if total == 0: # Choose next node probabilistically
                    next_node = random.choice(neighbors)
                else:
                    probs = [p / total for p in probs]
                    next_node = random.choices(neighbors, weights=probs, k=1)[0]

                # Update ant state  
                ant["path"].append(next_node)
                ant["cost"] += edge_cost(ant["node"], next_node)
                ant["node"] = next_node

            # Track best solution found so far
            if ant["node"] == goal and ant["cost"] < best_cost:
                best_cost = ant["cost"]
                best_path = ant["path"]
                stagnation = 0

        # Evaporation
        for edge in pheromone:
            pheromone[edge] *= (1 - rho)

        # Reinforcement
        for ant in ants:
            if ant["node"] == goal:
                deposit = Q / (ant["cost"] + 1e-6)
                for u, v in zip(ant["path"][:-1], ant["path"][1:]):
                    pheromone[(u, v)] += deposit

        # Elite Boost
        if best_path is not None:
            elite = 2.0 * (Q / (best_cost + 1e-6))
            for u, v in zip(best_path[:-1], best_path[1:]):
                pheromone[(u, v)] += elite
            stagnation += 1

        # Early Stop
        if stagnation > 15:
            break

    # If ACO fails, fall back to A* with same cost function
    if best_path is None:
        return nx.astar_path(
            G, start, goal,
            heuristic=lambda a, b: math.hypot(
                G.nodes[a]['x'] - G.nodes[b]['x'],
                G.nodes[a]['y'] - G.nodes[b]['y']
            ),
            weight=lambda u, v, d: lambda_weight * d.get("length", 1.0) + (1 - lambda_weight) * d.get("risk_weight", 0.0)
        )

    return best_path

# Routing
t0 = time.perf_counter()
shortest_path = ant_colony_path(H, start_node, finish_node, lambda_weight=1.0)
t1 = time.perf_counter()

t2 = time.perf_counter()
safest_path = ant_colony_path(H, start_node, finish_node, lambda_weight=0.0)
t3 = time.perf_counter()

shortest_time = t1 - t0
safest_time = t3 - t2

# Path Metrics
def path_metrics(G, path):
    edges = [G[u][v] for u,v in zip(path[:-1], path[1:])]
    total_distance = sum(e["length"] for e in edges)
    max_severity = max(e["severity"] for e in edges) if edges else 0
    log_safe = sum(math.log(1 - severity_to_probability(e["severity"])) for e in edges)
    return total_distance, max_severity, log_safe

def qualitative_safety(sev):
    if sev == 0:
        return "No recorded incidents"
    elif sev == 1:
        return "Very low risk"
    elif sev == 2:
        return "Low to moderate risk"
    elif sev == 3:
        return "Moderate to high risk"
    else:
        return "High risk"

sd, ss, slog = path_metrics(H, shortest_path)
fd, fs, flog = path_metrics(H, safest_path)

shortest_rating = qualitative_safety(ss)
safest_rating = qualitative_safety(fs)

# Risk Summary
threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0
def label_edges(G, path, threshold):
    return [1 if G[u][v].get("severity",0) > threshold else 0 for u,v in zip(path[:-1], path[1:])]
shortest_high = sum(label_edges(H, shortest_path, threshold))
safest_high = sum(label_edges(H, safest_path, threshold))
risk_reduction = 1 - (safest_high / shortest_high) if shortest_high else 0

# Map
def severity_colour(sev):
    if sev == 0:
        return "#2ca02c"
    elif sev == 1:
        return "#98df8a"
    elif sev == 2:
        return "#ffcc00"
    elif sev == 3:
        return "#ff7f0e"
    else:
        return "#d62728"

m = folium.Map(location=(H.nodes[start_node]['y'], H.nodes[start_node]['x']), zoom_start=14)

# Start / Finish markers
folium.Marker((H.nodes[start_node]['y'], H.nodes[start_node]['x']),
              popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker((H.nodes[finish_node]['y'], H.nodes[finish_node]['x']),
              popup="Finish", icon=folium.Icon(color="red")).add_to(m)

# Plot routes
def plot_path_with_risk(path, label, base_colour=None):
    for u,v in zip(path[:-1], path[1:]):
        sev = H[u][v].get("severity",0)
        coords = [(H.nodes[u]['y'], H.nodes[u]['x']), (H.nodes[v]['y'], H.nodes[v]['x'])]
        folium.PolyLine(
            coords,
            color=severity_colour(sev) if base_colour is None else base_colour,
            weight=6,
            opacity=0.8,
            popup=f"{label}: severity {sev}"
        ).add_to(m)

plot_path_with_risk(shortest_path, "Shortest Path", base_colour="blue")
plot_path_with_risk(safest_path, "Safest Path", base_colour="cyan")

output_file = f"{city_name.replace(',', '').replace(' ','_')}_ACO_routes.html"
m.save(output_file)

# Output
print(f"\n=== RESULTS FOR {city_name.upper()} ===")
print(f"Shortest Path: Distance = {sd:.2f} m | Max Severity = {ss} | Safety = {shortest_rating} | Log P(no incident) = {slog:.4f} | Time = {shortest_time:.4f} s")
print(f"Safest Path: Distance = {fd:.2f} m | Max Severity = {fs} | Safety = {safest_rating} | Log P(no incident) = {flog:.4f} | Time = {safest_time:.4f} s")
print("\n=== ROUTE RISK SUMMARY (THRESHOLD-BASED) ===")
print(f"High-risk edges on shortest path: {shortest_high}")
print(f"High-risk edges on safest path: {safest_high}")
print(f"Relative risk reduction: {risk_reduction:.2%}")
print(f"\nHTML map saved as: {output_file}\n")