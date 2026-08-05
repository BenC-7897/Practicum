import argparse # Handles command-line arguments
import time # Used to measure execution time
import math # Mathematical functions
import os # File path checks
import random # Random sampling for probabilistic transitions
import pandas as pd # CSV loading and data handling
import geopandas as gpd # Spatial/geographic data handling
from shapely.geometry import Point # Creates geometric point objects
import osmnx as ox # Downloads and processes OpenStreetMap road networks
import networkx as nx # Graph algorithms and data structures
import folium # Creates interactive HTML maps

# Settings
ox.settings.use_cache = False
ox.settings.log_console = False
random.seed(42)

# Command Line Interface
parser = argparse.ArgumentParser(
    description="Probabilistic Ant Colony Optimisation routing using incident severity with robust geocoding."
)
parser.add_argument("city", type=str, help="e.g., 'Dublin, Ireland'")
parser.add_argument("start_location", type=str, help="Start address or place name")
parser.add_argument("end_location", type=str, help="End address or place name")
args = parser.parse_args()

city_name = args.city

print("\n=== INPUT RECEIVED ===")
print(f"City: {city_name}")
print(f"Start location: {args.start_location}")
print(f"End location: {args.end_location}\n")

# Geocode
start_query = f"{args.start_location}, {city_name}"
end_query = f"{args.end_location}, {city_name}"

try:
    print("Geocoding start location...")
    start_lat, start_lon = ox.geocode(start_query)

    print("Geocoding finish location...")
    end_lat, end_lon = ox.geocode(end_query)

    print(f"Geocoded Start: {(start_lat, start_lon)}")
    print(f"Geocoded Finish: {(end_lat, end_lon)}\n")
except ox._errors.InsufficientResponseError as e: # Handles cases where Nominatim cannot interpret the address
    print("\n[CRITICAL ERROR] Geocoding Failed!")
    print("OpenStreetMap Nominatim engine could not resolve your location query string.")
    print("Suggestions:")
    print(" 1. Remove descriptive text inside parentheses like '(Macartney Bdg)'")
    print(" 2. Use a cleaner address line like 'Baggot Street Lower' or a known landmark.")
    print(f"\nDetails: {e}")
    raise SystemExit(1)
except Exception as e: # Catch-all for unexpected geocoding errors
    print(f"\n[CRITICAL ERROR] An unexpected error occurred during geocoding: {e}")
    raise SystemExit(1)

# Download Road Network
print(f"Downloading road network for {city_name}...")
Graph = ox.graph_from_place(
    city_name,
    network_type="drive",
    retain_all=True,
    simplify=True # Simplify geometry for faster ACO traversal
)
Graph = ox.distance.add_edge_lengths(Graph) # Add length attribute to edges

city_gdf = ox.geocode_to_gdf(city_name)
city_poly = city_gdf.geometry.union_all()

# Snap Nodes
start_node = ox.distance.nearest_nodes(Graph, start_lon, start_lat)
finish_node = ox.distance.nearest_nodes(Graph, end_lon, end_lat)

print(f"Nearest start node: {start_node}")
print(f"Nearest finish node: {finish_node}")

if not nx.has_path(Graph, start_node, finish_node): # Prevents ACO from running on disconnected components
    raise ValueError("No path exists between the selected points.")

# Helper Functions
def severity_to_prob(sev, p0=0.001, lam=0.02):
    return max(0.0, min(0.99, p0 + lam * sev))

def qualitative_safety(sev): # Human-readable interpretation of severity
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

def path_metrics(G, path):
    edges = [G[u][v] for u, v in zip(path[:-1], path[1:])]
    total_distance = sum(e["length"] for e in edges)
    max_severity = max(e["severity"] for e in edges) if edges else 0
    log_safe = sum(math.log(1 - severity_to_prob(e["severity"])) for e in edges)
    return total_distance, max_severity, log_safe

def severity_color(sev): # Colour scale for map visualisation
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

# Ant Colony Optimisation
def ant_colony_path(G, start, goal, lambda_weight=0.5):
    "ACO routing: multi-objective (distance and risk), pheromone evaporation/reinforcement and probabilistic transitions."

    # Core ACO hyperparameters controlling exploration/exploitation
    n_ants = 35
    max_iterations = 80
    alpha = 1.0
    beta = 2.5
    rho = 0.15
    Q = 100.0
    max_steps = 300
    k_neighbours = 6

    def edge_cost(u, v): # Weighted combination of distance and risk
        d = G[u][v].get("length", 1.0)
        r = G[u][v].get("risk_weight", 0.0)
        return lambda_weight * d + (1 - lambda_weight) * r

    def goal_heuristic(n): # Straight-line distance to goal node
        return math.hypot(
            G.nodes[n]['x'] - G.nodes[goal]['x'],
            G.nodes[n]['y'] - G.nodes[goal]['y']
        )

    def eta(u, v):
        return 1.0 / (edge_cost(u, v) + 1e-6)

    pheromone = {edge: 1.0 for edge in G.edges()} # Initialise pheromone levels on all edges
    best_path = None
    best_cost = float("inf")
    stagnation = 0 # Tracks lack of improvement

    for iteration in range(max_iterations):
        ants = [{ # Initialise ants at the start node
            "node": start,
            "path": [start],
            "cost": 0.0
        } for _ in range(n_ants)]

        for ant in ants:
            # Each ant performs a probabilistic walk through the graph
            for _ in range(max_steps):
                if ant["node"] == goal:
                    break # Ant reached destination

                neighbors = list(G.successors(ant["node"]))
                neighbors = [n for n in neighbors if n not in ant["path"]]
                if not neighbors:
                    break # Dead end

                if len(neighbors) > k_neighbours: # Randomly sample neighbours to reduce computation
                    neighbors = random.sample(neighbors, k_neighbours)

                probs = [] # Compute transition probabilities
                total = 0.0
                for n in neighbors:
                    tau = pheromone[(ant["node"], n)]
                    visibility = eta(ant["node"], n)
                    goal_bias = 1.0 / (goal_heuristic(n) + 1e-6)
                    val = (tau ** alpha) * ((visibility + 0.5 * goal_bias) ** beta) # Combined desirability score
                    probs.append(val)
                    total += val

                # Select next node probabilistically
                if total == 0:
                    next_node = random.choice(neighbors)
                else:
                    probs = [p / total for p in probs]
                    next_node = random.choices(neighbors, weights=probs, k=1)[0]

                # Update ant state
                ant["path"].append(next_node)
                ant["cost"] += edge_cost(ant["node"], next_node)
                ant["node"] = next_node

            # Track best ant of this iteration
            if ant["node"] == goal and ant["cost"] < best_cost:
                best_cost = ant["cost"]
                best_path = ant["path"]
                stagnation = 0

        for edge in pheromone: # Evaporate pheromone globally
            pheromone[edge] *= (1 - rho)

        for ant in ants: # Deposit pheromone for successful ants
            if ant["node"] == goal:
                deposit = Q / (ant["cost"] + 1e-6)
                for u, v in zip(ant["path"][:-1], ant["path"][1:]):
                    pheromone[(u, v)] += deposit

        if best_path is not None: # Elite reinforcement which boosts best path found so far
            elite = 2.0 * (Q / (best_cost + 1e-6))
            for u, v in zip(best_path[:-1], best_path[1:]):
                pheromone[(u, v)] += elite
            stagnation += 1

        if stagnation > 15: # Early stopping if no improvement
            break
    
    # Fallback to A* if ACO fails to find a path
    if best_path is None:
        return nx.astar_path(
            G, start, goal,
            heuristic=lambda a, b: math.hypot(
                G.nodes[a]['x'] - G.nodes[b]['x'],
                G.nodes[a]['y'] - G.nodes[b]['y']
            ),
            weight=lambda u, v, d: lambda_weight * d.get("length", 1.0)
                                  + (1 - lambda_weight) * d.get("risk_weight", 0.0)
        )

    return best_path

# Routing
def run_routing_pipeline(df_source, case_label):
    print("\n==============================================")
    print(f"RUNNING: {case_label}")
    print("==============================================")

    t_start = time.perf_counter()

    # Process Incident Data
    if df_source.empty:
        print("[INFO] Incident dataset is empty for this run.")
        gdf_inc = gpd.GeoDataFrame(columns=["Latitude", "Longitude", "severity"], geometry=[])
    else:
        gdf_inc = gpd.GeoDataFrame(
            df_source,
            geometry=[Point(xy) for xy in zip(df_source["Longitude"], df_source["Latitude"])],
            crs="EPSG:4326"
        )
        gdf_inc = gdf_inc[gdf_inc.within(city_poly)].copy()

    print(f"Incidents mapped inside {city_name}: {len(gdf_inc)}")

    # Clear old attributes and initialise edge severities directly on global Graph
    for _, _, data in Graph.edges(data=True):
        data["severity"] = 0

    if not gdf_inc.empty:
        gdf_inc["nearest_node"] = ox.distance.nearest_nodes(
            Graph, gdf_inc["Longitude"], gdf_inc["Latitude"]
        )
        severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict()
    else:
        severity_map = {}

    for u, v, data in Graph.edges(data=True):
        data["severity"] = max(severity_map.get(u, 0), severity_map.get(v, 0))

    # Calculate probabilistic risk weights directly on global Graph
    for _, _, data in Graph.edges(data=True):
        p_inc = severity_to_prob(data["severity"])
        data["risk_weight"] = -math.log(1.0 - p_inc)

    # Reconstruct DiGraph preserving geometry
    H = nx.DiGraph()
    H.add_nodes_from(Graph.nodes(data=True))

    for u, v, data in Graph.edges(data=True):
        H.add_edge(
            u, v,
            length=data.get("length", 1.0),
            risk_weight=data.get("risk_weight", 0.0),
            severity=data.get("severity", 0),
            geometry=data.get("geometry")
        )

    # Compute paths using Ant Colony Optimisation engine
    t0 = time.perf_counter()
    shortest_path = ant_colony_path(H, start_node, finish_node, lambda_weight=1.0)
    shortest_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    safest_path = ant_colony_path(H, start_node, finish_node, lambda_weight=0.0)
    safest_time = time.perf_counter() - t1

    pipeline_time = time.perf_counter() - t_start

    # Metrics
    sd, ss, slog = path_metrics(H, shortest_path)
    fd, fs, flog = path_metrics(H, safest_path)

    threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0
    shortest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    safest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(safest_path[:-1], safest_path[1:]))
    risk_reduction = 1 - (safest_high / shortest_high) if shortest_high else 0

    # Output
    print(f"\n=== ACO RESULTS FOR {city_name.upper()} ({case_label}) ===")
    print(f"Shortest Path: Distance = {sd:.2f} m | Max Severity = {ss} | Safety = {qualitative_safety(ss)} | Log P(no incident) = {slog:.4f} | Time = {shortest_time:.4f} s")
    print(f"Safest Path: Distance = {fd:.2f} m | Max Severity = {fs} | Safety = {qualitative_safety(fs)} | Log P(no incident) = {flog:.4f} | Time = {safest_time:.4f} s")

    print("\n=== ROUTE RISK SUMMARY (THRESHOLD-BASED) ===")
    print(f"High-risk edges on shortest path: {shortest_high}")
    print(f"High-risk edges on safest path: {safest_high}")
    print(f"Relative risk reduction: {risk_reduction:.2%}")

    # Map Creation
    m = folium.Map(location=(H.nodes[start_node]["y"], H.nodes[start_node]["x"]), zoom_start=14)

    folium.Marker((H.nodes[start_node]["y"], H.nodes[start_node]["x"]), popup="Start",
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker((H.nodes[finish_node]["y"], H.nodes[finish_node]["x"]), popup="Finish",
                  icon=folium.Icon(color="red")).add_to(m)

    def plot_path_with_risk(path, label, base_color=None):
        for u, v in zip(path[:-1], path[1:]):
            sev = H[u][v].get("severity", 0)
            geom = H[u][v].get("geometry")
            if geom is not None:
                coords = [(lat, lon) for lon, lat in geom.coords]
            else:
                coords = [(H.nodes[u]["y"], H.nodes[u]["x"]), (H.nodes[v]["y"], H.nodes[v]["x"])]
            folium.PolyLine(
                coords,
                color=severity_color(sev) if base_color is None else base_color,
                weight=6,
                opacity=0.8,
                popup=f"{label}: severity {sev}"
            ).add_to(m)

    plot_path_with_risk(shortest_path, "Shortest Path", base_color="blue")
    plot_path_with_risk(safest_path, "Safest Path", base_color="cyan")

    clean_label = case_label.lower().replace(" ", "_")
    output_file = f"{city_name.replace(',', '').replace(' ', '_')}_aco_{clean_label}_routes.html"
    m.save(output_file)
    print(f"\nHTML map saved as: {output_file}")

# Data Loading and City Exclusion
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

print("\nAll Ant Colony Optimisation routing comparisons completed successfully.")