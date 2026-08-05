import argparse # Handle command-line arguments
import time # Measure execution time
import math # Mathematical operations
import os # File system checks
import random # Randomness for WPO exploration
import pandas as pd # Tabular data handling
import geopandas as gpd # Spatial data handling
from shapely.geometry import Point # Geometry construction for incidents
import osmnx as ox # OpenStreetMap network download and geocoding
import networkx as nx # Graph algorithms
import folium # Interactive map visualisation

# Settings
ox.settings.use_cache = False # Disable caching for reproducibility
ox.settings.log_console = False # Silence OSMnx console logs
random.seed(42) # Fix random seed for deterministic WPO behaviour

# Command Line Interface
parser = argparse.ArgumentParser(
    description="Real-time WPO routing using incident severity comparison."
)
parser.add_argument("city", type=str, help="e.g., 'Dublin, Ireland'")
parser.add_argument("start_location", type=str, help="Start address or place name")
parser.add_argument("end_location", type=str, help="End address or place name")
args = parser.parse_args()

city_name = args.city # Store city name for reuse

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
    retain_all=True, # Keep disconnected components
    simplify=False # Preserve raw edges for severity mapping
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

# Shared Helpers
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

# Wolf Pack Optimisation Core Helpers
def remove_loops(path): # Loop removal
    seen = {}
    new_path = []
    for node in path:
        if node in seen:
            loop_start = seen[node] # Loop detected: cut path back to node's first occurrence
            new_path = new_path[:loop_start + 1]
            seen = {n: i for i, n in enumerate(new_path)} # Rebuild index map for remaining nodes
        else:
            seen[node] = len(new_path)
            new_path.append(node)
    return new_path

def random_path(G, start, goal, max_steps=400): # Random path generation
    for _ in range(40):
        current = start
        path = [current]
        visited = set(path)
        steps = 0

        while current != goal and steps < max_steps:
            neighbours = list(G.successors(current)) # Outgoing neighbours
            if not neighbours:
                break
            unvisited = [n for n in neighbours if n not in visited]
            next_node = random.choice(unvisited) if unvisited else random.choice(neighbours) # Prefer unvisited nodes: fall back to any neighbour
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            steps += 1

        if current == goal:
            return remove_loops(path)
    return None

def path_cost(G, path, weight): # Path cost
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            total += G[u][v][weight]
        else:
            return float("inf")
    return total

def mutate_path(G, path, goal): # Mutation
    if len(path) < 4:
        return path # Too short to meaningfully mutate
    cut_idx = random.randint(1, len(path) - 2) # Avoid cutting at endpoints
    base = path[:cut_idx] # Prefix to keep
    current = base[-1]
    visited = set(base)
    extension = []
    steps = 0
    max_steps = 150

    while current != goal and steps < max_steps:
        neighbors = list(G.successors(current))
        if not neighbors:
            break
        candidates = [n for n in neighbors if n not in visited]
        nxt = random.choice(candidates) if candidates else random.choice(neighbors)
        extension.append(nxt)
        visited.add(nxt)
        current = nxt
        steps += 1

    if current == goal:
        return remove_loops(base + extension)
    return path

def inherit_from_alpha(G, wolf_path, alpha_path): # inheritance from alpha
    common = list(set(wolf_path) & set(alpha_path))
    if not common:
        return wolf_path
    pivot = random.choice(common)
    wolf_idx = wolf_path.index(pivot)
    alpha_idx = alpha_path.index(pivot)
    new_path = wolf_path[:wolf_idx] + alpha_path[alpha_idx:]
    return remove_loops(new_path)

def local_refinement(G, path, weight): # local refinement
    if len(path) < 5:
        return path
    improved = path[:]
    i = 1
    while i < len(improved) - 2:
        u = improved[i - 1]
        v = improved[i + 1]
        if G.has_edge(u, v):
            old_cost = (
                G[improved[i - 1]][improved[i]][weight]
                + G[improved[i]][improved[i + 1]][weight]
            )
            new_cost = G[u][v][weight]
            if new_cost < old_cost:
                improved.pop(i)
                continue # Re-check at same index after removal
        i += 1
    return improved

# Guaranteed path (fallback) using Dijkstra
def guaranteed_path(G, start, goal, weight):
    try:
        return nx.dijkstra_path(G, start, goal, weight=weight)
    except:
        return None

# Wolf Pack Optimisation main routine
def wolf_pack_path(G, start, goal, weight):
    n_wolves = 40
    max_iter = 80
    elite_fraction = 0.2

    wolves = []
    attempts = 0
    max_attempts = n_wolves * 40

    while len(wolves) < n_wolves and attempts < max_attempts:
        p = random_path(G, start, goal) # Try random walk first

        if p is None: # If random walk fails, fall back to guaranteed Dijkstra path
            p = guaranteed_path(G, start, goal, weight)

        if p is not None:
            wolves.append({"path": p, "cost": path_cost(G, p, weight)})

        attempts += 1

    if not wolves:
        raise RuntimeError("WPO failed to initialise any valid paths.")

    # Ensure at least one guaranteed feasible path exists
    p = guaranteed_path(G, start, goal, weight)
    if p is not None:
        wolves.append({"path": p, "cost": path_cost(G, p, weight)})

    for iteration in range(max_iter): # Rank wolves by cost (ascending)
        wolves.sort(key=lambda w: w["cost"])
        alpha = wolves[0]
        beta = wolves[1] if len(wolves) > 1 else wolves[0]
        delta = wolves[2] if len(wolves) > 2 else wolves[0]
        elites = wolves[:max(1, int(n_wolves * elite_fraction))] # Elite preservation
        new_population = elites[:]

        while len(new_population) < n_wolves: # Generate new wolves until population is full
            parent = random.choice(elites)
            new_path = parent["path"][:]

            # Mutation (exploration)
            if random.random() < 0.8:
                new_path = mutate_path(G, new_path, goal)

            # Inheritance from alpha/beta/delta (exploitation)
            if random.random() < 0.6:
                leader = random.choice([alpha["path"], beta["path"], delta["path"]])
                new_path = inherit_from_alpha(G, new_path, leader)

            # Local refinement (intensification)
            if random.random() < 0.5:
                new_path = local_refinement(G, new_path, weight)

            # Validate path structure: all edges must exist
            valid = True
            for u, v in zip(new_path[:-1], new_path[1:]):
                if not G.has_edge(u, v):
                    valid = False
                    break

            # Accept only valid paths from start to goal
            if valid and new_path[0] == start and new_path[-1] == goal:
                new_population.append({
                    "path": new_path,
                    "cost": path_cost(G, new_path, weight)
                })

        wolves = new_population # Replace population with new generation

    # Return best-performing wolf (lowest cost)
    wolves.sort(key=lambda w: w["cost"])
    return wolves[0]["path"]

# Real-Time Wolf Pack Optimisation
def run_routing_pipeline(df_source, case_label):
    print("\n==============================================")
    print(f"RUNNING: {case_label}")
    print("==============================================")

    G_local = Graph.copy() # Work on a base graph copy to avoid cross-contamination between cases

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

    # Initialise edge severities
    for _, _, _, data in G_local.edges(keys=True, data=True):
        data.setdefault("severity", 0)

    if not gdf_inc.empty: # Map incidents to nearest graph nodes and build severity map
        gdf_inc["nearest_node"] = ox.distance.nearest_nodes(
            G_local, gdf_inc["Longitude"], gdf_inc["Latitude"]
        )
        severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict()
    else:
        severity_map = {}

    for u, v, key, data in G_local.edges(keys=True, data=True):
        data["severity"] = max(severity_map.get(u, 0), severity_map.get(v, 0))

    # Probabilistic risk weights
    for _, _, _, data in G_local.edges(keys=True, data=True):
        p_inc = severity_to_probability(data["severity"])
        data["risk_weight"] = -math.log(1.0 - p_inc) # Negative log survival probability

    # Collapse MultiDiGraph → DiGraph
    H = nx.DiGraph()
    H.add_nodes_from(G_local.nodes(data=True))

    for u, v, data in G_local.edges(data=True):
        length = data.get("length", 1.0)
        risk = data.get("risk_weight", 0.0)
        sev = data.get("severity", 0)

        if H.has_edge(u, v): # Keep shortest and lowest-risk edge, but highest severity
            H[u][v]["length"] = min(H[u][v]["length"], length)
            H[u][v]["risk_weight"] = min(H[u][v]["risk_weight"], risk)
            H[u][v]["severity"] = max(H[u][v]["severity"], sev)
        else:
            H.add_edge(u, v, length=length, risk_weight=risk, severity=sev)

    # Compute paths with WPO
    t0 = time.perf_counter()
    shortest_path = wolf_pack_path(H, start_node, finish_node, weight="length")
    shortest_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    safest_path = wolf_pack_path(H, start_node, finish_node, weight="risk_weight")
    safest_time = time.perf_counter() - t1

    # Metrics
    sd, ss, slog = path_metrics(H, shortest_path)
    fd, fs, flog = path_metrics(H, safest_path)

    # Threshold for high-risk edges: median incident severity
    threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0
    shortest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    safest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(safest_path[:-1], safest_path[1:]))
    risk_reduction = 1 - (safest_high / shortest_high) if shortest_high else 0

    # Output
    print(f"\n=== RESULTS FOR {city_name.upper()} ({case_label}) ===")
    print(f"Shortest Path (WPO): Distance = {sd:.2f} m | Max Severity = {ss} | "
          f"Safety = {qualitative_safety(ss)} | Log P(no incident) = {slog:.4f} | "
          f"Time = {shortest_time:.4f} s")
    print(f"Safest Path (WPO): Distance = {fd:.2f} m | Max Severity = {fs} | "
          f"Safety = {qualitative_safety(fs)} | Log P(no incident) = {flog:.4f} | "
          f"Time = {safest_time:.4f} s")

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

    plot_path(shortest_path, "Shortest Path (WPO)", "blue")
    plot_path(safest_path, "Safest Path (WPO)", "cyan")

    clean_label = case_label.lower().replace(" ", "_")
    output_file = f"{city_name.replace(',', '').replace(' ', '_')}_{clean_label}_wpo_routes.html"
    m.save(output_file)
    print(f"\nHTML map saved as: {output_file}")

# Data Loading and City Exclusion
file_path = "C:/Users/bencr/Downloads/Practicum/Incidents_With_OSM_IDs.csv"

if not os.path.exists(file_path): # Ensure base incident CSV exists
    print(f"[ERROR] Base incident CSV file not found at {file_path}.")
    raise SystemExit(1)

df_all = pd.read_csv(file_path) # Load full incident dataset
df_all = df_all.rename(columns={"Lat": "Latitude", "Long": "Longitude"})

gdf_all = gpd.GeoDataFrame( # Convert to GeoDataFrame for spatial exclusion
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

print("\nAll WPO routing comparisons completed successfully.")