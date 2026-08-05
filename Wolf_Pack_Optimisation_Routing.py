import argparse # Handles command-line arguments
import time # Used to measure execution time
import math # Mathematical functions
import random # Randomness for WPO exploration
import pandas as pd # CSV and data handling
import geopandas as gpd # Spatial/geographic data handling
from shapely.geometry import Point # Creates geometric point objects
import osmnx as ox # Downloads and processes OpenStreetMap networks
import networkx as nx # Graph algorithms
import folium # Creates interactive HTML maps

# Settings
ox.settings.use_cache = False # Disable caching for reproducibility
ox.settings.log_console = False # Disable console logging
random.seed(42) # Fix random seed for deterministic WPO behaviour

# Command Line Interface
parser = argparse.ArgumentParser(
    description="Probabilistic WPO routing using incident severity."
)
parser.add_argument("city", type=str) # City name
parser.add_argument("start_location", type=str) # Start address
parser.add_argument("end_location", type=str) # End address
args = parser.parse_args()

city_name = args.city # Store city name

# Print user inputs
print("\n=== INPUT RECEIVED ===")
print(f"City: {city_name}")
print(f"Start location: {args.start_location}")
print(f"End location: {args.end_location}\n")

# Geocode
start_query = f"{args.start_location}, {city_name}" # Add city for accuracy
end_query = f"{args.end_location}, {city_name}"

start_latitude, start_longitude = ox.geocode(start_query) # Convert to coordinates
end_latitude, end_longitude = ox.geocode(end_query)

print(f"Geocoded Start: {(start_latitude, start_longitude)}")
print(f"Geocoded Finish: {(end_latitude, end_longitude)}\n")

# Load Incident Data
file_path = "C:/Users/bencr/Downloads/Practicum/Incidents_With_OSM_IDs.csv"
df = pd.read_csv(file_path) # Load CSV
df = df.rename(columns={"Lat": "Latitude", "Long": "Longitude"}) # Standardise names

# Convert to GeoDataFrame with point geometry
gdf_incident = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["Longitude"], df["Latitude"])],
    crs="EPSG:4326"
)

# Download Network
print(f"Downloading road network for {city_name}...")
Graph = ox.graph_from_place(
    city_name,
    network_type="drive", # Drivable roads only
    retain_all=True, # Keep disconnected components
    simplify=False # Preserve raw edges for severity mapping
)

Graph = ox.distance.add_edge_lengths(Graph) # Add edge lengths
city_gdf = ox.geocode_to_gdf(city_name) # City boundary
city_poly = city_gdf.geometry.union_all()
gdf_inc = gdf_incident [gdf_incident.within(city_poly)].copy() # Keep only incidents inside city boundary

if gdf_inc.empty: 
    print(f"[INFO] No incidents found inside {city_name}. Routing will be distance-only.\n")

# Snap Nodes
start_node = ox.distance.nearest_nodes(Graph, start_longitude, start_latitude) # Snap start
finish_node = ox.distance.nearest_nodes(Graph, end_longitude, end_latitude) # Snap end

print(f"Nearest start node: {start_node}")
print(f"Nearest finish node: {finish_node}")

if not nx.has_path(Graph, start_node, finish_node): # Ensure connectivity
    raise ValueError("No path exists between the selected points.")

# Initialise Severity
for _, _, _, data in Graph.edges(keys=True, data=True):
    data.setdefault("severity", 0) # Default severity

if not gdf_inc.empty: # Map incidents to nearest graph nodes
    gdf_inc["nearest_node"] = ox.distance.nearest_nodes(
        Graph,
        gdf_inc["Longitude"],
        gdf_inc["Latitude"]
    )
    severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict()
else:
    severity_map = {}

for u, v, key, data in Graph.edges(keys=True, data=True): # Assign severity to edges based on node severity
    data["severity"] = max(
        severity_map.get(u, 0),
        severity_map.get(v, 0)
    )

# Probabilistic Model
def severity_to_probability(sev, p0=0.001, lam=0.02):
    return max(0.0, min(0.99, p0 + lam * sev)) # Clamp to avoid log(0)

for _, _, _, data in Graph.edges(keys=True, data=True):
    p_inc = severity_to_probability(data["severity"])
    data["risk_weight"] = -math.log(1.0 - p_inc) # Negative log survival probability

# Collapse Graph
H = nx.DiGraph() # Simple directed graph
H.add_nodes_from(Graph.nodes(data=True)) # Copy nodes
for u, v, data in Graph.edges(data=True): # Keep shortest and lowest-risk parallel edges
    length = data.get("length", 1.0)
    risk = data.get("risk_weight", 0.0)
    sev = data.get("severity", 0)
    if H.has_edge(u, v):
        H[u][v]["length"] = min(H[u][v]["length"], length)
        H[u][v]["risk_weight"] = min(H[u][v]["risk_weight"], risk)
        H[u][v]["severity"] = max(H[u][v]["severity"], sev)
    else:
        H.add_edge(u, v, length=length, risk_weight=risk, severity=sev)

# Wolf Pack Optimisation
def random_path(G, start, goal, max_steps=400):
    for _ in range(40): # Try up to 40 random-walk attempts
        current = start
        path = [current]
        visited = set(path)
        steps = 0

        while current != goal and steps < max_steps: # Random walk until reaching goal or exceeding step limit
            neighbours = list(G.successors(current)) # Outgoing edges
            if not neighbours:
                break # Dead end
            unvisited = [n for n in neighbours if n not in visited] # Prefer unvisited neighbors to avoid loops
            next_node = random.choice(unvisited) if unvisited else random.choice(neighbours)
            path.append(next_node) # Extend path
            visited.add(next_node)
            current = next_node
            steps += 1

        if current == goal: # If goal reached, clean up loops and return
            return remove_loops(path)
 
    return None # No valid path found

def remove_loops(path):
    seen = {}
    new_path = []
    for node in path:
        if node in seen:
            loop_start = seen[node] # Loop detected to cut back to first occurrence
            new_path = new_path[:loop_start + 1]
            seen = {n: i for i, n in enumerate(new_path)} # Rebuild index map after truncation
        else:
            seen[node] = len(new_path)
            new_path.append(node)
    return new_path

def path_cost(G, path, weight):
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            total += G[u][v][weight]
        else:
            return float("inf") # Invalid path segment
    return total

def mutate_path(G, path, goal):
    if len(path) < 4:
        return path # Too short to mutate
    cut_idx = random.randint(1, len(path) - 2) # Random cut point
    base = path[:cut_idx] # Keep prefix
    current = base[-1]
    visited = set(base)
    extension = []
    steps = 0
    max_steps = 150

    while current != goal and steps < max_steps: # Randomly regrow the tail of the path
        neighbors = list(G.successors(current))
        if not neighbors:
            break
        candidates = [n for n in neighbors if n not in visited] # Prefer unvisited nodes
        nxt = random.choice(candidates) if candidates else random.choice(neighbors)
        extension.append(nxt)
        visited.add(nxt)
        current = nxt
        steps += 1

    if current == goal: # If regrowth reached the goal, return cleaned path
        return remove_loops(base + extension)
    return path # Mutation failed to keep original

def inherit_from_alpha(G, wolf_path, alpha_path):
    common = list(set(wolf_path) & set(alpha_path))
    if not common:
        return wolf_path # No shared nodes to no crossover possible
    pivot = random.choice(common) # Choose shared node
    wolf_idx = wolf_path.index(pivot)
    alpha_idx = alpha_path.index(pivot)
    new_path = wolf_path[:wolf_idx] + alpha_path[alpha_idx:] # Combine wolf prefix with alpha suffix
    return remove_loops(new_path)

def local_refinement(G, path, weight):
    if len(path) < 5:
        return path
    improved = path[:]
    for i in range(1, len(path) - 2):
        u = improved[i - 1]
        v = improved[i + 1]
        if G.has_edge(u, v): # If shortcut exists, compare costs
            old_cost = (
                G[improved[i - 1]][improved[i]][weight]
                + G[improved[i]][improved[i + 1]][weight]
            )
            new_cost = G[u][v][weight]
            if new_cost < old_cost: # Remove middle node if shortcut is cheaper
                improved.pop(i)
    return improved
def guaranteed_path(G, start, goal):
    try:
        # Use a simple BFS to guarantee feasibility
        return nx.shortest_path(G, start, goal)
    except:
        return None

def wolf_pack_path(G, start, goal, weight):
    n_wolves = 40 # Population size
    max_iter = 80 # Number of optimisation iterations
    elite_fraction = 0.2 # Top-performing wolves retained each generation

    wolves = []
    attempts = 0
    max_attempts = n_wolves * 40 # Upper bound on initialisation attempts

    # Initialise wolves with random or fallback paths
    while len(wolves) < n_wolves and attempts < max_attempts:
        p = random_path(G, start, goal) # Try random walk

        # If random walk fails, fall back to A*
        if p is None:
            try:
                p = nx.astar_path(
                    G, start, goal,
                    heuristic=lambda a, b: math.hypot(
                        G.nodes[a]['x'] - G.nodes[b]['x'],
                        G.nodes[a]['y'] - G.nodes[b]['y']
                    ),
                    weight=weight
                )
            except:
                p = None

        if p is not None: # Accept valid path
            wolves.append({"path": p, "cost": path_cost(G, p, weight)})

        attempts += 1

    if not wolves:
        raise RuntimeError("WPO failed to initialise any valid paths.")

    # Ensure at least one guaranteed feasible path exists
    if not wolves:
        p = guaranteed_path(G, start, goal)
        if p is None:
            raise RuntimeError("Graph connectivity failure: no path exists.")
    wolves.append({"path": p, "cost": path_cost(G, p, weight)})

    # Main optimisation loop
    for iteration in range(max_iter):
        wolves.sort(key=lambda w: w["cost"]) # Rank by cost
        # Identify alpha, beta, delta wolves
        alpha = wolves[0]
        beta = wolves[1] if len(wolves) > 1 else wolves[0]
        delta = wolves[2] if len(wolves) > 2 else wolves[0]
        elites = wolves[:max(1, int(n_wolves * elite_fraction))] # Elite preservation
        new_population = elites[:]

        while len(new_population) < n_wolves: # Generate new wolves until population is full
            parent = random.choice(elites)
            new_path = parent["path"][:]

            if random.random() < 0.8: # Mutation (exploration)
                new_path = mutate_path(G, new_path, goal)

            if random.random() < 0.6: # Inheritance from alpha/beta/delta (exploitation)
                leader = random.choice([alpha["path"], beta["path"], delta["path"]])
                new_path = inherit_from_alpha(G, new_path, leader)

            if random.random() < 0.5: # Local refinement (intensification)
                new_path = local_refinement(G, new_path, weight)

            valid = True # Validate path structure 
            for u, v in zip(new_path[:-1], new_path[1:]):
                if not G.has_edge(u, v):
                    valid = False
                    break

            if valid and new_path[0] == start and new_path[-1] == goal: # Accept only valid paths from start to goal
                new_population.append({
                    "path": new_path,
                    "cost": path_cost(G, new_path, weight)
                })

        wolves = new_population # Replace population

    wolves.sort(key=lambda w: w["cost"]) # Return best-performing wolf
    return wolves[0]["path"]

# Routing
t0 = time.perf_counter()
shortest_path = wolf_pack_path(H, start_node, finish_node, "length") # Distance-optimal
t1 = time.perf_counter()

t2 = time.perf_counter()
safest_path = wolf_pack_path(H, start_node, finish_node, "risk_weight") # Risk-optimal
t3 = time.perf_counter()

shortest_time = t1 - t0
safest_time = t3 - t2

# Path Metrics
def path_metrics(G, path):
    edges = [G[u][v] for u, v in zip(path[:-1], path[1:])]
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

sd, ss, slog = path_metrics(H, shortest_path) # Shortest distance, safety-score and path log
fd, fs, flog = path_metrics(H, safest_path) # Safest distance, safety-score and path log

# Convert numeric safety scores into qualitative labels
shortest_rating = qualitative_safety(ss)
safest_rating = qualitative_safety(fs)

# Risk Summary
threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0

def label_edges(G, path, threshold):
    return [1 if G[u][v].get("severity", 0) > threshold else 0
            for u, v in zip(path[:-1], path[1:])]

shortest_high = sum(label_edges(H, shortest_path, threshold))
safest_high = sum(label_edges(H, safest_path, threshold))
risk_reduction = 1 - (safest_high / shortest_high) if shortest_high else 0

# Both Routes
def severity_color(sev):
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

m = folium.Map(
    location=(H.nodes[start_node]['y'], H.nodes[start_node]['x']),
    zoom_start=14
)

# Start marker
folium.Marker(
    (H.nodes[start_node]['y'], H.nodes[start_node]['x']),
    popup="Start",
    icon=folium.Icon(color="green")
).add_to(m)

# Finish marker
folium.Marker(
    (H.nodes[finish_node]['y'], H.nodes[finish_node]['x']),
    popup="Finish",
    icon=folium.Icon(color="red")
).add_to(m)

# Draw a path with a fixed colour
def draw_path(G, path, label, base_color):
    for u, v in zip(path[:-1], path[1:]):
        sev = G[u][v].get("severity", 0)
        coords = [
            (G.nodes[u]['y'], G.nodes[u]['x']),
            (G.nodes[v]['y'], G.nodes[v]['x'])
        ]
        folium.PolyLine(
            coords,
            color=base_color,
            weight=6,
            opacity=0.8,
            popup=f"{label}: severity {sev}"
        ).add_to(m)

# Draw both routes
draw_path(H, shortest_path, "Shortest Path", "blue")
draw_path(H, safest_path, "Safest Path", "cyan")

# Save combined map
output_file = f"{city_name.replace(',', '').replace(' ','_')}_WPO_routes.html"
m.save(output_file)

print(f"\n=== RESULTS FOR {city_name.upper()} ===")
print(f"Shortest Path: Distance = {sd:.2f} m | Max Severity = {ss} | "
      f"Safety = {shortest_rating} | Log P(no incident) = {slog:.4f} | "
      f"Time = {shortest_time:.4f} s")
print(f"Safest Path: Distance = {fd:.2f} m | Max Severity = {fs} | "
      f"Safety = {safest_rating} | Log P(no incident) = {flog:.4f} | "
      f"Time = {safest_time:.4f} s")

print("\n=== ROUTE RISK SUMMARY (THRESHOLD-BASED) ===")
print(f"High-risk edges on shortest path: {shortest_high}")
print(f"High-risk edges on safest path: {safest_high}")
print(f"Relative risk reduction: {risk_reduction:.2%}")

print(f"\nHTML map saved as: {output_file}\n")