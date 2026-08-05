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
random.seed(42) # Ensure deterministic behaviour

# Parse command-line inputs: city, start location, end location
parser = argparse.ArgumentParser(
    description="Probabilistic Particle Swarm Optimisation (PSO) routing using incident severity."
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

# Convert severity score → probability of incident
def severity_to_probability(sev, p0=0.001, lam=0.02):
    return max(0.0, min(0.99, p0 + lam * sev))

# Precompute risk weight for every edge (negative log-safe probability)
for _, _, _, data in Graph.edges(keys=True, data=True):
    p_inc = severity_to_probability(data["severity"])
    data["risk_weight"] = -math.log(1.0 - p_inc)

# Convert MultiDiGraph → DiGraph by keeping: shortest length, lowest risk and highest severity
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

# Particle Swarm Optimisation
def particle_swarm_path(G, start, goal, lambda_weight=0.5):
    # Parameters
    n_particles = 35 # Particle numbers
    max_iter = 80 # Maximum PSO iterations
    max_steps = 300 # Max steps when constructing a path
    k_neighbours = 6 # Random neighbour sampling for speed

    w = 0.6 # Inertia weight
    c1 = 1.4 # Cognitive coefficient
    c2 = 1.4 # Social coefficient

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

    # Construct a greedy-biased path
    def construct_path():
        node = start
        path = [start]

        for _ in range(max_steps):

            if node == goal:
                break

            neighbors = list(G.successors(node))
            neighbors = [n for n in neighbors if n not in path]

            if not neighbors:
                break

            if len(neighbors) > k_neighbours:
                neighbors = random.sample(neighbors, k_neighbours)

            scores = []
            for n in neighbors:
                c = edge_cost(node, n)
                h = goal_heuristic(n)
                scores.append((c + 0.5 * h, n))

            scores.sort(key=lambda x: x[0])
            node = scores[0][1]
            path.append(node)

        if path[-1] != goal:
            return None
        return path

    # Path cost
    def path_cost(path):
        if path is None or len(path) < 2:
            return float("inf")
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if not G.has_edge(u, v):
                return float("inf")
            total += edge_cost(u, v)
        return total

    # Mutation operator (velocity application)
    def mutate(path):
        if path is None or len(path) < 3:
            return path

        new_path = path[:]
        op = random.choice(["splice", "swap"])

        # Splice: replace a segment with a random walk
        if op == "splice":
            i = random.randint(0, len(new_path) - 2)
            j = random.randint(i + 1, len(new_path) - 1)
            u = new_path[i]
            v = new_path[j]

            node = u
            sub = [u]
            for _ in range(20):
                if node == v:
                    break
                neighbors = list(G.successors(node))
                if not neighbors:
                    break
                node = random.choice(neighbors)
                sub.append(node)
                if node == v:
                    break

            if sub[-1] == v:
                new_path = new_path[:i] + sub + new_path[j+1:]

        # Swap: swap two internal nodes
        elif op == "swap" and len(new_path) > 4:
            i, j = sorted(random.sample(range(1, len(new_path)-1), 2))
            new_path[i], new_path[j] = new_path[j], new_path[i]

        # Repair: ensure edges exist
        repaired = [new_path[0]]
        for u, v in zip(new_path[:-1], new_path[1:]):
            if G.has_edge(u, v):
                repaired.append(v)
            else:
                break

        if repaired[-1] != goal:
            return path
        return repaired

    # Apply velocity (number of mutations)
    def apply_velocity(path, v):
        new_path = path
        for _ in range(v):
            new_path = mutate(new_path)
        return new_path

    # Path distance for velocity update
    def path_distance(a, b):
        if a is None or b is None:
            return 0
        return abs(len(a) - len(b)) + sum(x != y for x, y in zip(a, b))

    # Swarm Initialisation
    particles = []
    global_best = None
    global_best_cost = float("inf")

    for _ in range(n_particles):
        p = construct_path()
        if p is None:
            continue
        c = path_cost(p)
        particle = {
            "position": p,
            "velocity": 1,
            "best_position": p[:],
            "best_cost": c
        }
        particles.append(particle)

        if c < global_best_cost:
            global_best_cost = c
            global_best = p[:]

    # Fallback if no valid initialisation
    if global_best is None:
        return nx.astar_path(
            G, start, goal,
            heuristic=lambda a, b: math.hypot(
                G.nodes[a]['x'] - G.nodes[b]['x'],
                G.nodes[a]['y'] - G.nodes[b]['y']
            ),
            weight=lambda u, v, d: lambda_weight * d.get("length", 1.0) + (1 - lambda_weight) * d.get("risk_weight", 0.0)
        )

    # Main Loop
    for _ in range(max_iter):
        for particle in particles:

            # Velocity Update
            r1 = random.random()
            r2 = random.random()

            dist_p = path_distance(particle["position"], particle["best_position"])
            dist_g = path_distance(particle["position"], global_best)

            v_new = (
                w * particle["velocity"] + c1 * r1 * dist_p + c2 * r2 * dist_g
            )
            particle["velocity"] = max(1, int(round(v_new)))

            # Postion Update
            new_pos = apply_velocity(particle["position"], particle["velocity"])
            new_cost = path_cost(new_pos)

            # Accept if better
            if new_cost < path_cost(particle["position"]):
                particle["position"] = new_pos

            # Update personal best
            if new_cost < particle["best_cost"]:
                particle["best_cost"] = new_cost
                particle["best_position"] = new_pos[:]

            # Update global best
            if new_cost < global_best_cost:
                global_best_cost = new_cost
                global_best = new_pos[:]

    # Fallback
    if global_best is None:
        return nx.astar_path(
            G, start, goal,
            heuristic=lambda a, b: math.hypot(
                G.nodes[a]['x'] - G.nodes[b]['x'],
                G.nodes[a]['y'] - G.nodes[b]['y']
            ),
            weight=lambda u, v, d: lambda_weight * d.get("length", 1.0) + (1 - lambda_weight) * d.get("risk_weight", 0.0)
        )

    return global_best

# Routing
t0 = time.perf_counter()
shortest_path = particle_swarm_path(H, start_node, finish_node, lambda_weight=1.0)
t1 = time.perf_counter()

t2 = time.perf_counter()
safest_path = particle_swarm_path(H, start_node, finish_node, lambda_weight=0.0)
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

m = folium.Map(location=(H.nodes[start_node]['y'], H.nodes[start_node]['x']), zoom_start=14)

# Start and Finish markers
folium.Marker((H.nodes[start_node]['y'], H.nodes[start_node]['x']),
              popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker((H.nodes[finish_node]['y'], H.nodes[finish_node]['x']),
              popup="Finish", icon=folium.Icon(color="red")).add_to(m)

# Plot routes
def plot_path_with_risk(path, label, base_color=None):
    for u,v in zip(path[:-1], path[1:]):
        sev = H[u][v].get("severity",0)
        coords = [(H.nodes[u]['y'], H.nodes[u]['x']), (H.nodes[v]['y'], H.nodes[v]['x'])]
        folium.PolyLine(
            coords,
            color=severity_color(sev) if base_color is None else base_color,
            weight=6,
            opacity=0.8,
            popup=f"{label}: severity {sev}"
        ).add_to(m)

plot_path_with_risk(shortest_path, "Shortest Path", base_color="blue")
plot_path_with_risk(safest_path, "Safest Path", base_color="cyan")

output_file = f"{city_name.replace(',', '').replace(' ','_')}_PSO_routes.html"
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