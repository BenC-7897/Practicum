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

# Settings
ox.settings.use_cache = False
ox.settings.log_console = False
random.seed(42)

# Command Line Interface
parser = argparse.ArgumentParser(
    description="Real-time PSO routing using incident severity comparison (Dijkstra-style pipeline)."
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

try:
    print("Geocoding start location...")
    start_latitude, start_longitude = ox.geocode(start_query)

    print("Geocoding finish location...")
    end_latitude, end_longitude = ox.geocode(end_query)

    print(f"Geocoded Start: {(start_latitude, start_longitude)}")
    print(f"Geocoded Finish: {(end_latitude, end_longitude)}\n")
except ox._errors.InsufficientResponseError as e: # Handle Nominatim geocoding failures gracefully
    print("\n[CRITICAL ERROR] Geocoding Failed!")
    print("OpenStreetMap Nominatim engine could not resolve your location query string.")
    print("Suggestions:")
    print(" 1. Remove descriptive text inside parentheses like '(Macartney Bdg)'")
    print(" 2. Use a cleaner address line like 'Baggot Street Lower' or a known landmark.")
    print(f"\nDetails: {e}")
    raise SystemExit(1)
except Exception as e: # Catch all for unexpected geocoding errors
    print(f"\n[CRITICAL ERROR] An unexpected error occurred during geocoding: {e}")
    raise SystemExit(1)

# MultiDiGraph is kept unsimplified to preserve full routing structure
print(f"Downloading road network for {city_name}...")
Graph = ox.graph_from_place(
    city_name,
    network_type="drive",
    retain_all=True,
    simplify=False # keep MultiDiGraph 
)
Graph = ox.distance.add_edge_lengths(Graph) # Add edge lengths in metres

# City polygon used for incident filtering
city_gdf = ox.geocode_to_gdf(city_name)
city_poly = city_gdf.geometry.union_all()

# Snap user locations to graph
start_node = ox.distance.nearest_nodes(Graph, start_longitude, start_latitude)
finish_node = ox.distance.nearest_nodes(Graph, end_longitude, end_latitude)

print(f"Nearest start node: {start_node}")
print(f"Nearest finish node: {finish_node}")

# Ensure a valid path exists
if not nx.has_path(Graph, start_node, finish_node):
    raise ValueError("No path exists between the selected points.")

# Helper Functions
def severity_to_probability(sev, p0=0.001, lam=0.02):
    return max(0.0, min(0.99, p0 + lam * sev)) # Convert severity score into incident probability using linear model

def qualitative_safety(sev): # Map severity values to human-readable risk categories
    if sev == 0: return "No recorded incidents"
    elif sev == 1: return "Very low risk"
    elif sev == 2: return "Low to moderate risk"
    elif sev == 3: return "Moderate to high risk"
    else: return "High risk"

def path_metrics(G, path): # Compute distance, max severity and log safety score for a given path
    edges = [G[u][v] for u, v in zip(path[:-1], path[1:])]
    total_distance = sum(e["length"] for e in edges)
    max_severity = max(e["severity"] for e in edges) if edges else 0
    log_safe = sum(math.log(1 - severity_to_probability(e["severity"])) for e in edges)
    return total_distance, max_severity, log_safe

# Real-Time Particle Swarm Optimisation
def particle_swarm_path(G, start, goal, lambda_weight=0.5):
    "PSO routing that optimises distance–risk trade‑offs using particle path updates."

    # PSO hyperparameters
    n_particles = 35
    max_iter = 80
    max_steps = 300
    k_neighbours = 6

    w = 0.6 # inertia
    c1 = 1.4 # cognitive attraction
    c2 = 1.4 # social attraction

    def edge_cost(u, v): # Weighted combination of distance and risk
        d = G[u][v].get("length", 1.0)
        r = G[u][v].get("risk_weight", 0.0)
        return lambda_weight * d + (1 - lambda_weight) * r

    def goal_heuristic(n): # Euclidean heuristic for goal proximity
        return math.hypot(
            G.nodes[n]['x'] - G.nodes[goal]['x'],
            G.nodes[n]['y'] - G.nodes[goal]['y']
        )

    # Greedy Path Construction
    def construct_path():
        node = start
        path = [start]

        for _ in range(max_steps):
            if node == goal:
                break

            neighbours = list(G.successors(node))
            neighbours = [n for n in neighbours if n not in path]

            if not neighbours:
                break

            if len(neighbours) > k_neighbours: # Limit branching factor
                neighbours = random.sample(neighbours, k_neighbours)

            scores = []
            for n in neighbours:
                c = edge_cost(node, n)
                h = goal_heuristic(n)
                scores.append((c + 0.5 * h, n)) # Combined cost

            scores.sort(key=lambda x: x[0])
            node = scores[0][1]
            path.append(node)

        if path[-1] != goal:
            return None
        return path

    def path_cost(path): # Compute total cost of a path
        if path is None or len(path) < 2:
            return float("inf")
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if not G.has_edge(u, v):
                return float("inf")
            total += edge_cost(u, v)
        return total

    # Mutation Operators
    def mutate(path):
        if path is None or len(path) < 3:
            return path

        new_path = path[:]
        op = random.choice(["splice", "swap"])

        if op == "splice": # Replace a segment with a random walk
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

        elif op == "swap" and len(new_path) > 4: # Swap two internal nodes
            i, j = sorted(random.sample(range(1, len(new_path)-1), 2))
            new_path[i], new_path[j] = new_path[j], new_path[i]

        # Repair path by ensuring edges exist
        repaired = [new_path[0]]
        for u, v in zip(new_path[:-1], new_path[1:]):
            if G.has_edge(u, v):
                repaired.append(v)
            else:
                break

        if repaired[-1] != goal:
            return path
        return repaired

    # Velocity Application
    def apply_velocity(path, v):
        new_path = path
        for _ in range(v):
            new_path = mutate(new_path)
        return new_path

    # Path Distance
    def path_distance(a, b):
        if a is None or b is None:
            return 0
        return abs(len(a) - len(b)) + sum(x != y for x, y in zip(a, b))

    # Initialise Particles
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

    if global_best is None: # Fallback to A* if PSO cannot initialise
        return nx.astar_path(
            G, start, goal,
            heuristic=lambda a, b: math.hypot(
                G.nodes[a]['x'] - G.nodes[b]['x'],
                G.nodes[a]['y'] - G.nodes[b]['y']
            ),
            weight=lambda u, v, d: lambda_weight * d.get("length", 1.0)
                                  + (1 - lambda_weight) * d.get("risk_weight", 0.0)
        )

    for _ in range(max_iter):
        for particle in particles:
            r1 = random.random()
            r2 = random.random()

            dist_p = path_distance(particle["position"], particle["best_position"])
            dist_g = path_distance(particle["position"], global_best)

            v_new = ( # Velocity update
                w * particle["velocity"]
                + c1 * r1 * dist_p
                + c2 * r2 * dist_g
            )
            particle["velocity"] = max(1, int(round(v_new)))

            new_pos = apply_velocity(particle["position"], particle["velocity"]) # Apply velocity (mutations)
            new_cost = path_cost(new_pos)

            if new_cost < path_cost(particle["position"]): # Update particle position
                particle["position"] = new_pos

            if new_cost < particle["best_cost"]: # Update personal best
                particle["best_cost"] = new_cost
                particle["best_position"] = new_pos[:]

            if new_cost < global_best_cost: # Update global best
                global_best_cost = new_cost
                global_best = new_pos[:]

    # Final fallback
    if global_best is None:
        return nx.astar_path(
            G, start, goal,
            heuristic=lambda a, b: math.hypot(
                G.nodes[a]['x'] - G.nodes[b]['x'],
                G.nodes[a]['y'] - G.nodes[b]['y']
            ),
            weight=lambda u, v, d: lambda_weight * d.get("length", 1.0)
                                  + (1 - lambda_weight) * d.get("risk_weight", 0.0)
        )

    return global_best

# Routing Function
def run_routing_pipeline(df_source, case_label):
    print("\n==============================================")
    print(f"RUNNING: {case_label}")
    print("==============================================")

    G_local = Graph.copy() # Work on a local copy to avoid mutating global graph

    # Process Incident Data
    if df_source.empty:
        print("[INFO] Incident dataset is empty for this run.")
        gdf_inc = gpd.GeoDataFrame(columns=["Latitude", "Longitude", "severity"], geometry=[])
    else:
        # Convert to GeoDataFrame
        gdf_inc = gpd.GeoDataFrame(
            df_source,
            geometry=[Point(xy) for xy in zip(df_source["Longitude"], df_source["Latitude"])],
            crs="EPSG:4326"
        )
        gdf_inc = gdf_inc[gdf_inc.within(city_poly)].copy() # Keep only incidents inside city polygon

    print(f"Incidents mapped inside {city_name}: {len(gdf_inc)}")

    # Initialise edge severities
    for _, _, _, data in G_local.edges(keys=True, data=True):
        data.setdefault("severity", 0)

    # Map incidents to nearest nodes
    if not gdf_inc.empty:
        gdf_inc["nearest_node"] = ox.distance.nearest_nodes(
            G_local, gdf_inc["Longitude"], gdf_inc["Latitude"]
        )
        severity_map = gdf_inc.set_index("nearest_node")["severity"].to_dict()
    else:
        severity_map = {}

    # Assign severity to edges
    for u, v, key, data in G_local.edges(keys=True, data=True):
        data["severity"] = max(severity_map.get(u, 0), severity_map.get(v, 0))

    # Convert severity to probability to risk weight
    for _, _, _, data in G_local.edges(keys=True, data=True):
        p_inc = severity_to_probability(data["severity"])
        data["risk_weight"] = -math.log(1.0 - p_inc) # Log-risk transformation

    # Collapse MultiDiGraph to DiGraph
    H = nx.DiGraph()
    H.add_nodes_from(G_local.nodes(data=True))

    for u, v, data in G_local.edges(data=True):
        length = data.get("length", 1.0)
        risk = data.get("risk_weight", 0.0)
        sev = data.get("severity", 0)

        if H.has_edge(u, v):
	    # Keep shortest length, lowest risk and highest severity
            H[u][v]["length"] = min(H[u][v]["length"], length)
            H[u][v]["risk_weight"] = min(H[u][v]["risk_weight"], risk)
            H[u][v]["severity"] = max(H[u][v]["severity"], sev)
        else:
            H.add_edge(u, v, length=length, risk_weight=risk, severity=sev)

    # Real-Time Particle Swarm Optimisation Routing
    t0 = time.perf_counter()
    shortest_path = particle_swarm_path(H, start_node, finish_node, lambda_weight=1.0)
    shortest_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    safest_path = particle_swarm_path(H, start_node, finish_node, lambda_weight=0.0)
    safest_time = time.perf_counter() - t1

    # Metrics
    sd, ss, slog = path_metrics(H, shortest_path)
    fd, fs, flog = path_metrics(H, safest_path)

    threshold = gdf_inc["severity"].median() if not gdf_inc.empty else 0
    shortest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(shortest_path[:-1], shortest_path[1:]))
    safest_high = sum(H[u][v]["severity"] > threshold for u, v in zip(safest_path[:-1], safest_path[1:]))
    risk_reduction = 1 - (safest_high / shortest_high) if shortest_high else 0

    # Output
    print(f"\n=== REAL-TIME PSO RESULTS FOR {city_name.upper()} ({case_label}) ===")
    print(f"Shortest Path: Distance = {sd:.2f} m | Max Severity = {ss} | Safety = {qualitative_safety(ss)} | "
          f"Log P(no incident) = {slog:.4f} | Time = {shortest_time:.4f} s")
    print(f"Safest Path: Distance = {fd:.2f} m | Max Severity = {fs} | Safety = {qualitative_safety(fs)} | "
          f"Log P(no incident) = {flog:.4f} | Time = {safest_time:.4f} s")

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

    def plot_path(path, label, color): # Draw each edge of the path as a polyline and attach a popup showing edge severity
        for u, v in zip(path[:-1], path[1:]):
            coords = [(H.nodes[u]["y"], H.nodes[u]["x"]), (H.nodes[v]["y"], H.nodes[v]["x"])]
            folium.PolyLine(coords, color=color, weight=6, opacity=0.8,
                            popup=f"{label}: severity {H[u][v]['severity']}").add_to(m)

    plot_path(shortest_path, "Shortest Path (PSO)", "blue")
    plot_path(safest_path, "Safest Path (PSO)", "cyan")

    # Save map to an HTML file named by city and case label
    clean_label = case_label.lower().replace(" ", "_")
    output_file = f"{city_name.replace(',', '').replace(' ', '_')}_realtime_pso_{clean_label}_routes.html"
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

print("\nAll real-time PSO routing comparisons completed successfully.")