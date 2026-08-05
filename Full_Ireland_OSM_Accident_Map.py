import osmnx as ox # Importing osmnx for downloading and working with OpenStreetMap data
import geopandas as gpd # For geospatial data handling
import polars as pl # Fast DataFrame library for CSV processing
import folium # Importing folium for creating interactive maps

ox.settings.use_cache = True # Enable caching so OSM data is stored locally
ox.settings.log_console = True # Enable console logging so progress messages are shown

# Geocode "Ireland" into a GeoDataFrame representing the country's boundary
ireland = ox.geocode_to_gdf("Ireland") 

# Download the full drivable road network within Ireland's boundary
G = ox.graph_from_polygon(
    ireland.geometry.iloc[0],
    network_type="drive",
    simplify=True
)

# Read the accident CSV file using Polars 
dataframe = pl.read_csv(
    "C:/Users/bencr/Downloads/Practicum/Incidents_Filtered_After2020.csv"
)

# Vectorised nearest-node OpenStreetMap for each accident point
osm_ids = ox.distance.nearest_nodes(
    G,
    dataframe["Long"].to_list(),
    dataframe["Lat"].to_list()
)

dataframe = dataframe.with_columns(pl.Series("OSM_ID", osm_ids)) # Add the nearest OSM node ID as a new column in the dataframe

# Convert textual accident outcomes into numeric severity scores
severity_map = {
    "Fatality": 4,
    "Serious injuries": 3,
    "Minor injuries": 2,
    "No injuries": 1,
    "Unknown": 0
}

# Map severity scores to line colours for visualisation
colour_map = {
    4: "red",
    3: "orange",
    2: "blue",
    1: "green"
}

# Create a new column "severity" by mapping Outcome → numeric score
dataframe = dataframe.with_columns(
    pl.col("Outcome")
      .map_elements(lambda x: severity_map.get(x, 0))
      .alias("severity")
)

# Group accidents by OSM node and take the maximum severity
street_severity = (
    dataframe.group_by("OSM_ID")
      .agg(pl.col("severity").max())
      .to_dict(as_series=False)
)

# Convert Polars output into a standard Python dictionary
street_severity = dict(zip(
    street_severity["OSM_ID"],
    street_severity["severity"]
))

edges = ox.graph_to_gdfs(G, nodes=False) # Convert the OSM graph into a GeoDataFrame of edges

# Calculate map centre using mean latitude and longitude of accidents
centre_latitude = dataframe["Lat"].mean()
centre_longitude = dataframe["Long"].mean()

# Initialise the interactive Folium map
m = folium.Map(
    location=[centre_latitude, centre_longitude],
    zoom_start=7,
    tiles="cartodbpositron"
)

# Loop through every road segment in Ireland
for _, edge in edges.iterrows():
    u, v = edge.name[:2] # Extract the start (u) and end (v) node IDs of the edge

    severity = max( # Determine the highest severity recorded on either end of the street
        street_severity.get(u, 0),
        street_severity.get(v, 0)
    )

    if severity == 0: 
        continue # skip non-incident streets

    coordinates = [(lat, lon) for lon, lat in edge.geometry.coords] # Convert geometry coordinates from (lon, lat) to (lat, lon) for Folium

    folium.PolyLine( # Draw the road segment on the map
        locations=coordinates,
        colour=colour_map[severity],
        weight=3,
        opacity=0.85
    ).add_to(m)

# Save the interactive HTML map to disk
m.save(
    "C:/Users/bencr/Downloads/Practicum/ireland_accident_street_map.html"
)

# Save the accident dataset with OSM node IDs added
dataframe.write_csv(
    "C:/Users/bencr/Downloads/Practicum/Incidents_With_OSM_IDs.csv"
)