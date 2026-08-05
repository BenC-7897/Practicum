import pandas as pd
from dateutil import parser

# Load the dataset
df = pd.read_csv("C:/Users/bencr/Downloads/Practicum/Incidents-Exportable_ForResearchers (18Sep2026)_Cleaned.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Define a function to standardise date format
def standardise_date(date_string):
    try:
        dt = parser.parse(str(date_string), fuzzy=True, default=pd.Timestamp("1900-01-01 00:00:00"))
        return dt.strftime("%y-%m-%d-%H-%M-%S")
    except Exception:
        return None

# Apply to the OccurredAt column
df['date_standardised'] = df['OccurredAt'].apply(standardise_date)

# Convert to datetime for filtering
df['date_standardised'] = pd.to_datetime(df['date_standardised'], format="%y-%m-%d-%H-%M-%S", errors='coerce')

# Filter entries after the year 2020
df_filtered = df[df['date_standardised'].dt.year > 2020]

# Save the filtered dataset
df_filtered.to_csv("C:/Users/bencr/Downloads/Practicum/Incidents_Filtered_After2020.csv", index=False)