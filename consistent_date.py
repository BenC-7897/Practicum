import pandas as pd
from dateutil import parser

# Load your CSV
data_frame = pd.read_csv("C:/Users/bencr/Downloads/Practicum/Incidents-Exportable_ForResearchers (18Sep2026).csv")

# Strip whitespace from column names to avoid KeyErrors
data_frame.columns = data_frame.columns.str.strip()

# Define a function to parse and reformat dates
def standardise_date(date_string):
    try:
        # Default time is midnight if missing
        dt = parser.parse(str(date_string), fuzzy=True, default=pd.Timestamp("1900-01-01 00:00:00"))
        return dt.strftime("%y-%m-%d-%H-%M-%S")
    except Exception:
        return None # or use 'Invalid' or pd.NaT

# Apply to the OccurredAt column
data_frame['date_standardised'] = data_frame['OccurredAt'].apply(standardise_date)

# Save to a new CSV
data_frame.to_csv("C:/Users/bencr/Downloads/Practicum/Incidents-Exportable_ForResearchers (18Sep2026)_Cleaned.csv", index=False)