import pandas as pd

# Load dataset
crime_data = pd.read_csv('UScrime.csv')

# Display the first few rows
print("Preview of the dataset:")
print(crime_data.head())

# Rename the unnamed column to State
if 'Unnamed: 0' in crime_data.columns:
    crime_data.rename(columns={'Unnamed: 0': 'State'}, inplace=True)

# Check for missing values
missing_values = crime_data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Drop rows with missing values (if any)
crime_data.dropna(inplace=True)

# Convert columns to appropriate data types (if necessary)
print("\nColumn data types before conversion:")
print(crime_data.dtypes)

# Assuming all columns except 'State' are numeric
for col in crime_data.columns:
    if col != 'State':
        crime_data[col] = pd.to_numeric(crime_data[col], errors='coerce')

# Display updated data types
print("\nColumn data types after conversion:")
print(crime_data.dtypes)

# Display summary statistics
print("\nSummary statistics:")
print(crime_data.describe())

# Ensure the data is in proper format for further analysis
print("\nFinal cleaned dataset preview:")
print(crime_data.head())

# Save the cleaned data (optional)
crime_data.to_csv('UScrime_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'UScrime_cleaned.csv'.")
