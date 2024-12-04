import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import zscore



# Load the cleaned CSV file
file_path = 'UScrime_cleaned.csv'
crime_data = pd.read_csv(file_path)

# List of crime columns
crime_columns = ['Murder', 'Assault', 'UrbanPop', 'Rape']

# Create individual visualizations for each crime
for crime in crime_columns:
    plt.figure(figsize=(12, 6))
    plt.bar(crime_data['State'], crime_data[crime], color='skyblue')
    plt.title(f'{crime} Rates by State', fontsize=16)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    plt.show()




# Create scatterplots for Urban Population vs each crime
crime_columns = ['Murder', 'Assault', 'Rape']
for crime in crime_columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=crime_data, x='UrbanPop', y=crime)
    plt.title(f'Urban Population vs {crime}')
    plt.xlabel('Urban Population (%)')
    plt.ylabel(f'{crime} Rate')
    plt.grid(True)
    plt.show()




# Correlation of UrbanPop with each crime
urban_corr = crime_data[['UrbanPop', 'Murder', 'Assault', 'Rape']].corr()
print("Correlation Matrix:")
print(urban_corr)
# Values close to 1 indicate a strong positive correlation.
# Values close to -1 indicate a strong negative correlation.
# Values near 0 suggest little to no linear relationship.

# Highlight UrbanPop correlations
print("\nUrbanPop correlations:")
print(urban_corr['UrbanPop'])

#anomoly detection

# Calculate z-scores for UrbanPop and each crime
crime_data['UrbanPop_z'] = zscore(crime_data['UrbanPop'])
for crime in crime_columns:
    crime_data[f'{crime}_z'] = zscore(crime_data[crime])

# Identify anomalies (e.g., z-score > 2 or < -2)
anomalies = crime_data[(crime_data['UrbanPop_z'].abs() > 2) | 
                       (crime_data[[f'{crime}_z' for crime in crime_columns]].abs().max(axis=1) > 2)]
print("\nAnomalies detected:")
print(anomalies[['State', 'UrbanPop', 'Murder', 'Assault', 'Rape']])

# will label anomoly state in the plot
for crime in crime_columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=crime_data, x='UrbanPop', y=crime)
    plt.title(f'Urban Population vs {crime}')
    plt.xlabel('Urban Population (%)')
    plt.ylabel(f'{crime} Rate')
    
    # Highlight anomalies
    anomalies_crime = anomalies[['State', 'UrbanPop', crime]]
    for _, row in anomalies_crime.iterrows():
        plt.text(row['UrbanPop'], row[crime], row['State'], color='red', fontsize=9)
    
    plt.grid(True)
    plt.show()

# regression line for visualization

# direction shows relationship if positve slope then positive relationship
# how steep the regression line is shows the strength of the relationship

for crime in crime_columns:
    # Prepare data
    X = crime_data[['UrbanPop']]
    y = crime_data[crime]
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Plot with regression line
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=crime_data, x='UrbanPop', y=crime)
    plt.plot(crime_data['UrbanPop'], y_pred, color='red', label='Regression Line')
    plt.title(f'Urban Population vs {crime} with Regression Line')
    plt.xlabel('Urban Population (%)')
    plt.ylabel(f'{crime} Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Display regression coefficients
    print(f"Regression analysis for {crime}:")
    print(f"  Coefficient: {model.coef_[0]:.2f}")
    print(f"  Intercept: {model.intercept_:.2f}")
