import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'corn_yield.csv'  # Ensure this file is in the same directory as this script
df = pd.read_csv(file_path)

# Preprocess the dataset
# Handle missing values by dropping rows with missing values
df = df.dropna(subset=['Value'])

# Convert 'Value' to numeric, forcing errors to NaN and then drop NaN values
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df = df.dropna(subset=['Value'])

# Select relevant columns
df = df[['Year', 'State', 'Commodity', 'Data Item', 'Value']]

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['State', 'Commodity', 'Data Item'])

# Define features (X) and target (y)
X = df.drop(['Value'], axis=1)
y = df['Value']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Calculate and print the Mean Squared Error and R² score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()
