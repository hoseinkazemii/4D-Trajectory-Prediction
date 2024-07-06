import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load your dataframe
df = pd.read_csv("LoadData_20240624150205.csv")

# Ensure the dataframe has the columns 'X', 'Y', 'Z'
print(df.head())



# Calculate the correlation matrix
correlation_matrix = df[['X', 'Y', 'Z']].corr()
print("Correlation Matrix:\n", correlation_matrix)

# Plot the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of X, Y, Z Coordinates")
plt.savefig("correlation.png", dpi=300)
plt.show()



# Prepare the data
X = df[['X', 'Z']]
y = df['Y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Get feature importances
feature_importances = model.feature_importances_
print("Feature Importances:\n", feature_importances)

# Plot feature importances
features = X.columns
importances = feature_importances
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig("RelativeImportance.png", dpi=300)
plt.show()