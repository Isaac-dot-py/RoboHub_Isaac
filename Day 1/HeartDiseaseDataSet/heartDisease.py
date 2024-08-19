import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
df = pd.read_csv('heartDisease.csv')

# Step 2: Prepare the features and target variable
X = df[['maximum heart rate achieved']]  # Predictor
y = df['number of major vessels (0-3) colored by flourosopy']  # Response

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


# Step 8: Visualize the data and the regression line
plt.figure(figsize=(10, 6))

# Plot the original data
sns.scatterplot(data=df, x='maximum heart rate achieved', y='number of major vessels (0-3) colored by flourosopy', label='Data', color='blue')

# Plot the regression line
heart_rate_range = np.linspace(X['maximum heart rate achieved'].min(), X['maximum heart rate achieved'].max(), 100)
predicted_vessels_range = model.predict(heart_rate_range.reshape(-1, 1))
plt.plot(heart_rate_range, predicted_vessels_range, color='red', linewidth=2, label='Regression Line')

# Highlight the prediction
plt.scatter(, , color='green', s=100, edgecolor='black', zorder=5, label='Prediction')

plt.xlabel('Maximum Heart Rate Achieved')
plt.ylabel('Number of Major Vessels Colored by Fluoroscopy')
plt.title('Prediction of Major Vessels Colored by Fluoroscopy')
plt.legend()
plt.grid(True)
plt.show()