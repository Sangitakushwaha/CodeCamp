# 1. Mean-Variance-Standard Deviation Calculator

import numpy as np

# Sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Calculate mean, variance, and standard deviation
mean = np.mean(data)
variance = np.var(data)
std_deviation = np.std(data)

# Print results
print("Mean-Variance-Standard Deviation Calculator:")
print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_deviation}")
print("\n")


# 2. Demographic Data Analyzer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the demographic dataset
df = pd.read_csv('demographic_data.csv')

# Display first few rows and data info
print("Demographic Data Analyzer:")
print(df.head())
print(df.info())

# Calculate mean age and income
mean_age = df['age'].mean()
mean_income = df['income'].mean()

# Count entries by gender
gender_counts = df['gender'].value_counts()

# Display results
print(f"Mean Age: {mean_age}")
print(f"Mean Income: {mean_income}")
print(f"Gender Distribution:\n{gender_counts}")

# Visualizing the data using histograms
df['age'].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Boxplot of income by gender
sns.boxplot(x='gender', y='income', data=df)
plt.title("Income by Gender")
plt.show()
print("\n")


# 3. Medical Data Visualizer

# Load the medical dataset
df = pd.read_csv('medical_data.csv')

# Display first few rows
print("Medical Data Visualizer:")
print(df.head())

# Visualize age vs cholesterol level with a scatter plot
sns.scatterplot(x='age', y='cholesterol', data=df)
plt.title("Age vs Cholesterol Level")
plt.show()

# Create a boxplot for blood pressure by gender
sns.boxplot(x='gender', y='blood_pressure', data=df)
plt.title("Blood Pressure by Gender")
plt.show()
print("\n")


# 4. Page View Time Series Visualizer

# Load page view data
df = pd.read_csv('page_view_data.csv', parse_dates=['date'], index_col='date')

# Display the first few rows
print("Page View Time Series Visualizer:")
print(df.head())

# Plotting the page views over time
df['page_views'].plot(figsize=(10, 6))
plt.title("Page Views Over Time")
plt.xlabel("Date")
plt.ylabel("Page Views")
plt.show()
print("\n")


# 5. Sea Level Predictor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load sea level data
df = pd.read_csv('sea_level_data.csv')

# Display the first few rows
print("Sea Level Predictor:")
print(df.head())

# Prepare the data (years and sea level)
X = df['Year'].values.reshape(-1, 1)  # Independent variable (Year)
y = df['Sea Level'].values  # Dependent variable (Sea Level)

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict sea levels for future years
future_years = pd.DataFrame({'Year': range(2020, 2051)})
predicted_sea_levels = model.predict(future_years)

# Plot the data and predictions
plt.scatter(df['Year'], df['Sea Level'], label='Actual Data')
plt.plot(df['Year'], model.predict(X), color='red', label='Fitted Line')
plt.plot(future_years['Year'], predicted_sea_levels, color='green', label='Predicted Sea Level')
plt.title("Sea Level Prediction")
plt.xlabel("Year")
plt.ylabel("Sea Level (inches)")
plt.legend()
plt.show()
print("\n")



