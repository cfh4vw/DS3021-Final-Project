# %% 
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

# %%
# Import Dataset
car_original = pd.read_csv('car_price_dataset.csv')
car = car_original.sample(n=1000, random_state=42)

# %%
# View
car.head()
# %%
# Drop unnecessary columns

target = car["Price($)"] #target variable

#keep Mileage(km), CarAge, Condition, AccidentHistory, FuelEfficiency(L/100km), Horsepower, BodyType, 
car_df = car.drop(columns=[
    'Brand', 'Model', 'Year', 'EngineSize(L)', 'Torque', 'Transmission',
    'DriveType', 'Doors', 'Seats', 'Color', 'Interior', 'Options', 'City',
    'Insurance', 'RegistrationStatus', 'FuelType', 'PricePerKm', 'Price($)'
])
# %%
car_df.head()

# %%
# One hot encode categorical variables
car_df = pd.get_dummies(car_df, columns=['Condition', 'BodyType', 'AccidentHistory'], drop_first=True, dtype=int)
# %%
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(car_df)

# %%
# kMeans with guess for k=3
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit model to training data
kmeans.fit(scaled_data)
# Predictions
clusters = kmeans.predict(scaled_data)
# Show clusters
car_df["Cluster"] = clusters
print(car_df["Cluster"])


# %%
# Group Clusters by mean
car_df.groupby("Cluster").mean()

#%%
#centers of clusters
print(kmeans.cluster_centers_)

# %%
# Visualization of top variables
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    car_df["Mileage(km)"],
    car_df["FuelEfficiency(L/100km)"],
    c=target,
    alpha = 0.5
)
plt.colorbar(scatter, label="Price ($)")
plt.xlabel("Mileage (km)")
plt.ylabel("FuelEfficiency(L/100km)")
plt.title("Price Clusters (Price as Color)")
plt.show()
# %%
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    np.log(car_df["Mileage(km)"]),
    np.log(car_df["FuelEfficiency(L/100km)"]),
    c=target,
    alpha=0.5
)
plt.colorbar(scatter, label="Price ($)")
plt.xlabel("Log Mileage (km)")
plt.ylabel("Log Fuel Efficiency (L/100km)")
plt.title("Price Clusters (Price as Color)")
plt.show()
# %%
