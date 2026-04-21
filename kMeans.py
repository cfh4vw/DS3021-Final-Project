# %% 
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# %%
# Import Dataset
car = pd.read_csv('car_price_dataset.csv')

# %%
# View
car.head()
# %%
# Drop unnecessary columns
# Keep CarAge, Condition, Mileage, Horsepower, Torque, Transmission, BodyType, Interior, AccidentHistory 
car_df = car.drop(['Color', 'Year','Interior', 'EngineSize(L)', 'Doors', 'Insurance', 'DriveType','Options', 'Seats','City', 'RegistrationStatus', 'FuelType'], axis=1)

# %%
car_df.head()

# %%
# One hot encode categorical variables
car_df = pd.get_dummies(car_df, columns=['Condition', 'BodyType', 'AccidentHistory'], drop_first=False, dtype=int)
# %%
scaler = StandardScaler()
scaled_data = scaler.fit_transform(car)
# %%
# kMeans with guess for k=3
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit model to training data
kmeans.fit(scaled_data)
# Predictions
clusters = kmeans.predict(scaled_data)
# Show clusters
car_df["Cluster"] = clusters

car_df.head()
# %%
# Group Clusters by mean
car_df.groupby("Cluster").mean()
# %%
# Visualization of top variables
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    car_df["AccidentHistory"],."],
    car_df["Mileage"],
    c=car_df["Price($)"]
)

plt.colorbar(scatter, label="Price")
plt.xlabel("Accident History")
plt.ylabel("Mileage")
plt.title("Price Clusters (Price as Color)")
plt.show()
# %%
