# %% 
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse

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
# scale numerical data with log to help with skew and make clearer clusters
car_df["Mileage(km)"] = np.log1p(car_df["Mileage(km)"])
car_df["FuelEfficiency(L/100km)"] = np.log1p(car_df["FuelEfficiency(L/100km)"])
car_df["Horsepower"] = np.log1p(car_df["Horsepower"])
car_df["CarAge"] = np.log1p(car_df["CarAge"])
# %%
scaler = StandardScaler()
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
# Evaluate quality of clusters using silhouette score
inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled_data)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_data, labels))

print("Total Variance for k=2-10:", inertias)
print("Silhouette Scores for k=2-10:", sil_scores)
print("Model Score for k=10:", km.score(scaled_data))
# %%
# Elbow plot to find ideal k (# of clusters)
plt.plot(k_range, inertias)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
# %%
# Silhouette scores for k=2-10
plt.plot(k_range, sil_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")
plt.show()
# %%
# Run final model with k=6
kmeans_final = KMeans(n_clusters=6, random_state=42)
car_df["Cluster_Final"] = kmeans_final.fit_predict(scaled_data)

plt.figure(figsize=(8,6))
final = plt.scatter(
    car_df["Mileage(km)"],
    car_df["FuelEfficiency(L/100km)"],
    c=target
)

plt.colorbar(final, label="Price ($)")
plt.xlabel("Mileage (km)")
plt.ylabel("Fuel Efficiency (L/100km)")
plt.title("Final Clusters")
plt.show()

# %%
# labeling clusters

plt.figure(figsize=(8,6))
final = plt.scatter(
    car_df["Mileage(km)"],
    car_df["FuelEfficiency(L/100km)"],
    c=car_df["Cluster_Final"]
)

plt.colorbar(final, label="Price ($)")
plt.xlabel("Mileage (km)")
plt.ylabel("Fuel Efficiency (L/100km)")
plt.title("Final Clusters")
plt.show()

# %%
# Evaluate final clusters
print("Final Inertia:", kmeans_final.inertia_)
print("Final Silhouette:", silhouette_score(scaled_data, car_df["Cluster_Final"]))
# %%
# convert numerical data back to original for summary interpretation
car_df["Mileage(km)"] = np.expm1(car_df["Mileage(km)"])
car_df["FuelEfficiency(L/100km)"] = np.expm1(car_df["FuelEfficiency(L/100km)"])
car_df["Horsepower"] = np.expm1(car_df["Horsepower"])
car_df["CarAge"] = np.expm1(car_df["CarAge"])

# %%
# Bring back make and model
car_df["Brand"] = car["Brand"]
car_df["Model"] = car["Model"]
# %%
# Use the model to find underpriced and overpriced cars based on quality
car_df["Price($)"] = target
car_df["Cluster"] = car_df["Cluster_Final"]
car_df["Difference"] = car_df["Price($)"] - car_df.groupby("Cluster")["Price($)"].transform("median")

# %%
# Underpriced cars
car_df.sort_values("Difference").head(10)

# %%
# Overpriced cars
car_df.sort_values("Difference", ascending=False).head(10)


# %%
