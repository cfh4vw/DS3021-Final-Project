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
car = car.drop(['Color', 'Interior', 'DriveType','Options', 'City', 'RegistrationStatus', 'FuelType'], axis=1)

# %%
car.head()
# %%
