import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('playlist_data.csv')

# Select features for PCA
features = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' ]]  # Replace with your feature columns

# Initialize PCA
pca = PCA(n_components=11)  
pca.fit(features)

# Access the components and inspect their importance
component_1 = pca.components_[0]  # First principal component
component_2 = pca.components_[1]  # Second principal component
component_3 = pca.components_[2]

# Example: Print the importance of original features in the first two components
print("Feature importance for component 1:")
for feature, importance in zip(features.columns, component_1):
    print(f"{feature}: {importance}")

print("\nFeature importance for component 2:")
for feature, importance in zip(features.columns, component_2):
    print(f"{feature}: {importance}")


print("\nFeature importance for component 3:")
for feature, importance in zip(features.columns, component_3):
    print(f"{feature}: {importance}")


