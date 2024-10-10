# plot_pca_3d_plotly_webgl_random_sample.py

import pickle
import os
import numpy as np
import plotly.graph_objects as go

# Load the pickled PCA-transformed data and target variable
export_folder = 'PCA_visualization'
X_pca_3d_path = os.path.join(export_folder, 'X_pca_3d.pkl')
y_ridge_path = os.path.join(export_folder, 'y_ridge.pkl')

with open(X_pca_3d_path, 'rb') as f:
    X_pca_3d = pickle.load(f)

with open(y_ridge_path, 'rb') as f:
    y_ridge = pickle.load(f)

# Convert y_ridge to numpy array for calculations
y_ridge = y_ridge.to_numpy()

# Step 1: Calculate mean and standard deviation of the target variable (y_ridge)
mean_y = np.mean(y_ridge)
std_y = np.std(y_ridge)

# Step 2: Define the range for values within 2 standard deviations of the mean
lower_bound = mean_y - 8 * std_y
upper_bound = mean_y + 8 * std_y

# Step 3: Filter for points within 2 standard deviations of the target variable
within_range = (y_ridge >= lower_bound) & (y_ridge <= upper_bound)
X_pca_3d_filtered = X_pca_3d[within_range]
y_ridge_filtered = y_ridge[within_range]

# Step 4: Randomly sample 100,000 points from the filtered data
sample_size = min(100000, len(X_pca_3d_filtered))  # Ensure sample_size does not exceed filtered data size
indices = np.random.choice(len(X_pca_3d_filtered), sample_size, replace=False)
X_pca_3d_sampled = X_pca_3d_filtered[indices, :]
y_ridge_sampled = y_ridge_filtered[indices]

# Step 5: Normalize the sampled target values (y_ridge) for coloring
y_ridge_norm = (y_ridge_sampled - np.min(y_ridge_sampled)) / (np.max(y_ridge_sampled) - np.min(y_ridge_sampled))

# Step 6: Create the 3D scatter plot using Plotly with WebGL
fig = go.Figure(data=[go.Scatter3d(
    x=X_pca_3d_sampled[:, 0],  # PC1
    y=X_pca_3d_sampled[:, 1],  # PC2
    z=X_pca_3d_sampled[:, 2],  # PC3
    mode='markers',
    marker=dict(
        size=2,  # Reduce size for better performance
        color=y_ridge_norm,  # Color based on the target variable
        colorscale='icefire',  # Red-to-blue colormap
        colorbar=dict(title="Target"),
        showscale=True,
        opacity=0.8  # Slight transparency for better visibility
    )
)])

# Step 7: Update layout for better visualization and add axis labels
fig.update_layout(
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3',
        xaxis=dict(backgroundcolor="black", color="white", gridcolor="gray"),
        yaxis=dict(backgroundcolor="black", color="white", gridcolor="gray"),
        zaxis=dict(backgroundcolor="black", color="white", gridcolor="gray"),
    ),
    margin=dict(l=0, r=0, b=0, t=0),  # Set the margins to fit the canvas
    paper_bgcolor="black",
    font=dict(color="white")
)

# Step 8: Show the plot
fig.show()
