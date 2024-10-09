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

# Normalize the target values (y_ridge) to range between 0 and 1 for coloring
y_ridge = y_ridge.to_numpy()
y_ridge_norm = (y_ridge - np.min(y_ridge)) / (np.max(y_ridge) - np.min(y_ridge))

# Randomly sample 100,000 points
sample_size = 100000
indices = np.random.choice(len(X_pca_3d), sample_size, replace=False)
X_pca_3d_sampled = X_pca_3d[indices, :]
y_ridge = y_ridge[indices]

# Create the 3D scatter plot using Plotly with WebGL
fig = go.Figure(data=[go.Scatter3d(
    x=X_pca_3d_sampled[:, 0],  # PC1
    y=X_pca_3d_sampled[:, 1],  # PC2
    z=X_pca_3d_sampled[:, 2],  # PC3
    mode='markers',
    marker=dict(
        size=2,  # Reduce size for better performance
        color=y_ridge,  # Color based on the target variable
        colorscale='icefire',  # Red-to-blue colormap
        colorbar=dict(title="Target"),
        showscale=True,
        opacity=0.8  # Slight transparency for better visibility
    )
)])

# Update layout for better visualization and add axis labels
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

# Show the plot
fig.show()
