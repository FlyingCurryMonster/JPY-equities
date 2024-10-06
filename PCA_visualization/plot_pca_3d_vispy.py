# plot_pca_3d_vispy.py

import pickle
import os
import numpy as np
from vispy import scene
from vispy import use
from vispy.scene import visuals
from vispy.color import Colormap

# Ensure VisPy uses the PyQt5 backend (you can try others if needed)
use('PyQt5')

# Define the folder and file paths for the pickled variables
export_folder = 'PCA_visualization'
X_pca_3d_path = os.path.join(export_folder, 'X_pca_3d.pkl')
y_ridge_path = os.path.join(export_folder, 'y_ridge.pkl')

# Load the pickled variables
with open(X_pca_3d_path, 'rb') as f:
    X_pca_3d = pickle.load(f)

with open(y_ridge_path, 'rb') as f:
    y_ridge = pickle.load(f)

# Step 1: Convert y_ridge to NumPy array and normalize
y_ridge = np.array(y_ridge)
y_ridge_norm = (y_ridge - np.min(y_ridge)) / (np.max(y_ridge) - np.min(y_ridge))

# Step 2: Set up VisPy 3D canvas
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(800, 600))
grid = canvas.central_widget.add_grid()
view = grid.add_view(row=0, col=1, camera='arcball')  # Arcball camera for rotation
view.camera.fov = 45  # Set field of view to see axes better

# Step 3: Create the colormap object
cmap = Colormap(['blue', 'red'])

# Step 4: Map the normalized target values to colors using the colormap
colors = cmap.map(y_ridge_norm)

# Step 5: Create scatter plot
scatter = visuals.Markers()
scatter.set_data(X_pca_3d, face_color=colors, size=5)

# Add scatter plot to the scene
view.add(scatter)

# Step 6: Configure the XYZ axes with labels and ticks
axis = visuals.XYZAxis(parent=view.scene)
axis.transform = scene.STTransform(translate=(0, 0, 0), scale=(1, 1, 1))

# Add labels for each axis (ensure pos is a 2D tuple)
x_axis_label = scene.Label('PC1', color='white', font_size=10, anchor_x='center')
y_axis_label = scene.Label('PC2', color='white', font_size=10, anchor_x='center')
z_axis_label = scene.Label('PC3', color='white', font_size=10, anchor_x='center')

# Position the axis labels near the plot, using 2D tuples for pos
x_axis_label.pos = (canvas.size[0] * 0.9, canvas.size[1] * 0.1)  # Adjust position
y_axis_label.pos = (canvas.size[0] * 0.1, canvas.size[1] * 0.9)
z_axis_label.pos = (canvas.size[0] * 0.5, canvas.size[1] * 0.9)

view.add(x_axis_label)
view.add(y_axis_label)
view.add(z_axis_label)

# Step 7: Add a color bar (Legend for target variable) wrapped inside a Widget
colorbar = scene.visuals.ColorBar(cmap=cmap, label="Target (beta_target_45)", orientation='right', size=(500, 50))
colorbar_widget = scene.Widget()
colorbar_widget._add_child(colorbar)
grid.add_widget(colorbar_widget, row=0, col=2)  # Position colorbar next to the plot

# Step 8: Run the VisPy app to visualize the plot
if __name__ == '__main__':
    canvas.app.run()
