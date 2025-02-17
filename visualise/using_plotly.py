import nibabel as nib
import numpy as np
import plotly.graph_objects as go

# Load the NIfTI file
file_path = "liver_1.nii"  # Change to your file path
nii_img = nib.load(file_path)  # Load NIfTI file
data = nii_img.get_fdata()  # Convert to NumPy array

# Get coordinates of segmented regions
x, y, z = np.where(data > 0)  # Extract only segmented regions

# Create a 3D scatter plot
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=1, color=z, colorscale="Viridis")
        )
    ]
)

fig.update_layout(
    title="3D Segmented Liver and Tumor",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
)

fig.show()
